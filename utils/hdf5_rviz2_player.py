import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header, Bool
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time
import struct
import threading
import time


class HDF5Player(Node):
    def __init__(self):
        super().__init__('hdf5_player')

        # Publisher
        self.lidar_publisher = self.create_publisher(PointCloud2, '/lidar_points', 10)
        self.uwb_publisher = self.create_publisher(PointStamped, '/uwb_position', 10)
        self.uwb_publisher_filtered = self.create_publisher(PointStamped, '/uwb_position_filtered', 10)
        self.position_publisher = self.create_publisher(PointStamped, '/ground_truth_position', 10)
        self.presence_publisher = self.create_publisher(Bool, '/driver_presence', 10)

        # Timer für die Wiedergabe
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        # Datenindex für die Wiedergabe
        self.current_index = 0
        self.playing = False
        self.loop = True  # Endlose Wiedergabe

        # load HDF5 Data
        #self.hdf5_file = 'dataset/test/dataset_val_cw.hdf5'
        #self.hdf5_file = 'dataset/test/dataset_val_ccw.hdf5'
        self.hdf5_file = '../dataset/test/dataset_val_line.hdf5'
        #self.hdf5_file = 'dataset/train_val_test/dataset_rosbag2_2025_06_17-11_07_48.hdf5'

        self.load_data()

        self.get_logger().info(f"HDF5 Player initialized. Loaded {self.total_samples} samples.")
        self.get_logger().info("Use 'ros2 service call /start_playback std_srvs/srv/Trigger' to start playback")

        # Service for start/stop
        self.start_service = self.create_service(Trigger, 'start_playback', self.start_playback_callback)
        self.stop_service = self.create_service(Trigger, 'stop_playback', self.stop_playback_callback)

    def load_data(self):
        try:
            self.hdf5 = h5py.File(self.hdf5_file, 'r')

            self.lidar_data = self.hdf5['lidar_data']
            self.uwb_data = self.hdf5['uwb_data']
            self.uwb_data_filtered = self.hdf5['uwb_data_filtered']
            self.position_data = self.hdf5['ground_truth']
            self.presence_data = self.hdf5['driver_present']

            # if all data have same length
            self.total_samples = len(self.lidar_data)

            self.get_logger().info(f"Loaded HDF5 data:")
            self.get_logger().info(f"  - Lidar data shape: {self.lidar_data.shape}")
            self.get_logger().info(f"  - UWB data shape: {self.uwb_data.shape}")
            self.get_logger().info(f"  - UWB data filterd shape: {self.uwb_data_filtered.shape}")
            self.get_logger().info(f"  - Position data shape: {self.position_data.shape}")
            self.get_logger().info(f"  - Presence data shape: {self.presence_data.shape}")

        except Exception as e:
            self.get_logger().error(f"Error loading HDF5 file: {e}")
            self.total_samples = 0

    def start_playback_callback(self, request, response):
        self.playing = True
        self.current_index = 0
        response.success = True
        response.message = "Playback started"
        self.get_logger().info("Playback started")
        return response

    def stop_playback_callback(self, request, response):
        self.playing = False
        response.success = True
        response.message = "Playback stopped"
        self.get_logger().info("Playback stopped")
        return response

    def timer_callback(self):
        if not self.playing or self.total_samples == 0:
            return

        if self.current_index >= self.total_samples:
            if self.loop:
                self.current_index = 0
                self.get_logger().info("Restarting playback from beginning")
            else:
                self.playing = False
                self.get_logger().info("Playback finished")
                return

        current_time = self.get_clock().now().to_msg()

        try:
            # publish LiDAR Data
            self.publish_lidar_data(self.current_index, current_time)

            # publish UWB Data
            self.publish_uwb_data(self.current_index, current_time)

            # publish filtered UWB Data
            self.publish_uwb_filtered_data(self.current_index, current_time)

            # publish gt Data
            self.publish_position_data(self.current_index, current_time)

            # publish Presence Data
            self.publish_presence_data(self.current_index, current_time)


            self.current_index += 1

            if self.current_index % 50 == 0:  # Alle 5 Sekunden bei 10Hz
                self.get_logger().info(f"Playing frame {self.current_index}/{self.total_samples}")

        except Exception as e:
            self.get_logger().error(f"Error publishing data at index {self.current_index}: {e}")

    def publish_lidar_data(self, index, timestamp):
        try:
            lidar_points = self.lidar_data[index]  # (4500, 3)

            # PointCloud2 Message
            msg = PointCloud2()
            msg.header = Header()
            msg.header.stamp = timestamp
            msg.header.frame_id = "base_link"

            msg.height = 1
            msg.width = len(lidar_points)

            # Point fields (x, y, z)
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]

            msg.is_bigendian = False
            msg.point_step = 12  # 3 * 4 bytes
            msg.row_step = msg.point_step * msg.width

            # convert data to bytes
            buffer = []
            for point in lidar_points:
                buffer.extend(struct.pack('fff', float(point[0]), float(point[1]), float(point[2])))

            msg.data = buffer
            msg.is_dense = True

            self.lidar_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing lidar data: {e}")

    def publish_uwb_data(self, index, timestamp):
        try:
            uwb_point = self.uwb_data[index]

            msg = PointStamped()
            msg.header = Header()
            msg.header.stamp = timestamp
            msg.header.frame_id = "base_link"

            # UWB Data (x, y)
            msg.point.x = float(uwb_point[0])
            msg.point.y = float(uwb_point[1])
            msg.point.z = 0.0  # 2D Data, z = 0

            self.uwb_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing UWB data: {e}")

    def publish_uwb_filtered_data(self, index, timestamp):
        try:
            uwb_point = self.uwb_data_filtered[index]

            msg = PointStamped()
            msg.header = Header()
            msg.header.stamp = timestamp
            msg.header.frame_id = "base_link"

            # UWB Data (x, y)
            msg.point.x = float(uwb_point[0])
            msg.point.y = float(uwb_point[1])
            msg.point.z = 0.0  # 2D Data, z = 0

            self.uwb_publisher_filtered.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing UWB data: {e}")

    def publish_position_data(self, index, timestamp):
        try:
            position = self.position_data[index]

            msg = PointStamped()
            msg.header = Header()
            msg.header.stamp = timestamp
            msg.header.frame_id = "base_link"

            msg.point.x = float(position[0])
            msg.point.y = float(position[1])
            msg.point.z = 0.0  # 2D Daten, z auf 0

            self.position_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing position data: {e}")

    def publish_presence_data(self, index, timestamp):
        try:
            presence = self.presence_data[index]

            msg = Bool()
            msg.data = bool(presence)

            self.presence_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing presence data: {e}")

    def destroy_node(self):
        """clean after stop"""
        if hasattr(self, 'hdf5'):
            self.hdf5.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = HDF5Player()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()