import os
import shutil
import rclpy
import numpy as np
import rosbag2_py
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import serialize_message, deserialize_message
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


class InteractiveBagLabeling(Node):
    def __init__(self):
        super().__init__('interactive_bag_labeling')

        # Subscribe to clicked points in RViz2
        self.subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.click_callback,
            10)

        # Subscribe to 2D goal pose to stop labeling
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.stop_labeling_callback,
            10)

        # Publishers to visualize data in RViz2
        self.lidar_publisher = self.create_publisher(PointCloud2, '/lidar_cluster_points', 10)
        self.tracking_publisher = self.create_publisher(PointStamped, '/deliveryman_tracking', 10)
        self.tracking_marvel_publisher = self.create_publisher(PointStamped, '/deliveryman_tracking_marvelmind', 10)

        # Read bag file
        self.input_bag = "/media/afius/Data1/0-rosbags_raw/rosbag2_2025_06_17-11_07_48"
        self.output_bag = "labeled_rosbag/rosbag2_2025_03_26-15_20_55"
        self.delete_existing_bag(self.output_bag)

        self.reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.input_bag, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions()
        self.reader.open(storage_options, converter_options)
        self.topic_types = self.reader.get_all_topics_and_types()

        self.new_topic_name = "/deliveryman_tracking_marvelmind"
        self.clicked_positions = []
        self.labeling_active = True
        self.delay = 10
        self.t_buffer = np.zeros((10))
        self.counter = 0
        self.old_pos_x = None
        self.x_offset = -0.64
        self.old_pos_y = None
        self.header = None

        # Initialize writer
        self.writer = rosbag2_py.SequentialWriter()
        writer_storage_options = rosbag2_py.StorageOptions(uri=self.output_bag, storage_id="sqlite3")
        writer_converter_options = rosbag2_py.ConverterOptions()
        self.writer.open(writer_storage_options, writer_converter_options)

        existing_topics = {topic.name for topic in self.topic_types}
        for topic in self.topic_types:
            if topic.name != self.new_topic_name:
                self.writer.create_topic(rosbag2_py.TopicMetadata(
                    name=topic.name, type=topic.type, serialization_format="cdr"
                ))
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.new_topic_name, type="geometry_msgs/msg/PointStamped", serialization_format="cdr"
        ))

        self.process_bag_messages()

    def delete_existing_bag(self, bag_path):
        if os.path.exists(bag_path):
            print(f"Deleting existing ROS2-Bag: {bag_path}")
            shutil.rmtree(bag_path)


    def stop_labeling_callback(self, msg):
        """Stops labeling when a 2D goal pose is received."""
        self.labeling_active = False
        self.get_logger().info("2D Goal Pose received. Stopping labeling and saving bag...")

    def process_bag_messages(self):
        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()

            if topic != self.new_topic_name:
                self.writer.write(topic, data, timestamp)

            if topic == '/lidar_cluster_points':
                msg = deserialize_message(data, PointCloud2)
                self.lidar_publisher.publish(msg)
            elif topic == self.new_topic_name:
                msg = deserialize_message(data, PointStamped)
                self.header = msg.header
                self.tracking_marvel_publisher.publish(msg)
            elif topic == '/deliveryman_tracking':
                msg = deserialize_message(data, PointStamped)
                self.header = msg.header
                self.tracking_publisher.publish(msg)
                self.t_buffer[self.counter % self.delay] = timestamp / 1e9
                self.counter += 1

                if self.counter % self.delay == 0:
                    self.get_logger().info("Bag paused. Click in RViz to label.")
                    rclpy.spin_once(self, timeout_sec=None)

    def click_callback(self, msg):
        """Handles clicked points from RViz and interpolates between tracking timestamps."""
        if not self.labeling_active:
            return

        x, y = msg.point.x, msg.point.y

        if self.old_pos_x is None:
            self.old_pos_x, self.old_pos_y = x, y
            self.write_msgs(self.t_buffer[-1], x, y)
        else:
            for i in range(1, self.delay - 1):
                t_interp = self.t_buffer[i]
                x_int, y_int = self.interpolate_positions(
                    (self.old_pos_x, self.old_pos_y),
                    (x, y),
                    self.t_buffer[0],
                    self.t_buffer[-1],
                    t_interp
                )
                self.write_msgs(t_interp, x_int, y_int)
            self.write_msgs(self.t_buffer[-1], x, y)
            self.old_pos_x, self.old_pos_y = x, y

        self.get_logger().info("Paused again. Click in RViz for next position.")

    def write_msgs(self, timestamp, x, y):
        """Writes labeled positions with tracking timestamps."""
        label_msg = PointStamped()
        label_msg.header = self.header
        label_msg.header.stamp = Time(sec=int(timestamp), nanosec=int((timestamp - int(timestamp)) * 1e9))
        label_msg.point.x = x + self.x_offset
        label_msg.point.y = y
        label_msg.point.z = 0.0
        self.writer.write(self.new_topic_name, serialize_message(label_msg), int(timestamp * 1e9))
        self.get_logger().info(f"Saved labeled point: x={x}, y={y} at t={timestamp}")

    def interpolate_positions(self, pos1, pos2, t1, t2, t):
        x = pos1[0] + (t - t1) / (t2 - t1) * (pos2[0] - pos1[0])
        y = pos1[1] + (t - t1) / (t2 - t1) * (pos2[1] - pos1[1])
        return (x, y)


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveBagLabeling()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()