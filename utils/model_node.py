#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
import tensorflow as tf
import numpy as np
from collections import deque
import struct
import threading
from builtin_interfaces.msg import Time

# Import your modules
import model as model_factory


# fasten inference speed
@tf.function
def infer(model, X_input):
    return model(X_input, training=False)


class ModelPredictionNode(Node):
    def __init__(self):
        super().__init__('model_prediction_node')

        # Model parameters - should match your training configuration
        self.grid_size = 500
        self.grid_resolution = 0.1
        self.seq_length = 10
        self.height_encoding = 'single_height_map'

        # Model setup
        self.model_path = ("saved_models/best_models/adaptive_fused_model/adaptive_fused_model.keras")
        self.model = None
        self.load_model()

        # Data buffers for sequence processing
        self.lidar_buffer = deque(maxlen=self.seq_length)
        self.uwb_buffer = deque(maxlen=self.seq_length)
        self.timestamp_buffer = deque(maxlen=self.seq_length)

        # Thread lock for data synchronization
        self.data_lock = threading.Lock()

        # Subscribers
        self.lidar_sub = self.create_subscription(PointCloud2, '/lidar_points', self.lidar_callback, 10)
        self.uwb_sub = self.create_subscription(PointStamped, '/uwb_position', self.uwb_callback, 10)

        # Publisher
        self.prediction_pub = self.create_publisher(PointStamped, '/predicted_position', 10)

        # Statistics
        self.prediction_count = 0
        self.last_prediction_time = None

        # compute grid range
        half_size = (self.grid_size * self.grid_resolution) / 2
        self.range_x = [-half_size, half_size]
        self.range_y = [-half_size, half_size]

        # Height encoding parameters
        self.ground_height_threshold = 0.2
        self.max_height = 3.0
        self.min_height = -1.0

        self.get_logger().info("Model Prediction Node initialized")
        self.get_logger().info(f"Grid size: {self.grid_size}x{self.grid_size}")
        self.get_logger().info(f"Sequence length: {self.seq_length}")
        self.get_logger().info(f"Height_encoding encoding: {self.height_encoding}")

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "TemporalAttention": model_factory.TemporalAttention,
                    "ResidualDKF": model_factory.ResidualDKF,
                    "AdaptiveSensorFusionDKF": model_factory.AdaptiveSensorFusionDKF,
                },
                compile=False,
                safe_mode=False
            )
            # Show the model architecture
            self.model.summary()

        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
            self.model = None

    def _get_num_channels(self):
            return 1

    def lidar_callback(self, msg):
        """Process incoming LiDAR data"""
        try:
            # Extract points from PointCloud2
            points = self.extract_points_from_pointcloud2(msg)

            with self.data_lock:
                self.lidar_buffer.append({
                    'points': points,
                    'timestamp': msg.header.stamp
                })

        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")

    def uwb_callback(self, msg):
        """Process incoming UWB data"""
        try:
            with self.data_lock:
                self.uwb_buffer.append({
                    'x': msg.point.x,
                    'y': msg.point.y,
                    'timestamp': msg.header.stamp
                })

            # predict position
            self.predict_position()

        except Exception as e:
            self.get_logger().error(f"Error processing UWB data: {e}")

    def extract_points_from_pointcloud2(self, msg):

        """Extract 3D points from PointCloud2 message"""
        points = []

        # Parse the binary data
        for i in range(msg.width):
            offset = i * msg.point_step
            x = struct.unpack('f', msg.data[offset:offset + 4])[0]
            y = struct.unpack('f', msg.data[offset + 4:offset + 8])[0]
            z = struct.unpack('f', msg.data[offset + 8:offset + 12])[0]
            points.append([x, y, z])

        return np.array(points)

    def create_enhanced_grid(self, lidar_points):

        mask = (lidar_points[:, 0] >= self.range_x[0]) & (lidar_points[:, 0] < self.range_x[1]) & \
               (lidar_points[:, 1] >= self.range_y[0]) & (lidar_points[:, 1] < self.range_y[1])
        valid_pts = lidar_points[mask]

        if len(valid_pts) == 0:
            return np.zeros((self.grid_size, self.grid_size, self.num_channels), dtype=np.float32)

        x_idx = ((valid_pts[:, 0] - self.range_x[0]) / self.grid_resolution).astype(np.int32)
        y_idx = ((valid_pts[:, 1] - self.range_y[0]) / self.grid_resolution).astype(np.int32)

        x_idx = np.clip(x_idx, 0, self.grid_size - 1)
        y_idx = np.clip(y_idx, 0, self.grid_size - 1)
        z_vals = valid_pts[:, 2]

        flat_idx = y_idx * self.grid_size + x_idx

        height_flat = np.full(self.grid_size * self.grid_size, self.min_height, dtype=np.float32)
        np.maximum.at(height_flat, flat_idx, z_vals)

        height_norm = np.clip((height_flat - self.min_height) / (self.max_height - self.min_height), 0, 1)

        occupied = np.zeros_like(height_flat, dtype=bool)
        occupied[flat_idx] = True

        final_h = height_norm.reshape(self.grid_size, self.grid_size)
        final_h[~occupied.reshape(self.grid_size, self.grid_size)] = 0

        return final_h[..., np.newaxis]






    def position_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        return int(x // self.grid_resolution + self.grid_size // 2), \
            int(y // self.grid_resolution + self.grid_size // 2)

    def predict_position(self):
        """Main prediction function called by uwb subscriber"""
        if self.model is None:
            return

        try:
            with self.data_lock:
                # Check if we have enough data for prediction
                if len(self.lidar_buffer) < self.seq_length or len(self.uwb_buffer) < self.seq_length:
                    return

                # Create sequence of grids and uwb
                grids = []
                for i in range(self.seq_length):
                    lidar_data = self.lidar_buffer[i]
                    grid = self.create_enhanced_grid(lidar_data['points'])
                    grids.append(grid)

                # Stack grids: (grid_size, grid_size, num_channels, seq_length) and uwb
                grid_sequence = np.stack(grids, axis=-1)

                # Add batch dimension: (1, grid_size, grid_size, num_channels, seq_length)
                grid_input = np.expand_dims(grid_sequence, axis=0).astype(np.float32)

                # Create UWB sequence: (seq_length, 2)
                uwb_sequence = np.array([[uwb_data['x'], uwb_data['y']]
                                         for uwb_data in self.uwb_buffer], dtype=np.float32)

                # Add batch dimension: (1, seq_length, 2)
                uwb_input = np.expand_dims(uwb_sequence, axis=0)

                # Create input dictionary matching model's expected format
                input_dict = {
                    'grid_input': grid_input,
                    'uwb_input': uwb_input
                }



            # Make prediction
            #prediction = self.model(input_dict, training=False)
            prediction = infer(self.model, input_dict)

            #pred_x, pred_y = prediction[0][0].numpy(), prediction[0][1].numpy()
            pred_x, pred_y = prediction['position'][0][0].numpy(), prediction['position'][0][1].numpy()

            self.publish_prediction(pred_x, pred_y)

            # Update statistics
            self.prediction_count += 1
            self.last_prediction_time = self.get_clock().now()

            if self.prediction_count % 50 == 0:
                self.get_logger().info(f"Published {self.prediction_count} predictions")

        except Exception as e:
            self.get_logger().error(f"Error during prediction: {e}")

    def publish_prediction(self, x, y):
        """Publish the predicted position"""
        try:
            # Create prediction message
            pred_msg = PointStamped()
            pred_msg.header.stamp = self.get_clock().now().to_msg()
            pred_msg.header.frame_id = "base_link"
            pred_msg.point.x = float(x)
            pred_msg.point.y = float(y)
            pred_msg.point.z = 0.0

            self.prediction_pub.publish(pred_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing prediction: {e}")


def main(args=None):
    # Set up GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

    rclpy.init(args=args)

    try:
        node = ModelPredictionNode()

        # Inform user about the topics
        node.get_logger().info("=== Topic Information ===")
        node.get_logger().info("Subscribed to:")
        node.get_logger().info("  - /lidar_points (sensor_msgs/PointCloud2)")
        node.get_logger().info("  - /uwb_position (geometry_msgs/PointStamped)")
        node.get_logger().info("Publishing to:")
        node.get_logger().info("  - /predicted_position (geometry_msgs/PointStamped)")
        node.get_logger().info("========================")

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()