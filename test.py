import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
from collections import deque
import os
from tqdm import tqdm
import data_generator as dataloader

# ==== Projektimports (anpassen falls Pfade anders) ====
import model as model_factory

# Externe Abhängigkeit: filterpy für den Unscented Kalman Filter
try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
except Exception as e:
    raise ImportError("Benötigt 'filterpy' (pip install filterpy). Fehler: {}".format(e))

# fasten inference speed
@tf.function
def infer(model, X_input):
    return model(X_input, training=False)


# ------------------ Konstant-beschleunigtes UKF-Modell ------------------
# Zustand: x = [px, py, vx, vy, ax, ay]
# Messung: z = [px, py]

def fx_ca(x, dt):
    """State transition for constant acceleration model."""
    px, py, vx, vy, ax, ay = x
    px_new = px + vx * dt + 0.5 * ax * (dt ** 2)
    py_new = py + vy * dt + 0.5 * ay * (dt ** 2)
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    ax_new = ax
    ay_new = ay
    return np.array([px_new, py_new, vx_new, vy_new, ax_new, ay_new])


def hx_ca(x):
    """Measurement function: return position only."""
    return np.array([x[0], x[1]])


class UWBUnscentedFilterCA:
    """Wrapper around filterpy UnscentedKalmanFilter with constant-acceleration model.

    - dim_x = 6 (px,py,vx,vy,ax,ay)
    - dim_z = 2 (px,py)
    """
    def __init__(self, dt=0.1,
                 R_std=0.3,
                 proc_var_pos=0.01,
                 proc_var_vel=0.1,
                 proc_var_acc=1e-3):
        self.dt = dt
        self.dim_x = 6
        self.dim_z = 2

        # Sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UnscentedKalmanFilter(dim_x=self.dim_x,
                                         dim_z=self.dim_z,
                                         dt=dt,
                                         fx=lambda x, dt: fx_ca(x, dt),
                                         hx=lambda x: hx_ca(x),
                                         points=points)

        # Initial state (will be initialized on first measurement)
        self.ukf.x = np.zeros(self.dim_x)
        self.ukf.P = np.eye(self.dim_x) * 1.0

        # Measurement noise
        self.ukf.R = np.diag([R_std**2, R_std**2])

        # Process noise Q: tuned per sub-block
        Q = np.zeros((self.dim_x, self.dim_x))
        Q[0:2, 0:2] = np.eye(2) * proc_var_pos
        Q[2:4, 2:4] = np.eye(2) * proc_var_vel
        Q[4:6, 4:6] = np.eye(2) * proc_var_acc
        self.ukf.Q = Q

        self.initialized = False

    def reset(self):
        self.ukf.x = np.zeros(self.dim_x)
        self.ukf.P = np.eye(self.dim_x) * 1.0
        self.initialized = False

    def update(self, meas):
        """Update UKF with a new measurement meas = [px, py].

        Returns filtered position [px, py].
        """
        meas = np.asarray(meas, dtype=np.float64)
        if not self.initialized:
            # Initialize position; velocities and acc remain zero
            self.ukf.x[0:2] = meas
            self.initialized = True
            return self.ukf.x[0:2].copy()

        # Predict + update
        self.ukf.predict()
        self.ukf.update(meas)
        return self.ukf.x[0:2].copy()


# ------------------ OfflineModelEvaluator (überarbeitet) ------------------
class OfflineModelEvaluator:
    def __init__(self, test_path, model_path,
                 grid_size=300, grid_resolution=0.1,
                 seq_length=10, vel_output=True, ukf_dt=0.1):

        self.test_path = test_path
        self.model_path = model_path
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.seq_length = seq_length
        self.vel_output = vel_output

        # Höhen- / Bodenschwellen
        self.min_height = -1.0
        self.max_height = 3.0
        self.ground_height_threshold = 0.2

        # UKF für UWB (konstant-beschleunigt)
        self.ukf = UWBUnscentedFilterCA(dt=ukf_dt)

        # Daten & Modell laden
        self._load_hdf5()
        self._load_model()

    # ----------------------------------------------------
    #  Daten laden
    # ----------------------------------------------------
    def _load_hdf5(self):
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"HDF5-Datei nicht gefunden: {self.test_path}")
        self.hdf5 = h5py.File(self.test_path, 'r')
        self.lidar_data = self.hdf5['lidar_data']
        self.uwb_data = self.hdf5['uwb_data']
        self.driver_present = self.hdf5['driver_present']

        # falls gefiltertes uwb schon im hdf5 vorhanden ist
        self.uwb_data_filtered = self.hdf5.get('uwb_data_filtered', self.uwb_data)
        self.position_data = self.hdf5['ground_truth']

        self.num_samples = len(self.lidar_data)
        print(f"✅ Loaded HDF5: {self.num_samples} frames")

    # ----------------------------------------------------
    #  Modell laden
    # ----------------------------------------------------
    def _load_model(self):
        print("Loading model...")
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
        print("✅ Model loaded successfully")
        try:
            self.model.summary()
        except Exception:
            pass

    def create_grid(self, lidar_points):
        return self._create_height_map_grid(lidar_points)

    def _create_height_map_grid(self, lidar_points):
        if len(lidar_points) == 0:
            return np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)

        # Berechne Grenzen (wie DataGenerator)
        half_size = (self.grid_size * self.grid_resolution) / 2
        range_x = [-half_size, half_size]
        range_y = [-half_size, half_size]

        # Filtere gültige Punkte
        mask = (lidar_points[:, 0] >= range_x[0]) & (lidar_points[:, 0] < range_x[1]) & \
               (lidar_points[:, 1] >= range_y[0]) & (lidar_points[:, 1] < range_y[1])
        valid_pts = lidar_points[mask]

        if len(valid_pts) == 0:
            return np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)

        # Berechne Indizes (EXAKT wie DataGenerator)
        x_idx = ((valid_pts[:, 0] - range_x[0]) / self.grid_resolution).astype(np.int32)
        y_idx = ((valid_pts[:, 1] - range_y[0]) / self.grid_resolution).astype(np.int32)

        x_idx = np.clip(x_idx, 0, self.grid_size - 1)
        y_idx = np.clip(y_idx, 0, self.grid_size - 1)
        z_vals = valid_pts[:, 2]

        # Flatten Index (WICHTIG: y * width + x wie im DataGenerator!)
        flat_idx = y_idx * self.grid_size + x_idx

        # Height Map mit Maximum-Aggregation
        height_flat = np.full(self.grid_size * self.grid_size, self.min_height, dtype=np.float32)
        np.maximum.at(height_flat, flat_idx, z_vals)

        # Normalisierung
        height_norm = np.clip(
            (height_flat - self.min_height) / (self.max_height - self.min_height),
            0, 1
        )

        # Occupancy Mask
        occupied = np.zeros_like(height_flat, dtype=bool)
        occupied[flat_idx] = True

        # Reshape und unoccupied auf 0 setzen
        final_h = height_norm.reshape(self.grid_size, self.grid_size)
        final_h[~occupied.reshape(self.grid_size, self.grid_size)] = 0

        return final_h[..., np.newaxis]  # Shape: (H, W, 1)


    # ----------------------------------------------------
    #  Inferenz
    # ----------------------------------------------------
    def run_inference(self):
        preds = []
        gts = []
        uwbs = []
        uwbs_filterd = []
        uwbs_ukf = []
        driver_present_array = []

        lidar_buffer = deque(maxlen=self.seq_length)
        uwb_buffer = deque(maxlen=self.seq_length)

        # Reset UKF zwischen Läufen
        self.ukf.reset()

        print("Running model inference...")
        for i in tqdm(range(self.num_samples)):
            lidar_points = self.lidar_data[i]
            uwb_point = self.uwb_data[i]
            uwb_point_filtered = self.uwb_data_filtered[i]
            gt_point = self.position_data[i]

            lidar_buffer.append(lidar_points)
            uwb_buffer.append(uwb_point)

            if len(lidar_buffer) < self.seq_length:
                # noch nicht genug für eine Sequenz
                continue

            # Grid sequence
            grids = [self.create_grid(l) for l in lidar_buffer]
            grid_input = np.stack(grids, axis=-1)[np.newaxis, ...]  # (1, H, W, C, T)

            # UWB sequence
            uwb_input = np.array(uwb_buffer, dtype=np.float32)[np.newaxis, ...]

            input_dict = {'grid_input': grid_input, 'uwb_input': uwb_input}

            # Modell-Inferenz
            pred = infer(self.model, input_dict)

            # check if model predicts velocity and position or position only
            if self.vel_output:
                try:
                    pred_x = float(pred['position'][0][0].numpy())
                    pred_y = float(pred['position'][0][1].numpy())
                except Exception:
                    # Fallback falls Struktur anders
                    pred_x, pred_y = float(pred[0][0].numpy()), float(pred[0][1].numpy())
            else:
                pred_x, pred_y = float(pred[0][0].numpy()), float(pred[0][1].numpy())

            preds.append([pred_x, pred_y])
            uwbs.append([uwb_point[0], uwb_point[1]])
            uwbs_filterd.append([uwb_point_filtered[0], uwb_point_filtered[1]])
            gts.append([gt_point[0], gt_point[1]])
            driver_present_array.append(self.driver_present[i])

            # === UKF Filtering (constant acceleration) ===
            uwb_ukf_pos = self.ukf.update([uwb_point[0], uwb_point[1]])
            uwbs_ukf.append([uwb_ukf_pos[0], uwb_ukf_pos[1]])

        # Ergebnisse als numpy arrays
        self.preds = np.array(preds)
        self.uwbs = np.array(uwbs)
        self.uwbs_filtered = np.array(uwbs_filterd)
        self.uwbs_ukf = np.array(uwbs_ukf)
        self.gts = np.array(gts)
        self.driver_present_array = np.array(driver_present_array)
        self.mask_lidar_and_uwb = self.driver_present_array == 1
        self.mask_uwb_only = self.driver_present_array == 0

        print("✅ Inference complete")

    def baseline_metrics_and_inference(self, dataset_folder):
        # Validation Generator
        test_gen = dataloader.DataGenerator(
            hdf5_folder=dataset_folder,
            grid_size=self.grid_size,
            grid_resolution=self.grid_resolution,
            seq_length=self.seq_length,
            batch_size=1,
            mode='test',
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            use_velocity_auxiliary=True
        )
        print("\n=== Metrics with Model Predictions ===")
        results = test_gen.compute_baseline_metrics(
            model=self.model,
            vel_output=self.vel_output  # True wenn Model Position + Velocity ausgibt
        )

        # Falls Timing-Stats zurückgegeben wurden
        if isinstance(results, tuple):
            metrics, timing_stats = results
            print("\n📊 Inference Timing Summary:")
            print(f"  Inference FPS: {timing_stats['inference']['fps']:.1f} Hz")
            print(f"  Total FPS (incl. preprocessing): {timing_stats['total_per_sample']['fps']:.1f} Hz")

    # ----------------------------------------------------
    #  Visualisierung
    # ----------------------------------------------------
    def _draw_sensor_platform(self, ax):
        """Draw sensor platform rectangle and direction triangle on given axes."""
        vehicle_length = 3.40  # 340 cm
        vehicle_width = 1.125  # 112.5 cm

        vehicle_x_rear = -2.76
        vehicle_x_front = 0.64
        vehicle_y_bottom = -vehicle_width / 2
        vehicle_y_top = vehicle_width / 2

        # Draw sensor platform rectangle
        vehicle_rect = Rectangle(
            (vehicle_x_rear, vehicle_y_bottom),
            vehicle_length,
            vehicle_width,
            linewidth=2.5,
            edgecolor='darkgreen',
            facecolor='white',
            alpha=0.3,
            zorder=10
        )
        ax.add_patch(vehicle_rect)

        # Draw sensor platform triangle (x-direction indicator)
        arrow_length = vehicle_width
        arrow_width = vehicle_width
        arrow_x_center = vehicle_x_front - arrow_length / 2

        triangle = Polygon(
            [
                [arrow_x_center + arrow_length / 2, 0],  # front
                [arrow_x_center - arrow_length / 2, arrow_width / 2],  # top edge
                [arrow_x_center - arrow_length / 2, -arrow_width / 2]  # bottom edge
            ],
            closed=True,
            linewidth=1.5,
            edgecolor='darkgreen',
            facecolor='white',
            alpha=0.6,
            zorder=10
        )
        ax.add_patch(triangle)

        return vehicle_x_rear, vehicle_x_front, vehicle_y_bottom, vehicle_y_top

    def _plot_trajectory_segments(self, ax, positions, driver_present):
        """Plot trajectory with color-coded segments based on sensor availability."""
        i = 0
        has_lidar = False
        has_uwb_only = False

        while i < len(positions):
            current_state = driver_present[i]
            j = i
            while j < len(positions) and driver_present[j] == current_state:
                j += 1

            # Include one extra point for smooth transition (if available)
            end_idx = min(j + 1, len(positions))
            segment = positions[i:end_idx]

            if current_state == 1:  # UWB + LiDAR
                ax.plot(segment[:, 0], segment[:, 1], 'b-', linewidth=1.5, zorder=5)
                has_lidar = True
            else:  # UWB only
                ax.plot(segment[:, 0], segment[:, 1], 'm-', linewidth=1.5, zorder=5)
                has_uwb_only = True

            i = j

        return has_lidar, has_uwb_only

    def plot_results(self):
        if not hasattr(self, 'preds'):
            raise RuntimeError("Run inference first")

        # === Plot 0: Validation pattern ===
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw sensor platform
        veh_x_rear, veh_x_front, veh_y_bottom, veh_y_top = self._draw_sensor_platform(ax)

        # Plot ground truth with segments
        has_lidar_gt, has_uwb_gt = self._plot_trajectory_segments(ax, self.gts, self.driver_present_array)

        # Legend
        legend_elements = [
            Line2D([0], [0], color='darkgreen', linewidth=2.5, label='Sensor platform (3.40m × 1.13m)')
        ]
        if has_lidar_gt:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2, label='Ground Truth (UWB + LiDAR)'))
        if has_uwb_gt:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2, label='Ground Truth (UWB only)'))

        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_title("Ground Truth", fontsize=14)
        ax.axis("equal")
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Adjust limits
        x_min = min(self.gts[:, 0].min(), self.preds[:, 0].min(), veh_x_rear - 0.5)
        x_max = max(self.gts[:, 0].max(), self.preds[:, 0].max(), veh_x_front + 0.5)
        y_min = min(self.gts[:, 1].min(), self.preds[:, 1].min(), veh_y_bottom - 0.5)
        y_max = max(self.gts[:, 1].max(), self.preds[:, 1].max(), veh_y_top + 0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()


        # === Plot 1: Ground Truth vs Prediction ===
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw sensor platform
        veh_x_rear, veh_x_front, veh_y_bottom, veh_y_top = self._draw_sensor_platform(ax)

        # Plot ground truth with segments
        has_lidar_gt, has_uwb_gt = self._plot_trajectory_segments(ax, self.gts, self.driver_present_array)

        # Plot prediction
        ax.plot(self.preds[:, 0], self.preds[:, 1], 'r--', linewidth=2, label='Prediction', zorder=6)

        # Legend
        legend_elements = [
            Line2D([0], [0], color='darkgreen', linewidth=2.5, label='Sensor platform (3.40m × 1.13m)'),
            Line2D([0], [0], color='r', linewidth=2, linestyle='--', label='Prediction')
        ]
        if has_lidar_gt:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2, label='Ground Truth (UWB + LiDAR)'))
        if has_uwb_gt:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2, label='Ground Truth (UWB only)'))

        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_title("Ground Truth vs. Model Prediction", fontsize=14)
        ax.axis("equal")
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Adjust limits
        x_min = min(self.gts[:, 0].min(), self.preds[:, 0].min(), veh_x_rear - 0.5)
        x_max = max(self.gts[:, 0].max(), self.preds[:, 0].max(), veh_x_front + 0.5)
        y_min = min(self.gts[:, 1].min(), self.preds[:, 1].min(), veh_y_bottom - 0.5)
        y_max = max(self.gts[:, 1].max(), self.preds[:, 1].max(), veh_y_top + 0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()

        # === Plot 2: Ground Truth vs UWB Raw ===
        fig, ax = plt.subplots(figsize=(12, 10))
        veh_x_rear, veh_x_front, veh_y_bottom, veh_y_top = self._draw_sensor_platform(ax)
        has_lidar_gt, has_uwb_gt = self._plot_trajectory_segments(ax, self.gts, self.driver_present_array)
        ax.plot(self.uwbs[:, 0], self.uwbs[:, 1], 'r--', linewidth=2, label='UWB raw', zorder=6)

        legend_elements = [
            Line2D([0], [0], color='darkgreen', linewidth=2.5, label='Sensor platform (3.40m × 1.13m)'),
            Line2D([0], [0], color='r', linewidth=2, linestyle='--', label='UWB raw')
        ]
        if has_lidar_gt:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2, label='Ground Truth (UWB + LiDAR)'))
        if has_uwb_gt:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2, label='Ground Truth (UWB only)'))

        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_title("Ground Truth vs. UWB Raw", fontsize=14)
        ax.axis("equal")
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        x_min = min(self.gts[:, 0].min(), self.uwbs[:, 0].min(), veh_x_rear - 0.5)
        x_max = max(self.gts[:, 0].max(), self.uwbs[:, 0].max(), veh_x_front + 0.5)
        y_min = min(self.gts[:, 1].min(), self.uwbs[:, 1].min(), veh_y_bottom - 0.5)
        y_max = max(self.gts[:, 1].max(), self.uwbs[:, 1].max(), veh_y_top + 0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()

        # === Plot 3: Ground Truth vs UWB Filtered (EKF) ===
        fig, ax = plt.subplots(figsize=(12, 10))
        veh_x_rear, veh_x_front, veh_y_bottom, veh_y_top = self._draw_sensor_platform(ax)
        has_lidar_gt, has_uwb_gt = self._plot_trajectory_segments(ax, self.gts, self.driver_present_array)
        ax.plot(self.uwbs_filtered[:, 0], self.uwbs_filtered[:, 1], 'r--', linewidth=2, label='UWB filtered (orig)',
                zorder=6)

        legend_elements = [
            Line2D([0], [0], color='darkgreen', linewidth=2.5, label='Sensor platform (3.40m × 1.13m)'),
            Line2D([0], [0], color='r', linewidth=2, linestyle='--', label='UWB filtered (orig)')
        ]
        if has_lidar_gt:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2, label='Ground Truth (UWB + LiDAR)'))
        if has_uwb_gt:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2, label='Ground Truth (UWB only)'))

        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_title("Ground Truth vs. UWB Filtered (EKF)", fontsize=14)
        ax.axis("equal")
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        x_min = min(self.gts[:, 0].min(), self.uwbs_filtered[:, 0].min(), veh_x_rear - 0.5)
        x_max = max(self.gts[:, 0].max(), self.uwbs_filtered[:, 0].max(), veh_x_front + 0.5)
        y_min = min(self.gts[:, 1].min(), self.uwbs_filtered[:, 1].min(), veh_y_bottom - 0.5)
        y_max = max(self.gts[:, 1].max(), self.uwbs_filtered[:, 1].max(), veh_y_top + 0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()

        # === Plot 4: Ground Truth vs UWB UKF ===
        fig, ax = plt.subplots(figsize=(12, 10))
        veh_x_rear, veh_x_front, veh_y_bottom, veh_y_top = self._draw_sensor_platform(ax)
        has_lidar_gt, has_uwb_gt = self._plot_trajectory_segments(ax, self.gts, self.driver_present_array)
        ax.plot(self.uwbs_ukf[:, 0], self.uwbs_ukf[:, 1], 'r--', linewidth=2, label='UWB UKF CA', zorder=6)

        legend_elements = [
            Line2D([0], [0], color='darkgreen', linewidth=2.5, label='Sensor platform (3.40m × 1.13m)'),
            Line2D([0], [0], color='r', linewidth=2, linestyle='--', label='UWB UKF CA')
        ]
        if has_lidar_gt:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2, label='Ground Truth (UWB + LiDAR)'))
        if has_uwb_gt:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2, label='Ground Truth (UWB only)'))

        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_title("Ground Truth vs. UWB Filtered (UKF - const accel)", fontsize=14)
        ax.axis("equal")
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        x_min = min(self.gts[:, 0].min(), self.uwbs_ukf[:, 0].min(), veh_x_rear - 0.5)
        x_max = max(self.gts[:, 0].max(), self.uwbs_ukf[:, 0].max(), veh_x_front + 0.5)
        y_min = min(self.gts[:, 1].min(), self.uwbs_ukf[:, 1].min(), veh_y_bottom - 0.5)
        y_max = max(self.gts[:, 1].max(), self.uwbs_ukf[:, 1].max(), veh_y_top + 0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()

        # === Error Plots (keep original style) ===
        # Prediction Error
        errors = np.linalg.norm(self.preds - self.gts, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(errors, label='Position Error [m]')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("Prediction Error over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # UWB Raw Error
        errors = np.linalg.norm(self.uwbs - self.gts, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(errors, label='UWB raw Error [m]', color='g')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("UWB Raw Error over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # UWB Filtered Error
        errors = np.linalg.norm(self.uwbs_filtered - self.gts, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(errors, label='UWB filtered (orig) Error [m]', color='y')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("UWB Filtered (orig) Error over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # UWB UKF Error
        errors = np.linalg.norm(self.uwbs_ukf - self.gts, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(errors, label='UWB UKF (CA) Error [m]', color='c')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("UWB UKF (constant accel) Error over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    # ----------------------------------------------------
    #  Metriken berechnen
    # ----------------------------------------------------

    def compute_metrics(self):
        if not hasattr(self, 'preds'):
            raise RuntimeError("Run inference first")

        def compute_error_stats(estimates, gt):
            if len(estimates) == 0:
                return {
                    "Mean [m]": np.nan,
                    "RMSE [m]": np.nan,
                    "Median [m]": np.nan,
                    "P90 [m]": np.nan,
                    "Max [m]": np.nan,
                    "Count": 0
                }
            errors = np.linalg.norm(estimates - gt, axis=1)
            return {
                "Mean [m]": np.mean(errors),
                "RMSE [m]": np.sqrt(np.mean(errors ** 2)),
                "Median [m]": np.median(errors),
                "P90 [m]": np.percentile(errors, 90),
                "Max [m]": np.max(errors),
                "Count": len(errors)
            }


        results = {
            "Model Prediction": compute_error_stats(self.preds, self.gts),
            "Model Prediction UWB + Lidar": compute_error_stats(self.preds[self.mask_lidar_and_uwb], self.gts[self.mask_lidar_and_uwb]),
            "Model Prediction UWB only": compute_error_stats(self.preds[self.mask_uwb_only], self.gts[self.mask_uwb_only]),
            "UWB Raw": compute_error_stats(self.uwbs, self.gts),
            "UWB Filtered (orig)": compute_error_stats(self.uwbs_filtered, self.gts),
            "UWB UKF (const accel)": compute_error_stats(self.uwbs_ukf, self.gts),
        }

        print("\n========== Evaluation Metrics ==========")
        for key, stats in results.items():
            print(f"\n--- {key} ---")
            for stat_name, value in stats.items():
                print(f"{stat_name}: {value:.4f}" if isinstance(value, float) else f"{stat_name}: {value}")

        print("\n========================================\n")

        return results

# ------------------ Ausführung ------------------
if __name__ == "__main__":

    # folder for dataset location (train,val,test) -> test split is used for test and inference
    train_val_test_folder = "dataset/train_val_test/"

    # folder for test data -> cw or line check
    test_folder = "dataset/test/"

    # path for test data -> cw or line check
    test_path = test_folder + "dataset_val_cw.hdf5"
    #test_path = test_folder + "dataset_val_line.hdf5"


    # model path
    # model_path = "saved_models/minimal_multimodal_model.keras"
    # model_path= "saved_models/kalman_multimodal_model.keras"
    # model_path= "saved_models/best_models/fused_kalman_multimodal_model/fused_kalman_multimodal_model.keras"
    # model_path="saved_models/adaptive_fused_model.keras"


    #model_path="saved_models/best_models/minimal_multimodal_model/minimal_multimodal_model.keras"
    #model_path="saved_models/best_models/kalman_multimodal_model/kalman_multimodal_model.keras"
    #model_path="saved_models/best_models/fused_kalman_multimodal_model/fused_kalman_multimodal_model.keras"
    model_path="saved_models/best_models/adaptive_fused_model/adaptive_fused_model.keras"


    # create evaluator
    evaluator = OfflineModelEvaluator(
        test_path=test_path, # test data
        model_path=model_path,
        grid_size=500,
        grid_resolution=0.1,
        seq_length=10,
        vel_output=True,  # predict velocity and position -> set true, pos only false
        ukf_dt=0.08  # 12,5 Hz -> 0.08
    )

    # test model with training dataset
    #evaluator.baseline_metrics_and_inference(dataset_folder=train_val_test_folder)

    # test model with test data -> cw or line
    evaluator.run_inference()
    evaluator.plot_results()
    metrics = evaluator.compute_metrics()
