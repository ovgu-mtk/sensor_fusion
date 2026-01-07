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
        """Grid-Erstellung IDENTISCH zum DataGenerator"""
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
    def plot_results(self):
        if not hasattr(self, 'preds'):
            raise RuntimeError("Run inference first")

        plt.figure(figsize=(8, 8))
        plt.plot(self.gts[:, 0], self.gts[:, 1], 'b-', label='Ground Truth')
        plt.plot(self.preds[:, 0], self.preds[:, 1], 'r--', label='Prediction')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.title("Trajectory Comparison: True vs Predicted")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.plot(self.gts[:, 0], self.gts[:, 1], 'b-', label='Ground Truth')
        plt.plot(self.uwbs[:, 0], self.uwbs[:, 1], 'g--', label='UWB raw')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.title("Trajectory Comparison: True vs UWB")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.plot(self.gts[:, 0], self.gts[:, 1], 'b-', label='Ground Truth')
        plt.plot(self.uwbs_filtered[:, 0], self.uwbs_filtered[:, 1], 'y--', label='UWB filtered (orig)')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.title("Trajectory Comparison: True vs UWB filtered (orig)")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.plot(self.gts[:, 0], self.gts[:, 1], 'b-', label='Ground Truth')
        plt.plot(self.uwbs_ukf[:, 0], self.uwbs_ukf[:, 1], 'm--', label='UWB UKF CA')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.title("Trajectory Comparison: True vs UWB UKF (const accel)")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Fehler über Zeit Predicted
        errors = np.linalg.norm(self.preds - self.gts, axis=1)
        plt.figure()
        plt.plot(errors, label='Position Error [m]')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("Prediction Error over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Fehler über Zeit UWB
        errors = np.linalg.norm(self.uwbs - self.gts, axis=1)
        plt.figure()
        plt.plot(errors, label='UWB raw Error [m]')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("UWB Raw Error over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Fehler über Zeit UWB Filtered (orig)
        errors = np.linalg.norm(self.uwbs_filtered - self.gts, axis=1)
        plt.figure()
        plt.plot(errors, label='UWB filtered (orig) Error [m]')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("UWB Filtered (orig) Error over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Fehler über Zeit UWB UKF
        errors = np.linalg.norm(self.uwbs_ukf - self.gts, axis=1)
        plt.figure()
        plt.plot(errors, label='UWB UKF (CA) Error [m]')
        plt.xlabel("Frame")
        plt.ylabel("Euclidean Error [m]")
        plt.title("UWB UKF (constant accel) Error over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_validation_pattern(self):
        gts = []

        for i in tqdm(range(self.num_samples)):
            gt_point = self.position_data[i]

            gts.append([gt_point[0], gt_point[1]])


        # Ergebnis als numpy arrays
        self.gts = np.array(gts)

        plt.figure(figsize=(8, 8))
        plt.plot(self.gts[:, 0], self.gts[:, 1], 'b-', label='Ground Truth')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
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

    def plot_validation_pattern_split(self):
        gts = []
        driver_present_array = []

        for i in tqdm(range(self.num_samples)):
            gt_point = self.position_data[i]
            gts.append([gt_point[0], gt_point[1]])
            driver_present_array.append(self.driver_present[i])

        # Ergebnis als numpy arrays
        gts = np.array(gts)
        driver_present_array = np.array(driver_present_array)

        plt.figure(figsize=(10, 10))

        # Segmente basierend auf Zustandswechseln erstellen
        i = 0
        has_lidar = False
        has_uwb_only = False

        while i < len(gts):
            # Aktuellen Zustand ermitteln
            current_state = driver_present_array[i]

            # Finde Ende des aktuellen Segments (gleicher Zustand)
            j = i
            while j < len(gts) and driver_present_array[j] == current_state:
                j += 1

            # Segment plotten
            segment = gts[i:j]
            if current_state == 1:  # UWB + LiDAR
                plt.plot(segment[:, 0], segment[:, 1], 'b-', linewidth=2)
                has_lidar = True
            else:  # Nur UWB
                plt.plot(segment[:, 0], segment[:, 1], 'm-', linewidth=2)
                has_uwb_only = True

            i = j

        # Legende nur mit tatsächlich vorhandenen Zuständen
        from matplotlib.lines import Line2D
        legend_elements = []
        if has_lidar:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2,
                                          label='UWB + LiDAR (Zielperson sichtbar)'))
        if has_uwb_only:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2,
                                          label='Nur UWB (Zielperson nicht sichtbar)'))

        plt.legend(handles=legend_elements, fontsize=10)
        plt.xlabel("X [m]", fontsize=12)
        plt.ylabel("Y [m]", fontsize=12)
        plt.title("Ground Truth Trajektorie - LiDAR Sichtbarkeit", fontsize=14)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Statistik ausgeben
        total_samples = len(driver_present_array)
        lidar_samples = np.sum(driver_present_array == 1)
        uwb_only_samples = np.sum(driver_present_array == 0)

        print(f"\n=== Validierungsdaten Statistik ===")
        print(f"Gesamt Samples: {total_samples}")
        print(f"UWB + LiDAR: {lidar_samples} ({100 * lidar_samples / total_samples:.1f}%)")
        print(f"Nur UWB: {uwb_only_samples} ({100 * uwb_only_samples / total_samples:.1f}%)")
        print(f"===================================\n")

    def plot_validation_pattern_split_detailed(self):
        """Erweiterte Version mit mehr Details"""
        gts = []
        driver_present_array = []

        for i in tqdm(range(self.num_samples)):
            gt_point = self.position_data[i]
            gts.append([gt_point[0], gt_point[1]])
            driver_present_array.append(self.driver_present[i])

        gts = np.array(gts)
        driver_present_array = np.array(driver_present_array)

        fig, ax = plt.subplots(figsize=(14, 10))

        # Fahrzeugabmessungen
        vehicle_length = 3.40
        vehicle_width = 1.125
        vehicle_x_rear = -3.00
        vehicle_x_front = 0.40
        vehicle_y_bottom = -vehicle_width / 2
        vehicle_y_top = vehicle_width / 2

        # Fahrzeug zeichnen
        vehicle_rect = Rectangle(
            (vehicle_x_rear, vehicle_y_bottom),
            vehicle_length,
            vehicle_width,
            linewidth=2.5,
            edgecolor='darkgreen',
            facecolor='lightgreen',
            alpha=0.4,
            zorder=10
        )
        ax.add_patch(vehicle_rect)

        # Fahrtrichtungspfeil
        arrow_length = 0.8
        arrow_width = 0.4
        arrow_x_center = vehicle_x_front - 0.5

        triangle = Polygon(
            [
                [arrow_x_center - arrow_length / 2, 0],
                [arrow_x_center + arrow_length / 2, arrow_width / 2],
                [arrow_x_center + arrow_length / 2, -arrow_width / 2]
            ],
            closed=True,
            linewidth=2,
            edgecolor='darkgreen',
            facecolor='green',
            alpha=0.7,
            zorder=11
        )
        ax.add_patch(triangle)

        # Origin
        ax.scatter(0, 0, c='red', marker='x', s=250, linewidths=4,
                   zorder=12, label='Origin')
        ax.plot([0, 0], [-0.3, 0.3], 'r--', linewidth=1, alpha=0.5, zorder=12)
        ax.plot([-0.3, 0.3], [0, 0], 'r--', linewidth=1, alpha=0.5, zorder=12)

        # Fahrzeug-Dimensionslinien
        ax.annotate('', xy=(vehicle_x_front, vehicle_y_bottom - 0.3),
                    xytext=(vehicle_x_rear, vehicle_y_bottom - 0.3),
                    arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
        ax.text((vehicle_x_rear + vehicle_x_front) / 2, vehicle_y_bottom - 0.5,
                f'{vehicle_length:.2f}m', ha='center', fontsize=9,
                color='darkgreen', fontweight='bold')

        # Trajektorie plotten
        i = 0
        has_lidar = False
        has_uwb_only = False

        while i < len(gts):
            current_state = driver_present_array[i]
            j = i
            while j < len(gts) and driver_present_array[j] == current_state:
                j += 1

            segment = gts[i:j]
            if current_state == 1:
                ax.plot(segment[:, 0], segment[:, 1], 'b-', linewidth=2.5,
                        alpha=0.8, zorder=5)
                has_lidar = True
            else:
                ax.plot(segment[:, 0], segment[:, 1], 'm-', linewidth=2.5,
                        alpha=0.8, zorder=5)
                has_uwb_only = True
            i = j

        # Legende
        legend_elements = [
            Line2D([0], [0], color='darkgreen', linewidth=2.5,
                   label=f'Fahrzeug ({vehicle_length}m × {vehicle_width}m)'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                   markersize=12, label='Origin (0,0)', linestyle='None')
        ]
        if has_lidar:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2.5,
                                          label='UWB + LiDAR'))
        if has_uwb_only:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2.5,
                                          label='Nur UWB'))

        ax.legend(handles=legend_elements, fontsize=11, loc='upper right',
                  framealpha=0.9)

        ax.set_xlabel("X [m] (Längsrichtung / Fahrtrichtung →)", fontsize=13,
                      fontweight='bold')
        ax.set_ylabel("Y [m] (Querrichtung)", fontsize=13, fontweight='bold')
        ax.set_title("Ground Truth Trajektorie mit Fahrzeugkoordinatensystem",
                     fontsize=15, fontweight='bold', pad=20)
        ax.axis("equal")
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Achsenlimits
        x_min = min(gts[:, 0].min(), vehicle_x_rear - 1)
        x_max = max(gts[:, 0].max(), vehicle_x_front + 1)
        y_min = min(gts[:, 1].min(), vehicle_y_bottom - 1)
        y_max = max(gts[:, 1].max(), vehicle_y_top + 1)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

        # Statistik
        total_samples = len(driver_present_array)
        lidar_samples = np.sum(driver_present_array == 1)
        uwb_only_samples = np.sum(driver_present_array == 0)

        print(f"\n{'=' * 55}")
        print(f"📊 VALIDIERUNGSDATEN STATISTIK")
        print(f"{'=' * 55}")
        print(f"Gesamt Samples:    {total_samples:6d}")
        print(f"UWB + LiDAR:       {lidar_samples:6d} ({100 * lidar_samples / total_samples:5.1f}%)")
        print(f"Nur UWB:           {uwb_only_samples:6d} ({100 * uwb_only_samples / total_samples:5.1f}%)")
        print(f"\n🚗 FAHRZEUGKOORDINATENSYSTEM:")
        print(f"  Länge (x):       {vehicle_length:.2f} m (340.0 cm)")
        print(f"  Breite (y):      {vehicle_width:.3f} m (112.5 cm)")
        print(f"  Origin:          (0.0, 0.0) m")
        print(f"  Heck (x_min):    {vehicle_x_rear:.2f} m")
        print(f"  Front (x_max):   {vehicle_x_front:.2f} m")
        print(f"  Links (y_min):   {vehicle_y_bottom:.3f} m")
        print(f"  Rechts (y_max):  {vehicle_y_top:.3f} m")
        print(f"{'=' * 55}\n")

    def plot_validation_pattern_split_2(self):
        gts = []
        driver_present_array = []

        for i in tqdm(range(self.num_samples)):
            gt_point = self.position_data[i]
            gts.append([gt_point[0], gt_point[1]])
            driver_present_array.append(self.driver_present[i])

        # Ergebnis als numpy arrays
        gts = np.array(gts)
        driver_present_array = np.array(driver_present_array)

        plt.figure(figsize=(12, 10))

        # === FAHRZEUG ZEICHNEN ===
        # Fahrzeugabmessungen in Metern
        vehicle_length = 3.40  # 340 cm
        vehicle_width = 1.125  # 112.5 cm

        # Origin liegt bei: x = -3.00m (40cm vor Fahrzeugheck), y = 0 (Mitte)
        # Das bedeutet: Fahrzeugheck bei x=-3.00m, Fahrzeugfront bei x=0.40m
        vehicle_x_rear = -3.00
        vehicle_x_front = 0.40
        vehicle_y_bottom = -vehicle_width / 2
        vehicle_y_top = vehicle_width / 2

        # Fahrzeug-Rechteck zeichnen
        vehicle_rect = Rectangle(
            (vehicle_x_rear, vehicle_y_bottom),
            vehicle_length,
            vehicle_width,
            linewidth=2.5,
            edgecolor='darkgreen',
            facecolor='lightgreen',
            alpha=0.3,
            zorder=10,
            label='Versuchsfahrzeug'
        )
        plt.gca().add_patch(vehicle_rect)

        # Fahrtrichtungspfeil (Dreieck) - zeigt nach rechts (positive x-Richtung)
        arrow_length = 0.8  # 80 cm
        arrow_width = 0.4  # 40 cm
        arrow_x_center = vehicle_x_front - 0.5  # Mittig vorne im Fahrzeug

        triangle = Polygon(
            [
                [arrow_x_center - arrow_length / 2, 0],  # Hintere Spitze
                [arrow_x_center + arrow_length / 2, arrow_width / 2],  # Obere Ecke
                [arrow_x_center + arrow_length / 2, -arrow_width / 2]  # Untere Ecke
            ],
            closed=True,
            linewidth=1.5,
            edgecolor='darkgreen',
            facecolor='green',
            alpha=0.6,
            zorder=11
        )
        plt.gca().add_patch(triangle)

        # Origin-Markierung (x=0, y=0)
        plt.scatter(0, 0, c='red', marker='x', s=200, linewidths=3,
                    zorder=12, label='Origin (0,0)')

        # === TRAJEKTORIE PLOTTEN ===
        i = 0
        has_lidar = False
        has_uwb_only = False

        while i < len(gts):
            current_state = driver_present_array[i]

            j = i
            while j < len(gts) and driver_present_array[j] == current_state:
                j += 1

            segment = gts[i:j]
            if current_state == 1:  # UWB + LiDAR
                plt.plot(segment[:, 0], segment[:, 1], 'b-', linewidth=2, zorder=5)
                has_lidar = True
            else:  # Nur UWB
                plt.plot(segment[:, 0], segment[:, 1], 'm-', linewidth=2, zorder=5)
                has_uwb_only = True

            i = j

        # === LEGENDE ===
        legend_elements = []
        legend_elements.append(Line2D([0], [0], color='darkgreen', linewidth=2.5,
                                      label='Versuchsfahrzeug (3.40m × 1.13m)'))
        legend_elements.append(Line2D([0], [0], marker='x', color='w',
                                      markerfacecolor='red', markersize=10,
                                      label='Origin (0,0)', linestyle='None'))
        if has_lidar:
            legend_elements.append(Line2D([0], [0], color='b', linewidth=2,
                                          label='UWB + LiDAR (Zielperson sichtbar)'))
        if has_uwb_only:
            legend_elements.append(Line2D([0], [0], color='m', linewidth=2,
                                          label='Nur UWB (Zielperson nicht sichtbar)'))

        plt.legend(handles=legend_elements, fontsize=10, loc='upper right')

        # === ACHSEN & GRID ===
        plt.xlabel("X [m] (Fahrtrichtung →)", fontsize=12)
        plt.ylabel("Y [m] (Querrichtung)", fontsize=12)
        plt.title("Ground Truth Trajektorie mit Fahrzeugposition", fontsize=14, fontweight='bold')
        plt.axis("equal")
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Achsen-Limits anpassen, damit Fahrzeug gut sichtbar ist
        x_min, x_max = gts[:, 0].min(), gts[:, 0].max()
        y_min, y_max = gts[:, 1].min(), gts[:, 1].max()

        # Erweitere Limits um Fahrzeug
        x_min = min(x_min, vehicle_x_rear - 0.5)
        x_max = max(x_max, vehicle_x_front + 0.5)
        y_min = min(y_min, vehicle_y_bottom - 0.5)
        y_max = max(y_max, vehicle_y_top + 0.5)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

        # === STATISTIK ===
        total_samples = len(driver_present_array)
        lidar_samples = np.sum(driver_present_array == 1)
        uwb_only_samples = np.sum(driver_present_array == 0)

        print(f"\n{'=' * 50}")
        print(f"📊 Validierungsdaten Statistik")
        print(f"{'=' * 50}")
        print(f"Gesamt Samples:  {total_samples:6d}")
        print(f"UWB + LiDAR:     {lidar_samples:6d} ({100 * lidar_samples / total_samples:5.1f}%)")
        print(f"Nur UWB:         {uwb_only_samples:6d} ({100 * uwb_only_samples / total_samples:5.1f}%)")
        print(f"\n🚗 Fahrzeugposition:")
        print(f"  Länge: {vehicle_length:.2f} m (340 cm)")
        print(f"  Breite: {vehicle_width:.3f} m (112.5 cm)")
        print(f"  Origin: (0.0, 0.0)")
        print(f"  Heck: x = {vehicle_x_rear:.2f} m")
        print(f"  Front: x = {vehicle_x_front:.2f} m")
        print(f"{'=' * 50}\n")


# ------------------ Ausführung ------------------
if __name__ == "__main__":

    # folder for dataset location (train,val,test) -> test split is used for test and inference
    train_val_test_folder = "dataset/train_val_test/"

    # path for test data -> cw or line check
    test_folder = "dataset/test/"
    #test_path = test_folder + "dataset_val_cw.hdf5"
    test_path = test_folder + "dataset_val_line.hdf5"


    # model path
    # model_path = "saved_models/minimal_multimodal_model.keras"
    # model_path= "saved_models/kalman_multimodal_model.keras"
    # model_path= "saved_models/best_models/fused_kalman_multimodal_model/fused_kalman_multimodal_model.keras"
    # model_path="saved_models/adaptive_fused_model.keras"



    model_path="saved_models/best_models/minimal_multimodal_model/minimal_multimodal_model.keras"
    #model_path="saved_models/best_models/kalman_multimodal_model/kalman_multimodal_model.keras"
    #model_path="saved_models/best_models/fused_kalman_multimodal_model/fused_kalman_multimodal_model.keras"
    #model_path="saved_models/best_models/adaptive_fused_model/adaptive_fused_model.keras"


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

    # plot ground truth
    evaluator.plot_validation_pattern()
    evaluator.plot_validation_pattern_split()
    evaluator.plot_validation_pattern_split_2()
    evaluator.plot_validation_pattern_split_detailed()

    # run and test
    #evaluator.run_inference()
    #evaluator.plot_results()
    #metrics = evaluator.compute_metrics()
