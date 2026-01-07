import h5py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class DataGenerator:
    def __init__(self, hdf5_folder, grid_size=500, grid_resolution=0.1, seq_length=10,
                 prediction_horizon=1, batch_size=40, mode='train',
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 shuffle=True, lidar_dropout_prob=0.05,
                 use_velocity_auxiliary=True, augment_uwb=True):

        self.hdf5_folder = hdf5_folder
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.lidar_dropout_prob = lidar_dropout_prob
        self.use_velocity_auxiliary = use_velocity_auxiliary
        self.augment_uwb = augment_uwb

        # Validate the ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must be 1.0! Current sum: {train_ratio + val_ratio + test_ratio}")

        # compute grid range
        half_size = (grid_size * grid_resolution) / 2
        self.range_x = [-half_size, half_size]
        self.range_y = [-half_size, half_size]

        # Height Encoding Parameter
        self.ground_height_threshold = 0.2
        self.max_height = 3.0
        self.min_height = -1.0

        # 1. scan meta data
        self.samples = self._scan_hdf5_files()
        self.total_samples = len(self.samples)

        self.num_channels = self._get_num_channels()

        # 2. create dataset
        self.dataset = self._build_tf_dataset()

        print(f"Generator initialized. Mode: {mode}, "
              f"Samples: {self.total_samples}, Batch: {batch_size}, "
              f"Velocity Auxiliary: {use_velocity_auxiliary}")

    def __len__(self):
        return int(np.ceil(self.total_samples / self.batch_size))

    def _scan_hdf5_files(self):
        valid_samples = []
        files = sorted([f for f in os.listdir(self.hdf5_folder) if f.endswith('.hdf5')])

        if not files:
            raise ValueError(f"No HDF5 files found in {self.hdf5_folder}")

        print(f"Scanning {len(files)} files for {self.mode} split...")

        for f_name in files:
            path = os.path.join(self.hdf5_folder, f_name)
            try:
                with h5py.File(path, 'r') as f:
                    n_total = len(f['ground_truth'])
                    max_start = n_total - self.seq_length - self.prediction_horizon

                    # compute split indices
                    train_end = int(n_total * self.train_ratio)
                    val_end = int(n_total * (self.train_ratio + self.val_ratio))

                    # use indices based on mode
                    if self.mode == 'train':
                        indices = range(0, train_end)
                    elif self.mode == 'test':
                        indices = range(train_end, val_end)
                    elif self.mode == 'val':
                        indices = range(val_end, n_total)
                    else:
                        raise ValueError(f"Unknown mode: {self.mode}. Use 'train', 'test', or 'val'")

                    # add valid sequences only
                    for i in indices:
                        if i <= max_start:
                            valid_samples.append((path, i))

            except Exception as e:
                print(f"Error scanning {f_name}: {e}")

        if self.shuffle and self.mode == 'train':
            np.random.shuffle(valid_samples)

        return valid_samples

    def _augment_uwb(self, uwb_sequence):
        """Simulate UWB Noise, Spikes und Dropouts."""

        aug_uwb = uwb_sequence.copy()
        seq_len = aug_uwb.shape[0]

        # 1. Noise
        noise = np.random.normal(0, 0.08, aug_uwb.shape)
        aug_uwb += noise

        # 2. Random Spikes (Multipath) - 10%
        if np.random.rand() < 0.1:
            start = np.random.randint(0, seq_len - 1)
            duration = np.random.randint(1, 4)
            end = min(start + duration, seq_len)
            spike_val = np.random.uniform(0.5, 2.0, size=(1, 2)) * np.random.choice([-1, 1], size=(1, 2))
            aug_uwb[start:end] += spike_val

        # 3. Dropouts (Signal freeze) - 5%
        if np.random.rand() < 0.05:
            start = np.random.randint(0, seq_len - 2)
            aug_uwb[start:] = aug_uwb[start]

        return aug_uwb

    def _vectorized_grid_generation(self, lidar_points):

        if len(lidar_points) == 0 or (self.mode == 'train' and np.random.rand() < self.lidar_dropout_prob):
            return np.zeros((self.grid_size, self.grid_size, self.num_channels), dtype=np.float32)

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

    def _data_generator(self):

        for file_path, start_idx in self.samples:
            try:
                with h5py.File(file_path, 'r') as f:
                    seq_end = start_idx + self.seq_length
                    target_idx = seq_end + self.prediction_horizon - 1

                    # --- 1. UWB ---
                    uwb_seq = f['uwb_data'][start_idx:seq_end]
                    # check if mode is train and augmentation param is true
                    if self.mode == 'train' and self.augment_uwb:
                        uwb_seq = self._augment_uwb(uwb_seq)

                    # --- 2. Grid Generation ---
                    grids = []
                    lidar_ds = f['lidar_data']

                    for t in range(start_idx, seq_end):
                        points = lidar_ds[t]
                        grid = self._vectorized_grid_generation(points)
                        grids.append(grid)

                    grid_stack = np.stack(grids, axis=0)
                    grid_input = np.transpose(grid_stack, (1, 2, 3, 0))

                    # --- 3. Targets ---
                    gt_pos = f['ground_truth'][target_idx]

                    # compute always velocity
                    if target_idx > 0:
                        prev_pos = f['ground_truth'][target_idx - 1]
                        velocity = (gt_pos - prev_pos) / 0.1
                    else:
                        velocity = np.zeros(2)

                    # --- 4. Output based on Flag ---
                    if self.use_velocity_auxiliary:
                        yield (
                            {'grid_input': grid_input.astype(np.float32),
                             'uwb_input': uwb_seq.astype(np.float32)},
                            {'position': gt_pos.astype(np.float32),
                             'velocity_aux': velocity.astype(np.float32)}
                        )
                    else:
                        yield (
                            {'grid_input': grid_input.astype(np.float32),
                             'uwb_input': uwb_seq.astype(np.float32)},
                            gt_pos.astype(np.float32)
                        )

            except Exception as e:
                print(f"Error yielding sample from {file_path}: {e}")
                continue

    def _build_tf_dataset(self):
        grid_shape = (self.grid_size, self.grid_size, self.num_channels, self.seq_length)
        uwb_shape = (self.seq_length, 2)

        input_sig = {
            'grid_input': tf.TensorSpec(shape=grid_shape, dtype=tf.float32),
            'uwb_input': tf.TensorSpec(shape=uwb_shape, dtype=tf.float32)
        }

        if self.use_velocity_auxiliary:
            output_sig = {
                'position': tf.TensorSpec(shape=(2,), dtype=tf.float32),
                'velocity_aux': tf.TensorSpec(shape=(2,), dtype=tf.float32)
            }
        else:
            output_sig = tf.TensorSpec(shape=(2,), dtype=tf.float32)

        dataset = tf.data.Dataset.from_generator(
            self._data_generator,
            output_signature=(input_sig, output_sig)
        )

        # CRITICAL FIX: Add repeat() for training mode
        if self.mode == 'train':
            # Repeat indefinitely for training
            dataset = dataset.repeat()

        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _get_num_channels(self):
        return 1



    def load_all_ground_truth_and_driver_status(self):

        all_gt = []
        all_driver_present = []
        all_uwb_raw = []
        all_uwb_filtered = []
        all_lidar = []

        files = sorted([f for f in os.listdir(self.hdf5_folder) if f.endswith('.hdf5')])

        print(f"Loading validation data from {len(files)} files...")

        for f_name in files:
            path = os.path.join(self.hdf5_folder, f_name)
            try:
                with h5py.File(path, 'r') as f:
                    n_total = len(f['ground_truth'])

                    # Bestimme den richtigen Split
                    split_idx = int(n_total * (self.train_ratio+self.val_ratio))
                    start, end = split_idx, n_total

                    # Lade Daten für diesen Split
                    gt = f['ground_truth'][start:end]
                    driver_present = f['driver_present'][start:end]
                    uwb_raw = f['uwb_data'][start:end]

                    # Prüfe ob gefiltertes UWB vorhanden ist
                    if 'uwb_data_filtered' in f:
                        uwb_filtered = f['uwb_data_filtered'][start:end]
                    else:
                        uwb_filtered = uwb_raw.copy()

                    # Lade LiDAR Daten (für Model Inference)
                    lidar_data = [f['lidar_data'][i] for i in range(start, end)]

                    all_gt.append(gt)
                    all_driver_present.append(driver_present)
                    all_uwb_raw.append(uwb_raw)
                    all_uwb_filtered.append(uwb_filtered)
                    all_lidar.extend(lidar_data)

            except Exception as e:
                print(f"Error loading {f_name}: {e}")
                continue

        ground_truth = np.vstack(all_gt)
        driver_present = np.concatenate(all_driver_present)
        uwb_raw = np.vstack(all_uwb_raw)
        uwb_filtered = np.vstack(all_uwb_filtered)

        print(f"✅ Loaded {len(ground_truth)} validation samples")

        return ground_truth, driver_present, uwb_raw, uwb_filtered, all_lidar

    def run_model_inference(self, model, vel_output=True):

        import time
        from collections import deque
        from tqdm import tqdm

        # Load all data
        gt, driver_present, uwb_raw, uwb_filtered, lidar_data = self.load_all_ground_truth_and_driver_status()

        # TensorFlow function for fast inference
        @tf.function
        def infer(model_fn, X_input):
            return model_fn(X_input, training=False)

        predictions = []
        lidar_buffer = deque(maxlen=self.seq_length)
        uwb_buffer = deque(maxlen=self.seq_length)

        # Timing-variables
        preprocessing_times = []
        inference_times = []
        total_start = time.time()

        print(f"Running model inference on {len(lidar_data)} samples...")

        for i in tqdm(range(len(lidar_data))):
            lidar_points = lidar_data[i]
            uwb_point = uwb_raw[i]

            lidar_buffer.append(lidar_points)
            uwb_buffer.append(uwb_point)

            if len(lidar_buffer) < self.seq_length:
                continue

            # === PREPROCESSING TIMING ===
            prep_start = time.time()

            # create grid sequence
            grids = []
            for lidar_pts in lidar_buffer:
                grid = self._vectorized_grid_generation(lidar_pts)
                grids.append(grid)

            grid_stack = np.stack(grids, axis=0)
            grid_input = np.transpose(grid_stack, (1, 2, 3, 0))[np.newaxis, ...]  # (1, H, W, C, T)

            # UWB sequence
            uwb_input = np.array(uwb_buffer, dtype=np.float32)[np.newaxis, ...]  # (1, T, 2)

            # Input dict
            input_dict = {
                'grid_input': tf.convert_to_tensor(grid_input),
                'uwb_input': tf.convert_to_tensor(uwb_input)
            }

            prep_time = time.time() - prep_start
            preprocessing_times.append(prep_time)

            # === INFERENCE TIMING ===
            inf_start = time.time()

            # Modell-Inference
            pred = infer(model, input_dict)

            inf_time = time.time() - inf_start
            inference_times.append(inf_time)

            # extract prediction
            if vel_output:
                try:
                    pred_x = float(pred['position'][0][0].numpy())
                    pred_y = float(pred['position'][0][1].numpy())
                except (KeyError, TypeError):
                    # Fallback
                    pred_x, pred_y = float(pred[0][0].numpy()), float(pred[0][1].numpy())
            else:
                pred_x, pred_y = float(pred[0][0].numpy()), float(pred[0][1].numpy())

            predictions.append([pred_x, pred_y])

        total_time = time.time() - total_start

        # fill missing values
        predictions_array = np.array(predictions)
        if len(predictions_array) < len(gt):
            missing = len(gt) - len(predictions_array)
            first_pred = predictions_array[0:1].repeat(missing, axis=0)
            predictions_array = np.vstack([first_pred, predictions_array])

        # compute timing statistics
        preprocessing_times = np.array(preprocessing_times) * 1000  # Convert to ms
        inference_times = np.array(inference_times) * 1000  # Convert to ms
        total_times = preprocessing_times + inference_times

        timing_stats = {
            'total_samples': len(predictions_array),
            'total_time_s': total_time,
            'preprocessing': {
                'mean_ms': np.mean(preprocessing_times),
                'median_ms': np.median(preprocessing_times),
                'std_ms': np.std(preprocessing_times),
                'min_ms': np.min(preprocessing_times),
                'max_ms': np.max(preprocessing_times),
            },
            'inference': {
                'mean_ms': np.mean(inference_times),
                'median_ms': np.median(inference_times),
                'std_ms': np.std(inference_times),
                'min_ms': np.min(inference_times),
                'max_ms': np.max(inference_times),
                'fps': 1000.0 / np.mean(inference_times),
            },
            'total_per_sample': {
                'mean_ms': np.mean(total_times),
                'median_ms': np.median(total_times),
                'fps': 1000.0 / np.mean(total_times),
            }
        }

        # Ausgabe der Timing-Statistiken
        print(f"\n{'=' * 60}")
        print(f"⏱️  INFERENCE TIMING STATISTICS")
        print(f"{'=' * 60}")
        print(f"Total samples processed: {timing_stats['total_samples']}")
        print(f"Total time: {timing_stats['total_time_s']:.2f} s")
        print(f"\n--- Preprocessing (Grid Generation) ---")
        print(f"  Mean:   {timing_stats['preprocessing']['mean_ms']:.2f} ms")
        print(f"  Median: {timing_stats['preprocessing']['median_ms']:.2f} ms")
        print(f"  Std:    {timing_stats['preprocessing']['std_ms']:.2f} ms")
        print(f"  Min:    {timing_stats['preprocessing']['min_ms']:.2f} ms")
        print(f"  Max:    {timing_stats['preprocessing']['max_ms']:.2f} ms")
        print(f"\n--- Model Inference ---")
        print(f"  Mean:   {timing_stats['inference']['mean_ms']:.2f} ms")
        print(f"  Median: {timing_stats['inference']['median_ms']:.2f} ms")
        print(f"  Std:    {timing_stats['inference']['std_ms']:.2f} ms")
        print(f"  Min:    {timing_stats['inference']['min_ms']:.2f} ms")
        print(f"  Max:    {timing_stats['inference']['max_ms']:.2f} ms")
        print(f"  FPS:    {timing_stats['inference']['fps']:.1f} Hz")
        print(f"\n--- Total (Preprocessing + Inference) ---")
        print(f"  Mean:   {timing_stats['total_per_sample']['mean_ms']:.2f} ms")
        print(f"  Median: {timing_stats['total_per_sample']['median_ms']:.2f} ms")
        print(f"  FPS:    {timing_stats['total_per_sample']['fps']:.1f} Hz")
        print(f"{'=' * 60}\n")

        print(f"✅ Model inference complete: {len(predictions_array)} predictions")

        return predictions_array, timing_stats

    def compute_baseline_metrics(self, model=None, vel_output=True):

        gt, driver_present, uwb_raw, uwb_filtered, lidar_data = self.load_all_ground_truth_and_driver_status()

        # UWB with UKF
        from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

        def fx_ca(x, dt):
            px, py, vx, vy, ax, ay = x
            return np.array([
                px + vx * dt + 0.5 * ax * (dt ** 2),
                py + vy * dt + 0.5 * ay * (dt ** 2),
                vx + ax * dt,
                vy + ay * dt,
                ax, ay
            ])

        def hx_ca(x):
            return np.array([x[0], x[1]])

        # UKF Setup
        dt = 0.08  # frequency 12.5 hz
        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2.0, kappa=0.0)
        ukf = UnscentedKalmanFilter(
            dim_x=6, dim_z=2, dt=dt,
            fx=lambda x, dt: fx_ca(x, dt),
            hx=lambda x: hx_ca(x),
            points=points
        )
        ukf.x = np.zeros(6)
        ukf.P = np.eye(6) * 1.0
        ukf.R = np.diag([0.3 ** 2, 0.3 ** 2])
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = np.eye(2) * 0.01
        Q[2:4, 2:4] = np.eye(2) * 0.1
        Q[4:6, 4:6] = np.eye(2) * 1e-3
        ukf.Q = Q

        uwb_ukf = []
        initialized = False

        for uwb_point in uwb_raw:
            if not initialized:
                ukf.x[0:2] = uwb_point
                initialized = True
                uwb_ukf.append(uwb_point)
            else:
                ukf.predict()
                ukf.update(uwb_point)
                uwb_ukf.append(ukf.x[0:2].copy())

        uwb_ukf = np.array(uwb_ukf)

        # create masks
        mask_lidar_and_uwb = driver_present == 1
        mask_uwb_only = driver_present == 0

        def compute_error_stats(estimates, gt_data, mask=None):
            if mask is not None:
                estimates = estimates[mask]
                gt_data = gt_data[mask]

            if len(estimates) == 0:
                return {
                    "Mean [m]": np.nan,
                    "RMSE [m]": np.nan,
                    "Median [m]": np.nan,
                    "P90 [m]": np.nan,
                    "Max [m]": np.nan,
                    "Count": 0
                }

            errors = np.linalg.norm(estimates - gt_data, axis=1)
            return {
                "Mean [m]": np.mean(errors),
                "RMSE [m]": np.sqrt(np.mean(errors ** 2)),
                "Median [m]": np.median(errors),
                "P90 [m]": np.percentile(errors, 90),
                "Max [m]": np.max(errors),
                "Count": len(errors)
            }

        results = {}

        # UWB Baselines
        results["UWB Raw"] = compute_error_stats(uwb_raw, gt)
        results["UWB Filtered (orig)"] = compute_error_stats(uwb_filtered, gt)
        results["UWB UKF (const accel)"] = compute_error_stats(uwb_ukf, gt)

        # Model Predictions
        predictions = None
        timing_stats = None

        if model is not None:
            print("\n🔄 Running model inference...")
            predictions, timing_stats = self.run_model_inference(model, vel_output=vel_output)

            # Predictions must be the same length as gt
            if len(predictions) != len(gt):
                print(f"⚠️ Warning: Predictions length {len(predictions)} != GT length {len(gt)}")
                min_len = min(len(predictions), len(gt))
                predictions = predictions[:min_len]
                gt_trimmed = gt[:min_len]
                mask_lidar_trimmed = mask_lidar_and_uwb[:min_len]
                mask_uwb_trimmed = mask_uwb_only[:min_len]
            else:
                gt_trimmed = gt
                mask_lidar_trimmed = mask_lidar_and_uwb
                mask_uwb_trimmed = mask_uwb_only

            results["Model Prediction"] = compute_error_stats(predictions, gt_trimmed)
            results["Model Prediction UWB + LiDAR"] = compute_error_stats(
                predictions, gt_trimmed, mask_lidar_trimmed
            )
            results["Model Prediction UWB only"] = compute_error_stats(
                predictions, gt_trimmed, mask_uwb_trimmed
            )

        # output
        print("\n========== Validation Metrics ==========")
        for key, stats in results.items():
            print(f"\n--- {key} ---")
            for stat_name, value in stats.items():
                if isinstance(value, float):
                    print(f"{stat_name}: {value:.4f}")
                else:
                    print(f"{stat_name}: {value}")
        print("\n========================================\n")

        # Return with Timing-Stats if exist
        if timing_stats is not None:
            return results, timing_stats
        return results

    def plot_validation_pattern(self):
        gt, _, _, _, _ = self.load_all_ground_truth_and_driver_status()

        plt.figure(figsize=(8, 8))
        plt.plot(gt[:, 0], gt[:, 1], 'b-', label='Ground Truth')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.title(f"Ground Truth Trajectory ({self.mode.capitalize()} Split)")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_validation_pattern_split(self):
        gt, driver_present, _, _, _ = self.load_all_ground_truth_and_driver_status()

        plt.figure(figsize=(10, 10))

        i = 0
        has_lidar = False
        has_uwb_only = False

        while i < len(gt):
            current_state = driver_present[i]

            j = i
            while j < len(gt) and driver_present[j] == current_state:
                j += 1

            segment = gt[i:j]
            if current_state == 1:
                plt.plot(segment[:, 0], segment[:, 1], 'b-', linewidth=2)
                has_lidar = True
            else:
                plt.plot(segment[:, 0], segment[:, 1], 'm-', linewidth=2)
                has_uwb_only = True

            i = j

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
        plt.title(f"Ground Truth - LiDAR Sichtbarkeit ({'Train' if self.mode else 'Val'} Split)", fontsize=14)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        total_samples = len(driver_present)
        lidar_samples = np.sum(driver_present == 1)
        uwb_only_samples = np.sum(driver_present == 0)

        print(f"\n=== {'Train' if self.mode else 'Validation'} Data Statistics ===")
        print(f"Total Samples: {total_samples}")
        print(f"UWB + LiDAR: {lidar_samples} ({100 * lidar_samples / total_samples:.1f}%)")
        print(f"UWB only: {uwb_only_samples} ({100 * uwb_only_samples / total_samples:.1f}%)")
        print(f"============================================\n")


    def visualize_sample(self, idx_in_list=0):
        if idx_in_list >= len(self.samples):
            print("Index out of range")
            return

        file_path, start_idx = self.samples[idx_in_list]
        print(f"Visualizing from {os.path.basename(file_path)}, Index {start_idx}")

        with h5py.File(file_path, 'r') as f:
            seq_end = start_idx + self.seq_length
            target_idx = seq_end + self.prediction_horizon - 1

            points = f['lidar_data'][start_idx]
            uwb = f['uwb_data'][start_idx]
            gt = f['ground_truth'][start_idx]
            gt_target = f['ground_truth'][target_idx]

            if target_idx > 0:
                prev_pos = f['ground_truth'][target_idx - 1]
                velocity = (gt_target - prev_pos) / 0.1
            else:
                velocity = np.zeros(2)

            grid = self._vectorized_grid_generation(points)

            fig, ax = plt.subplots(1, 2)

            if len(points) > 0:
                ax[0].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1)
            ax[0].scatter(uwb[0], uwb[1], c='r', marker='*', s=100, label='UWB')
            ax[0].scatter(gt[0], gt[1], c='g', marker='x', s=100, label='GT')
            ax[0].arrow(gt[0], gt[1], velocity[0] * 0.1, velocity[1] * 0.1,
                        head_width=0.1, head_length=0.05, fc='blue', ec='blue', label='Velocity')
            ax[0].set_title("Raw LiDAR + UWB + Velocity")
            ax[0].legend()
            ax[0].axis('equal')

            ax[1].imshow(grid[:, :, 0], origin='lower')
            ax[1].set_title("Heigth Grid")

            plt.tight_layout()
            plt.show()

            print(f"Position: {gt_target}")
            print(f"Velocity: {velocity} m/s (Magnitude: {np.linalg.norm(velocity):.2f} m/s)")

    def visualize_lidar_to_grid(self, idx_in_list=0):
        if idx_in_list >= len(self.samples):
            print("Index out of range")
            return

        file_path, start_idx = self.samples[idx_in_list]
        print(f"Visualizing from {os.path.basename(file_path)}, Index {start_idx}")

        with h5py.File(file_path, 'r') as f:

            points = f['lidar_data'][start_idx]
            gt = f['ground_truth'][start_idx]

            grid = self._vectorized_grid_generation(points)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            if len(points) > 0:
                ax[0].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1)
            ax[0].scatter(gt[0], gt[1], c='r', marker='x', s=100, label='GT')
            ax[0].set_title("Raw LiDAR + Ground Truth(GT)")
            ax[0].legend()
            ax[0].set_ylim(7, -7)
            ax[0].set_xlim(-1, 15)
            ax[0].set_aspect('equal', adjustable='box')

            im = ax[1].imshow(grid[:, :, 0], cmap='hot', origin='lower', aspect='auto')
            ax[1].set_title("Height Grid")
            plt.colorbar(im, ax=ax[1], label='Height [m]')

            plt.tight_layout()
            plt.show()

    def visualize_batch(self, batch_idx=0, save_path=None):
        for idx, (X, y) in enumerate(self.dataset):
            if idx == batch_idx:
                batch_size = X['grid_input'].shape[0]
                seq_length = X['grid_input'].shape[-1]

                fig, axes = plt.subplots(batch_size, min(seq_length, 3),
                                         figsize=(15, 4 * batch_size))
                fig.suptitle(f'Batch {batch_idx} | Batch Size: {batch_size}', fontsize=14, fontweight='bold')

                if batch_size == 1:
                    axes = axes.reshape(1, -1)
                elif min(seq_length, 3) == 1:
                    axes = axes.reshape(-1, 1)

                for b in range(batch_size):
                    for t in range(min(seq_length, 3)):
                        ax = axes[b, t]
                        frame = X['grid_input'][b, :, :, :, t].numpy()
                        frame_sum = np.sum(frame, axis=-1)
                        im = ax.imshow(frame_sum, cmap='hot', origin='lower')
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Batch visualization saved to: {save_path}")

                plt.show()
                return

        print(f"Batch {batch_idx} not found in dataset")



# Usage example
if __name__ == "__main__":
    dataset_folder = "dataset/train_val_test"

    # Train Generator
    train_gen = DataGenerator(
        hdf5_folder="dataset/train_val_test",
        mode='train',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        grid_size=500,
        seq_length=10,
        batch_size=8
    )


    print("Visualizing sample with velocity...")
    for i in range(1):
        #train_gen.visualize_sample(idx_in_list=10)
        train_gen.visualize_lidar_to_grid(idx_in_list=1)
        #train_gen.visualize_batch(batch_idx=8)

    # Test Dataset
    for batch_inputs, batch_outputs in train_gen.dataset.take(1):
        print("\n=== Batch Test ===")
        print(f"Grid Input Shape: {batch_inputs['grid_input'].shape}")
        print(f"UWB Input Shape: {batch_inputs['uwb_input'].shape}")
        print(f"Position Output Shape: {batch_outputs['position'].shape}")
        print(f"Velocity Aux Output Shape: {batch_outputs['velocity_aux'].shape}")
        print(f"Sample Velocity: {batch_outputs['velocity_aux'][0].numpy()}")
        break

    # Validation Generator
    test_gen = DataGenerator(
        hdf5_folder=dataset_folder,
        grid_size=500,
        grid_resolution=0.1,
        seq_length=10,
        batch_size=40,
        mode='test',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        use_velocity_auxiliary=True
    )

    # Validation Metrics ===
    print("\n=== Validation Pattern Visualization ===")
    test_gen.plot_validation_pattern()
    test_gen.plot_validation_pattern_split()

    print("\n=== Baseline Metrics (ohne Model) ===")
    baseline_metrics = test_gen.compute_baseline_metrics()

    # === With Model Predictions ===
    # load trained model
    import model as model_factory

    model = tf.keras.models.load_model(
        "saved_models/minimal_multimodal_model.keras",
        custom_objects={
            "TemporalAttention": model_factory.TemporalAttention,
            "ResidualDKF": model_factory.ResidualDKF,
            "AdaptiveSensorFusionDKF": model_factory.AdaptiveSensorFusionDKF,
        },
        compile=False,
        safe_mode=False
    )

    print("\n=== Metrics with Model Predictions ===")
    results = test_gen.compute_baseline_metrics(
        model=model,
        vel_output=True  # True if Model Position + Velocity
    )


    if isinstance(results, tuple):
        metrics, timing_stats = results
        print("\n📊 Inference Timing Summary:")
        print(f"  Inference FPS: {timing_stats['inference']['fps']:.1f} Hz")
        print(f"  Total FPS (incl. preprocessing): {timing_stats['total_per_sample']['fps']:.1f} Hz")
    else:
        metrics = results


    print("\n=== Sample Visualization ===")
    test_gen.visualize_sample(idx_in_list=10)
    test_gen.visualize_batch(batch_idx=0)
