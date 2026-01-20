import os
import argparse
import tensorflow as tf
import data_generator as dataloader
import model as model_factory
import test as tester

def str_to_bool(value):
    """Converts 1/0 or true/false strings to actual boolean."""
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(value).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value (1/0 or true/false) expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Train semantic segmentation model.")
    parser.add_argument('--gpu', type=int, choices=range(0, 10), default=0, help='GPU index to use (0-9), faalback to gpu 0. Default: 0')
    parser.add_argument('--model', type=int, choices=range(0, 4), default=0, help='Model index. '
                                                                                  '0-Minimal, '
                                                                                  '1-Kalman, '
                                                                                  '2-Fused-Kalman'
                                                                                  '3-Adaptive-Kalman. Default: 0')
    parser.add_argument('--ds_folder', default="dataset/train_val_test/",
                        help='Dataset location, default: "dataset/train_val_test/"')
    parser.add_argument('--grid_resolution', type=float, default=0.1, help='Grid resolution. Default: 0.1')
    parser.add_argument('--seq_length', type=int, default=10, help='Length of past sequence. Default: 10')
    parser.add_argument('--grid_size', type=int, default=500, help='Grid size for height map. Default: 500')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='The number of future time steps the model predicts into the future . Default: 1')
    parser.add_argument('--augment_uwb', type=str_to_bool, default=True,
                        help='Set augmentation of UWB signal true(1)/false(0). Default: 1')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs. Default: 100')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size. Default: 40')
    return parser.parse_args()

def set_gpu(gpu_index):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        except Exception as e:
            print(f"Failed to use GPU {gpu_index}, falling back to GPU 0. Error: {e}")
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)



if __name__ == "__main__":
    args = parse_args()

    print(f"tensorflow version: {tf.__version__}")

    # Set GPU
    set_gpu(args.gpu)

    # ========================================
    # DATASET CONFIGURATION
    # ========================================
    dataset_folder = args.ds_folder
    grid_resolution = args.grid_resolution
    seq_length = args.seq_length
    grid_size = args.grid_size
    prediction_horizon = args.prediction_horizon
    augment_uwb = args.augment_uwb
    batch_size = args.batch_size
    epochs = args.epochs

    # print tf version and dataset size
    print(f"tensorflow verion: {tf.__version__}")

    # velocity for model aoutput
    use_velocity_auxiliary = True
    if args.model == 0:
        use_velocity_auxiliary = False


    # Prediction horizon (wichtig für Velocity-Modell!)
    prediction_horizon = 1  # 0.1s in die Zukunft bei 10Hz

    train_gen = dataloader.DataGenerator(
        hdf5_folder=dataset_folder,
        grid_size=grid_size,
        grid_resolution=grid_resolution,
        seq_length=seq_length,
        batch_size=batch_size,
        mode='train',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        lidar_dropout_prob=0.05,  # 5 % lidar dropout to simulate no lidar data available
        prediction_horizon=prediction_horizon,
        use_velocity_auxiliary=use_velocity_auxiliary,
    )

    val_gen = dataloader.DataGenerator(
        hdf5_folder=dataset_folder,
        grid_size=grid_size,
        grid_resolution=grid_resolution,
        seq_length=seq_length,
        batch_size=batch_size,
        mode='val',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        lidar_dropout_prob=0.05,  # 5 % lidar dropout to simulate no lidar data available
        prediction_horizon=prediction_horizon,
        use_velocity_auxiliary=use_velocity_auxiliary,
    )


    # Datasets extrahieren
    train_dataset = train_gen.dataset
    val_dataset = val_gen.dataset

    model_name = "model_v1"
    save_model_path = "saved_models/" + str(model_name) + ".keras"

    # create model from args
    if args.model == 0:
        model = model_factory.create_minimal_multimodal_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length, gru_units=64)
    elif args.model == 1:
        model = model_factory.create_kalman_multimodal_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length,latent_dim=64,use_velocity_auxiliary=use_velocity_auxiliary,gru_units=96)
    elif args.model == 2:
        model = model_factory.create_fused_kalman_multimodal_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length,latent_dim=64,use_velocity_auxiliary=use_velocity_auxiliary,gru_units=96)
    elif args.model == 3:
        model = model_factory.create_adaptive_fused_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length,latent_dim=64,use_velocity_auxiliary=use_velocity_auxiliary,gru_units=96)
    else:
        print("wrong index: 0-Minimal, 1-Kalman, 2-Fused-Kalman, 3-Adaptive-Kalman")


    # Modell kompilieren
    if use_velocity_auxiliary:
        model = model_factory.TrainingSetup.compile_model(model)
        monitor_metric = 'val_position_loss'
    else:
        model_factory.TrainingSetup.compile_model(model=model, loss='huber')
        monitor_metric = 'val_loss'

    # ========================================
    # CALLBACKS
    # ========================================

    # Monitor: Bei Multi-Output ist es 'val_loss' (kombiniert) oder 'val_position_loss' (spezifisch)
    if use_velocity_auxiliary:
        monitor_metric = 'val_position_loss'  # Fokus auf Position
    else:
        monitor_metric = 'val_loss'

    callbacks = model_factory.TrainingSetup.get_callbacks(
        save_model_path,
        patience=8,  # Etwas mehr Geduld
        monitor=monitor_metric
    )

    # ========================================
    # MODEL SUMMARY
    # ========================================

    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    model.summary()

    # Parameter count
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")

    # ========================================
    # TRAINING
    # ========================================

    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Velocity Auxiliary: {use_velocity_auxiliary}")
    print(f"Monitor Metric: {monitor_metric}")
    print("=" * 50 + "\n")


    history = model.fit(
        train_dataset,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=len(val_gen),
        callbacks=callbacks,
        verbose=1
    )

    # ========================================
    # TRAINING HISTORY
    # ========================================

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    if use_velocity_auxiliary:
        final_train_pos_loss = history.history['position_loss'][-1]
        final_val_pos_loss = history.history['val_position_loss'][-1]
        final_train_vel_loss = history.history['velocity_aux_loss'][-1]
        final_val_vel_loss = history.history['val_velocity_aux_loss'][-1]

        print(f"Final Training Position Loss: {final_train_pos_loss:.4f}")
        print(f"Final Validation Position Loss: {final_val_pos_loss:.4f}")
        print(f"Final Training Velocity Loss: {final_train_vel_loss:.4f}")
        print(f"Final Validation Velocity Loss: {final_val_vel_loss:.4f}")

        best_val_loss = min(history.history['val_position_loss'])
        best_epoch = history.history['val_position_loss'].index(best_val_loss) + 1
    else:
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")

        best_val_loss = min(history.history['val_loss'])
        best_epoch = history.history['val_loss'].index(best_val_loss) + 1

    print(f"\nBest Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")

    # ========================================
    # EVALUATION
    # ========================================

    print("\n" + "=" * 50)
    print("STARTING EVALUATION")
    print("=" * 50)

    # ========================================
    # Evaluation
    # ========================================
    evaluator = tester.OfflineModelEvaluator(
        hdf5_path="dataset/test/dataset_val_cw.hdf5",
        model_path=save_model_path,
        grid_size=grid_size,
        grid_resolution=grid_resolution,
        seq_length=seq_length,
        vel_output=use_velocity_auxiliary  # predict velocity and position -> set true, pos only false
    )

    evaluator.run_inference()
    # evaluator.plot_results()
    metrics = evaluator.compute_metrics()