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




def train_single_model(model_config,
                       train_gen,
                       val_gen,
                       grid_size,
                       grid_resolution,
                       seq_length,
                       batch_size,
                       epochs,
                       use_velocity_auxiliary):

    """
    Trains a single model using the given configuration
    """
    model_name = model_config['name']
    model_creator = model_config['creator']

    print("\n" + "=" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 70)

    train_dataset = train_gen.dataset
    val_dataset = val_gen.dataset

    # Create model
    model = model_creator(
        grid_size=grid_size,
        num_layers=train_gen.num_channels,
        sequence_length=seq_length,
        latent_dim=64,
        use_velocity_auxiliary=use_velocity_auxiliary,
        gru_units=96
    )

    # Compile model
    if use_velocity_auxiliary:
        model = model_factory.TrainingSetup.compile_model(model)
        monitor_metric = 'val_position_loss'
    else:
        model_factory.TrainingSetup.compile_model(model=model, loss='huber')
        monitor_metric = 'val_loss'

    # Save path
    save_model_path = f"saved_models/{model_name}.keras"

    # Callbacks
    callbacks = model_factory.TrainingSetup.get_callbacks(
        save_model_path,
        patience=8,
        monitor=monitor_metric
    )

    # Model summary
    print("\n" + "-" * 50)
    print("MODEL SUMMARY")
    print("-" * 50)
    model.summary()

    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")

    # Training
    print("\n" + "-" * 50)
    print("STARTING TRAINING")
    print("-" * 50)
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Velocity Auxiliary: {use_velocity_auxiliary}")
    print(f"Monitor Metric: {monitor_metric}")
    print("-" * 50 + "\n")

    history = model.fit(
        train_dataset,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=len(val_gen),
        callbacks=callbacks,
        verbose=1
    )

    # Training results
    print("\n" + "-" * 50)
    print(f"TRAINING COMPLETED: {model_name}")
    print("-" * 50)

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
    print(f"Model saved to: {save_model_path}")

    # Clear session to free memory
    tf.keras.backend.clear_session()

    # Evaluation
    """
    Evaluates a trained model
    """
    print("\n" + "-" * 50)
    print(f"EVALUATING MODEL: {save_model_path}")
    print("-" * 50)

    evaluator = tester.OfflineModelEvaluator(
        hdf5_path="dataset/test/dataset_val_cw.hdf5",
        model_path=save_model_path,
        grid_size=grid_size,
        grid_resolution=grid_resolution,
        seq_length=seq_length,
        vel_output=True
    )

    evaluator.run_inference()
    metrics = evaluator.compute_metrics()

    # Save evaluation metrics log
    log_path = os.path.join("saved_models", f"{model_name}_evaluation_log.txt")

    with open(log_path, "w") as f:
        f.write(f"MODEL NAME: {model_name}\n")
        f.write(f"MODEL PATH: {save_model_path}\n")
        f.write(f"BEST VALIDATION LOSS: {best_val_loss:.6f}\n\n")
        f.write("EVALUATION METRICS:\n")
        f.write("===================\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"\nEvaluation log saved to: {log_path}")

    results[model_name] = {
        'model_path': save_model_path,
        'best_val_loss': best_val_loss,
        'metrics': metrics,
        'history': history
    }

    # Clear session to free memory
    tf.keras.backend.clear_session()

    return save_model_path, history, best_val_loss, results


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
    # default for (almost) all models; the first model is explicitly overridden
    use_velocity_auxiliary_default = True
    batch_size = args.batch_size
    epochs = args.epochs


    # ========================================
    # MODEL CONFIGURATIONS
    # ========================================
    models_to_train = [
        {
            'name': 'minimal_multimodal_model',
            'creator': lambda **kwargs: model_factory.create_minimal_multimodal_model(
                grid_size=kwargs['grid_size'],
                num_layers=kwargs['num_layers'],
                sequence_length=kwargs['sequence_length'],
                gru_units=64  # Different GRU units for this model
            )
        },
        {
            'name': 'kalman_multimodal_model',
            'creator': model_factory.create_kalman_multimodal_model
        },
        {
            'name': 'fused_kalman_multimodal_model',
            'creator': model_factory.create_fused_kalman_multimodal_model
        },
        {
            'name': 'adaptive_fused_model',
            'creator': model_factory.create_adaptive_fused_model
        }
    ]

    # ========================================
    # TRAIN ALL MODELS
    # ========================================
    results = {}

    for idx, model_config in enumerate(models_to_train):
        # Disable velocity auxiliary for the first model (index 0)
        model_use_velocity_auxiliary = False if idx == 0 else use_velocity_auxiliary_default

        try:
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
                lidar_dropout_prob=0.05,  # 5% lidar dropout to simulate missing lidar data
                prediction_horizon=prediction_horizon,
                use_velocity_auxiliary=model_use_velocity_auxiliary,
                augment_uwb=augment_uwb
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
                lidar_dropout_prob=0.05,  # 5% lidar dropout to simulate missing lidar data
                prediction_horizon=prediction_horizon,
                use_velocity_auxiliary=model_use_velocity_auxiliary,
            )

            model_path, history, best_val_loss, results = train_single_model(
                model_config=model_config,
                train_gen=train_gen,
                val_gen=val_gen,
                grid_size=grid_size,
                grid_resolution=grid_resolution,
                seq_length=seq_length,
                batch_size=batch_size,
                epochs=epochs,
                use_velocity_auxiliary=model_use_velocity_auxiliary
            )

        except Exception as e:
            print(f"\nERROR training {model_config['name']}: {str(e)}")
            continue

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY - ALL MODELS")
    print("=" * 70)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Path: {result['model_path']}")
        print(f"  Best Val Loss: {result['best_val_loss']:.4f}")

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETED!")
    print("=" * 70)