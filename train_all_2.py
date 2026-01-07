import os
import tensorflow as tf
import data_generator as dataloader
import model as model_factory
import test as tester

# set cuda device 0-3 -> fallback gpu0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    if len(gpus) == 1:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Force TensorFlow to use GPU 1
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    else:
        try:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')  # Force TensorFlow to use GPU 1
            tf.config.experimental.set_memory_growth(gpus[1], True)
        except RuntimeError as e:
            print(e)


def train_single_model(model_config,
                       train_gen,
                       val_gen,
                       grid_size,
                       grid_resolution,
                       seq_length,
                       grid_type,
                       batch_size,
                       epochs,
                       use_velocity_auxiliary):

    """
    Trainiert ein einzelnes Modell mit der gegebenen Konfiguration
    """
    model_name = model_config['name']
    model_name = model_name + "_2"
    model_creator = model_config['creator']

    print("\n" + "=" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 70)

    train_dataset = train_gen.dataset
    val_dataset = val_gen.dataset

    # Modell erstellen
    model = model_creator(
        grid_size=grid_size,
        num_layers=train_gen.num_channels,
        sequence_length=seq_length,
        latent_dim=64,
        use_velocity_auxiliary=use_velocity_auxiliary,
        gru_units=96
    )

    # Modell kompilieren
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

    # Model Summary
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

    # Training Results
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
        Evaluiert ein trainiertes Modell
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
        grid_type=grid_type,
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

    print(f"tensorflow version: {tf.__version__}")

    # ========================================
    # DATASET CONFIGURATION
    # ========================================
    dataset_folder = "dataset/train_val_test/"
    grid_resolution = 0.1
    seq_length = 10
    grid_size = 500
    batch_size = 40
    epochs = 100
    augment_uwb = False
    # default für (fast) alle Modelle; das erste Modell wird explizit überschrieben
    use_velocity_auxiliary_default = True
    grid_type = 'single_height_map'
    prediction_horizon = 1

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
                gru_units=64  # Unterschiedliche GRU units für dieses Modell
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
        # Für das erste Modell (Index 0) velocity auxiliary ausschalten
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
                lidar_dropout_prob=0.0,  # 5 % lidar dropout to simulate no lidar data available
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
                mode='test',
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                lidar_dropout_prob=0.05,  # 5 % lidar dropout to simulate no lidar data available
                prediction_horizon=prediction_horizon,
                use_velocity_auxiliary=model_use_velocity_auxiliary,
            )


            model_path, history, best_val_loss, results = train_single_model(
                model_config=model_config,
                train_gen=train_gen,
                val_gen=val_gen,
                grid_size=grid_size,
                grid_resolution=grid_resolution,
                grid_type=grid_type,
                seq_length=seq_length,
                batch_size=batch_size,
                epochs=epochs,
                use_velocity_auxiliary=model_use_velocity_auxiliary
            )

        except Exception as e:
            print(f"\n ERROR training {model_config['name']}: {str(e)}")
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