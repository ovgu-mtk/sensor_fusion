import os
import tensorflow as tf
import data_generator as dataloader
import model as model_factory
import test as tester


# set cuda device 0-3
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Force TensorFlow to use GPU 1
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)



if __name__ == "__main__":

    # print tf version and dataset size
    print(f"tensorflow verion: {tf.__version__}")

    # Path to dataset
    dataset_folder = "dataset/train_val_test/"
    grid_resolution = 0.1
    seq_length = 10
    grid_size = 500
    batch_size = 40
    epochs = 100
    #use_velocity_auxiliary = True
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
        mode='test',
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

    # set use_velocity_auxiliary to false
    model = model_factory.create_minimal_multimodal_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length, gru_units=64)

    # set use_velocity_auxiliary to true
    #model = model_factory.create_kalman_multimodal_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length,latent_dim=64,use_velocity_auxiliary=use_velocity_auxiliary,gru_units=96)
    #model = model_factory.create_fused_kalman_multimodal_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length,latent_dim=64,use_velocity_auxiliary=use_velocity_auxiliary,gru_units=96)
    #model = model_factory.create_adaptive_fused_model(grid_size=grid_size,num_layers=train_gen.num_channels,sequence_length=seq_length,latent_dim=64,use_velocity_auxiliary=use_velocity_auxiliary,gru_units=96)



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
        epochs=epochs,
        validation_data=val_dataset,
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