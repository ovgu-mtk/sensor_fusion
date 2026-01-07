Model: "fused_kalman_multimodal_model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 grid_input (InputLayer)     [(None, 500, 500, 1, 10)]    0         []                            
                                                                                                  
 permute (Permute)           (None, 10, 500, 500, 1)      0         ['grid_input[0][0]']          
                                                                                                  
 time_distributed (TimeDist  (None, 10, 125, 125, 32)     832       ['permute[0][0]']             
 ributed)                                                                                         
                                                                                                  
 time_distributed_1 (TimeDi  (None, 10, 125, 125, 32)     0         ['time_distributed[0][0]']    
 stributed)                                                                                       
                                                                                                  
 time_distributed_2 (TimeDi  (None, 10, 32)               0         ['time_distributed_1[0][0]']  
 stributed)                                                                                       
                                                                                                  
 uwb_input (InputLayer)      [(None, 10, 2)]              0         []                            
                                                                                                  
 grid_embedding (TimeDistri  (None, 10, 32)               1056      ['time_distributed_2[0][0]']  
 buted)                                                                                           
                                                                                                  
 sensor_fusion (Concatenate  (None, 10, 34)               0         ['uwb_input[0][0]',           
 )                                                                   'grid_embedding[0][0]']      
                                                                                                  
 fused_dkf (ResidualDKF)     (None, 10, 72)               30280     ['sensor_fusion[0][0]']       
                                                                                                  
 bi_gru (Bidirectional)      (None, 10, 192)              97920     ['fused_dkf[0][0]']           
                                                                                                  
 temp_attn (TemporalAttenti  ((None, 192),                24833     ['bi_gru[0][0]']              
 on)                          (None, 10, 1))                                                      
                                                                                                  
 decoder (Dense)             (None, 64)                   12352     ['temp_attn[0][0]']           
                                                                                                  
 dropout (Dropout)           (None, 64)                   0         ['decoder[0][0]']             
                                                                                                  
 position (Dense)            (None, 2)                    130       ['dropout[0][0]']             
                                                                                                  
 velocity_aux (Lambda)       (None, 2)                    0         ['fused_dkf[0][0]']           
                                                                                                  
==================================================================================================
Total params: 167403 (653.92 KB)
Trainable params: 167403 (653.92 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
Scanning 7 files for test split...
Generator initialized. Mode: test, Samples: 5486, Batch: 1, Velocity Auxiliary: True

=== Metrics with Model Predictions ===
Loading validation data from 7 files...
✅ Loaded 5491 validation samples

🔄 Running model inference...
Loading validation data from 7 files...
✅ Loaded 5491 validation samples
Running model inference on 5491 samples...
  0%|          | 0/5491 [00:00<?, ?it/s]2026-01-05 13:04:16.030133: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 13:04:16.203965: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 5491/5491 [01:48<00:00, 50.62it/s]

============================================================
⏱️  INFERENCE TIMING STATISTICS
============================================================
Total samples processed: 5491
Total time: 108.48 s

--- Preprocessing (Grid Generation) ---
  Mean:   15.38 ms
  Median: 13.30 ms
  Std:    4.67 ms
  Min:    10.72 ms
  Max:    84.75 ms

--- Model Inference ---
  Mean:   2.61 ms
  Median: 2.27 ms
  Std:    17.47 ms
  Min:    1.87 ms
  Max:    1294.91 ms
  FPS:    383.4 Hz

--- Total (Preprocessing + Inference) ---
  Mean:   17.99 ms
  Median: 15.68 ms
  FPS:    55.6 Hz
============================================================

✅ Model inference complete: 5491 predictions

========== Validation Metrics ==========

--- UWB Raw ---
Mean [m]: 1.0301
RMSE [m]: 1.2427
Median [m]: 0.8981
P90 [m]: 1.8655
Max [m]: 5.9958
Count: 5491

--- UWB Filtered (orig) ---
Mean [m]: 1.0296
RMSE [m]: 1.2410
Median [m]: 0.8977
P90 [m]: 1.8725
Max [m]: 5.9958
Count: 5491

--- UWB UKF (const accel) ---
Mean [m]: 0.9630
RMSE [m]: 1.1466
Median [m]: 0.8409
P90 [m]: 1.7553
Max [m]: 4.8452
Count: 5491

--- Model Prediction ---
Mean [m]: 0.5071
RMSE [m]: 0.6340
Median [m]: 0.4265
P90 [m]: 0.9450
Max [m]: 4.9481
Count: 5491

--- Model Prediction UWB + LiDAR ---
Mean [m]: 0.4601
RMSE [m]: 0.5479
Median [m]: 0.4111
P90 [m]: 0.8055
Max [m]: 4.3673
Count: 4254

--- Model Prediction UWB only ---
Mean [m]: 0.6690
RMSE [m]: 0.8672
Median [m]: 0.4874
P90 [m]: 1.3645
Max [m]: 4.9481
Count: 1237

========================================


📊 Inference Timing Summary:
  Inference FPS: 383.4 Hz
  Total FPS (incl. preprocessing): 55.6 Hz
100%|██████████| 408/408 [00:00<00:00, 39762.90it/s]
100%|██████████| 408/408 [00:00<00:00, 79306.52it/s]
  0%|          | 0/408 [00:00<?, ?it/s]
=== Validierungsdaten Statistik 1. Trajektorie ===
Gesamt Samples: 408
UWB + LiDAR: 131 (32.1%)
Nur UWB: 277 (67.9%)
===================================

Running model inference...
100%|██████████| 408/408 [00:10<00:00, 39.06it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 0.7765
RMSE [m]: 0.8579
Median [m]: 0.7807
P90 [m]: 1.2261
Max [m]: 1.5671
Count: 399

--- Model Prediction UWB + Lidar ---
Mean [m]: 0.5251
RMSE [m]: 0.6074
Median [m]: 0.5069
P90 [m]: 0.9675
Max [m]: 1.3387
Count: 131

--- Model Prediction UWB only ---
Mean [m]: 0.8994
RMSE [m]: 0.9567
Median [m]: 0.9469
P90 [m]: 1.2625
Max [m]: 1.5671
Count: 268

--- UWB Raw ---
Mean [m]: 1.0677
RMSE [m]: 1.3235
Median [m]: 0.9071
P90 [m]: 2.1173
Max [m]: 4.4353
Count: 399

--- UWB Filtered (orig) ---
Mean [m]: 0.9535
RMSE [m]: 1.0870
Median [m]: 0.8618
P90 [m]: 1.6837
Max [m]: 2.4052
Count: 399

--- UWB UKF (const accel) ---
Mean [m]: 0.9656
RMSE [m]: 1.1401
Median [m]: 0.8310
P90 [m]: 1.8073
Max [m]: 2.9277
Count: 399

========================================

=== Validierungsdaten Statistik 2. Trajektorie ===
Gesamt Samples: 507
UWB + LiDAR: 268 (52.9%)
Nur UWB: 239 (47.1%)
===================================

Running model inference...
  0%|          | 0/507 [00:00<?, ?it/s]2026-01-05 14:53:08.868079: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 14:53:08.965474: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 507/507 [00:12<00:00, 41.74it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 1.3784
RMSE [m]: 1.7880
Median [m]: 0.9092
P90 [m]: 3.0301
Max [m]: 4.5018
Count: 498

--- Model Prediction UWB + Lidar ---
Mean [m]: 0.7312
RMSE [m]: 0.9560
Median [m]: 0.5575
P90 [m]: 1.3888
Max [m]: 3.3721
Count: 259

--- Model Prediction UWB only ---
Mean [m]: 2.0797
RMSE [m]: 2.3814
Median [m]: 2.1267
P90 [m]: 3.8299
Max [m]: 4.5018
Count: 239

--- UWB Raw ---
Mean [m]: 3.1027
RMSE [m]: 8.1062
Median [m]: 1.9432
P90 [m]: 4.4061
Max [m]: 117.2779
Count: 498

--- UWB Filtered (orig) ---
Mean [m]: 2.6749
RMSE [m]: 4.6134
Median [m]: 1.6528
P90 [m]: 4.8004
Max [m]: 35.5913
Count: 498

--- UWB UKF (const accel) ---
Mean [m]: 2.6400
RMSE [m]: 5.0259
Median [m]: 1.7629
P90 [m]: 3.8510
Max [m]: 49.6033
Count: 498

========================================