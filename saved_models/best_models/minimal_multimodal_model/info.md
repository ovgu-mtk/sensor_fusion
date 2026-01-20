Model: "minimal_multimodal_model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 grid_input (InputLayer)     [(None, 500, 500, 1, 10)]    0         []                            
                                                                                                  
 permute (Permute)           (None, 10, 500, 500, 1)      0         ['grid_input[0][0]']          
                                                                                                  
 time_distributed (TimeDist  (None, 10, 125, 125, 32)     832       ['permute[0][0]']             
 ributed)                                                                                         
                                                                                                  
 uwb_input (InputLayer)      [(None, 10, 2)]              0         []                            
                                                                                                  
 time_distributed_1 (TimeDi  (None, 10, 125, 125, 32)     0         ['time_distributed[0][0]']    
 stributed)                                                                                       
                                                                                                  
 gru (GRU)                   (None, 10, 64)               13056     ['uwb_input[0][0]']           
                                                                                                  
 time_distributed_2 (TimeDi  (None, 10, 32)               0         ['time_distributed_1[0][0]']  
 stributed)                                                                                       
                                                                                                  
 concatenate (Concatenate)   (None, 10, 96)               0         ['gru[0][0]',                 
                                                                     'time_distributed_2[0][0]']  
                                                                                                  
 gru_1 (GRU)                 (None, 64)                   31104     ['concatenate[0][0]']         
                                                                                                  
 dense (Dense)               (None, 32)                   2080      ['gru_1[0][0]']               
                                                                                                  
 position (Dense)            (None, 2)                    66        ['dense[0][0]']               
                                                                                                  
==================================================================================================
Total params: 47138 (184.13 KB)
Trainable params: 47138 (184.13 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
Scanning 7 files for test split...
Generator initialized. Mode: test, Samples: 5486, Batch: 1, Velocity Auxiliary: True

=== Metrics with Model Predictions ===
Loading validation data from 7 files...
✅ Loaded 5491 validation samples

🔄 Running model inference...
Loading validation data from 7 files...
  0%|          | 0/5491 [00:00<?, ?it/s]✅ Loaded 5491 validation samples
Running model inference on 5491 samples...
2026-01-05 10:39:45.088202: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 10:39:45.289714: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 5491/5491 [02:14<00:00, 40.76it/s]
100%|██████████| 408/408 [00:00<00:00, 214876.45it/s]

============================================================
⏱️  INFERENCE TIMING STATISTICS
============================================================
Total samples processed: 54865
Total time: 2836.55 s

--- Preprocessing (Grid Generation) ---
  Mean:   24.45 ms
  Median: 24.22 ms
  Std:    1.21 ms
  Min:    22.24 ms
  Max:    35.60 ms

--- Model Inference ---
  Mean:   24.18 ms
  Median: 23.36 ms
  Std:    7.82 ms
  Min:    18.84 ms
  Max:    1771.34 ms
  FPS:    41.3 Hz

--- Total (Preprocessing + Inference) ---
  Mean:   48.64 ms
  Median: 48.02 ms
  FPS:    20.6 Hz
============================================================

✅ Model inference complete: 54865 predictions


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
Mean [m]: 0.4838
RMSE [m]: 0.5968
Median [m]: 0.4018
P90 [m]: 0.9373
Max [m]: 4.5029
Count: 5491

--- Model Prediction UWB + LiDAR ---
Mean [m]: 0.4470
RMSE [m]: 0.5328
Median [m]: 0.3962
P90 [m]: 0.8071
Max [m]: 3.6949
Count: 4254

--- Model Prediction UWB only ---
Mean [m]: 0.6103
RMSE [m]: 0.7777
Median [m]: 0.4372
P90 [m]: 1.3146
Max [m]: 4.5029
Count: 1237

========================================


📊 Inference Timing Summary:
  Inference FPS: 527.4 Hz
  Total FPS (incl. preprocessing): 44.2 Hz
100%|██████████| 408/408 [00:00<00:00, 24330.36it/s]

=== Validierungsdaten Statistik 1. Trajektorie ===
Gesamt Samples: 408
UWB + LiDAR: 131 (32.1%)
Nur UWB: 277 (67.9%)
===================================

Running model inference...
100%|██████████| 408/408 [00:08<00:00, 47.83it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 0.8369
RMSE [m]: 0.9350
Median [m]: 0.8141
P90 [m]: 1.4042
Max [m]: 1.6737
Count: 399

--- Model Prediction UWB + Lidar ---
Mean [m]: 0.5645
RMSE [m]: 0.6406
Median [m]: 0.5544
P90 [m]: 0.9662
Max [m]: 1.4101
Count: 131

--- Model Prediction UWB only ---
Mean [m]: 0.9700
RMSE [m]: 1.0493
Median [m]: 1.0175
P90 [m]: 1.5185
Max [m]: 1.6737
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
2026-01-05 14:58:30.759300: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 14:58:30.856311: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 507/507 [00:14<00:00, 36.06it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 1.9442
RMSE [m]: 2.7103
Median [m]: 1.1226
P90 [m]: 5.3629
Max [m]: 9.1592
Count: 498

--- Model Prediction UWB + Lidar ---
Mean [m]: 0.9491
RMSE [m]: 1.3689
Median [m]: 0.6878
P90 [m]: 1.9805
Max [m]: 5.8970
Count: 259

--- Model Prediction UWB only ---
Mean [m]: 3.0227
RMSE [m]: 3.6436
Median [m]: 2.1631
P90 [m]: 6.1877
Max [m]: 9.1592
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