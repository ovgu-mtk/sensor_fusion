Model: "kalman_multimodal_model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 grid_input (InputLayer)     [(None, 500, 500, 1, 10)]    0         []                            
                                                                                                  
 uwb_input (InputLayer)      [(None, 10, 2)]              0         []                            
                                                                                                  
 permute (Permute)           (None, 10, 500, 500, 1)      0         ['grid_input[0][0]']          
                                                                                                  
 dkf_velocity (ResidualDKF)  (None, 10, 72)               22088     ['uwb_input[0][0]']           
                                                                                                  
 time_distributed (TimeDist  (None, 10, 125, 125, 32)     832       ['permute[0][0]']             
 ributed)                                                                                         
                                                                                                  
 uwb_features (Dense)        (None, 10, 64)               4672      ['dkf_velocity[0][0]']        
                                                                                                  
 time_distributed_1 (TimeDi  (None, 10, 125, 125, 32)     0         ['time_distributed[0][0]']    
 stributed)                                                                                       
                                                                                                  
 layer_normalization (Layer  (None, 10, 64)               128       ['uwb_features[0][0]']        
 Normalization)                                                                                   
                                                                                                  
 time_distributed_2 (TimeDi  (None, 10, 32)               0         ['time_distributed_1[0][0]']  
 stributed)                                                                                       
                                                                                                  
 fusion (Concatenate)        (None, 10, 96)               0         ['layer_normalization[0][0]', 
                                                                     'time_distributed_2[0][0]']  
                                                                                                  
 dense (Dense)               (None, 10, 128)              12416     ['fusion[0][0]']              
                                                                                                  
 dropout (Dropout)           (None, 10, 128)              0         ['dense[0][0]']               
                                                                                                  
 bi_gru (Bidirectional)      (None, 10, 192)              130176    ['dropout[0][0]']             
                                                                                                  
 temp_attn (TemporalAttenti  ((None, 192),                24833     ['bi_gru[0][0]']              
 on)                          (None, 10, 1))                                                      
                                                                                                  
 decoder (Dense)             (None, 64)                   12352     ['temp_attn[0][0]']           
                                                                                                  
 dropout_1 (Dropout)         (None, 64)                   0         ['decoder[0][0]']             
                                                                                                  
 position (Dense)            (None, 2)                    130       ['dropout_1[0][0]']           
                                                                                                  
 velocity_aux (Lambda)       (None, 2)                    0         ['dkf_velocity[0][0]']        
                                                                                                  
==================================================================================================
Total params: 207627 (811.04 KB)
Trainable params: 207627 (811.04 KB)
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
  0%|          | 0/5491 [00:00<?, ?it/s]2026-01-05 11:00:30.525650: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 11:00:30.664276: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 5491/5491 [02:50<00:00, 32.27it/s]
100%|██████████| 408/408 [00:00<00:00, 115806.73it/s]

============================================================
⏱️  INFERENCE TIMING STATISTICS
============================================================
Total samples processed: 5491
Total time: 170.15 s

--- Preprocessing (Grid Generation) ---
  Mean:   25.21 ms
  Median: 24.49 ms
  Std:    6.39 ms
  Min:    12.91 ms
  Max:    66.02 ms

--- Model Inference ---
  Mean:   3.69 ms
  Median: 3.17 ms
  Std:    21.29 ms
  Min:    2.12 ms
  Max:    1578.80 ms
  FPS:    270.6 Hz

--- Total (Preprocessing + Inference) ---
  Mean:   28.91 ms
  Median: 27.92 ms
  FPS:    34.6 Hz
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
Mean [m]: 0.4888
RMSE [m]: 0.6242
Median [m]: 0.4044
P90 [m]: 1.0008
Max [m]: 4.7947
Count: 5491

--- Model Prediction UWB + LiDAR ---
Mean [m]: 0.4395
RMSE [m]: 0.5337
Median [m]: 0.3832
P90 [m]: 0.7908
Max [m]: 4.3009
Count: 4254

--- Model Prediction UWB only ---
Mean [m]: 0.6586
RMSE [m]: 0.8658
Median [m]: 0.4808
P90 [m]: 1.3736
Max [m]: 4.7947
Count: 1237

========================================


📊 Inference Timing Summary:
  Inference FPS: 270.6 Hz
  Total FPS (incl. preprocessing): 34.6 Hz
100%|██████████| 408/408 [00:00<00:00, 24188.33it/s]

=== Validierungsdaten Statistik 1. Trajektorie ===
Gesamt Samples: 408
UWB + LiDAR: 131 (32.1%)
Nur UWB: 277 (67.9%)
===================================

Running model inference...
100%|██████████| 408/408 [00:14<00:00, 27.32it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 0.7274
RMSE [m]: 0.8194
Median [m]: 0.6770
P90 [m]: 1.2511
Max [m]: 1.6907
Count: 399

--- Model Prediction UWB + Lidar ---
Mean [m]: 0.5344
RMSE [m]: 0.6083
Median [m]: 0.5081
P90 [m]: 0.9676
Max [m]: 1.3842
Count: 131

--- Model Prediction UWB only ---
Mean [m]: 0.8218
RMSE [m]: 0.9049
Median [m]: 0.7722
P90 [m]: 1.3540
Max [m]: 1.6907
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
  0%|          | 0/507 [00:00<?, ?it/s]2026-01-05 14:55:43.680765: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 14:55:43.809451: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 507/507 [00:21<00:00, 23.80it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 1.8345
RMSE [m]: 2.3606
Median [m]: 1.2564
P90 [m]: 4.2164
Max [m]: 5.7392
Count: 498

--- Model Prediction UWB + Lidar ---
Mean [m]: 0.9138
RMSE [m]: 1.2333
Median [m]: 0.7055
P90 [m]: 1.8443
Max [m]: 4.5201
Count: 259

--- Model Prediction UWB only ---
Mean [m]: 2.8323
RMSE [m]: 3.1565
Median [m]: 2.6740
P90 [m]: 5.0800
Max [m]: 5.7392
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