Model: "adaptive_fused_model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 grid_input (InputLayer)     [(None, 500, 500, 1, 10)]    0         []                            
                                                                                                  
 permute (Permute)           (None, 10, 500, 500, 1)      0         ['grid_input[0][0]']          
                                                                                                  
 time_distributed (TimeDist  (None, 10, 125, 125, 32)     832       ['permute[0][0]']             
 ributed)                                                                                         
                                                                                                  
 time_distributed_1 (TimeDi  (None, 10, 125, 125, 32)     0         ['time_distributed[0][0]']    
 stributed)                                                                                       
                                                                                                  
 uwb_input (InputLayer)      [(None, 10, 2)]              0         []                            
                                                                                                  
 time_distributed_2 (TimeDi  (None, 10, 32)               0         ['time_distributed_1[0][0]']  
 stributed)                                                                                       
                                                                                                  
 adaptive_dkf (AdaptiveSens  (None, 10, 72)               37900     ['uwb_input[0][0]',           
 orFusionDKF)                                                        'time_distributed_2[0][0]']  
                                                                                                  
 bi_gru (Bidirectional)      (None, 10, 192)              97920     ['adaptive_dkf[0][0]']        
                                                                                                  
 temp_attn (TemporalAttenti  ((None, 192),                24833     ['bi_gru[0][0]']              
 on)                          (None, 10, 1))                                                      
                                                                                                  
 decoder (Dense)             (None, 64)                   12352     ['temp_attn[0][0]']           
                                                                                                  
 dropout (Dropout)           (None, 64)                   0         ['decoder[0][0]']             
                                                                                                  
 position (Dense)            (None, 2)                    130       ['dropout[0][0]']             
                                                                                                  
 velocity_aux (Lambda)       (None, 2)                    0         ['adaptive_dkf[0][0]']        
                                                                                                  
==================================================================================================
Total params: 173967 (679.56 KB)
Trainable params: 173967 (679.56 KB)
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
  0%|          | 0/5491 [00:00<?, ?it/s]2026-01-05 13:10:25.623850: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 13:10:25.717868: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 5491/5491 [02:19<00:00, 39.29it/s]

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
Mean [m]: 0.5040
RMSE [m]: 0.6302
Median [m]: 0.4198
P90 [m]: 0.9678
Max [m]: 4.7792
Count: 5491

--- Model Prediction UWB + LiDAR ---
Mean [m]: 0.4554
RMSE [m]: 0.5440
Median [m]: 0.4029
P90 [m]: 0.8063
Max [m]:   
Count: 4254

--- Model Prediction UWB only ---
Mean [m]: 0.6713
RMSE [m]: 0.8632
Median [m]: 0.4793
P90 [m]: 1.3897
Max [m]: 4.7792
Count: 1237

========================================


📊 Inference Timing Summary:
  Inference FPS: 295.7 Hz
  Total FPS (incl. preprocessing): 42.5 Hz
100%|██████████| 408/408 [00:00<00:00, 26791.01it/s]
100%|██████████| 408/408 [00:00<00:00, 24422.03it/s]

=== Validierungsdaten Statistik 1. Trajektorie ===
Gesamt Samples: 408
UWB + LiDAR: 131 (32.1%)
Nur UWB: 277 (67.9%)
===================================

Running model inference...
100%|██████████| 408/408 [00:09<00:00, 41.82it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 0.7696
RMSE [m]: 0.8356
Median [m]: 0.8081
P90 [m]: 1.1466
Max [m]: 1.5698
Count: 399

--- Model Prediction UWB + Lidar ---
Mean [m]: 0.5953
RMSE [m]: 0.6859
Median [m]: 0.5833
P90 [m]: 1.0173
Max [m]: 1.5698
Count: 131

--- Model Prediction UWB only ---
Mean [m]: 0.8548
RMSE [m]: 0.8997
Median [m]: 0.8732
P90 [m]: 1.1633
Max [m]: 1.5687
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
  0%|          | 0/507 [00:00<?, ?it/s]2026-01-05 14:48:31.232536: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8904
2026-01-05 14:48:31.327054: I external/local_tsl/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
100%|██████████| 507/507 [00:14<00:00, 36.16it/s]
✅ Inference complete

========== Evaluation Metrics ==========

--- Model Prediction ---
Mean [m]: 1.7017
RMSE [m]: 2.3315
Median [m]: 1.0337
P90 [m]: 4.1903
Max [m]: 9.7719
Count: 498

--- Model Prediction UWB + Lidar ---
Mean [m]: 1.0264
RMSE [m]: 1.7334
Median [m]: 0.5728
P90 [m]: 2.0345
Max [m]: 9.7719
Count: 259

--- Model Prediction UWB only ---
Mean [m]: 2.4334
RMSE [m]: 2.8410
Median [m]: 2.2842
P90 [m]: 4.6855
Max [m]: 6.8887
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

