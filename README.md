# Sensor Fusion 

Relative position estimation based on multimodal sensor fusion of Ultra-Wideband (UWB) and LiDAR data

> 📧 **Note**: Only sample data is included here. Full dataset is available here:  
> [Download full dataset](https://cloud.ovgu.de/s/i32f79eCJCHp9Kn)  
> or use helper scripts: download_dataset.sh (shell) download_dataset.py (python)

> Academic or research purposes, please contact:  
> [stefan.sass@ovgu.de](mailto:stefan.sass@ovgu.de)  
> [markus.hoefer@ovgu.de](mailto:markus.hoefer@ovgu.de)

## download dataset

- dependencies:

``` bash
sudo apt install wget unzip rsync
```

- shell script

``` bash
chmod +x download_dataset.sh
./download_dataset.sh
```

- python script:

``` python
python download_dataset.py
```

## 🗂️ Folder Structure

```
├── config
│   ├── rviz.rviz             # config for rviz2 (ros2) 
├── dataset
│   ├── test                  # test data ->
│   └── train_val_test        # trainings data (can be split in train/test/val via data generator)
├── utils
│   ├── hdf5_rviz2_player.py   # plays hdf5 files like rosbags to visualize in rviz2 (ros2) 
│   └── interactive_marker.py  # used for annotating the ground truth in raw rosbags
│   └── model_node.py          # node for use trained model in ros2
├── data_generator.py          # dataset creator
├── model.py                   # tensorflow models
├── test.py                    # script for test trained models
├── train.py                   # script for train model
├── train_all.py               # script for train all models sequencially     
```


## 🔬 Baseline Models

Baseline models are provided using TensorFlow. Each task comes with:

- Custom dataloaders

- Model architecture

- Training and testing scripts

- Pretrained models 

You can find them under:

- saved_models/best_models/



## 🔗 Citation

