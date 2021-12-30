# Driver-drowsiness-classification-based-on-reaction-time
Classifiying driver drowsiness based on the dataset from the paper ["Multi-channel EEG recordings during a sustained-attention driving task"](https://www.nature.com/articles/s41597-019-0027-4.pdf). 

## Project Topic & Goal
The goal of this project is the development of subject-dependent EEG-based driver drowsiness state (i.e., awake, drowsy) classification model.
- Estimating driver fatigue is an important issue for traffic safety and user-centered brain-computer interface.
- The drowsy state is caused by various factors such as fatigue, workload, and distraction.
- Physiological features are used to assess drowsiness because they are continuously available and could be considered as an objective, more direct, measure of the mental state.

## Dataset

Title: Multi-channel EEG recordings during a sustained-attention driving task

Author: Z. Cao, C.-H. Chuang, J.-K. King, and C.-T. Lin

Journal: Scientific data

Year: 2019

Original dataset: https://figshare.com/articles/dataset/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task/6427334/5

Preprocessed dataset: https://figshare.com/articles/dataset/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task_preprocessed_dataset_/7666055/3

### Experimental protocol
- 60-90 minutes simulated driving on an empty highway
- Measure reaction time while performing a sustained-attention task
  >- Steer the wheel when a participant recognizes a random lane-deviation (5-10s)
  >- Reaction time (RT) = response onset - deviation onset
- 30-channel EEG data for 27 subjects (22-28 years)
  >- FP1, FP2, F7, F3, FZ, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, CZ, C4, T4, TP7, CP3, CPZ, CP4, TP8, T5, P3, PZ, P4, T6, O1, OZ, O2

### Data selection and labeling
- Select 14 suhbject data
- Data preprocessing:
  >- 1-50Hz bandpass filtering for the data pre-processing
  >- Down-sample to 250Hz
  >- Linearly interpolate the RT between the lane-deviation happening
  >- Segmentation


## Method & Performance
To improve the performance of the classification, we utilized the ESTCNN with minor modifications, which was derived the higher performance in the drowsiness classification.
![drow_1](https://user-images.githubusercontent.com/57162425/147727872-c6fcadc6-259b-4493-a1c8-1d5df88747fd.png)

### Experimental Setup
1. We DO NOT change the "dataloader" for a fair comparison with baseline performance.
2. Best model selection was performed from the balanced accuracy in the validation set.
3. Hyperparameters:
  >- Epochs: 100
  >- Batch size: 8
  >- Channels: 30
  >- Learning rate: 1e-3
  >- Time window size: 750

### Quantitative classification performance
![drow_table](https://user-images.githubusercontent.com/57162425/147727870-57762414-43ff-4587-86cb-f5c47ebd3ddf.png)

## Rquirements

python ≥ 3.8.10   numpy ≥ 1.20.3  pandas ≥ 1.2.5  scipy ≥ 1.6.2   torch ≥ 1.9.0

sklearn   random  os  argparse  collection

-------------------
The predicted values for test are stored in files under each model name + "prediction".
