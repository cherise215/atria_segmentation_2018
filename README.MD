# Multi-Task Convolutional Neural Networks for Atria Segmentation Challenge

- This repository contains code for [Atrial Segmentation challenge 2018](http://atriaseg2018.cardiacatlas.org/)
- Training and testing data is available on the [web](http://atriaseg2018.cardiacatlas.org/data/)


# Install and Prepare
## Environment
- Python 3.5
- Install Anaconda
- Install CUDA (cuda 9.0), if your machine has a CUDA-enabled GPU.
- create an Anaconda virtual environment to install allthe dependencies packages below.
    - Follow the instructions from [How to create an anaconda virtual environment](https://dziganto.github.io/data%20science/python/anaconda/Creating-Conda-Environments/)
    - Install Pytorch. pytorch==0.4.0 , e.g. `conda install pytorch torchvision -c pytorch` [ref](https://pytorch.org/get-started/locally/)

## Dependencies
- after activated virtual enviroment under project directory, `pip install -r requirements.txt`
    - pandas==0.22.0
    - matplotlib==2.2.2
    - torchsample==0.1.3
    - nipy==0.4.2
    - MedPy==0.3.0
    - scipy==1.0.1
    - tqdm==4.23.0
    - scikit_image==0.13.1
    - numpy==1.14.2
    - SimpleITK==1.1.0
    - scikit-image
    - tensorboardX==1.4


## Data
### Download all training and testing data.
 - for test data, simply unzip the data you download.
 - for trainining data:
    - create a directory and then 2 sub directory `train` and `validate` set under it.
    - randomly split training data into training set and validation set and then put it under its corresponding directory.


# Test
- using a single model to predict:
    - open terminal under project dir,
    - type: `python predict_nrrd.py --model $model_path$  --input $test_dataset$ --save_name $predicted_result_name$ --force_norm`
        - model_path: the path you saved your trained model for predict
        - test_dataset: the parent dir which saves all patients data.
            - test_dataset
                - patient_1
                - patient_2
                - patient_XX
        - save_name: the name for the predicted result when saving. The predicted result will be saved under the same directory of the input patient_dir.
        - *e.g.`python predict_nrrd.py --model 'home/Atria_Seg/atria_mt_model.pkl' --input "home/AtriaSeg_2018_testing/"  --save_name "predict.nrrd" --force_norm`*
    - if use post-process: append '--post' to the tail of the above command. This will do morphological operations for smoothing and keep the largest component in a volume.
        - e.g.`python predict_nrrd.py --model 'home/Atria_Seg/atria_mt_model.pkl' --input "home/AtriaSeg_2018_testing/"  --save_name "predict.nrrd" --force_norm --post`


- Ensemble of multi models result.
    - `python ensemble.py`
        - Before run: please change $test_dir$ and those paths in the $model_dict$ to your test dataset path and your downloaded models paths respectively.


# Train
- `python train_atria_seg.py --name '$model_save_name$' -b 6  --root_dir '$your training dataset path$' --csv_path $csv_path$ `
    - $model_save_name$ is a name used to save the model you trained
    - $your training dataset path$: is the training dataset directory which contains *train* and *validate* two subdirs.
        - training dataset path
            - `train`
                - patient_1
                - patient_2
            - `validate`
                - patient_1
                - patient_2
    - $csv_path$: specifies an csv file path which contains extra pre/post ablation label for each patient. This file should be download with the training data. Or you can find it under the $data directory.

        - e.g. `python  train_atria_seg_mt.py --name 'atria_mt_model'  --root_dir 'home/AtriaSeg_2018_training/cross_validation_dataset/0' --csv_path 'home/AtriaSeg_2018_training/data/pre post ablation recrods.csv'`


# Visualization of training and validation loss and iou during training procedure.
- open the terminal under your project dir.
- run `tensorboard --logdir runs`