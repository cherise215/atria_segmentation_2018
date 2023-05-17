# Multi-Task Convolutional Neural Networks for Atria Segmentation Challenge
- The preprint version of this paper is available at [arxiv](https://arxiv.org/abs/1810.13205)
- This repository contains code for [Atrial Segmentation challenge 2018](http://atriaseg2018.cardiacatlas.org/)
- Training and testing data is available on the [web](https://drive.google.com/file/d/19uaNSIu3BtXakRECnziG6cnYQVKikwiB/view?usp=share_link)

## Paper
Please cite our paper if you find it useful for your research.

```
@InProceedings{10.1007/978-3-030-12029-0_32,
author="Chen, Chen
and Bai, Wenjia
and Rueckert, Daniel",
title="Multi-task Learning for Left Atrial Segmentation on GE-MRI",
booktitle="Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="292--301",
isbn="978-3-030-12029-0"
}
```

# Install and Prepare
## Environment
- Python 3.6
- Install Anaconda
- Install CUDA (cuda 9.0), if your machine has a CUDA-enabled GPU.
- create an Anaconda virtual environment to install all the dependencies packages below.
    - Follow the instructions from [How to create an anaconda virtual environment](https://dziganto.github.io/data%20science/python/anaconda/Creating-Conda-Environments/)
    - Install Pytorch 1.0 `conda install pytorch torchvision -c pytorch` [ref](https://pytorch.org/get-started/locally/)

## Dependencies
- after having activated your virtual enviroment under this project directory (`conda activate $your_virtual environment`), run `pip install -r requirements.txt`
    - pandas==0.22.0
    - matplotlib==2.2.2
    - nipy==0.4.2
    - MedPy==0.3.0
    - scipy==1.0.1
    - tqdm==4.23.0
    - numpy==1.14.2
    - SimpleITK==1.1.0
    - scikit-image
    - tensorboardX==1.4
- and then install an adapted version of torch sample via : `pip install git+https://github.com/ozan-oktay/torchsample/`


## Data
### Download all training and testing data.
 - for the testing data, simply unzip the data you download.
 - for the trainining data:
    - create a directory contains 2 sub directories: a `train` set and a `validate` set.
    - randomly split the training data into this training set and the validation set with a ratio of 4:1.


# Test
- You can eiter use a single model to predict:
    - open terminal under project dir,
    - type: `python predict_nrrd.py --model $model_path$  --input $test_dataset$ --save_name $predicted_result_name$ --force_norm`
        - model_path: the path you saved your trained model for predicting.
        - test_dataset: the path where you unzipped the testing data.
            - test_dataset
                - patient_1
                - patient_2
                - patient_XX
        - save_name: the name for the predicted result when saving. The predicted result will be saved under the same directory of the input patient_dir.
        - *e.g.`python predict_nrrd.py --model 'home/Atria_Seg/atria_mt_model.pkl' --input "home/AtriaSeg_2018_testing/"  --save_name "predict.nrrd" --force_norm`*
    - if use post-process: append '--post' to the tail of the above command. This will do morphological operations for smoothing and keep the largest component in a volume.
        - e.g.`python predict_nrrd.py --model 'home/Atria_Seg/atria_mt_model.pkl' --input "home/AtriaSeg_2018_testing/"  --save_name "predict.nrrd" --force_norm --post`


- Or use multiple models to get ensembled result with improved robustness.
    - `python ensemble.py`
        - Before run: please change $test_dir$ and those paths in the $model_dict$ to your test dataset path and your models paths, respectively.


# Train
- `python train_atria_seg.py --name '$predictor_path$' -b 6  --root_dir '$your training dataset path$' --csv_path $csv_path$ `
    - $predictor_path$ is your saved model's abs path.
    - $your training dataset path$: is the training dataset directory which contains *train* and *validate* dirs.
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
- open a new terminal under your project dir before training.
- run `tensorboard --logdir runs`

