

import os
import time
import re
import argparse
import pandas as pd
from medpy.metric.binary import  dc,jc
import numpy as np


from common.measures import hd,assd
from common.file_io import load_nrrd

HEADER = ["Name", "Dice","JC","HD","ASSD"
          ]
VOXEL_SPACING = (0.625, 0.625, 0.625)


#
# Utils functions used to sort strings into a natural order
#
def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


#
# Functions to process files, directories and metrics
#
def metrics(img_gt, img_pred, voxel_size):
    global VOXEL_SPACING
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """
    print (img_gt.shape)
    print (img_pred.shape)
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [255]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        hd_value=hd(gt_c_i,pred_c_i,voxelspacing=VOXEL_SPACING,connectivity=1)
        assd_value = assd(gt_c_i, pred_c_i, voxelspacing=VOXEL_SPACING,connectivity=1)
        jd=jc(gt_c_i,pred_c_i)

        # Compute volume
        # volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        # volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.
       # res+=[dice,jd]
        res += [dice,jd,hd_value,assd_value]#,volpred, volpred-volgt]

    return res


def compute_metrics_on_files(path_gt, path_pred):
    """
    Function to give the metrics for two files

    Parameters
    ----------

    path_gt: string
    Path of the ground truth image.

    path_pred: string
    Path of the predicted image.
    """
    gt,img= load_nrrd(path_gt)
    pred,_ = load_nrrd(path_pred)
    name = os.path.basename(path_gt)
    name = name.split('.')[0]
    res = metrics(gt, pred, img.GetSpacing())
    print ( 'spacing',img.GetSpacing())
    print (res)

def compute_metrics_on_directories(root_dir,gt_name,pred_name,save_name):
    """
    Function to generate a csv file for each images of two directories.

    Parameters
    ----------

    path_gt: string
    Directory of the ground truth segmentation maps.

    path_pred: string
    Directory of the predicted segmentation maps.
    """
    patient_path_list=sorted([os.path.join(root_dir,i) for i in os.listdir(root_dir)])

    res = []
    for p_path in patient_path_list :
        p_name=p_path.split('/')[-1]
        print(p_name)
        gt, gt_image= load_nrrd(os.path.join(p_path,gt_name))
        pred, _= load_nrrd(os.path.join(p_path,pred_name))
        zooms = gt_image.GetSpacing()
        values=np.unique(pred)
        res.append(metrics(gt, pred, zooms))

    lst_name_gt = [gt.split("/")[-1] for gt in (patient_path_list)]
    res = [[n,] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(res, columns=HEADER)
    print (df.describe(include=[np.number]))
    df.to_csv(save_name+"_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")), index=False)

def main(root_path,gt_name,pred_name,save_name):
    """
    Main function to select which method to apply on the input parameters.
    """

    compute_metrics_on_directories(root_path,gt_name,pred_name,save_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compute Atrial challenge metrics.")
    parser.add_argument("--root_path", type=str, help="rootpath")
    parser.add_argument("--gt", type=str,default='laendo.nrrd' ,help="gt_name")
    parser.add_argument("--pred", type=str,default='predict.nrrd', help="pred_name")
    parser.add_argument("--save_name", type=str,default='results_atrial', help="save_csvname")


    args = parser.parse_args()
    main(args.root_path,args.gt,args.pred,args.save_name)
