# used to predict all validate images and encode results in a csv file

import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
from os.path import  join
from common.file_io import load_nrrd

# this function encodes a 2D file into run-length-encoding format (RLE)
# 	the inpuy is a 2D binary image (1 = positive), the output is a string of the RLE
def run_length_encoding(input_mask):
    dots = np.where(input_mask.T.flatten() == 1)[0]

    run_lengths, prev = [], -2

    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1
        prev = b

    return (" ".join([str(i) for i in run_lengths]))



if __name__ == '__main__':

    ### a sample script to produce a prediction

    # load the image file and reformat such that its axis are consistent with the MRI

    mask_format_name= "predict.nrrd"
    validation_dir = 'home/AtriaSeg_2018_testing'
    encode_cavity = []
    image_ids=[]
    for patient_name in sorted(os.listdir(validation_dir)):
        print ('encode ',patient_name)
        mask_path=join(*(validation_dir,patient_name,mask_format_name))
        if not os.path.isdir(os.path.join(validation_dir,patient_name)):
            continue
        mask,_=load_nrrd(mask_path)
        mask[mask>0]=1
    # ***
        # encode in RLE
        image_ids.extend([patient_name+"_Slice_" + str(i) for i in range(mask.shape[0])])

        for i in range(mask.shape[0]):
            encode_cavity.append(run_length_encoding(mask[i, :, :]))

    # output to csv file
    csv_output = pd.DataFrame(data={"ImageId": image_ids, 'EncodeCavity': encode_cavity},
                              columns=['ImageId', 'EncodeCavity'])
    csv_output.to_csv("submission.csv", sep=",", index=False)
