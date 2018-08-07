# Reference:
# ref: https://stackoverflow.com/questions/29738822/how-to-convert-mha-file-to-nii-file-in-python-without-using-medpy-or-c#31620239
# https://stackoverflow.com/questions/7157340/renaming-filenames-using-python
# ref: https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python
# ref: https://stackoverflow.com/questions/29738822/how-to-convert-mha-file-to-nii-file-in-python-without-using-medpy-or-c/45125966

# Outline:
# 1. Convert .nii.gz to .mha automatically
# 2. Convert fcsv to csv automatically
# 3. Change the folder by user

import os
import numpy as np
import medpy
from medpy import *
from nibabel.testing import data_path
import nibabel as nib
import skimage.io as io
from shutil import copyfile
import SimpleITK as sitk

def nii2mha(dirnum):

    # MRI-resampled
    # Convert nii.gz file to .mha file
    MRI_nii_file = '/home/maguangshen/Desktop/Testdata/' + dirnum + '/Case' + dirnum + '-FLAIR-Resampled.nii.gz'
    MRI_output_file = '/home/maguangshen/Desktop/Testdata/' + dirnum + '/Case' + dirnum + '-FLAIR-Resampled.mha'
    img = sitk.ReadImage(MRI_nii_file)
    sitk.WriteImage(img, MRI_output_file)

    # US-before
    # Convert nii.gz file to .mha file
    US_nii_file = '/home/maguangshen/Desktop/Testdata/' + dirnum + '/Case' + dirnum + '-US-before.nii.gz'
    US_output_file = '/home/maguangshen/Desktop/Testdata/' + dirnum + '/Case' + dirnum + '-US-before.mha'
    img = sitk.ReadImage(US_nii_file)
    sitk.WriteImage(img, US_output_file)

def fcsv2csv(dirnum):
    dirpath = '/home/maguangshen/Desktop/Testdata/' + dirnum + '/'
    print(os.listdir(dirpath))
    for filename in os.listdir(dirpath):
        filename_split = os.path.splitext(filename)
        if(filename_split[1] == '.fcsv'):
            filename_zero, fileext = filename_split
            copyfile(dirpath + filename, dirpath + filename_zero + '_new' + fileext)
            print(filename_zero)
            os.rename(dirpath + filename_zero + '_new' + fileext, dirpath + filename_zero + ".csv")

if __name__ == "__main__":

    dirnum = '12'   # The for loop can be designed here
    nii2mha(dirnum)
    fcsv2csv(dirnum)
