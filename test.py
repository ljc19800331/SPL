# This code is used to test multiple functions included in SimpleITK
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import MI_3D
from MI_3D import *

# Read the MRI fiducial
csv_MRI = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case23-MRI.csv"
fiducial_MRI, fiducial_MRI_list = Read_tsv(csv_MRI)
print("The fiducials of MRI are ", fiducial_MRI_list)

# Read the US fiducial
csv_US = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case23-beforeUS.csv"
fiducial_US, fiducial_US_list = Read_tsv(csv_US)
print("The ficucials of US are ", fiducial_US_list)

# Test the results

MAT_test = [[1,  0,  0,  0],
         [0,  1,  0,  0],
         [0,  0,  1, 0],
         [0,  0,  0,  1]]

MAT_org = [[0.999258,  -0.0335698,  -0.0188681,  0.174432],
         [0.0334974,  0.99943,  -0.0041386,  -5.73273],
         [0.0189962,  0.0035035,  0.999813,  2.67365],
         [0,  0,  0,  1]]

MAT_inv = np.linalg.inv(MAT_org)

test = np.asarray(MAT_inv)

R = test[0:3,0:3]
T = test[0:3,3]

print(test)
print(R)
print(T)

# def Tuple2Npy(Tuplelist):
# Transfer the tuple value to the numpy value
MRI_list = [np.asarray(p) for p in fiducial_MRI_list]
MRI_npy = np.zeros((len(fiducial_MRI_list), 3))

US_list = [np.asarray(p) for p in fiducial_US_list]
US_npy = np.zeros((len(fiducial_US_list), 3))

for idx, item in enumerate(fiducial_US_list):
   US_npy[idx, :] = item

for idx, item in enumerate(fiducial_MRI_list):
   MRI_npy[idx, :] = item

# for item in points:
# print(points_list)
print(MRI_npy)
# Calculate the reference distances -- after transformation
US_after = np.matmul(US_npy, R) + np.tile(T, (len(fiducial_MRI_list), 1))
print(US_after)

dist_fiducials = np.asarray(MRI_npy) - np.asarray(US_after)
dis_after = np.sum(np.sqrt(np.square(dist_fiducials[:,0]) + np.square(dist_fiducials[:,1]) + np.square(dist_fiducials[:,2])))/len(dist_fiducials)
print('The dis of two US is ', dis_after)
