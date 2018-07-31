# This script is used to learn the 3D mutual information/ 2D mutual information
# Change the parameters to see which can perform a better registration
# Mutual information as an image matching metric
# Im Ref: https://matthew-brett.github.io/teaching/
# Ref: https://matthew-brett.github.io/teaching/mutual_information.html
# Database: http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
# Other reference: https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429#file-mutual_info-py-L9

from __future__ import print_function   # python3 to python2
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

# Load the data
t1_img = nib.load('/home/maguangshen/PycharmProjects/SPL/Data_SPL/mni_icbm152_t1_tal_nlin_sym_09a.nii')
t1_data = t1_img.get_data()
t2_img = nib.load('/home/maguangshen/PycharmProjects/SPL/Data_SPL/mni_icbm152_t2_tal_nlin_sym_09a.nii')
t2_data = t2_img.get_data()

# Show the images -- we can define the number of slice on this image
t1_slice = t1_data[:, :, 94]
t2_slice = t2_data[:, :, 94]
plt.imshow(np.hstack((t1_slice, t2_slice)))
plt.title("The slice images from T1 and T2")

# One-dimensional histogram of the example slices
# The index of the histogram values has changed
fig, axes = plt.subplots(1, 2)
axes[0].hist(t1_slice.ravel(), bins=20)
axes[0].set_title('T1 slice histogram (before)')
axes[1].hist(t2_slice.ravel(), bins=20)
axes[1].set_title('T2 slice histogram (before)')

# Change the slice to lower such that the MI will change
t2_slice_moved = np.zeros(t2_slice.shape)
t2_slice_moved[15:, :] = t2_slice[:-15, :]
fig, axes = plt.subplots(1, 2)
axes[0].hist(t1_slice.ravel(), bins=20)
axes[0].set_title('T1 slice histogram (after)')
axes[1].hist(t2_slice_moved.ravel(), bins=20)
axes[1].set_title('T2 slice histogram (after)')

# plot T1 against T2 to see what is the relationship (linear, non-linear or others) -- to measure the similarity
plt.figure(4)
plt.plot(t1_slice.ravel(), t2_slice.ravel(), '.')
plt.xlabel('T1 signal')
plt.ylabel('T2 signal')
plt.title('T1 vs T2 signal')
coeff_before = np.corrcoef(t1_slice.ravel(), t2_slice.ravel())[0, 1]
print("The coefficient before registration is ", coeff_before)

# Plot the edge(region) of the image
t1_20_30 = (t1_slice >= 20) & (t1_slice <= 30)
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
axes[0].imshow(t1_slice)
axes[0].set_title('T1 slice')
axes[1].imshow(t1_20_30)
axes[1].set_title('20<=T1<=30')
axes[2].imshow(t2_slice)
axes[2].set_title('T2 slice')

# Plot the signal bin in 2D to measure the similarity (mutual information)
plt.figure(5)
hist_2d, x_edges, y_edges = np.histogram2d(t1_slice.ravel(), t2_slice.ravel(), bins=20)
plt.imshow(hist_2d.T, origin='lower')
plt.xlabel('T1 signal bin')
plt.ylabel('T2 signal bin')

# Show the log histogram
plt.figure(6)
hist_2d_log = np.zeros(hist_2d.shape)
non_zeros = hist_2d != 0
hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
plt.imshow(hist_2d_log.T, origin='lower')
plt.xlabel('T1 signal bin')
plt.ylabel('T2 signal bin')
plt.show()

# Mutual information for joint histogram
def mutual_information(hgram):
   # Convert bins counts to probability values
   pxy = hgram / float(np.sum(hgram))
   px = np.sum(pxy, axis=1) # marginal for x over y
   py = np.sum(pxy, axis=0) # marginal for y over x
   px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
   # Now we can do the calculation using the pxy, px_py 2D arrays
   nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
   return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

coeff_mutul = mutual_information(hist_2d)
print("The coefficient after mutual information is ", coeff_mutul)
