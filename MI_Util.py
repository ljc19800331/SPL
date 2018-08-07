# http://simpleitk.readthedocs.io/en/master/Examples/DemonsRegistration2/Documentation.html
# https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/tree/master/Python
# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/62_Registration_Tuning.html
# plot in non-blocking way: https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
# Python and Slicer
# https://www.slicer.org/wiki/Documentation/Nightly/Developers/Python_scripting
# https://www.slicer.org/wiki/Slicer3:Python#Status_of_Python_in_Slicer
# Test different parameters
# Python code for utility -- plot and visualization of the process

# This is the utility code for reference
# This code is not developed by Guangshen Ma, it is developed from the reference
# This code is only for studying

import csv
import os
import warnings
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import registration_callbacks as rc
import registration_utilities as ru
from IPython.display import clear_output
from ipywidgets import interact, fixed
from downloaddata import fetch_data as fdata

# Define the output directory
OUTPUT_DIR = '/home/maguangshen/PycharmProjects/SPL/'
# remove the warning problem due to the scipy and number versions misalignment
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# The two images (moving and fixed)
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):

   # Create a figure with two subplots and the specified size.
   plt.subplots(1, 2, figsize=(10, 8))

   # Draw the fixed image in the first subplot.
   plt.subplot(1, 2, 1)
   plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r)
   plt.title('fixed image')
   plt.axis('off')

   # Draw the moving image in the second subplot.
   plt.subplot(1, 2, 2)
   plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r)
   plt.title('moving image')
   plt.axis('off')
   plt.draw()
   plt.pause(0.01)
# Display the image patch with alpha coefficient
def display_images_with_alpha(image_z, alpha, fixed, moving):

   img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
   plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
   plt.axis('off')
   # plt.show()
   plt.draw()
   plt.pause(0.01)
# callback invoked -- set up new data
def start_plot():

   global metric_values, multires_iterations
   metric_values = []
   multires_iterations = []
# callback invoked -- clean up new data
def end_plot():

   global metric_values, multires_iterations
   del metric_values
   del multires_iterations
   # Close figure, we don't want to get a duplicate of the plot latter on.
   plt.close()
# Callback invoked when the iteration event happens, update the data and renew figure
def plot_values(registration_method):

   global metric_values, multires_iterations
   metric_values.append(registration_method.GetMetricValue())

   # Clear the output area (wait=True, to reduce flickering), and plot current data
   clear_output(wait=True)

   # Plot the similarity metric values
   plt.plot(metric_values, 'r')
   plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
   plt.xlabel('Iteration Number', fontsize=12)
   plt.ylabel('Metric Value', fontsize=12)
   plt.draw()
   plt.pause(0.01)
def update_multires_iterations():

   global metric_values, multires_iterations
   multires_iterations.append(len(metric_values))
def register_images(fixed_image, moving_image, initial_transform, interpolator):        # Design the parameters

   registration_method = sitk.ImageRegistrationMethod()
   registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
   registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
   registration_method.SetMetricSamplingPercentage(0.01)
   registration_method.SetInterpolator(interpolator)
   registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000)
   registration_method.SetOptimizerScalesFromPhysicalShift()
   registration_method.SetInitialTransform(initial_transform, inPlace=False)
   final_transform = registration_method.Execute(fixed_image, moving_image)

   return (final_transform, registration_method.GetOptimizerStopConditionDescription())
def command_iteration():
   print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),
                                    filter.GetMetric()))
def Memory_Tradeoff():

   a = 1
def Read_tsv(filename):

   # Read the tsv file and output the 3D coordinates
   # filename = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case23-MRI.csv"
   with open(filename) as csvfile:
       reader = csv.DictReader(csvfile, dialect='excel-tab')
       count = 0
       N_fiducial = 16
       flag_coord = 0
       data_cood = np.zeros((N_fiducial, 3))
       list_cood = [] # The list of the tuple for the fiducial points
       # print(len(reader))
       for row in reader:
           # print(row['# Markups fiducial file version = 4.7'])
           if count > 1:
               mylist = row['# Markups fiducial file version = 4.7']       # This is a string object
               [x.strip() for x in mylist.split(',')]
               list_1 = mylist.split(',')
               data_cood[flag_coord, :] = np.asarray([list_1[1], list_1[2], list_1[3]])
               list_cood.append(  tuple([  np.float(list_1[1]), np.float(list_1[2]), np.float(list_1[3])  ]   )    )
               # print ( tuple([  np.float(list_1[1]), np.float(list_1[2]), np.float(list_1[3])  ]   )  )
               flag_coord += 1
           count += 1
       # print(data_cood)
       # print(data_cood.shape)
   return data_cood, list_cood
def test():

   # show the standard fiducial list
   fixed_fiducial_points, moving_fiducial_points = ru.load_RIRE_ground_truth(fdata("ct_T1.standard"))
   print "The standard fiducial points are ", fixed_fiducial_points
   print "The standard fiducial points shape ", type(moving_fiducial_points)

   # test_data = fdata("training_001_ct.mha")
   fixed_data = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/test_resampled_MRI.mha"
   fixed_img = sitk.ReadImage(fixed_data, sitk.sitkFloat32)

   moving_data = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/test_US_before.mha"
   moving_img = sitk.ReadImage(moving_data, sitk.sitkFloat32)

   # This is the list -- can be used for numpy array
   # Read the MRI fiducial
   csv_MRI = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case23-MRI.csv"
   fiducial_MRI, fiducial_MRI_list = Read_tsv(csv_MRI)
   print "The fiducials of MRI are ", fiducial_MRI_list

   # Read the US fiducial
   csv_US = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case23-beforeUS.csv"
   fiducial_US, fiducial_US_list = Read_tsv(csv_US)
   print "The ficucials of US are ", fiducial_US_list

   R, t = ru.absolute_orientation_m(fiducial_MRI_list, fiducial_US_list)
   print "The testing R is ", R
   print "The testing T is ", t
   print "The R flatten is ", R.flatten()

   # Set up the registration parameters
   reference_transform = sitk.Euler3DTransform()       # Euler 3D Transform
   reference_transform.SetMatrix(R.flatten())          # Flatten the R matrix
   reference_transform.SetTranslation(t)               # Set the translation vector
   reference_errors_mean, reference_errors_std, _, reference_errors_max, _ = ru.registration_errors(reference_transform, fiducial_MRI_list, fiducial_US_list)
   print('Reference data errors (FRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(reference_errors_mean, reference_errors_std, reference_errors_max))
   # Viz the results to map all the points

   # Generate a reference dataset from the reference transformation
   # Apply the transformation first based on the fiducial registration results
   fixed_points = ru.generate_random_pointset(image = fixed_img, num_points=100)          # generate random point sets from the volume dataset
   moving_points = [reference_transform.TransformPoint(p) for p in fixed_points]

   # Compute the TRE prior to the registration
   pre_errors_mean, pre_errors_std, pre_errors_min, pre_errors_max, _ = ru.registration_errors(sitk.Euler3DTransform(), fixed_points, moving_points, display_errors=True)
   print('Before registration, errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(pre_errors_mean, pre_errors_std, pre_errors_max))

   ## Initial alignment
   # Initialization of the transform
   initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_img, moving_img.GetPixelID()), moving_img, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
   initial_errors_mean, initial_errors_std, initial_errors_min, initial_errors_max, _ = ru.registration_errors(
   initial_transform, fixed_points, moving_points, min_err=pre_errors_min, max_err=pre_errors_max, display_errors=True)
   print('After initialization, errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(
       initial_errors_mean, initial_errors_std, initial_errors_max))

   # Begin registration
   registration_method = sitk.ImageRegistrationMethod()
   registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)        # Define the number of bins as 50
   registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)               # Define the
   registration_method.SetMetricSamplingPercentage(0.01)                                   # Default sampling percentage is 0.01
   registration_method.SetInterpolator(sitk.sitkNearestNeighbor)                           # Replace with sitkLinear
   registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)  # Increase to 1000
   registration_method.SetOptimizerScalesFromPhysicalShift()                               # Understand the optimization method
   registration_method.SetInitialTransform(initial_transform, inPlace=False)               # Apply the initial transform

   # Add the callback to display the similarity value
   registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
   registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
   registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))

   final_transform_single_scale = registration_method.Execute(sitk.Cast(fixed_img, sitk.sitkFloat32), sitk.Cast(moving_img, sitk.sitkFloat32))
   print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
   print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
   final_errors_mean, final_errors_std, _, final_errors_max, _ = ru.registration_errors(final_transform_single_scale,
                                                                                        fixed_points, moving_points,
                                                                                        min_err=initial_errors_min,
                                                                                        max_err=initial_errors_max,
                                                                                        display_errors=True)
   print('After registration, errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean,
                                                                                                    final_errors_std,
                                                                                                    final_errors_max))
   final_errors_mean, final_errors_std, _, final_errors_max, _ = ru.registration_errors(final_transform_single_scale,
                                                                                        fixed_points, moving_points,
                                                                                        display_errors=True)