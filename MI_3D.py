# http://simpleitk.readthedocs.io/en/master/Examples/DemonsRegistration2/Documentation.html
# https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/tree/master/Python
# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/62_Registration_Tuning.html
# plot in non-blocking way: https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib

# Goals
# Test different parameters

import SimpleITK as sitk
import matplotlib.pyplot as plt
import registration_callbacks as rc
import registration_utilities as ru
import numpy as np
import warnings
import csv
import sys
import os
from downloaddata import fetch_data as fdata
from ipywidgets import interact, fixed
from IPython.display import clear_output

# Define the output directory
OUTPUT_DIR = '/home/maguangshen/PycharmProjects/SPL/Data_SPL/'

# Remove the
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

def MI_2D():

   # Read the images
   fixed_filename = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/test_resampled_MRI.mha"
   moving_filename = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/test_US_before.mha"

   # fixed_filename = fdata("training_001_ct.mha")
   # moving_filename = fdata("training_001_mr_T1.mha")

   fixed_image = sitk.ReadImage(fixed_filename, sitk.sitkFloat32)
   moving_image = sitk.ReadImage(moving_filename, sitk.sitkFloat32)

   # print (fdata("training_001_ct.mha"))
   interact(display_images, fixed_image_z=(0, fixed_image.GetSize()[2] - 1),
            moving_image_z=(0, moving_image.GetSize()[2] - 1),
            fixed_npa=fixed(sitk.GetArrayViewFromImage(fixed_image)),
            moving_npa=fixed(sitk.GetArrayViewFromImage(moving_image)))

   # Initial Alignment (by definition)
   initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                         moving_image,
                                                         sitk.Euler3DTransform(),
                                                         sitk.CenteredTransformInitializerFilter.GEOMETRY)
   moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0,
                                    moving_image.GetPixelID())
   interact(display_images_with_alpha, image_z=(0, fixed_image.GetSize()[2]), alpha=(0.0, 1.0, 0.05),
            fixed=fixed(fixed_image), moving=fixed(moving_resampled))

   registration_method = sitk.ImageRegistrationMethod()

   # Similarity metric settings.
   registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
   registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
   registration_method.SetMetricSamplingPercentage(0.01)

   registration_method.SetInterpolator(sitk.sitkLinear)

   # Optimizer settings.
   registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                     convergenceMinimumValue=1e-6, convergenceWindowSize=10)
   registration_method.SetOptimizerScalesFromPhysicalShift()

   # Setup for the multi-resolution framework.
   registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
   registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
   registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

   # Don't optimize in-place, we would possibly like to run this cell multiple times.
   registration_method.SetInitialTransform(initial_transform, inPlace=False)

   # Connect all of the observers so that we can perform plotting during registration.
   registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
   registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
   registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
   registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

   final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                 sitk.Cast(moving_image, sitk.sitkFloat32))

   print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
   print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

   moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                    moving_image.GetPixelID())

   interact(display_images_with_alpha, image_z=(0, fixed_image.GetSize()[2]), alpha=(0.0, 1.0, 0.05),
            fixed=fixed(fixed_image), moving=fixed(moving_resampled))

   sitk.WriteImage(moving_resampled, os.path.join(OUTPUT_DIR, 'RIRE_test_US2MRI.mha'))
   sitk.WriteTransform(final_transform, os.path.join(OUTPUT_DIR, 'RIRE_test_US2MRI.tfm'))

def MI_3D():

   # Load the dataset
   fixed_filename = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-FLAIR-Resampled.mha"
   moving_filename = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-US-before.mha"

   # fixed_filename = fdata("training_001_ct.mha")
   # moving_filename = fdata("training_001_mr_T1.mha")

   # Load the fixed and moving images
   fixed_image = sitk.ReadImage(fixed_filename, sitk.sitkFloat32)
   moving_image = sitk.ReadImage(moving_filename, sitk.sitkFloat32)

   # Our fiducials points (cases)

   # Read the MRI fiducial
   csv_MRI = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-MRI.csv"
   fiducial_MRI, fiducial_MRI_list = Read_tsv(csv_MRI)
   print "The fiducials of MRI are ", fiducial_MRI_list

   # Read the US fiducial
   csv_US = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-beforeUS.csv"
   fiducial_US, fiducial_US_list = Read_tsv(csv_US)
   print "The ficucials of US are ", fiducial_US_list

   # Fiducials
   # Standard and reference fiducial points
   # fixed_fiducial_points, moving_fiducial_points = ru.load_RIRE_ground_truth(fdata("ct_T1.standard"))
   # print( "The reference fixed fiducial points are ", fixed_fiducial_points)

   fixed_fiducial_points = fiducial_MRI_list
   moving_fiducial_points = fiducial_US_list

   # Check reference transform by calculating the FRE values
   R, t = ru.absolute_orientation_m(fixed_fiducial_points, moving_fiducial_points)         # Solved by SVD -- considered as LS problem
   print "The R is ", R
   print "The t is ", t

   # conbine into single matrix
   Mat_RT = np.zeros((4,4))
   Mat_RT[0:3,0:3] = R
   Mat_RT[0:3,3] = np.transpose(t)
   Mat_RT[3,0:3] = np.zeros((1,3))
   Mat_RT[3,3] = 1
   print('The Mat_RT is ', Mat_RT)
   print('The inverse of Mat_RT is ', np.linalg.inv(Mat_RT))

   # Calculate the initial distances between two fiducials before registration
   dist_fiducials = np.asarray(fixed_fiducial_points) - np.asarray(moving_fiducial_points)
   dis_before = np.sum(np.sqrt(np.square(dist_fiducials[:,0]) + np.square(dist_fiducials[:,1]) + np.square(dist_fiducials[:,2])))/len(dist_fiducials)
   print('The distance of two fiducials before registration is ', dis_before)

   # Set up the reference transform -- for this case
   reference_transform = sitk.Euler3DTransform()
   reference_transform.SetMatrix(R.flatten())
   reference_transform.SetTranslation(t)
   reference_errors_mean, reference_errors_std, _, reference_errors_max, _ = ru.registration_errors(
                                                     reference_transform, fixed_fiducial_points, moving_fiducial_points)
   print('Reference data errors (fiducials) (FRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(reference_errors_mean,
                                                                                                 reference_errors_std,
                                                                                                 reference_errors_max))

   # Generate a reference dataset from the fixed image
   # fixed_points = ru.generate_random_pointset(image = fixed_image, num_points=100)     # random dataset from the fixed image
   # moving_points = [reference_transform.TransformPoint(p) for p in fixed_points]       # transfer the dataset to the target region
   fixed_points = fixed_fiducial_points
   moving_points = moving_fiducial_points
   print ("The random point sets from fixed image is ", fixed_points)
   print ("The random point sets from moving image is ", moving_points)

   # Compute the TRE prior to registration (random point sets). -- before registration
   # QS: what is the rule of the random point sets?
   pre_errors_mean, pre_errors_std, pre_errors_min, pre_errors_max, _ = ru.registration_errors(sitk.Euler3DTransform(),
                                                                                                 fixed_points,
                                                                                                 moving_points,
                                                                                                 display_errors=True)
   print('Before registration (reference dataset), errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(pre_errors_mean,
                                                                                                       pre_errors_std,
                                                                                                       pre_errors_max))
   # Initial alignment -- set up the initial position by setting the center of the two 3D images
   initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image, moving_image.GetPixelID()),
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)             # Center the two region together
   initial_errors_mean, initial_errors_std, initial_errors_min, initial_errors_max, _ = ru.registration_errors(
        initial_transform, fixed_points, moving_points, min_err=pre_errors_min, max_err=pre_errors_max,
        display_errors=True)
   print('After initialization, errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(
        initial_errors_mean, initial_errors_std, initial_errors_max))

   registration_method = sitk.ImageRegistrationMethod()
   registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
   registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
   registration_method.SetMetricSamplingPercentage(0.01)
   registration_method.SetInterpolator(sitk.sitkLinear)  # 2. Replace with sitkLinear
   registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
   registration_method.SetOptimizerScalesFromPhysicalShift()
   registration_method.SetInitialTransform(initial_transform, inPlace=False)
   registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
   registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
   registration_method.AddCommand(sitk.sitkIterationEvent,
                                   lambda: rc.metric_and_reference_plot_values(registration_method, moving_points,
                                                                               fixed_points))

   registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
   registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
   registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

   final_transform_single_scale = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                               sitk.Cast(moving_image, sitk.sitkFloat32))
   print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
   print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
   final_errors_mean, final_errors_std, _, final_errors_max, _ = ru.registration_errors(final_transform_single_scale,
                                                                                         moving_points, fixed_points,
                                                                                         min_err=initial_errors_min,
                                                                                         max_err=initial_errors_max,
                                                                                         display_errors=True)
   print('After registration, errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean,
                                                                                                     final_errors_std,
                                                                                                     final_errors_max))
   # print('The original transform R is ', R)
   # print('The original transform T is ', t)
   # print('The after transform R is ', final_transform_single_scale)
   # print('The after transform T is ', final_transform_single_scale)
   sitk.WriteTransform(final_transform_single_scale, os.path.join(OUTPUT_DIR, 'RIRE_US2MRI.tfm'))

   read_result = sitk.ReadTransform(os.path.join(OUTPUT_DIR, 'RIRE_US2MRI.tfm'))

   print('The final transform is ', read_result)

   # print('finished')

def Demon_1():

   # Demon Registration method 1
   a = 1

   # if len(sys.argv) < 4:
   #     print("Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile>".format(sys.argv[0]))
   #     sys.exit(1)

   # fixed_data = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/test_resampled_MRI.mha"
   # fixed_img = sitk.ReadImage(fixed_data, sitk.sitkFloat32)
   # moving_data = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/test_US_before.mha"
   # moving_img = sitk.ReadImage(moving_data, sitk.sitkFloat32)

   # Example case
   fixed_img = sitk.ReadImage(fdata("training_001_ct.mha"), sitk.sitkFloat32)
   moving_img = sitk.ReadImage(fdata("training_001_mr_T1.mha"), sitk.sitkFloat32)

   # Design the matcher
   matcher = sitk.HistogramMatchingImageFilter()
   matcher.SetNumberOfHistogramLevels(1024)
   matcher.SetNumberOfMatchPoints(7)
   matcher.ThresholdAtMeanIntensityOn()
   moving = matcher.Execute(moving_img, fixed_img)

   # Design the demons filter
   demons = sitk.DemonsRegistrationFilter()
   demons.SetNumberOfIterations(50)
   demons.SetStandardDeviations(1.0)

   # demon registration operator
   demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
   displacementField = demons.Execute(fixed_img, moving_img)                       # Design the displacement field

   print("-------")
   print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
   print(" RMS: {0}".format(demons.GetRMSChange()))

   # Output the results
   # outTx = sitk.DisplacementFieldTransform(displacementField)

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

   # This function is used to test the some functions

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

if __name__ == "__main__":

   # Mutual Information for the 2D case -- two MRI from different modality T1 and T2
   # MI_2D()

   # Mutual Information for the 3D case -- CT and MRI(T1) registration
   MI_3D()

   # Test cases for using the mutual information 3D volume to 3D volume
   # test()

   # Test the case with the demon_1 -- solution -- change the data set
   # Demon_1()
