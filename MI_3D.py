# Mutual information for 3D multi-modality volumes d
# MRI and US

# This code is not developed by Guangshen Ma, it is developed from the reference
# This code is only for studying

import MI_Util
from MI_Util import *

def MI_3D_V1(fixed_filename, moving_filename):

   # Mutual information for Version 1 -- with 3D multi-modality image

   # Read the images -- from the tutorial
   # fixed_filename = "/home/maguangshen/Desktop/Testdata/23/Case23-FLAIR-Resampled.mha"
   # moving_filename = "/home/maguangshen/Desktop/Testdata/23/Case23-US-before.mha"
   # fixed_filename = fdata("training_001_ct.mha")
   # moving_filename = fdata("training_001_mr_T1.mha")

   fixed_image = sitk.ReadImage(fixed_filename, sitk.sitkFloat32)
   moving_image = sitk.ReadImage(moving_filename, sitk.sitkFloat32)

   # print(fixed_image.GetSize())

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
   registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
   registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
   # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
   registration_method.SetMetricSamplingPercentage(1.0)

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

    # Save the output volume
   # sitk.WriteImage(moving_resampled, os.path.join(OUTPUT_DIR, 'RIRE_test_US2MRI.mha'))
   sitk.WriteTransform(final_transform, os.path.join(OUTPUT_DIR, 'RIRE_test_US2MRI.tfm'))

def MI_3D_V2(fixed_filename, moving_filename, csv_MRI, csv_US):

   # This code is only used for Mutual Information based registration -- multil 3D image -- version 2
   # Fiducial is shown as well

   # Load the dataset -- from the tutorial
   # fixed_filename = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-FLAIR-Resampled.mha"
   # moving_filename = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-US-before.mha"
   # fixed_filename = fdata("training_001_ct.mha")
   # moving_filename = fdata("training_001_mr_T1.mha")

   # Load the fixed and moving images
   fixed_image = sitk.ReadImage(fixed_filename, sitk.sitkFloat32)
   moving_image = sitk.ReadImage(moving_filename, sitk.sitkFloat32)

   # Our fiducials points (cases)

   # Read the MRI fiducial
   # csv_MRI = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-MRI.csv"
   fiducial_MRI, fiducial_MRI_list = Read_tsv(csv_MRI)
   print "The fiducials of MRI are ", fiducial_MRI_list

   # Read the US fiducial
   # csv_US = "/home/maguangshen/PycharmProjects/SPL/Data_SPL/Case12-beforeUS.csv"
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

if __name__ == "__main__":

   # Mutual Information for the 3D case -- two MRI from different modality T1 and T2
   dirnum = '12'
   dirfolder = '/home/maguangshen/Desktop/Testdata/'
   fixed_filename = dirfolder + dirnum + '/Case' + dirnum + '-FLAIR-Resampled.mha'
   moving_filename = dirfolder + dirnum + '/Case' + dirnum + '-US-before.mha'
   MI_3D_V1(fixed_filename, moving_filename)

   # Mutual Information for the 3D case -- CT and MRI(T1) registration
   # MI_3D_V2()

   # Test cases for using the mutual information 3D volume to 3D volume
   # test()

   # Test the case with the demon_1 -- solution -- change the data set
   # Demon_1()
