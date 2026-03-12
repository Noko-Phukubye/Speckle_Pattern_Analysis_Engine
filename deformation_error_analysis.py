
"""
Analytical vs Discrete deformation
===================================

This script generates and deforms a set of patterns using both analytical and
discrete deformation. The script then evaluates the discrepencies between their 
results and attempts to correct them using Support Vector Regression 
and SLSQP optimisation.

The pipeline includes:

1. Creating an Excel workbook used for storing pattern metrics and results.
2. Generating synthetic speckle patterns using several pattern generation
   approaches.
3. Performing speckle pattern analysis using multiple image quality metrics.
4. Loading Finite Element (FE) displacement fields from Nastran output.
5. Deforming reference images using the FE displacement field.
6. Running SUN-DIC to estimate displacement fields.
7. Performing error analysis between FE and DIC displacement results.
8. Performing optimisation studies on speckle pattern parameters.

Each block of the workflow is controlled by a flag to prevent unnecessary
re-computation and to reduce the risk of overwriting intermediate data.

The script is designed to run multiple *batches* of pattern classes in order
to compare different speckle morphologies under identical deformation and
DIC conditions.
"""


# =============================================================================
# IMPORTS
# =============================================================================

# Import libraries
import numpy as np
import time as time
from random import randint
import matplotlib.pyplot as plt
import image_process_tool_box as ipt
import os
from speckle_pattern import generate_and_save
import cv2
import time
import subprocess
from tqdm import tqdm
import file_paths as path
from noise import pnoise2
import shutil
import sundic.sundic as sdic
import sundic.post_process as sdpp
import sundic.settings as sdset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
from scipy.optimize import minimize
from speckle_pattern import generate_and_save, generate_lines, generate_checkerboard
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import traceback
import sys
import subprocess
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split




# =============================================================================
# WORKFLOW DESCRIPTION
# =============================================================================
"""
General workflow implemented in this script.

1. Generates N images and deforms them using the analytical deformation method.
2. Deforms the same images that were already generated but this time using the 
   discrete deformation method.
3. 
"""

# =============================================================================
# PLOTTING STYLE
# =============================================================================
# Plot settings (IEEE)
plt.style.use(['science', 'no-latex','ieee', 'grid'])

plt.rcParams.update({
    'font.family': 'Calibri',
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 1.25,
    'lines.markersize': 3,

    # Border styling
    'axes.linewidth': 1.0,     
    'xtick.major.width': 1.0,    
    'ytick.major.width': 1.0,    
    'xtick.minor.width': 0.6,    
    'ytick.minor.width': 0.6,

    'xtick.major.size': 5,      
    'ytick.major.size': 5,
    'xtick.minor.size': 3,      
    'ytick.minor.size': 3,
})

# Image dimensions (pixels)
image_height = 500
image_width = 2000

# =========================================================================
# CONSTANT DIRECTORY PATH DEFINITIONS
# =========================================================================
DIC_settings_path       = r"settings.ini"
debugg_folder           = r"output\Debugging"
Contour_path            = r"output\DIC_contour"
znssd_figure_path       = r"output\DIC_contour\ZNSSD"
power_spec              = r"output\spectral_analysis"
autocorrelation_path    = r"output\Autocorrelation"
optimised_save          = r"data\speckle_pattern_img\Optimised\data_corrected"
excel_pathh             = r"output\excel_docs\error_separation"


# =========================================================================
# DIRECTORY PATH DEFINITIONS
# =========================================================================
directories = [excel_pathh,
               debugg_folder,znssd_figure_path,
               Contour_path,power_spec,autocorrelation_path,optimised_save]

for dir_path in directories:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

tic = time.time()


# =============================================================================
# WORKFLOW FLAGS
# =============================================================================
# Flags for generating patterns and evaluating errors
flags = {
    'generate_2N_images': True,
    'error_dist': True,
    'error_separation': True,
    'separate_errors': False,
    'error_correction_opt': True
}

# =============================================================================
# Perlin noise class selector
# =============================================================================
textures = [
    # 'none',
    'sinusoidal',
    # 'cubic',
    # 'perlin_blobs'
]

# Excel files are selected based on the index of the last-created file. This means that the index of the 
# file that is selected is found by using the number of files that are already existing in the folder.
# The override_doc is for the user to force open a specific file number.
override_doc = 59

# The part of the code loops through the list of textures and performs the analyticalVSdiscrete 
# analysis for each of them. I will admit that I only really added this part towards the end.
# So I dont remember if it works perfectly but it should since its the same 
# logic as when I was running each texture at a time.
for iteration,texture in enumerate(textures):

    # =========================================================================
    # DYNAMIC DIRECTORY PATH DEFINITIONS
    # =========================================================================
    # The dynamic subfolders generated in this script are specified by 
    # deform_analysis_{texture} strings where {texture} is one of the 
    # strings in the "textures" list above.
     
    temp_ref_image_folder = rf"data\speckle_pattern_img\reference_im\Temp_reference_folder\deform_analysis_{texture}"
    temp_deform_image_folder = rf"data\speckle_pattern_img\deformed_im\Temp_deformed_folder\deform_analysis_{texture}"
    err_plots_correct       = rf"output\plots\Error_corrected\deform_analysis_{texture}"
    numpy_files             = rf"output\numpy_files\discanaly\deform_analysis_{texture}"
    scatter_plots           = rf"output\slices\slice_scatter_plots\deform_analysis_{texture}"
    reference_image_path    = rf"data\speckle_pattern_img\reference_im\discrete_anly\deform_analysis_{texture}"
    deformed_image_path     = rf"data\speckle_pattern_img\deformed_im\discrete_anly\deform_analysis_{texture}"
    sundic_binary_folder    = rf"output\sundic_bin\deform_analysis_{texture}"
    slice_path              = rf"output\Slices\deform_analysis_{texture}"

    # Generate folders
    dynamic_directories = [temp_ref_image_folder, 
                           temp_deform_image_folder,
                           err_plots_correct,
                           numpy_files,
                           scatter_plots,
                           reference_image_path,
                           deformed_image_path,
                           sundic_binary_folder,
                           slice_path
                           ]

    for dir_path in dynamic_directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # =========================================================================
    # GENERATE SPECKLE PATTERNS AND RUN DIC
    # =========================================================================
    if flags["generate_2N_images"]:

        '''
        This part of the code generates the images for the analysis.
        First the images are created in pairs through the application of Perlin noise.
        The image pairs are initially stored in temporary folders.
        The reference images


        '''
        # If the images already exist and are in the correct folders then 
        # there is no need to generate. Only run DIC analysis.
        just_dic = False

        # Else generate the images
        if not just_dic:

            # ------------------------------------------------------------------------------------
            # Create excel sheet for storing pattern generation and metric data and error results
            # -----------------------------------------------------------------------------------
            path_2_doc = ipt.make_xl(excel_pathh, file_name="my_doc")

            # -------------------------------------------------------------------------
            # Generate N images. These include reference and deformed images
            # -------------------------------------------------------------------------
            image_width = 2000
            image_height = 500
            pattern_count = 5

            # -------------------------------------------------------------------------
            # Load FE data and create RBF interpolator objects
            # -------------------------------------------------------------------------            
            op2_path = path.flat30_4mm_op2
            bdf_path = path.flat30_4mm_bdf

            nodes_2d, deformed_nodes_2d = ipt.load_fe_nodes(bdf_path, op2_path)

            # Scale the data to fit the image
            max_x_FE = np.max(nodes_2d[:, 0])
            x_scale  = image_width / max_x_FE
            nodes_2d = nodes_2d * x_scale
            deformed_nodes_2d = deformed_nodes_2d * x_scale
            displacements_2d = deformed_nodes_2d - nodes_2d

            # Create matrix for sizing boundaries of the displacement field
            matrix = np.full((image_height,image_width), np.nan)
            disc_dx,disc_dy,interp_rbfx, interp_rbfy = ipt.smooth_field(
                matrix, 
                nodes_2d, 
                deformed_nodes_2d, 
                3
                )

            # -------------------------------------------------------------------------
            # Generate Perlin noise images. Deform during generation (Analytical)
            # -------------------------------------------------------------------------
            # The images are stored in temporary folders, not the analysis folders
            ipt.generate_perlin_pair(
                image_height,image_width, 
                interp_rbfx, interp_rbfy,
                path_2_doc,
                temp_ref_image_folder,
                temp_deform_image_folder,
                number_of_images=pattern_count,
                texture_fun=texture
            )

            # -------------------------------------------------------------------------
            # Analyse patterns
            # -------------------------------------------------------------------------
            print('\n3. Analysing speckle patterns...\n')
            ipt.evaluate_patterns(
                excel_pathh,
                temp_ref_image_folder,
                power_spec,
                autocorrelation_path,
                doc_name='my_doc',
                doc_number=None
                )

            # -------------------------------------------------------------------------
            # Copy reference images from temporary folder to reference analysis folder
            # -------------------------------------------------------------------------
            # Get names
            reference_image_files = ipt.get_image_strings(temp_ref_image_folder, 
                                                        imagetype='tif', 
                                                        parity='even')
            
            # Get last number. The reference images will be duplicated. The images 
            # use an even numbered-prefix naming system. The last number is used to
            # initialise the naming of the duplicate images.
            last_nummber = int(reference_image_files[-1].split('_')[0])             # Last index in list
            print('\nLast prefix:', last_nummber)

            # Move items
            for image_name in reference_image_files:

                # Source and destination paths
                src = os.path.join(temp_ref_image_folder, image_name)
                dst = os.path.join(reference_image_path, image_name)

                # Copy files
                shutil.copy2(src, dst)
                print(f"| Copied: {image_name}")

            # -------------------------------------------------------------------------
            # Renumber reference images starting at N + 2 before deformation
            # -------------------------------------------------------------------------
            # The analytical-deformation reference images will be numbered 0 through N 
            # and their associated deformed images are numbered 1 through N + 1.
            # The discrete-deformation reference images will be numbered N + 2 through 2N.
            # Their associated deformed images will be numbered N + 2 + 1 through 2N + 1.

            ipt.ordered_prefix(
                reference_image_path,
                start_at=(last_nummber+2)
                )

            # -------------------------------------------------------------------------
            # Deform the reference images after renumbering (discrete)
            # -------------------------------------------------------------------------
            image_files = ipt.get_image_strings(reference_image_path)

            # Loop through each image and deform
            for k, ref_image_file in enumerate(image_files):
                
                # Get reference image
                ref_img_path = os.path.join(reference_image_path, ref_image_file)
                reference_image = cv2.imread(ref_img_path)
                if reference_image is None:
                    print(f"Warning: Could not read reference image {ref_img_path}")
                    continue

                # Create deformed image name (odd number).
                ref_num = int(ref_image_file.split('_')[0])
                deform_num = ref_num + 1
                deformed_name = f"{deform_num}_Generated_spec_image.tif"

                # Deform images
                tic_def = time.time()
                #-----------------------------DEFORM------------------------------------------------
                # Deform using the interpolators 
                # ipt.img_deform(reference_image, nodes_2d, deformed_nodes_2d,3) 
                transformed_image, difference_image = ipt.img_deform(
                    reference_image, 
                    dx = disc_dx, 
                    dy = disc_dy
                    )
                #-----------------------------------------------------------------------------------

                # Apply name to the deformed image and save
                deformed_path = os.path.join(deformed_image_path, deformed_name)
                cv2.imwrite(deformed_path, transformed_image)
                toc_def = time.time()


            # -------------------------------------------------------------------------
            # Copy the reference images in the temporary reference image folder to the 
            # analysis folder again.
            # -------------------------------------------------------------------------
            # These will serve as the analytical-deformation reference images. 
            # This will result in 2N reference images in the folder.

            for image_name, image_index in enumerate(reference_image_files):          
                src = os.path.join(temp_ref_image_folder, image_name)
                dst = os.path.join(reference_image_path, image_name)
                shutil.copy2(src, dst)
                print(f"Copied: {image_name}")

            # -------------------------------------------------------------------------
            # Similarly, copy the (analytically) deformed images in the temporary reference 
            # image folder to the deformed images analysis folder. This will result in a total
            # of 2N deformed images in the folder.
            # -------------------------------------------------------------------------
            deformed_image_files = ipt.get_image_strings(
                temp_deform_image_folder, 
                imagetype='tif', 
                parity='odd')
            
            for image_name in deformed_image_files:                                 # Perlin deformed images
                src3 = os.path.join(temp_deform_image_folder, image_name)           # Origin
                dst3 = os.path.join(deformed_image_path, image_name)                # Destination
                shutil.copy2(src3, dst3)
                print(f"Copied: {image_name}")

        # -------------------------------------------------------------------------
        # Apply blur to both reference and deformed images (optional)
        # -------------------------------------------------------------------------
        Def_flags = {"Image_blur": True}
        
        ipt.flag_status(Def_flags,wait_time=1.5)

        # Add Gaussian blur
        if Def_flags['Image_blur']:
            print('\nReference image prefilter')
            ipt.gaussian_blur_images(
                reference_image_path, 
                size = 5, 
                sig_y = 1.0, 
                par = 'even')
            
            print('\nDeformed image prefilter')
            ipt.gaussian_blur_images(
                deformed_image_path, 
                size = 5, 
                sig_y = 1.0, 
                par = 'odd')


        # -------------------------------------------------------------------------
        # Run DIC on all the images
        # -------------------------------------------------------------------------
        DIC_flags = {"run_analysis": True}
        
        ipt.flag_status(DIC_flags,wait_time=1.5)
        print("\n|Running DIC analysis...\n")

        # Check for config file
        if not os.path.exists(DIC_settings_path):
            print(f"Settings file not found at: {DIC_settings_path}")
        settings = sdset.Settings.fromSettingsFile(DIC_settings_path)

        roi = 'Auto'
        # Automatic ROI size based on image dimensions.
        start_x = 30
        start_y = 30

        if roi.lower() == 'auto':
            width = image_width - (2 * start_x)
            height = image_height - (2 * start_y)
        else:
            width = 1990
            height = 490

        settings.ROI = [start_x, start_y, width, height]
        settings.virtSubsetSize = 33
        settings.StepSize = 11
        settings.GaussianBlurSize = 5
        settings.GaussianBlurStdDev = 0.0
        settings.DebugLevel = 2
        settings.CPUCount = 8
        settings.ReferenceStrategy = "Absolute"
        settings.OptimizationAlgorithm = "IC-GN"

        print(settings.__repr__())
        print("Settings loaded successfully\n")

        # Check ray status and create headnode if uninitialised
        try:
            print('Checking ray...')
            subprocess.run('ray status', shell=True, check=True)
            time.sleep(5)
        # If not initialised
        except subprocess.CalledProcessError:
            print('Starting Ray...')
            subprocess.run('ray start --head --num-cpus=8', shell=True)
            time.sleep(5)   
            subprocess.run('ray status', shell=True)
        
        # Run analysis
        if DIC_flags["run_analysis"]:
            ipt.run_dic(
                settings,
                reference_image_path,
                deformed_image_path,
                sundic_binary_folder,
                debugg_folder,
                Contour_path,
                znssd_figure_path,
                start_index=0
                )

        plt.close('all')

    # -------------------------------------------------------------------------
    # Process DIC results and save error files in binary for later access
    # -------------------------------------------------------------------------
    # Loop through the numpy files (from the DIC analysis) and perform error analysis.
    # Ignore files that have f.split('_')[1] == '_T' as these are the subpixel translations

    if flags['error_dist']:
        print('\n| Error analysis...\n')
        
        # DIC binary files
        sundic_binary_files = sorted(
            [f for f in os.listdir(sundic_binary_folder)
            if f.endswith('results.sdic')],
            key=lambda x: int(x.split('_')[0])
        )
        print(f"\nSUNDIC files found: {sundic_binary_files[0]} through {sundic_binary_files[-1]}")

        # The prefix-based numbering system is used for images and results files. The 
        # reference images use even-numbered prefixes while the deformed images use odd 
        # numbers. The associated files use an even-number prefix system just like 
        # the reference images. Here the expected prefixes are initialised based on the number of 
        # existing files in the sundic results folder to be used to call each file.

        all_expected_prefixesbin = ipt.expected_prefixes(
            sundic_binary_folder,odd=False,skip=True)
        
        # Each prefix will have an index. Should a file not be found at that prefix the \
        # Associated index will be used for reporting purposes.
        prefix_positionsbin = {prefix: i for i, prefix in enumerate(all_expected_prefixesbin)}


        # -------------------------------------------------------------------------
        # Load FEA result files if they weren't loaded upstream
        # -------------------------------------------------------------------------
        # The FE data is required for error analysis. If "just_dic" is true
        # then the data is not loaded upstream. But this is also for in case the
        # data was not loaded for any other reason.

        if "nodes_2d" not in globals():
            op2_path = path.flat30_4mm_op2
            bdf_path = path.flat30_4mm_bdf
            nodes_2d, deformed_nodes_2d = ipt.load_fe_nodes(bdf_path, op2_path)

            # 3 Scale the data to fit the image (might automate)
            max_x_FE = np.max(nodes_2d[:, 0])
            x_scale  = image_width / max_x_FE
            deformed_nodes_2d = deformed_nodes_2d * x_scale
            displacements_2d = deformed_nodes_2d - nodes_2d
            # 4 Create matrix for sizing boundaries of the displacement field
            matrix = np.full((image_height,image_width), np.nan)

            # Get RBF interpolator objects
            disc_dx,disc_dy,interp_rbfx, interp_rbfy = ipt.smooth_field(
                matrix, 
                nodes_2d, 
                deformed_nodes_2d, 
                3)        

        fem_xcoord = nodes_2d[:, 0]
        fem_ycoord = nodes_2d[:, 1]
        fem_x_disp = displacements_2d[:, 0] 
        fem_y_disp = displacements_2d[:, 1]
        fem_mag = np.sqrt(fem_x_disp**2 + fem_y_disp**2)

        # -------------------------------------------------------------------------
        # Coordinate system correction
        # -------------------------------------------------------------------------
        lag_dx = interp_rbfx(np.column_stack([fem_xcoord, fem_ycoord]))
        lag_dy = interp_rbfy(np.column_stack([fem_xcoord, fem_ycoord]))

        fem_xcoord_lag = fem_xcoord - lag_dx
        fem_ycoord_lag = fem_ycoord - lag_dy
        fem_points = np.column_stack((fem_xcoord_lag, fem_ycoord_lag))

        # Initialising i manually. Struggling with enumerate() for some reason
        i = 0
        for prefix_num in all_expected_prefixesbin:
            try:

                print(f'\nCurrent prefix number is {prefix_num}')

                # Expected file name
                sunfile = f'{prefix_num}_results.sdic'
                file_number = prefix_num
                sundic_data_path = os.path.join(sundic_binary_folder, sunfile)
                print(f'Reading DIC data: file {sundic_data_path}')

                if not os.path.exists(sundic_data_path):
                    print(f"\n| ERROR: File path {sundic_data_path} not found. \nMoving to next prefix\n")
                    continue

                # -------------------------------------------------------------------------
                # Load DIC results object
                # -------------------------------------------------------------------------
                sundic_data, nRows, nCols = sdpp.getDisplacements(
                    sundic_data_path,
                    -1, 
                    smoothWindow=0
                    )

                # Organise results and associated coordinate locations into vectors
                x_coord, y_coord = sundic_data[:, 0], sundic_data[:, 1]
                X_disp, Y_disp = sundic_data[:, 3], sundic_data[:, 4]
                dic_mag = np.sqrt(X_disp**2 + Y_disp**2)

                # Create matrix from vectors
                sundic_points = np.column_stack((x_coord,y_coord))


                # -------------------------------------------------------------------------
                # Interpolation of displacements for error calculation
                # -------------------------------------------------------------------------
                # Select displacement quantities to analyse
                value = 0
                if value == 0:
                    fem_value = fem_x_disp
                    dic_value = X_disp
                    string = 'X_disp'
                elif value == 1:
                    fem_value = fem_y_disp
                    dic_value = Y_disp
                    string = 'Y_disp'
                elif value == 2:
                    fem_value = fem_mag
                    dic_value = dic_mag
                    string = 'Magnitude'
                else:
                    raise ValueError("Invalid value")

                # -------------------------------------------------------------------------
                # Interpolate FE displacements on to the SUNDIC coordinate frame
                # -------------------------------------------------------------------------
                interpolated_FEM_values = griddata(
                    fem_points, 
                    fem_value, 
                    sundic_points, 
                    method='cubic'
                    )

                # Reshape data into 2D matrix
                dic_value = dic_value.reshape(nCols,nRows)
                interpolated_FEM_values = interpolated_FEM_values.reshape(nCols,nRows)

                # -------------------------------------------------------------------------
                # Calculate the error grid
                # -------------------------------------------------------------------------
                errors2 = interpolated_FEM_values - dic_value

                # -------------------------------------------------------------------------
                # Save error grid as binary files (same naming convention as reference images)
                # for later use
                # -------------------------------------------------------------------------
                numpath = os.path.join(numpy_files,f"{file_number}_errors.npy")       
                np.save(numpath, errors2)
                #----------------------------------------------------

                # Increment i to move on to the next file
                i = i + 1

                # -------------------------------------------------------------------------
                # Create y-averaged slices of both FE and DIC displacements 
                # -------------------------------------------------------------------------
                # The SUNDIC coordinate grid is used for both. The FE data has been 
                # interpolated on to the DIC frame so the shapes are compatible
                x_grid = x_coord.reshape(nCols, nRows)

                # Collapse error grid along y (axis = 1)
                collapsed_errorgrid = np.nanmean(errors2, axis=1)

                # Collapse FE displacement data along axis = 1
                FE_value_collapse = np.nanmean(interpolated_FEM_values, axis=1)

                # Collapse DIC displacement data along axis = 1
                DIC_collapse = np.nanmean(dic_value, axis = 1)

                # Domain
                x_line = np.mean(x_grid, axis=1)
                
                # Save error grid as binary file
                #----------------------------------------------------
                slice_numpy = np.column_stack((
                    collapsed_errorgrid,
                    x_line,
                    FE_value_collapse,
                    DIC_collapse
                    ))
           
                # Save the slice matrix as a .npy file in 
                save_collape = os.path.join(slice_path,f'{file_number}_slice_{string}.npy')
                np.save(save_collape,slice_numpy)
                #----------------------------------------------------

                plt.figure(figsize=(5, 3))
                plt.plot(x_line,collapsed_errorgrid, color='black', linewidth=1.5)
                plt.title("Collapsed Error Grid")
                plt.xlabel("Pixels")
                plt.ylabel(f"Error {string}")
                plt.ylim(-0.05, 0.05)
                plt.grid(True)
                slice_save = os.path.join(
                    slice_path, f'{file_number}_slice_{string}.png')
                plt.savefig(slice_save)
                plt.close()

            except Exception as e:
                # Extract from traceback object
                tb = traceback.extract_tb(sys.exc_info()[2])
                filename, line_number, function_name, text = tb[-1]  # Last traceback entry
                print(f'Error with file: {sundic_data_path},\nMessage {str(e)}\nLine: {line_number}')
                

    # -------------------------------------------------------------------------
    # Separate the error results into two groups
    # -------------------------------------------------------------------------
    # At this point the error files are all saved in the same location. This part of the script
    # groups the files into their associated deformation method to be able to compare
    # them. It is not difficult to achieve the separation because the 2N image files
    # repeat after N images. Therefor the error results, which use the same prefix-based 
    # naming convention, are split based after the N'th prefix.

    if flags['error_separation']:
                                                  
        # Read the current excel workbook file
        path_2_doc = ipt.excel_doc_path(excel_pathh, 
                                        doc_name = "my_doc",
                                        override_doc_num = None)

        # Half the reference images (split at N)
        image_list = ipt.get_image_strings(reference_image_path)
        iter_range = int(len(image_list) / 2)      
        print(f'range = {iter_range}')

        # Initiate vectors to store the RMSE and percentage differences
        rms_diffr = np.full((int(iter_range),1), np.nan)
        percent_diffr = np.full((iter_range, 1), np.nan)

        # -------------------------------------------------------------------------
        # Loop through each analytical/ discrete deformation data pair
        # -------------------------------------------------------------------------
        for i in range(int(iter_range)):

            try:

                # The index is doubled to make it even because the file names use
                # even-numbered prefixes
                iter = i*2

                an_file_prefix = iter

                # Prefix number doubled to access the data associated with the 
                # analytical deformation.
                dis_file_prefix = an_file_prefix + iter_range*2 

                # Use both index groups to access data files using their names
                an_error_name = f"{an_file_prefix}_errors.npy"
                dis_error_name = f"{dis_file_prefix}_errors.npy"

                file_1_path = os.path.join(numpy_files,an_error_name)
                file_2_path = os.path.join(numpy_files,dis_error_name)

                file_1_error_grid = np.load(file_1_path)
                file_2_error_grid = np.load(file_2_path)

                # Find the RMSE for each file
                RMSE_file_1 = np.sqrt(np.mean(file_1_error_grid**2))
                RMSE_file_2 = np.sqrt(np.mean(file_2_error_grid**2))

                # print(f"--------\nRMSE discrete: {RMSE_file_1}")
                # print(f"RMSE analytical: {RMSE_file_2}\n--------")

                # Determine the amplification ratio
                amplification_ratio = RMSE_file_2/RMSE_file_1

                # Ratio as a percentage
                percent_diff = 100 * (amplification_ratio)

                # Store results as vectors in the initialised objects
                rms_diffr[i] = amplification_ratio
                percent_diffr[i] = percent_diff

            except Exception:
                print("| Error: Issue with the error grid binary file ")


        # Read excel document. This is where the '
        p_metrics,meas_error,p_param,nans,_ = ipt.read_spec_excel(
            excel_pathh,doc_num=None, doc_name='my_doc')
        
        metric_strings = ("MSF", "MIG", "$E_f$", "MIOSD",
                                "Shannon entropy", "PSA", 
                                "SSSIG", "$R_{peak}$")

        # there are only N metric entries in the excel sheet but there are 2N error entries.
        # Limit p_metrics to only the first N entries

        image_list = ipt.get_image_strings(reference_image_path)
        iter_range = int(len(image_list)/2)       # Loop over half of the reference images
        print(f'range = {iter_range}')
        save_fig = True


        for number,metric in enumerate(metric_strings):

            plot_title = "RMSE_difference"
            x_value = p_metrics[:iter_range, number]

            x_label = f'{metric}'
            y_value = rms_diffr
            y_label = '$A_{amp}$'

            # Fix: x_value is already 1D, no extra index
            x_value_1d = x_value[:iter_range].ravel()
            y_value_1d = np.abs(rms_diffr).ravel()

            print(f"It does not exist: {rms_diffr.shape}")

            # Remove NaNs and 3-sigma outliers
            valid_mask = ~np.isnan(y_value_1d)
            # mean, std = y_value_1d[valid_mask].mean(), y_value_1d[valid_mask].std()
            # valid_mask &= (y_value_1d >= mean - 3*std) & (y_value_1d <= mean + 3*std)

            # Correlation
            # r = np.corrcoef(x_value_1d[valid_mask], y_value_1d[valid_mask])

            # Plot
            plt.figure(figsize=(5,3),dpi=300)
            plt.scatter(x_value_1d[valid_mask], y_value_1d[valid_mask], color='black')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout(pad=1.5)   

            # plt.title(plot_title)
            # plt.text(
            #     0.95, 1.02, 
            #     f'R = {r[0,1]:.2f}', 
            #     transform=plt.gca().transAxes, 
            #     ha='right'
            # )
            # plt.ylim([0,12.1])
            plt.grid('on')
            plt.tight_layout()

            plt.draw()
            plt.get_current_fig_manager().window.raise_()

            if save_fig:
                os.makedirs(scatter_plots, exist_ok=True)
                plot_save = os.path.join(scatter_plots,f'{plot_title}_{x_label}vs{y_label}.png')
                plt.savefig(plot_save)

            plt.close()

        # -------------------------------------------------------------------------
        # Plot percent error change against metrics
        # -------------------------------------------------------------------------
        # For every metric in the list:
        for number, metric in enumerate(metric_strings):
            
            # Create a scatter plot
            plot_title = f"Percent change {metric}"

            # Get the first N entries in the specific metric column of the excel sheet 
            x_value = p_metrics[:iter_range, number]
            y_value = percent_diffr
            x_label = metric
            y_label = 'Amplitude difference (%)'

            # Flatten to 1D
            x_value_1d = x_value[:iter_range].ravel()  # fix in case x_value is already 1D
            y_value_1d = y_value.ravel()               # flatten y_value

            # Remove NaNs and 3-sigma outliers
            valid_mask = ~np.isnan(y_value_1d)
            # mean, std = y_value_1d[valid_mask].mean(), y_value_1d[valid_mask].std()
            # valid_mask &= (y_value_1d >= mean - 2*std) & (y_value_1d <= mean + 2*std)

            # Correlation
            # r = np.corrcoef(x_value_1d[valid_mask], y_value_1d[valid_mask])

            # Plot
            plt.figure(figsize=(5, 3),dpi=300)
            plt.scatter(x_value_1d[valid_mask], y_value_1d[valid_mask], color='darkred')
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            # Show more detail
            # plt.title(plot_title)
            # plt.text(
            #     0.95, 1.02,
            #     f'R = {r[0, 1]:.2f}',
            #     transform=plt.gca().transAxes,
            #     ha='right'
            # )
            plt.grid(True)
            plt.tight_layout()

            if save_fig:
 
                plot_save = os.path.join(
                    scatter_plots,
                    f'{plot_title}_{x_label}vs{y_label}.png'
                )
                plt.savefig(plot_save)

            # plt.pause(1)
            plt.close()

        # 3 Save to excel
        ipt.write_data_to_excel(path_2_doc, rms_diffr, percent_diffr)


    if flags["separate_errors"]:
            
        # 2. Separate the interpolation errors
        path_2_doc = ipt.excel_doc_path(excel_pathh, 
                                        doc_name = "my_doc",
                                        override_doc_num = None)
        ref_path = r"data\speckle_pattern_img\reference_im"
        image_list = ipt.get_image_strings(ref_path)
        iter_range = int(len(image_list) / 2)       # Loop over half of the reference images
        print(f'range = {iter_range}')

        # Initialise objects for storing values pertaining to 
        # the differences between the image pairs
        rms_diffr = np.full((int(iter_range),1), np.nan)
        percent_diffr = np.full((iter_range, 1), np.nan)


        for i in range(int(iter_range)):

            iter = i * 2
            print('iter = ', iter)
            if iter % 2 != 0:
                continue
                
            file_1 = iter                           # image 0, 2, 4, ...,N-2 -> first half of the image set -> analytical
            file_2 = file_1 + iter_range*2          # image N, N + 2, N + 4, ...., N + iter_range*2 -> corresponding images in the second set
            string = 'X_disp'

            # Load slice data and compute difference
            path_to_slice_bin = r"output\Slices\slice_binaries"         # Where the numpy slices data is saved
            path_2 = r"output\Slices\differences"
            if not os.path.exists(path_2):
                os.makedirs(path_2)

            # Load corresponding numpy file pairs
            file_1_path = os.path.join(path_to_slice_bin, f'{file_1}_slice_{string}.npy')
            file_2_path = os.path.join(path_to_slice_bin, f'{file_2}_slice_{string}.npy')

            if not os.path.exists(file_1_path) or not os.path.exists(file_2_path):
                continue

            print('file 1 prefix =', file_1)
            print('file 2 prefix =', file_2)

            file_1_dat = np.load(file_1_path, allow_pickle=False)
            file_2_dat = np.load(file_2_path, allow_pickle=False)

            # Point-wise differences for plotting
            difference = file_2_dat[:,0] - file_1_dat[:,0]                     

            # Calculate RMS of reference error (file_1)
            rms_ref = np.sqrt(np.mean(file_1_dat[:, 0]**2)) + 1e-8  # Small offset to avoid division by zero

            # Calculate RMS of the difference between signals
            rms_diff = np.sqrt(np.mean((file_2_dat[:, 0] - file_1_dat[:, 0])**2))

            # Mean absolute error (amplitude differences)
            mae_diff = np.mean(np.abs(file_2_dat[:, 0] - file_1_dat[:, 0]))         # Absolute amplitude difference
            mae_ref = np.mean(np.abs(file_1_dat[:, 0])) + 1e-12                     # Absolute file 1 mean amplitude 
            mae_def = np.mean(np.abs(file_2_dat[:, 0]))                             # Absolute file 2 mean amplitude

            deviation_ratio = mae_def / mae_ref

            # Compute percentage difference relative to reference RMS
            percent_diff = 100 * (deviation_ratio)

            # Store results 
            rms_diffr[i]        = deviation_ratio
            percent_diffr[i]    = percent_diff

            print(f'\nRMS difference = {deviation_ratio:.4f}') 
            print(f'Percentage difference (relative to ref RMS) = {percent_diff:.2f}%')


            x_line = file_1_dat[:,1]

            plt.figure(figsize=(10, 12))
            #-------------------------------------------------------------
            # Analytical image
            plt.subplot(3,1,1)
            plt.plot(x_line,file_1_dat[:,0], color='blue', linewidth=1.5)
            plt.title(f"Collapsed error {file_1} (Analytical image deformation)")
            plt.xlabel("Pixels")
            plt.ylabel(f"Error {string}")
            # Nan axis limits not permitted
            if np.isnan(np.max(file_1_dat[:,0])):
                plt.ylim(-0.015,0.015)
            else:
                plt.ylim(-np.max(file_1_dat[:,0])*1.5, np.max(file_1_dat[:,0])*1.5)
            plt.grid(True)

            #-------------------------------------------------------------
            # Combined errors
            plt.subplot(3,1,2)
            plt.plot(x_line,file_2_dat[:,0], color='blue', linewidth=1.5)
            plt.title(f"Collapsed error {file_2} (with pre-DIC interpolation)")
            plt.xlabel("Pixels")
            plt.ylabel(f"Error {string}")
            if np.isnan(np.max(file_1_dat[:,0])):
                plt.ylim(-0.015,0.015)
            else:
                plt.ylim(-np.max(file_1_dat[:,0])*1.5, np.max(file_1_dat[:,0])*1.5)
            plt.grid(True)

            #-------------------------------------------------------------
            # Difference between the two (separated)
            plt.subplot(3,1,3)
            plt.plot(x_line,difference, color='blue', linewidth=1.5)
            plt.title(f"Collapsed error difference betweeen {file_2} and {file_1}")
            plt.xlabel("Pixels")
            plt.ylabel(f"Error {string}")
            if np.isnan(np.max(file_1_dat[:,0])):
                plt.ylim(-0.015,0.015)
            else:
                plt.ylim(-np.max(file_1_dat[:,0])*1.5, np.max(file_1_dat[:,0])*1.5)
            plt.grid(True)
            plt.tight_layout()

            slice_save = os.path.join(path_2, f'{file_2},{file_1}_difference_{string}.png')
            plt.savefig(slice_save)
            # plt.show(block=False)
            # plt.pause(0.05)
            plt.close()

        print(f'Mean difference = {rms_diffr}')
                            
        p_metrics,meas_error,p_param,nans,_ = ipt.read_spec_excel(excel_pathh,doc_num=None, doc_name='my_doc')
        metric_strings = ("Mean subset fluctuation (MSF)", "Mean intensity gradient (MIG)", "E_f", "Mean intensity of the second derivative (MIOSD)",
                                "Shannon entropy", "Power spectrum", "SSSIG", "Autocorrelation peak radius")

        # there are only N metric entries in the excel sheet but there are 2N error entries.
        # Limit p_metrics to only the first N entries
        image_list = ipt.get_image_strings(reference_image_path)
        iter_range = int(len(image_list) / 2)       # Loop over half of the reference images
        print(f'range = {iter_range}')
        save_fig = True


        for number,metric in enumerate(metric_strings):

            plot_title = "Relative RMS difference between images"
            x_value =  p_metrics[:iter_range, number]

            # x_value = meas_error[mask,4]
            x_label = f'{metric}'
            y_value = rms_diffr
            y_label = 'Normalised root means squared difference'

            # Correlation coefficient
            # Correlation
            valid_mask = ~np.isnan(rms_diffr).ravel()
            r = np.corrcoef(x_value[valid_mask], np.abs(rms_diffr).ravel()[valid_mask])

            plt.figure(figsize=(7,6))
            plt.scatter(x_value,np.abs(y_value), color='black')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(plot_title)
            plt.text(
                0.95, 1.02, 
                f'R = {r[0,1]:.2f}', 
                transform=plt.gca().transAxes, 
                ha='right'
            )
            # plt.ylim(0, 0.06)  # Set y-axis limit
            plt.grid('on')    
            plt.draw()
            plt.get_current_fig_manager().window.raise_()

            if save_fig:
                # old plot_path -> plot_path
                if not os.path.exists(scatter_plots):
                    os.makedirs(scatter_plots)
                plot_save = os.path.join(scatter_plots,f'{plot_title}_{x_label}vs{y_label}.png')
                plt.savefig(plot_save)

            plt.pause(1)
            plt.close()

        # --- Plot percent error change against metrics ---
        for number, metric in enumerate(metric_strings):

            plot_title = "Percent change in image pairs"
            x_value = p_metrics[:iter_range, number]
            y_value = percent_diffr
            x_label = metric
            y_label = 'Mean amplitude change (%)'

            # Correlation (ignoring NaNs)
            valid_mask = ~np.isnan(y_value).ravel()
            r = np.corrcoef(x_value[valid_mask], y_value.ravel()[valid_mask])

            plt.figure(figsize=(7, 6))
            plt.scatter(x_value, y_value, color='darkred')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(plot_title)
            plt.text(
                0.95, 1.02,
                f'R = {r[0, 1]:.2f}',
                transform=plt.gca().transAxes,
                ha='right'
            )
            plt.grid(True)
            plt.tight_layout()

            if save_fig:
                if not os.path.exists(scatter_plots):
                    os.makedirs(scatter_plots)
                plot_save = os.path.join(
                    scatter_plots,
                    f'{plot_title}_{x_label}vs{y_label}.png'
                )
                plt.savefig(plot_save)

            plt.close()

        # 3 Save to excel
        ipt.write_data_to_excel(path_2_doc, rms_diffr, percent_diffr)


        # -------------------------------------------------------------------------
        # Fit a model to the amplitude ratio
        # -------------------------------------------------------------------------    if flags['error_correction_opt']:
        # Amplitude_ratio = f(metric_1, ......, metric 7)
        # Uses SVR with GridSearchCV

        FLAG = {
    
            'Compare_correction': True,
            'compare_with_analytical': False,
            'run_optimisation': False
        }
        
        # Read current excel workbook
        myd_p_metrics, change, _, _, _ = ipt.read_spec_excel(
            excel_pathh, 
            doc_num=None, 
            doc_name='my_doc',
            
            )
        
        # Model uses metric results as the (X) input
        x_1_traincorr = myd_p_metrics[:, 0]
        x_2_traincorr = myd_p_metrics[:, 1]
        x_3_traincorr = myd_p_metrics[:, 2]
        x_4_traincorr = myd_p_metrics[:, 3]
        x_5_traincorr = myd_p_metrics[:, 4]
        x_6_traincorr = myd_p_metrics[:, 5]
        x_7_traincorr = myd_p_metrics[:, 6]

        # Ignore NaN and zero results. The zeros indicate a failed input from the upstream processes
        zmask = (~np.isnan(change[:, 3])) & (change[:, 3] > 0)
        z_correction = change[zmask,3]

        # Stack the input data into a multidimensional object
        coordinate_arr_train = np.vstack((
            x_1_traincorr[zmask], 
            x_2_traincorr[zmask], 
            x_3_traincorr[zmask], 
            x_4_traincorr[zmask], 
            x_5_traincorr[zmask], 
            x_6_traincorr[zmask], 
            x_7_traincorr[zmask]
            )
            ).T

        # -------------------------------------------------------------------------
        # Surrogate modelling of the error amplification
        # -------------------------------------------------------------------------
        # SVR hyperparameters
        p_grid_corr = {
            'kernel': ['rbf'],  
            'C': np.logspace(-3, 3, 7),
            'gamma': ('auto', 'scale')
        }

        # Scale input
        scaler_X_correction = StandardScaler()
        X_scaled_correction = scaler_X_correction.fit_transform(coordinate_arr_train)

        # Scale output
        scaler_z_corr = StandardScaler()
        z_scaled_corr = scaler_z_corr.fit_transform(z_correction.reshape(-1, 1)).ravel()

        # Initialise SVR model
        svr_model_corr= GridSearchCV(
            SVR(), 
            p_grid_corr, 
            cv=5)
        
        # Train SVR model
        svr_model_corr.fit(X_scaled_correction, z_scaled_corr)

        # Model scoring
        print(f"Best R² (CV): {svr_model_corr.best_score_:.4f}")
        print(f"Train R²: {svr_model_corr.best_estimator_.score(X_scaled_correction, z_scaled_corr):.4f}")

        # Amp ratio prediction using trained SVR. Input is 
        # unscaled. Output is unscaled prediction. 
        def amp_change_value(X):
            """Predict using trained SVR model"""
            X_scaled_corrr = scaler_X_correction.transform(X)
            z_pred_scaled_corr = svr_model_corr.predict(X_scaled_corrr)
            return scaler_z_corr.inverse_transform(z_pred_scaled_corr.reshape(-1, 1)).ravel()

        # Get Analytically deformed results ( If applicable)
        excel_already_correct = r"output\excel_docs"
        
        # -------------------------------------------------------------------------
        # Data correction using model
        # -------------------------------------------------------------------------
        if FLAG['compare_with_analytical']:

            # Load analytically deformed data for comparison
            _, ANmeas_error, _, _, _ = ipt.read_spec_excel(excel_already_correct, doc_num=None)

        # Correct data from batch studies
        for batch in range(0, 12):

            # Make batch-specific file paths
            optimised_sav_b = os.path.join(optimised_save,f'batch_{batch}')
            if not os.path.exists(optimised_sav_b):
                os.makedirs(optimised_sav_b)

            # Configuration for batch processing
            excel_data_to_correct = rf"output\excel_docs\excel_{batch}"

            # Read measurement data
            print(f"\nReading uncorrected sheet: \n{excel_data_to_correct}\n")
            p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(
                excel_data_to_correct, doc_num=None)
            print('\n\reading:', indicators)

            # Extract coordinate features from measurement data
            x_1 = p_metrics[:, 0]
            x_2 = p_metrics[:, 1]
            x_3 = p_metrics[:, 2]
            x_4 = p_metrics[:, 3]
            x_5 = p_metrics[:, 4]
            x_6 = p_metrics[:, 5]
            x_7 = p_metrics[:, 6]

            coordarr_for_correction = np.vstack((
                x_1, 
                x_2, 
                x_3, 
                x_4, 
                x_5, 
                x_6, 
                x_7
                )
                ).T

            # Initialize correction arrays
            z = meas_error[:, 0]
            corrected_error = np.zeros(len(z))

            # Apply amplitude correction to each data point by looking at the metric values at that point
            for interr in range(len(z)):

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                # Skip rows with NaN values
                if np.any(np.isnan(coordarr_for_correction[interr])):
                    print(f"NaNs found in row {interr}: {coordarr_for_correction[interr]}")
                    continue

                # Calculate amplitude ratio and apply correction
                amplitude_ratio = float(amp_change_value(coordarr_for_correction[interr].reshape(1, -1))[0])
                if amplitude_ratio < 0:
                    amplitude_ratio = 1
                corrected_error[interr] = z[interr] / (amplitude_ratio)


            # Save corrected data to new Excel file
            corrected_excel_path = rf"output\excel_docs\after_corr_{texture}"
            if not os.path.exists(corrected_excel_path):
                os.makedirs(corrected_excel_path)

            # Create copy of original measurement error array and replace with corrected values
            meas_error_corrected = meas_error.copy()
            meas_error_corrected[:, 0] = corrected_error

            # Save to new Excel file
            corrected_filename = os.path.join(corrected_excel_path, f'corrected_batch_{batch}.xlsx')
            ipt.write_spec_excel(
                corrected_filename,
                p_metrics=p_metrics,
                meas_error=meas_error_corrected,
                p_param=p_param,
                nans=nans,
                indicators=indicators
            )
            print(f"Saved corrected data to: {corrected_filename}")
            # Apply data filtering and outlier removal
            threshold = 0.001
            outlier_idx = ipt.array_difference_outlier(meas_error[:, 0], meas_error[:, 2], threshold)
            mask = np.array([i not in outlier_idx for i in range(meas_error.shape[0])]) & (meas_error[:, 0] > 0) & (meas_error[:, 0] != 0) & (np.abs(meas_error[:, 0] - np.nanmean(meas_error[:, 0])) <= 3 * np.nanstd(meas_error[:, 0]))
            nonzero_mask = meas_error[:, 0] != 0


            # Plots and analysis
            figflag = {
                '2d scatter': True,
                '3d scatter': False,
                'xtreme_index': False
            }

            # 2D scatter plot of original and corrected data
            if figflag['2d scatter']:

                metric_strings = (
                    "Mean subset fluctuation (MSF)", 
                    "MIG", 
                    "E_f",
                    "Mean intensity of the second derivative (MIOSD)", 
                    "Shannon entropy",
                    "Power area", 
                    "SSSIG", 
                    "Autocorrelation peak radius"
                )
                error_strings = ("RMSE [pixel]", "Mean bias error")
                save_fig = True

                for number, metric in enumerate(metric_strings):
                    plot_title = f"Corrected_{indicators[0,0]}_{indicators[1,0]}"
                    x_value = p_metrics[mask, number]
                    x_label = f'{metric}'
                    y_label = error_strings[0]

                    P_change = (np.mean(meas_error[mask, 0]) - np.mean(corrected_error[mask])) / np.mean(meas_error[mask, 0]) * 100
                    plt.figure(figsize=(4, 3))

                    if FLAG['compare_with_analytical']:
                        # Corrected data
                        plt.scatter(x_value, corrected_error[mask], alpha=0.5, color='black', label='Corrected data')
                        # True analytical data
                        plt.scatter(x_value, ANmeas_error[mask,0], color='red', label='Analytical data')
                        # Data with grid-based interpolation artifacts
                        plt.scatter(x_value, meas_error[mask, 0], color='blue', label='Uncorrected data')

                    elif FLAG['Compare_correction']:
                        plt.scatter(x_value, corrected_error[mask], alpha=0.5, color='black', label='Corrected data')
                        plt.scatter(x_value, meas_error[mask, 0], color='blue', label='Uncorrected data')

                    else:
                        plt.scatter(x_value, corrected_error[mask], alpha=1, color='black', label='Analytical data')

                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.ylim([0,0.06])
                    plt.legend()
                    # plt.title(plot_title)
                    # plt.text(
                    #     0.98, 1.02,
                    #     f'Change = {P_change:.2f}%',
                    #     transform=plt.gca().transAxes,
                    #     ha='right'
                    # )
                    plt.grid('on')
                    plt.draw()
                    plt.get_current_fig_manager().window.raise_()

                    if save_fig:
                        plot_save = os.path.join(err_plots_correct, f'{plot_title}_{x_label}vs{y_label}.png')
                        plt.savefig(plot_save)

                    # plt.pause(0.2)
                    plt.close()


toc = time.time()

print(f'\n\nRuntime:{toc-tic:.3f} seconds\n-------------------------')
