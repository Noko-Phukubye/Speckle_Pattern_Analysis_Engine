# region Import libraries
import numpy as np
import sundic.sundic as sdic
import sundic.post_process as sdpp
import sundic.settings as sdset
import configparser
import matplotlib.pyplot as plt
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
from numpy.linalg import norm
import file_paths as path
import os
import image_process_tool_box as imgan
# import image_analysis_toolbox as imgan
from scipy.interpolate import griddata
import traceback
import sys
# endregion

# region Paths
error_hist_path = r"output\Histograms"
DIC_contour_path = r"output\DIC_contour"
error_heatmap_path = r"output\Heatmaps"
save_contor = r'output\DIC_contour'
debugg_folder = r'output\Debugging'
excel_path = r"output\excel_docs"                             
gen_speckle_pattern_save = r"data\speckle_pattern_img\reference_im"      
deformed_image_path = r"data\speckle_pattern_img\deformed_im"
reference_image_path = r"data\speckle_pattern_img\reference_im"
debugging_folder = r'output\debugging'
DIC_settings_path = r"settings.ini"
sundic_save = r'output\pyth'
Contour_path = r"output\DIC_contour"
spt_img_path = r'data\speckle_pattern_img\Rigid translation'
sundic_binary_folder = r'output\sundic_bin'
op2_path = path.flat30_4mm_op2
bdf_path = path.flat30_4mm_bdf
sundic_binary_folder = r'output\sundic_bin'
slice_path = r"output\Slices"


# endregion


FE_flags = {'show meshes':False,
            'method_1': False,
            'method_2': True,
            'method_3': False}
# region Load FE data
# Display figure of mesh nodes if necessary


print('\n4. Loading FEA data...\n')

model = OP2()
model.read_op2(op2_path)
bdf = BDF()
bdf.read_bdf(bdf_path)

# Extract the displacement data from the op2 file. Assuming isubcase = 1 and static analysis
itime = 0
isubcase = 1
disp = model.displacements[isubcase]
# Extract the translational displacements
txyz = disp.data[itime, :, :3]

# Calculate the total deflection of the vector (from documentation)
total_xyz = norm(txyz, axis=1)
nnodes = disp.data.shape[1]

# Get the node positions from the BDF file
nodes = np.array([bdf.nodes[nid].get_position() for nid in bdf.nodes])
# Extract only the x and y components for 2D visualization (2d dic)
nodes_2d = nodes[:, :2]
displacements_2d = txyz[:, :2]
# Calculate the deformed positions by adding displacements to original node positions [The shapes of the objects from the bdf and the op2 are (nnodes, 3)]
# deformed_nodes = nodes + txyz
# Calculate the deformed positions by adding displacements to original node positions (2D)
deformed_nodes_2d = nodes_2d + displacements_2d

# Plotting the original and deformed mesh in 2D
if FE_flags['show meshes']:
    plt.figure(figsize=(12, 6))
    # Original mesh
    plt.subplot(1, 2, 1)
    plt.scatter(nodes_2d[:, 0] , nodes_2d[:, 1], c='b', marker='o', label='Original')
    plt.title('Original Mesh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    # Deformed mesh
    plt.subplot(1, 2, 2)
    plt.scatter(deformed_nodes_2d[:, 0], deformed_nodes_2d[:, 1], c='r', marker='o', label='Deformed')
    plt.title('Deformed Mesh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

# FEA points and alignment
x_scale = 1000
# y_scale = 1000
original_points = nodes_2d * x_scale
new_points = deformed_nodes_2d * x_scale
print('\nFEA data loaded successfully.')
# endregion


# region Method 1
if FE_flags['method_1']:

    # This variable selects the folders in which the resulting images are saved
    meth = 1

    # Creating the heatmap save folder (if one does not exist already)
    globals()[f'method_{meth}_heat'] = os.path.join(error_heatmap_path,f'meth_{meth}')
    if not os.path.exists(globals()[f'method_{meth}_heat']):
        os.makedirs(globals()[f'method_{meth}_heat'])
        print('Directory generated')

    print('\n8. Error distribution analysis...\n')

    # Uses string filtering to get a list of files to open from the binary folder
    sundic_binary_files = sorted(
        [f for f in os.listdir(sundic_binary_folder)
        if f.endswith('results.sdic')],
        key=lambda x: int(x.split('_')[0])
    )

    # Gets a list of prefixes based on the contents of the binary files folder. The
    # prefixes are used to assume file names to open files
    all_expected_prefixesbin = imgan.expected_prefixes(sundic_binary_folder,odd=False,skip=True)
    prefix_positionsbin = {prefix: i for i, prefix in enumerate(all_expected_prefixesbin)}

    # FE data
    # Data exists as 1D vectors of shape (1224,)
    fem_xcoord = nodes_2d[:, 0] * x_scale
    fem_ycoord = nodes_2d[:, 1] * x_scale
    fem_x_disp = displacements_2d[:, 0] * x_scale  # x-direction displacement
    fem_y_disp = displacements_2d[:, 1] * x_scale
    fem_mag = np.sqrt(fem_x_disp**2 + fem_y_disp**2)
    fem_points = np.column_stack((fem_xcoord, fem_ycoord))  # Creates a 2D coordinate array of x in the first column and y in the second column
    # fem_points = np.column_stack((fem_ycoord, fem_xcoord))
    print(f'FEM points shape = {fem_points.shape}')

    # Cycle through binary folder and open each file. Assume the file name based on the expected prefix and 
    # check if it exists.
    i = 0
    for prefix_num in all_expected_prefixesbin:
        try:

            # Expected file name based on prefixes
            sunfile = f'{prefix_num}_results.sdic'
            file_number = prefix_num
            sundic_data_path = os.path.join(sundic_binary_folder, sunfile)
            print(f'\nReading DIC data: file {sundic_data_path}')

            if not os.path.exists(sundic_data_path):
                print(f"\nFile path: {sundic_data_path} not found. Moving to next prefix\n")
                continue

            # Load path and file name for DIC data
            sundic_data, nRows, nCols = sdpp.getDisplacements(sundic_data_path,-1, smoothWindow=25)
            print(f'nRows = {nRows}\n nCols = {nCols}')

            # sundic_data = np.load(sundic_data_path)
            # Data is in the form of 1-D arrays just like the FE data (31324,)
            # Confirmed that data is read correctly
            x_coord, y_coord = sundic_data[:, 0], sundic_data[:, 1]
            X_disp, Y_disp = sundic_data[:, 3], sundic_data[:, 4]
            print(f'X_disp.shape={X_disp.shape}')
            print(f'X-coord.shape={x_coord.shape}')
            dic_mag = np.sqrt(X_disp**2 + Y_disp**2)

            # Coordinate array for interpolation
            # sundic_points = np.column_stack((x_coord_filtered, y_coord_filtered))
            sundic_points = np.column_stack((x_coord,y_coord))    # 2D coordinate array (x and y in first and second column like FE data)
            print(f'DIC points shape = {sundic_points.shape}')
            print(f'reshaped DIC data (Z) = {(dic_mag.reshape(nCols,nRows)).shape}')

            # === Interpolation and Error Analysis ===
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

            # Only interpolate FE-grid and subtract DIC data
            #----------------------------------------------------
            interpolated_FEM_values = griddata(fem_points, fem_value, sundic_points, method='cubic')

            # Reshape data
            dic_value = dic_value.reshape(nCols,nRows)
            interpolated_FEM_values = interpolated_FEM_values.reshape(nCols,nRows)

            errors = interpolated_FEM_values - dic_value
            # Reshape into 2D error grid
            error_mag = errors
            #----------------------------------------------------

            # Quick contour plot
            # First, reshape your coordinates and errors correctly
            x_grid = x_coord.reshape(nCols, nRows)  # Note: rows first, columns second
            y_grid = y_coord.reshape(nCols, nRows)
            errors = errors.reshape(nCols, nRows)  # Assuming errors has the same length initially

            # Contour plot 
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = ax.contourf(x_grid, y_grid, errors, alpha=1, zorder=2, cmap='jet')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_aspect('equal')  
            fig.colorbar(contour, ax=ax)
            plt.show()

            # For the line plot - average errors along y-axis (rows)
            error_line_in_x = np.mean(errors, axis=1)  # Average across rows (y-direction)

            # Create a proper x-axis for plotting
            x_line = np.mean(x_grid, axis=1) # Create evenly spaced points matching your column count

            plt.figure(figsize=(10, 8))
            plt.plot(x_line, error_line_in_x, color='black', linewidth=1.5)
            plt.xlabel('x (pixels)')
            plt.ylabel('Mean error')
            plt.title('Mean Error Along x')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # #-------------------------------------------------------------------------------------------------
            # Plotting is the same for all methods
            plt.figure(figsize=(10, 8))
            plt.imshow(errors.T, cmap='jet', interpolation='none', 
                    extent=(x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()), 
                    origin='lower')  # ,vmin=-0.003, vmax=0.003
            plt.colorbar(label='Error magnitude')
            plt.title(f'Filtered Error Distribution: Pattern {file_number}')
            plt.xlabel('X')
            plt.ylabel('Y')
            Heatmap_path = os.path.join(globals()[f'method_{meth}_heat'], f'{file_number}_heatmap_{string}.png')
            plt.savefig(Heatmap_path)
            plt.close()
           

            # #-------------------------------------------------------------------------------------------------
            # # Create and save histogram
            # # Histogram path
            # globals()[f'method_{meth}_histogram'] = os.path.join(error_hist_path,f'meth_{meth}')
            # if not os.path.exists(globals()[f'method_{meth}_histogram']):
            #     os.makedirs(globals()[f'method_{meth}_histogram'])

            # plt.figure(figsize=(12, 10))
            # plt.hist(errors[~np.isnan(errors)], bins=100, density=True)  # Use only non-NaN values
            # plt.title(f'Error Distribution: Pattern {file_number}', fontsize=22)
            # plt.xlabel(f'Error {string}', fontsize=24)
            # plt.ylabel('Frequency', fontsize=24)
            # plt.tick_params(axis='both', which='major', labelsize=20)
            # plt.xlim(-0.05, 0.05)
            # plt.ylim(0, 300)
            # histogrampath = os.path.join(globals()[f'method_{meth}_histogram'], f'{file_number}_histogram_{string}.png')
            # plt.savefig(histogrampath)
            # plt.close()

            # #-------------------------------------------------------------------------------------------------
            # Collapse grid
            globals()[f'method_{meth}_slices'] = os.path.join(slice_path,f'meth_{meth}')
            if not os.path.exists(globals()[f'method_{meth}_slices']):
                os.makedirs(globals()[f'method_{meth}_slices'])

            collapsed_errorgrid = np.mean(errors, axis=1)
            x_line = np.mean(x_grid, axis=1) 

            plt.figure(figsize=(10, 6))
            plt.plot(x_line,collapsed_errorgrid, color='blue', linewidth=1.5)
            plt.title("Collapsed Error Grid")
            plt.xlabel("Pixels")
            plt.ylabel(f"Error {string}")
            # plt.ylim(-0.02, 0.02)
            plt.grid(True)
            slice_save = os.path.join(globals()[f'method_{meth}_slices'], f'{file_number}_Slice_{string}.png')
            plt.savefig(slice_save)
            plt.close()
            #-------------------------------------------------------------------------------------------------

        except Exception as e:
            # Extract from traceback object
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line_number, function_name, text = tb[-1]  # Last traceback entry
            print(f'Error with file: {sundic_data_path},\nMessage {str(e)}\nLine: {line_number}')   
# endregion



# region method 2
if FE_flags['method_2']:

    meth = 2
    
    print('\n8. Error distribution analysis...\n')

    sundic_binary_files = sorted(
        [f for f in os.listdir(sundic_binary_folder)
        if f.endswith('results.sdic')],
        key=lambda x: int(x.split('_')[0])
    )
    all_expected_prefixesbin = imgan.expected_prefixes(sundic_binary_folder,odd=False,skip=True)
    prefix_positionsbin = {prefix: i for i, prefix in enumerate(all_expected_prefixesbin)}

    # FE data
    fem_xcoord = nodes_2d[:, 0] * x_scale
    fem_ycoord = nodes_2d[:, 1] * x_scale
    fem_x_disp = displacements_2d[:, 0] * x_scale  # x-direction displacement
    fem_y_disp = displacements_2d[:, 1] * x_scale
    fem_mag = np.sqrt(fem_x_disp**2 + fem_y_disp**2)
    fem_points = np.column_stack((fem_xcoord, fem_ycoord))
    print(f'FEM points shape = {fem_points.shape}')

    # Initialising i manually. Struggling with enumerate() for some reason
    i = 0
    for prefix_num in all_expected_prefixesbin:
        try:

            # Expected file name
            sunfile = f'{prefix_num}_results.sdic'
            file_number = prefix_num
            sundic_data_path = os.path.join(sundic_binary_folder, sunfile)
            print(f'\nReading DIC data: file {sundic_data_path}')

            if not os.path.exists(sundic_data_path):
                print(f"File path: {sundic_data_path} not found. Moving to next prefix\n")
                continue

            # Load path and file name for DIC data
            sundic_data, nRows, nCols = sdpp.getDisplacements(sundic_data_path,-1, smoothWindow=25)

            # sundic_data = np.load(sundic_data_path)
            x_coord, y_coord = sundic_data[:, 0], sundic_data[:, 1]
            X_disp, Y_disp = sundic_data[:, 3], sundic_data[:, 4]
            print(f'X_disp.shape={X_disp.shape}')
            print(f'X-coord.shape={x_coord.shape}')
            dic_mag = np.sqrt(X_disp**2 + Y_disp**2)

            # Coordinate array for interpolation
            sundic_points = np.column_stack((x_coord,y_coord))
            print(f'DIC points shape = {sundic_points.shape}')
            print(f'reshaped DIC data (Z) = {(dic_mag.reshape(nCols,nRows)).shape}')

            # === Interpolation and Error Analysis ===
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

            # Interpolate both data-sets on to the same grid
            #----------------------------------------------------
            # x_min, x_max = np.min(x_coord), np.max(x_coord)
            # y_min, y_max = np.min(y_coord), np.max(y_coord)
            x_min, x_max = 0, 2000
            y_min, y_max = 0, 500

            # Grid sizes
            grid_size_x = nCols  
            grid_size_y = nRows  

            x_grid = np.linspace(x_min, x_max, grid_size_x)
            y_grid = np.linspace(y_min, y_max, grid_size_y)
            xx, yy = np.meshgrid(x_grid, y_grid)

            # Interpolate FEM data onto regular grid
            fem_grid_values = griddata(fem_points, fem_value, (xx, yy), method='linear')

            # Interpolate DIC data onto regular grid
            dic_grid_values = griddata(sundic_points, dic_value, (xx, yy), method='linear')

            # Calculate error on regular grid
            errors2 = fem_grid_values - dic_grid_values
            #----------------------------------------------------

            #-------------------------------------------------------------------------------------------------
            # Create and save heatmap
            globals()[f'method_{meth}_heat'] = os.path.join(error_heatmap_path,f'meth_{meth}')
            if not os.path.exists(globals()[f'method_{meth}_heat']):
                os.makedirs(globals()[f'method_{meth}_heat'])

            plt.figure(figsize=(10, 8))
            # errors2 = np.nan_to_num(errors2, nan=0.0) 
            plt.imshow(errors2, cmap='jet', interpolation='none', 
                    extent=(x_min, x_max, y_min, y_max), 
                    origin='lower')
            plt.colorbar(label='Error magnitude')
            plt.title(f'Filtered Error Distribution: Pattern {file_number}')
            plt.xlabel('X')
            plt.ylabel('Y')
            Heatmap_path = os.path.join( globals()[f'method_{meth}_heat'], f'{file_number}_heatmap_{string}.png')
            plt.savefig(Heatmap_path)
            plt.close()

            #-------------------------------------------------------------------------------------------------
            # Create and save histogram
            globals()[f'method_{meth}_histogram'] = os.path.join(error_hist_path,f'meth_{meth}')
            if not os.path.exists(globals()[f'method_{meth}_histogram']):
                os.makedirs(globals()[f'method_{meth}_histogram'])
            plt.figure(figsize=(12, 10))
            plt.hist(errors2[~np.isnan(errors2)].flatten(), bins=1000, density=True)  # Flatten and use only non-NaN values
            plt.title(f'Error Distribution: Pattern {file_number}', fontsize=22)
            plt.xlabel(f'Error {string}', fontsize=24)
            plt.ylabel('Frequency', fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.xlim(-0.05, 0.05)
            plt.ylim(0, 300)
            histogrampath = os.path.join(globals()[f'method_{meth}_histogram'], f'{file_number}_histogram_{string}.png')
            plt.savefig(histogrampath)
            plt.close()

            #-------------------------------------------------------------------------------------------------
            # Collapse grid
            # globals()[f'method_{meth}_slices'] = os.path.join(slice_path,f'meth_{meth}')
            # if not os.path.exists(globals()[f'method_{meth}_slices']):
            #     os.makedirs(globals()[f'method_{meth}_slices'])
            # ticks = 9
            # x_ticks_custom = np.linspace(0, 2000, num=ticks)
            # x_tick_labels = [f"{int(x)}" for x in x_ticks_custom]
            # collapsed_errorgrid = np.nanmean(error_grid_filtered, axis=0).reshape(1, error_grid_filtered.shape[1])
            # plt.figure(figsize=(10, 6))
            # plt.plot(collapsed_errorgrid.flatten(), color='blue', linewidth=1.5)
            # plt.title("Collapsed Error Grid")
            # plt.xlabel("Pixels")
            # plt.ylabel(f"Error {string}")
            # # plt.ylim(-0.02, 0.02)
            # plt.grid(True)
            # plt.xticks(np.linspace(0, error_grid_filtered.shape[1] - 1, num=ticks), x_tick_labels)
            # slice_path = r"output\Slices"
            # slice_save = os.path.join(globals()[f'method_{meth}_slices'], f'{file_number}_Slice_{string}.png')
            # plt.savefig(slice_save)
            # plt.close()

            globals()[f'method_{meth}_slices'] = os.path.join(slice_path,f'meth_{meth}')
            if not os.path.exists(globals()[f'method_{meth}_slices']):
                os.makedirs(globals()[f'method_{meth}_slices'])

            ticks = 9
            x_ticks_custom = np.linspace(0, 2000, num=ticks)
            x_tick_labels = [f"{int(x)}" for x in x_ticks_custom]

            collapsed_errorgrid = np.nanmean(errors2, axis=0).reshape(1, errors2.shape[1])
            plt.figure(figsize=(10, 6))
            plt.plot(collapsed_errorgrid.flatten(), color='blue', linewidth=1.5)
            plt.title("Collapsed Error Grid")
            plt.xlabel("Pixels")
            plt.ylabel("RMSE")
            # plt.ylim(-0.02, 0.02)
            plt.grid(True)
            plt.xticks(np.linspace(0, errors2.shape[1] - 1, num=ticks), x_tick_labels)
            slice_path = r"output\Slices"
            slice_save = os.path.join(globals()[f'method_{meth}_slices'], f'{file_number}_Slice_{string}.png')
            plt.savefig(slice_save)
            plt.close()
            #-------------------------------------------------------------------------------------------------


        except Exception as e:
            # Extract from traceback object
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line_number, function_name, text = tb[-1]  # Last traceback entry
            print(f'Error with file: {sundic_data_path},\nMessage {str(e)}\nLine: {line_number}')    
# endregion



# region method 3
if FE_flags['method_3']:

    meth = 3
    
    print('\n8. Error distribution analysis...\n')

    sundic_binary_files = sorted(
        [f for f in os.listdir(sundic_binary_folder)
        if f.endswith('results.sdic')],
        key=lambda x: int(x.split('_')[0])
    )
    all_expected_prefixesbin = imgan.expected_prefixes(sundic_binary_folder,odd=False,skip=True)
    prefix_positionsbin = {prefix: i for i, prefix in enumerate(all_expected_prefixesbin)}

    # FE data
    fem_xcoord = nodes_2d[:, 0] * x_scale
    fem_ycoord = nodes_2d[:, 1] * x_scale
    fem_x_disp = displacements_2d[:, 0] * x_scale  # x-direction displacement
    fem_y_disp = displacements_2d[:, 1] * x_scale
    fem_mag = np.sqrt(fem_x_disp**2 + fem_y_disp**2)
    fem_points = np.stack((fem_xcoord, fem_ycoord)).T
    print(f'FEM points shape = {fem_points.shape}')

    # Initialising i manually. Struggling with enumerate() for some reason
    i = 0
    for prefix_num in all_expected_prefixesbin:
        try:

            # Expected file name
            sunfile = f'{prefix_num}_results.sdic'
            file_number = prefix_num
            sundic_data_path = os.path.join(sundic_binary_folder, sunfile)
            print(f'Reading DIC data: file {sundic_data_path}')

            if not os.path.exists(sundic_data_path):
                print(f"File path: {sundic_data_path} not found. Moving to next prefix\n")
                continue

            # Load path and file name for DIC data
            sundic_data, nRows, nCols = sdpp.getDisplacements(sundic_data_path,-1, smoothWindow=25)

            # sundic_data = np.load(sundic_data_path)
            x_coord, y_coord = sundic_data[:, 0], sundic_data[:, 1]
            X_disp, Y_disp = sundic_data[:, 3], sundic_data[:, 4]
            print(f'X_disp.shape={X_disp.shape}')
            print(f'X-coord.shape={x_coord.shape}')
            dic_mag = np.sqrt(X_disp**2 + Y_disp**2)

            # Exclude points with displacement magnitude above threshold (for outliers)
            threshold = 9999999999
            filtered_indices = dic_mag < threshold

            # 1-D arrays of filtered DIC coordinates and data
            x_coord_filtered = x_coord[filtered_indices]
            y_coord_filtered = y_coord[filtered_indices]
            X_disp_filtered = X_disp[filtered_indices]
            Y_disp_filtered = Y_disp[filtered_indices]
            madlol_filtered = dic_mag[filtered_indices]
            # Coordinate array for interpolation
            sundic_points = np.vstack((x_coord_filtered, y_coord_filtered)).T
            print(f'DIC points shape = {sundic_points.shape}')
            print(f'reshaped DIC data (Z) = {(dic_mag.reshape(nCols,nRows)).shape}')

            # === Interpolation and Error Analysis ===
            value = 2
            if value == 0:
                fem_value = fem_x_disp
                dic_value = X_disp_filtered
                string = 'X_disp'
            elif value == 1:
                fem_value = fem_y_disp
                dic_value = Y_disp_filtered
                string = 'Y_disp'
            elif value == 2:
                fem_value = fem_mag
                dic_value = madlol_filtered
                string = 'Magnitude'
            else:
                raise ValueError("Invalid value")

            # interpolate twice (old method)
            #----------------------------------------------------
            interpolated_FEM_values = griddata(fem_points, fem_value, sundic_points, method='cubic')

            # Remove nans 
            valid_indices = ~np.isnan(interpolated_FEM_values)
            # Initialize errors with NaNs
            errors = np.empty_like(interpolated_FEM_values)
            errors[:] = np.nan
            errors[valid_indices] = interpolated_FEM_values[valid_indices] - dic_value[valid_indices]

            #  error threshold
            error_threshold = 0.2  
            valid_error_indices = valid_indices & (np.abs(errors) < error_threshold)

            x_valid = x_coord_filtered[valid_error_indices]
            y_valid = y_coord_filtered[valid_error_indices]
            errors_valid = errors[valid_error_indices]

            # limits based on DIC coordinates
            x_min, x_max = np.min(x_coord), np.max(x_coord)
            y_min, y_max = np.min(y_coord), np.max(y_coord)
            grid_size_x = 500 # Grid control
            grid_size_y = 125  
            x_grid = np.linspace(x_min, x_max, grid_size_x)
            y_grid = np.linspace(y_min, y_max, grid_size_y)
            xx, yy = np.meshgrid(x_grid, y_grid)

            # Final smoothed error grid
            error_grid_smooth = griddata((x_valid, y_valid), errors_valid, (xx, yy), method='cubic')
            #----------------------------------------------------


            globals()[f'method_{meth}_heat'] = os.path.join(error_heatmap_path,f'meth_{meth}')
            if not os.path.exists(globals()[f'method_{meth}_heat']):
                os.makedirs(globals()[f'method_{meth}_heat'])

            plt.figure(figsize=(10, 8))
            plt.imshow(error_grid_smooth, cmap='jet', interpolation='none', 
                    extent=(x_min, x_max, y_min, y_max), 
                    origin='lower')
            plt.colorbar(label='Error magnitude')
            plt.title(f'Filtered Error Distribution: Pattern {file_number}')
            plt.xlabel('X')
            plt.ylabel('Y')
            mean_err = np.nanmean(np.abs(error_grid_smooth))
            text = plt.figtext(0.5, 0.02, f'Mean error: {mean_err:.4f}', size=14)
            Heatmap_path = os.path.join( globals()[f'method_{meth}_heat'], f'{file_number}_heatmap_{string}.png')
            plt.savefig(Heatmap_path)
            plt.close()

            # Create and save histogram
            globals()[f'method_{meth}_histogram'] = os.path.join(error_hist_path,f'meth_{meth}')
            if not os.path.exists(globals()[f'method_{meth}_histogram']):
                os.makedirs(globals()[f'method_{meth}_histogram'])

            plt.figure(figsize=(12, 10))
            plt.hist(errors[valid_error_indices], bins=100, density=True)
            plt.title(f'Error Distribution: Pattern {file_number}', fontsize=22)
            plt.xlabel('Error magnitude [mm]', fontsize=24)
            plt.ylabel('Frequency', fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.xlim(-0.07, 0.07)
            plt.ylim(0, 300)
            histogrampath = os.path.join(globals()[f'method_{meth}_histogram'], f'{file_number}_histogram_{string}.png')
            plt.savefig(histogrampath)
            plt.close()

            # Collapse grid for the line plot
            globals()[f'method_{meth}_slices'] = os.path.join(slice_path,f'meth_{meth}')
            if not os.path.exists(globals()[f'method_{meth}_slices']):
                os.makedirs(globals()[f'method_{meth}_slices'])

            ticks = 9
            x_ticks_custom = np.linspace(0, 2000, num=ticks)
            x_tick_labels = [f"{int(x)}" for x in x_ticks_custom]

            collapsed_errorgrid = np.nanmean(error_grid_smooth, axis=0).reshape(1, error_grid_smooth.shape[1])
            plt.figure(figsize=(10, 6))
            plt.plot(collapsed_errorgrid.flatten(), color='blue', linewidth=1.5)
            plt.title("Collapsed Error Grid")
            plt.xlabel("Pixels")
            plt.ylabel("RMSE")
            plt.ylim(-0.02, 0.02)
            plt.grid(True)
            plt.xticks(np.linspace(0, error_grid_smooth.shape[1] - 1, num=ticks), x_tick_labels)
            slice_path = r"output\Slices"
            slice_save = os.path.join(globals()[f'method_{meth}_slices'], f'{file_number}_Slice_{string}.png')
            plt.savefig(slice_save)
            plt.close()

        except Exception as e:
            # Extract from traceback object
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line_number, function_name, text = tb[-1]  # Last traceback entry
            print(f'Error with file: {sundic_data_path},\nMessage {str(e)}\nLine: {line_number}')   
# endregion
