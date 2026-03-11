import matplotlib.pyplot as plt
import image_process_tool_box as ipt
import  numpy as np
import os
from scipy.interpolate import RBFInterpolator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import image_process_tool_box as ipt
from scipy import stats
import numpy as np
import cv2
import pandas as pd


plt.style.use(['science', 'no-latex','ieee', 'grid'])

plt.rcParams.update({
    'font.family': 'Calibri',
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
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

flags = {'Interclass_comparisons': False,
         'compare_optimum': False,
         'compare_KDE': False,
         'Image_matrix': False,
         'Distributions_of_parameters': False,
         'model_evaluation': False,
         'Sample comparisons': False,
         'High_level_analysis': False,
         'In-sample compare': False,
         'Metric-error metamodel': False,
         "IQRvsRMSE": False,
         'Miscellenious': False,
         'MetricPercentiles': False,
         }

ipt.flag_status(flags,0.25)


if flags['Interclass_comparisons']:

    # Paths
    save_dir = r"output\plots\Feb_10\ord"
    os.makedirs(save_dir, exist_ok=True)

    # load data
    # excel_paths = [
    #     rf"output\excel_docs\excel_{i}" for i in range(12)
    #     ]

    wait_time = 0.025

    # patt_class = [6,7,8,9,10,11,0,1,2,3,4,5]  

    patt_class = [6,7,8,9,10,11,0,1,2,3,4,5]   

    # patt_class = [0,2,3,4,5,6,8,9,10,11]
    # patt_class = [8,9,10,11,2,3,4,5,]
    
    # Create a list of excel path strings (for each pattern class)
    excel_paths = [
        rf"output\excel_docs\excel_{i}" for i in patt_class
        ]

    # Initialise containers
    p_metrics_all = []
    meas_errors_all = []
    p_params_all = []
    nans_all = []
    indicators_all = []

    # Loop through paths and load data
    for path in excel_paths:

        # Temporary Doc number
        # existing_files = [f for f in os.listdir(path)
        #                 if f.startswith('Pattern_evaluation') and f.endswith(".xlsx")]
        # temp_doc_number = len(existing_files) - 1
        p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(path, doc_num=None)


        # p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(path, doc_num=None)
        p_metrics_all.append(p_metrics)             # Matrix of metris. Each class occupies column
        meas_errors_all.append(meas_error)
        p_params_all.append(p_param)
        nans_all.append(nans)
        indicators_all.append(indicators)


    # Box plots labels
    scatter_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']
    scatter_colors = [
            '#FF0000',  # Pure Red
            '#0000FF',  # Pure Blue
            '#00FF00',  # Pure Green
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFFF00',  # Yellow
            '#FF8000',  # Orange
            '#8000FF',  # Purple
            '#00FF80',  # Spring Green
            '#FF0080',  # Deep Pink
            '#0080FF',  # Sky Blue
            '#80FF00',  # Chartreuse
    ]

    pattern_classes = ['S', 'Ch', 'P', 'PC', 'Pbl', 'PS','RS', 'RCh', 'RP', 'RPc', 'RPbl', 'RPS']
    
    # pattern_classes = ['Class 1', 'Class 2', 'Class 3', 
    #                 'Class 4', 'Class 5', 'Class 6',
    #                 'Class 7', 'Class 8', 'Class 9', 
    #                 'Class 10', 'Class 11', 'Class 12']
    
    # pattern_classes = ['Class 1', 'Class 2', 'Class 3',         # No checkerboard
    #                 'Class 4', 'Class 5', 'Class 6',
    #                 'Class 7', 'Class 8', 'Class 9', 
    #                 'Class 10', 'Class 11', 'Class 12']

    selected_labels = [pattern_classes[i] for i,_ in enumerate(patt_class)]

    metric_names = ['MSF', 'MIG', '$E_f$', 
                    'MIOSD', 'Shannon entropy', 'PSA', 'SSSIG', "$R_{peak}$"]  



    error_names = ['RMSE [pixel]','IQR','Total systematic error', 'Total variance error']

    # box_color = '#E68A00'
    box_color = '#D97904'
    median_color = 'black'
     

    # 1. Boxplot for RMSE per pattern class
    specific = [0,1,4,6]

    for i, error_column in enumerate(specific):

        error_per_class = []

        for idx, j in enumerate(patt_class):

            errs = np.abs(meas_errors_all[idx][:, error_column])
            mean_val = np.nanmean(errs)
            std_val = np.nanstd(errs)
            # mask = (~np.isnan(errs)) & (errs != 0) & (np.abs(errs - mean_val) <= 3 * std_val)
            mask = (errs != 0) 
            # & (np.abs(errs - mean_val) <= 3 * std_val)

            error_per_class.append(errs[mask])  # filtered error data
        

        plt.figure(figsize=(5,3))
        bp = plt.boxplot(
            error_per_class, 
            tick_labels=selected_labels, 
            showfliers=False, 
            patch_artist=True)
            
        for patch in bp['boxes']:
            patch.set_facecolor(box_color)
        for median in bp['medians']:
            median.set(color=median_color)
        plt.ylabel(error_names[i])
        # plt.title(f"{error_names[i]} per Pattern Class")
        # plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{error_names[i]}.png"), dpi=300)
        plt.show(block=False)
        plt.pause(wait_time)
        plt.close()

        # --- Quartile + Min/Max values Figure (for error boxplot) ---
        # Initialise empty lists for the values
        mins, q1s, meds, q3s, maxs, iqrs = [], [], [], [], [], []   

        # Get relevent data from all errors above and populate the lists
        for filtered in error_per_class:
            if len(filtered) > 0:
                # If the data exists
                mins.append(float(f"{np.min(filtered):.6f}"))
                q1 = float(f"{np.percentile(filtered, 25):.6f}")
                med = float(f"{np.percentile(filtered, 50):.6f}")
                q3 = float(f"{np.percentile(filtered, 75):.6f}")
                maxs.append(float(f"{np.max(filtered):.6f}"))
                q1s.append(q1)
                meds.append(med)
                q3s.append(q3)
                iqrs.append(float(f"{q3 - q1:.6f}"))
            else:
                # Otherwise NAN 
                mins.append(np.nan)
                q1s.append(np.nan)
                meds.append(np.nan)
                q3s.append(np.nan)
                maxs.append(np.nan)
                iqrs.append(np.nan)

        plt.figure()
        # Loop through all lists of quartile data simultaneously
        table_data = list(zip(mins, q1s, meds, q3s, maxs, iqrs))
        col_labels = ["Min", "Q1", "Median", "Q3", "Max", "IQR"]
        plt.axis("off")
        plt.table(
            cellText=np.round(table_data, 7),
            rowLabels=selected_labels,
            colLabels=col_labels,
            loc="center"
        )
        plt.title(f"Statistics for {error_names[i]}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Stats_{error_names[i]}.png"), dpi=300)
        plt.show(block=False)
        plt.pause(wait_time)
        plt.close()

    # 2. Scatter plot RMSE vs selected metric
    for metric_index in range(8):
        metric_per_class = [p_metrics_all[idx][:, metric_index] for idx in range(len(patt_class))]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=1000)
        
        for idx, i in enumerate(patt_class):
            # mask per class
            # error_mask = (meas_errors_all[idx][:, 0] < 0.25) & (meas_errors_all[idx][:, 0] != 0)
            error_mask = ((meas_errors_all[idx][:, 0] < 0.25) 
                          & (meas_errors_all[idx][:, 0] != 0) 
                          & (np.abs(meas_errors_all[idx][:, 0] - np.nanmean(meas_errors_all[idx][:, 0])) <= 3 * np.nanstd(meas_errors_all[idx][:, 0])))

            x = p_metrics_all[idx][error_mask, metric_index]
            y = np.abs(meas_errors_all[idx][error_mask, 0])

            # eish
            mask = y != 0
            
            ax.scatter(x[mask], y[mask], 
                    label=selected_labels[idx], 
                    alpha=0.85,
                    marker=scatter_markers[idx % len(scatter_markers)],
                    color=scatter_colors[idx % len(scatter_colors)],
                    s=8,
                    edgecolors='black',
                    linewidths=0.5)
        
            ax.minorticks_on()
            ax.grid(True, which='major', alpha=0.3, linewidth=0.8)
            ax.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor('black')
                spine.set_visible(True)
            
            # Configure tick parameters
            ax.tick_params(axis='both', which='major', 
                        width=1.0,
                        length=5,
                        labelsize=10,
                        direction='in')
            ax.tick_params(axis='both', which='minor', 
                        width=0.6,
                        length=3,
                        direction='in')
            
        ax.set_xlabel(metric_names[metric_index])
        ax.set_ylabel("RMSE [pixel]")

        # Scientific notation x-axis
        ax.ticklabel_format(
            axis='x', 
            style='sci', 
            scilimits=(0,0)
            )

        ax.grid(True, alpha=0.5)
        ax.grid(True, alpha=0.5)
        
        # Export legend separately (only once, not for every metric)
        if metric_index == 0:
            # Create a separate figure for the legend
            figlegend = plt.figure(figsize=(5, 2))
            handles, labels = ax.get_legend_handles_labels()
            figlegend.legend(handles, labels, loc='center', ncol=6, frameon=True, 
                            fontsize=12, markerscale=3)
            figlegend.savefig(os.path.join(save_dir, "legend_separate.png"), dpi=300, bbox_inches='tight')
            plt.close(figlegend)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"scatter_RMSE_vs_{metric_names[metric_index].replace(' ', '_')}.png"), dpi=300)
        plt.show(block=False)
        plt.pause(wait_time)
        plt.close()

        # Show oultiers - Properties
        flier_properties = {
            'marker': 'o',          
            'markerfacecolor': 'white',
            'markersize': 4,      
            'markeredgecolor': 'black',
            'alpha': 0.95            
        }
        # 3. Boxplot for selected metric
        plt.figure(figsize=(5,3))
        bp = plt.boxplot(
            metric_per_class, 
            tick_labels=selected_labels, 
            flierprops=flier_properties,
            showfliers=True, 
            patch_artist=True
            )
        
        for patch in bp['boxes']:
            patch.set_facecolor(box_color)
        for median in bp['medians']:
            median.set(color=median_color)
        plt.ylabel(metric_names[metric_index])
        # plt.title(f"{metric_names[metric_index]} per Pattern Class")
        plt.grid(True)
        # plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"boxplot_{metric_names[metric_index].replace(' ', '_')}_per_class.png"), dpi=1000)
        plt.show(block=False)
        plt.pause(wait_time)
        plt.close()

        # --- Quartile + Min/Max values Figure (for metric boxplot) ---
        mins, q1s, meds, q3s, maxs, iqrs = [], [], [], [], [], []
        for filtered in metric_per_class:
            if len(filtered) > 0:
                mins.append(np.min(filtered))
                q1 = np.percentile(filtered, 25)
                med = np.percentile(filtered, 50)
                q3 = np.percentile(filtered, 75)
                maxs.append(np.max(filtered))
                q1s.append(q1)
                meds.append(med)
                q3s.append(q3)
                iqrs.append(q3 - q1)
            else:
                mins.append(np.nan)
                q1s.append(np.nan)
                meds.append(np.nan)
                q3s.append(np.nan)
                maxs.append(np.nan)
                iqrs.append(np.nan)

        plt.figure()
        table_data = list(zip(mins, q1s, meds, q3s, maxs, iqrs))
        col_labels = ["Min", "Q1", "Median", "Q3", "Max", "IQR"]
        plt.axis("off")
        plt.table(
            cellText=np.round(table_data, 4),
            rowLabels=selected_labels,
            colLabels=col_labels,
            loc="center"
        )
        plt.title(f"Statistics for {metric_names[metric_index]}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Stats_{metric_names[metric_index]}.png"), dpi=300)
        plt.show(block=False)
        plt.pause(wait_time)
        plt.close()



if flags['compare_optimum']:

    save_directory = r'output\plots\Scatter_plots\optimised_comparison'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # If RTG, indicate
    striiing = True
    if striiing:
        add_string = 'Rtg'
        excel_doc = [0,1,2,3,4,5]
    else:
        add_string = None 
        excel_doc = [6,7,8,9,10,11]

    for docnumber in excel_doc:
        optimised_patterns_data = r'output\excel_docs'

        # Get specific optimised pattern data 
        # Select optimised pattern to analyse. Prefix number is used. 
        # Make sure path to optimised patterns excel sheet is correct 
        # and that all optimised patterns have been evaluated.
        pattern_prefix = docnumber * 2

        #----------------------------------
        index = (pattern_prefix / 2)               # RMSE optimised patterns
        print('index:', index)

        # mbindex = index + 24/2                       

        msfindex = index + 24/2   

        migindex = index + 48/2                    

        efindex =  index + 72/2                        

        miosdindex = index + 96/2   

        shanindex = index + 120/2                   

        powerindex = index + 144/2   

        sssigindex = index + 168/2   

        autoindex = index + 192/2

        # Fixed indices for reference patterns (not batch-dependent)
        star1index = 216 / 2

        randomindex = 218 / 2

        # iqr_index = index + 240/2               


        #----------------------------------
        

        # Accessing excel documents. The first line is for reading the excel document for a specific pattern clasee
        # the second line reads the document containing the data for the analysed optimised pattern
        # the pattern will be selected based on its prefix value. It is important to ensure that the selected pattern
        # is of the same class as the class-specific excel data against which it is compared.
        excel_folder = rf'output\excel_docs\excel_{docnumber}'

        p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(excel_folder, doc_num = None)
        op_metrics, omeas_error, op_param, onans, oindicators = ipt.read_spec_excel(optimised_patterns_data, doc_num = 165)

        errors = meas_error[:, 0]
        mask = (~np.isnan(errors)) & (errors != 0) & (np.abs(errors - np.nanmean(errors)) <= 3 * np.nanstd(errors))
        

        metric_names = ['Mean subset fluctuation', 'Mean intensity gradient', 'E_f', 
                        'MIOSD', 'Shannon', 'Power area', 'SSSIG', 'Autocorrelation peak']  
        error_names = ['RMSE','IQR','Systematic error', 'Standard deviation error']
        error_columns = [0,1,4,6]

    
        annotate_points = True 

        # For each speckle pattern metric
        for metric_col in range(p_metrics.shape[1]):

            # Each each error category
            for error_name_index, error_col in enumerate(error_columns):

                # Dictionary for getting the x and y values for plotting
                # Optimised patterns are denoted by blocks that are indicated by the prefix in the name
                # In each iteration we collect the patterns that are optimised towards specific objectives
                # by indicating the index at which the block starts and offsetting that number by the batch
                key_points = {

                    "RMSE": [op_metrics[int(index), metric_col], omeas_error[int(index), error_col]],
                    # "Bias": [op_metrics[int(mbindex), metric_col], omeas_error[int(mbindex), error_col]],
                    "MSF": [op_metrics[int(msfindex), metric_col], omeas_error[int(msfindex), error_col]],
                    "MIG": [op_metrics[int(migindex), metric_col], omeas_error[int(migindex), error_col]],
                    "Ef": [op_metrics[int(efindex), metric_col], omeas_error[int(efindex), error_col]],
                    "Miosd": [op_metrics[int(miosdindex), metric_col], omeas_error[int(miosdindex), error_col]],
                    "Shannon": [op_metrics[int(shanindex), metric_col], omeas_error[int(shanindex), error_col]],
                    "Power": [op_metrics[int(powerindex), metric_col], omeas_error[int(powerindex), error_col]],
                    "SSSIG": [op_metrics[int(sssigindex), metric_col], omeas_error[int(sssigindex), error_col]],
                    "Autocorr": [op_metrics[int(autoindex), metric_col], omeas_error[int(autoindex), error_col]],
                    # "Star-1": [op_metrics[int(star1index), metric_col], omeas_error[int(star1index), error_col]],
                    # "Random": [op_metrics[int(randomindex), metric_col], omeas_error[int(randomindex), error_col]],
                    # "IQR": [op_metrics[int(iqr_index), metric_col], omeas_error[int(iqr_index), error_col]],
                }

                # Distinct markers and colors. Might need to look into better representations or markers
                markers = {
                    "RMSE": ('D', 'red'),
                    "MSF": ('v', 'cyan'),
                    "MIG": ('X', 'blue'),       
                    "Ef": ('P', 'brown'),
                    "Miosd": ('H', 'orange'),
                    "Shannon": ('s', 'green'),
                    "Power": ('X', 'darkorange'),
                    "SSSIG": ('*', 'gold'),
                    "Autocorr": ('^', 'purple'),
                    # "Star-1": ('<', 'olive'),
                    # "Random": ('>', 'magenta'),
                }

                # Reference patterns (non-optimised)
                reference_patterns = ["Star-1", "Random"]

                # Plotting
                plt.figure(figsize=(8, 6))
                plt.scatter(
                    p_metrics[mask, metric_col],
                    meas_error[mask, error_col],
                    color='black',
                    alpha=0.5,
                    s=20,
                    label='All patterns'
                )

                # Plot all optimisation points with labels in legend
                for label, (marker, color) in markers.items():
                    x, y = key_points[label]
                    
                    # Check if it's a reference pattern or optimised pattern
                    if label in reference_patterns:
                        legend_label = f'{label} (reference)'
                    else:
                        legend_label = f'{label} optimised'
                    
                    plt.plot(
                        x, y,
                        marker=marker,
                        color=color,
                        markersize=8,
                        markeredgecolor='black',
                        linestyle='None',
                        label=legend_label
                    )

                # Axis labels and plot title
                x_label = metric_names[metric_col]
                y_label = error_names[error_name_index]
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title(f"{y_label} vs {x_label}", fontsize=12)

                # Legend and grid
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend(loc='best', frameon=True)

                # Save location and file naming
                if add_string is None and indicators[1, 0] != 0:
                    pre_string = f'Optim_{indicators[0, 0]}_{indicators[1, 0]}'
                elif add_string is None:
                    pre_string = f'Optim_{indicators[0, 0]}'
                elif indicators[1, 0] != 0:
                    pre_string = f'Optim_{add_string}_{indicators[0, 0]}_{indicators[1, 0]}'
                else:
                    pre_string = f'Optim_{add_string}_{indicators[0, 0]}'

                class_directory = os.path.join(save_directory, pre_string)
                os.makedirs(class_directory, exist_ok=True)

                safe_x = x_label.replace(" ", "_").replace("/", "_")
                safe_y = y_label.replace(" ", "_").replace("/", "_")
                plot_path = os.path.join(class_directory, f'{safe_x}_vs_{safe_y}.png')

                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                plt.close()


if flags["Image_matrix"]:
    ref_image_path = r"data\speckle_pattern_img\reference_im"
    znssd_dit = r"output\DIC_contour\ZNSSD"
    contour = r"output\DIC_contour"
    error_hist = r"output\histograms"
    error_map = r"output\Heatmaps"
    error_slice = r"output\Slices"

    # index = 1
    for index in range(7):
        imnumber = index * 2
        
        ipt.create_image_matrix(
            image_paths=[ref_image_path, 
                         znssd_dit, 
                         contour, 
                        error_hist, 
                        error_map, 
                        error_slice
                        ],

            image_indices=[index, index, index, index, index, index],

            output_path= os.path.join(r"output\image_matrix", f"{imnumber}_pattern_results.png")
            )


if flags["Distributions_of_parameters"]:

    # To see how the results from the analysis of checkerboard patterms behave

    # Evaluation of optimisation
    import image_process_tool_box as ipt
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from scipy.stats import gaussian_kde


    for ii in range(1,2):

        check_borad = rf"output\excel_docs\excel_{ii}"

        p_metrics,meas_error,p_param,nans,indicators = ipt.read_spec_excel(check_borad,doc_num=1)

        # Domian
        x = np.arange(0,p_param.shape[0])

        rms = meas_error[:,0]
        param_1 = p_param[:,0]
        param_2 = p_param[:,1]

        # Specifically for the checkerboard
        #----------------------------------
        N_lines = 500 // 2 * param_2
        #----------------------------------

        mask = rms[:] == 0.079296

        ylabell = 'RMS error'
        xlabell = 'Pattern index'

        error_var = np.std(rms)**2
        data_var = np.std(param_1)**2

        print('\n----------------\n',indicators[0,0])

        print('\nVariance in error:', error_var,'\nData variance:', data_var,'\n----------------')

        print('Variance ratio:', data_var/error_var,'\n----------------')

        print('\nNumber of unique values:')
        print('\nUnique error values:',len(np.unique(rms)) )
        print('Unique parameter 1 values:',len(np.unique(param_1)) )
        print('Unique parameter 2 values:',len(np.unique(N_lines)) )

        ratioo = (param_2/param_1)
        print('Unique values in ratio of parameters', len(np.unique((ratioo).round(decimals=0))))

        if ii == 1 or ii == 7:
            # plt.scatter(x,param_ratio, label='Parameter ratio')
            plt.scatter(x,N_lines, label = 'Error')
            plt.xlabel(xlabell)
            plt.ylabel(ylabell)
            plt.legend()
            plt.show(block=False)
            plt.pause(25)
            plt.close() 

        # KDE of errors
        kde = gaussian_kde(rms)
        x_vals = np.linspace(np.min(rms), np.max(rms), 1000)
        y_vals = kde(x_vals)

        plt.plot(x_vals,y_vals,'k-')
        plt.title('KDE distribution of errors')
        plt.xlabel("RMS")
        plt.ylabel('Frequency')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # KDE of parameters
        kde = gaussian_kde(param_1)
        x_vals = np.linspace(np.min(param_1), np.max(param_1), 1000)
        y_vals = kde(x_vals)

        plt.plot(x_vals,y_vals,'k-')
        plt.title('KDE distribution of parameter 1')
        plt.xlabel("Parameter 1 value")
        plt.ylabel('Frequency')
        plt.show(block=False)
        plt.pause(20)
        plt.close()

        # KDE of metric values
        kde = gaussian_kde(p_metrics[:,1])
        x_vals = np.linspace(np.min(p_metrics[:,1]), np.max(p_metrics[:,1]), 1000)
        y_vals = kde(x_vals)

        plt.plot(x_vals,y_vals,'k-')
        plt.title('KDE distribution of MIG')
        plt.xlabel("Metric value")
        plt.ylabel('Frequency')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


if flags['model_evaluation']:

    # Testing and evaluating different surrogate models
    # Evaluate the model for a single parameter or multiple parameters

    patt_class = 11

    excel_path = rf'output\excel_docs\excel_{patt_class}'
    p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(
        excel_path, doc_num=None
    )

    metric_strings = (
                        "Mean subset fluctuation (MSF)", 
                        "Mean intensity gradient (MIG)", 
                        "E_f", 
                        "Mean intensity of the second derivative (MIOSD)",
                        "Shannon entropy", 
                        "Power area", 
                        "SSSIG", 
                        "Autocorrelation peak radius"
                    )

    # Prepare data
    x_1 = p_param[:, 0]
    x_2 = p_param[:, 1]
    x_3 = p_param[:, 2]
    x_4 = p_param[:, 3]
    x_5 = p_param[:, 4]
    coordinate_arr = np.vstack((x_1, x_2, x_3, x_4)).T
    # coordinate_arr = np.vstack((x_1, x_3, x_4, x_5)).T
    # coordinate_arr = np.vstack((x_1, x_2)).T


    cols = [0,1,2,3,4,5,6,7]
    col_err = [0,4]
    oplys = [3]                 # or set to range(max_oply)
    # cols = [0,6]


    for col in col_err:
    # for col in cols:

        best_poly_rscore = np.zeros(7)
        mean_rscore = [-np.inf, None]

        for oply in oplys:
            z = meas_error[:, col]
            # z = p_metrics[:, col]

            # Split data
            valid_mask = ~np.isnan(z)
            X_clean = coordinate_arr[valid_mask]
            z_clean = z[valid_mask]

            X_train, X_test, z_train, z_test = train_test_split(
                X_clean, z_clean, test_size=0.2, random_state=10
            )


            # ============================================================================
            # MODEL 1: POLYNOMIAL REGRESSION
            # ============================================================================
            poly = PolynomialFeatures(degree=oply)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)
            poly_model = LinearRegression()
            poly_model.fit(X_poly_train, z_train)
            poly_pred = poly_model.predict(X_poly_test)
            poly_r2 = r2_score(z_test, poly_pred)

            best_poly_rscore[oply] = poly_r2 

            # ============================================================================
            # MODEL 2: RBF (RADIAL BASIS FUNCTIONS)
            # ============================================================================
            rbf_model = RBFInterpolator(
                X_train, z_train, kernel='thin_plate_spline', smoothing=0.1
            )
            rbf_pred = rbf_model(X_test)
            rbf_r2 = r2_score(z_test, rbf_pred)


        
            new_mean_rscore = [poly_r2, oply]

            print(f'Column {col} → Polynomial order {oply}\n R² = {new_mean_rscore[0]:.4f}')

            if new_mean_rscore[0] > mean_rscore[0]:
                mean_rscore = new_mean_rscore

        # print(f'\nBest polynomial order for column {col}: {mean_rscore[1]}')
        # print(f'Best R²: {mean_rscore[0]:.4f}\n\n')


        # Extended
        #------------------------------------------------------------
        for oply in range(7):                   # For a range of poly degrees
            for column_number in  col_err:      # For each metric/error
                # z = meas_error[:, 0]
                z = p_metrics[:, column_number]

                # Split data
                X_train, X_test, z_train, z_test = train_test_split(
                    coordinate_arr, z, test_size=0.2, random_state=42
                )

                # ============================================================================
                # MODEL 1: POLYNOMIAL REGRESSION
                # ============================================================================
                poly = PolynomialFeatures(degree=oply)  
                X_poly_train = poly.fit_transform(X_train)
                X_poly_test = poly.transform(X_test)
                poly_model = LinearRegression()
                poly_model.fit(X_poly_train, z_train)
                poly_pred = poly_model.predict(X_poly_test)
                poly_r2 = r2_score(z_test, poly_pred)

                best_poly_rscore[column_number] = poly_r2

                # ============================================================================
                # MODEL 2: RBF (RADIAL BASIS FUNCTIONS)
                # ============================================================================
                rbf_model = RBFInterpolator(
                    X_train, z_train, kernel='thin_plate_spline', smoothing=0.1
                )
                rbf_pred = rbf_model(X_test)
                rbf_r2 = r2_score(z_test, rbf_pred)

                # ============================================================================
                # RESULTS
                # ============================================================================
                # print("Model Performance (R² scores):")
                # print(f"Polynomial (degree {ply}): {poly_r2:.4f}")
                # print(f"RBF:                   {rbf_r2:.4f}")

                # Choose best model
                scores = {'Polynomial': poly_r2, 'RBF': rbf_r2}
                best_model = max(scores, key=scores.get)
                print(f"\nBest model: {best_model} (R² = {scores[best_model]:.4f})")

                # Plot comparison
                # ============================================================================
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                ax1.scatter(z_test, poly_pred, alpha=0.6)
                ax1.plot([z_test.min(), z_test.max()], [z_test.min(), z_test.max()], 'r--')
                ax1.set_title(f'Polynomial (R² = {poly_r2:.3f})')
                ax1.set_xlabel(f'Actual {metric_strings[column_number]}')
                ax1.set_ylabel(f'Predicted {metric_strings[column_number]}')
                ax1.grid(True, alpha=0.3)

                ax2.scatter(z_test, rbf_pred, alpha=0.6)
                ax2.plot([z_test.min(), z_test.max()], [z_test.min(), z_test.max()], 'r--')
                ax2.set_title(f'RBF (R² = {rbf_r2:.3f})')
                ax2.set_xlabel(f'Actual {metric_strings[column_number]}')
                ax2.set_ylabel(f'Predicted {metric_strings[column_number]}')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

                # Plot the worst model - look at after inner loop, find lowest value in best_rscore and plot corresponding

            # Done with all 8 metrics for this degree
            new_mean_rscore = [np.mean(best_poly_rscore), oply]

            print(f'Polynomial order {oply}\n mean R² = {new_mean_rscore[0]:.4f}')

            if new_mean_rscore[0] > mean_rscore[0]:
                mean_rscore = new_mean_rscore

        print(f'\nBest polynomial order is {mean_rscore[1]}')
        print(f'Mean R² = {mean_rscore[0]:.4f}')



if flags['Sample comparisons']: 

    """
    Manual evaluation of pattern optimisation results.
    Compare optimised pattern metrics to batch results and compare
    their statistical significance using percentile ranks.

    User: Select the pattern class by specifying the batch number
          Select the optimisation category by specifying the start_index_in_opt_category
    """
    image_path = r"C:\Users\General User\nokop\pattern2\data\speckle_pattern_img\reference_im\Not_noiseless"
    list_of_images = ipt.get_image_strings(image_path, imagetype='tif', parity='even')


    # ==================== Load data ====================
    # Select pattern class to evaluate
    batch = 11
    excel_folder_address = rf'output\excel_docs\excel_{batch}'
    optimised_patterns_data = r'output\excel_docs'

    # Read batch and optimised data as vectors 
    p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(excel_folder_address, doc_num=None)
    op_metrics, omeas_error, op_param, onans, oindicators = ipt.read_spec_excel(optimised_patterns_data, doc_num=86)

    # ==================== Create mask for values ====================
    threshold_diff = 0.01
    threshold_y = 0.02
    outlier_idx = ipt.array_difference_outlier(meas_error[:, 0], meas_error[:, 2], threshold_diff)  # Check if FEA and DIC interp errors are too different
    outlier_idx_y = [i for i, val in enumerate(meas_error[:, 0]) if abs(val) > threshold_y]
    combined_outliers = set(outlier_idx + outlier_idx_y)
    errors = meas_error[:, 0]
    mymask = (~np.isnan(errors)) & (errors != 0) & (np.abs(errors - np.nanmean(errors)) <= 4 * np.nanstd(errors))

    # ==================== Analysis Loop ====================
    metric_names = ("MSF", "MIG", "E_f", "MIOSD", "Shannon", "Power", "SSSIG", "Peak radius")

    # The assumption is that the images follow an even numbered naming convention as usual. 
    # If there are 9 optimisation objectives then there should be (9 X number_of_pattern_classes) images that are numbered accordingly
    # Order of optimisation objectives:
    number_of_objectives = 9
    # RMSE -> Bias -> MSF -> MIG -> Ef -> MIOSD -> Shannon -> Power -> SSSIG -> Autocorrelation peak radius
    # RMSE optimised patterns start at 0 and end at 22
    # Bias optimised patterns start at 24 and end at 46
    # MSF optimised patterns start at 48 and end at 70
    # And so on

    # The complicated part is here
    # When I said RMSE optimised patterns start at 0 and MSF optimised patterns start at 48, 
    # 0 and 48 are the values that are entered in "start_index_in_opt_category"
    # That means you are evaluating the patterns that were optimised for the objective that is associted with that number
    start_index_in_opt_category = 216

    # The correct image is found using the batch number
    # E.g. If you want to analyse a pattern from the traditional speckle class that was 
    # optimised for RMSE, you need to specify "6" in "batch" and "0" in "start_index_in_opt_category"
    image_index = (start_index_in_opt_category + (batch * 2))
    print(f"\n-------------\nImage prefix: {image_index}")
    excel_index = int(0.5*image_index)

    print(f"Excel row: {excel_index}")
    print(f'Loading image: {list_of_images[image_index//2]}')

    for col in range(1):

        # print(f"------------------\nResults after objective {col+1}")
        # print(f'\n--- {metric_names[col]} ---')

        # batch_metric_values = p_metrics[mymask, col]
        batch_errors = meas_error[mymask, 0]
        batch_bias_errors = meas_error[mymask,4]
        all_iqr = meas_error[mymask,1]

        # Optimised metric and its error
        # sample_x = op_metrics[excel_index, col]       
        sample_error = omeas_error[excel_index, 0]   
        iqrr =    omeas_error[excel_index, 1]     

        # min_metric_value = np.min(batch_metric_values)
        # max_metric_value = np.max(batch_metric_values)
        # mean_metric = np.nanmean(batch_metric_values)

        # print(f"Average metric = {mean_metric}")
        # print(f"Maximum metric = {max_metric_value}")
        # print(f"Minimum metric = {min_metric_value}")


        # print(f"Optimised metric = {sample_x}")
        # Percentile ranks
        # percentile_sample_metric = stats.percentileofscore(batch_metric_values, sample_x)
        # print(f"Metric percentile: {percentile_sample_metric:.1f}th")

        print(f"\n\nSmallest class error = {np.min(batch_errors):.5f}")
        # print(f'Average class error = {np.mean(batch_errors)}')

        print(f"Optimised error = {sample_error:.5f}")
        percentile_min_metric_in_error = stats.percentileofscore(batch_errors, sample_error)
        print(f"Error percentile: {100 - percentile_min_metric_in_error:.1f}th")

        print("-------------------------------------------------")
        print(f"Smallest class iqr = {np.min(all_iqr):.5f}")
        # print(f'Average class error = {np.mean(batch_errors)}')

        print(f"Optimised iqr = {iqrr:.5f}")
        percentile_min_metric_in_error = stats.percentileofscore(all_iqr, iqrr)
        print(f"Error percentile: {100 - percentile_min_metric_in_error:.1f}th")



# Ranking patterns (Low level analysis)
if flags["compare_KDE"]:

    # Make directory for saving plots
    kde_plots = r"C:\Users\General User\nokop\pattern2\output\numpy_files\kde_plots\October9\RMSE2"
    if not os.path.exists(kde_plots):
        os.makedirs(kde_plots)

    numpy_files  = r"C:\Users\General User\nokop\pattern2\output\numpy_files"
    # random = r"C:\Users\General User\nokop\pattern2\output\numpy_files\subset_size"

    # Access the image files using even numbers because that is how they are named
    numbers_file = list(range(0,220, 2))

    # numbers_file2 = [204,42,22,18,84,46]    
    numbers_file2 = [60,180,174,22,132]    


    kde_file_name = 'all5'
    imgs = 0

    if imgs == 1:
        for filenum in numbers_file2:
            # Look at specfic patterns

            refere = r'C:\Users\General User\nokop\pattern2\data\speckle_pattern_img\reference_im\NOISELESS'
            stri = ipt.get_image_strings(refere,parity='even')

            filenum = int(filenum/2)

            print(stri[filenum])
            to_image = os.path.join(refere,stri[filenum])
            mimage = ipt.readImage(to_image)

            first_part  = os.path.splitext(stri[filenum])[0]

            # Get image subset through slicing
            subsets = ipt.img_subsets(mimage, 100,25)
            subset = subsets[5,2,:,:]
            plt.imshow( subset, cmap = 'grey')
            to_sav = os.path.join(kde_plots,f'{stri[filenum]}.png')
            plt.imsave(to_sav,subset,cmap='grey')
            plt.show(block=False)
            plt.close()

        # Kernel density estimate
        ipt.plot_error_kdes(
            numpy_files,
            numbers_file2,
            save_loc=kde_plots,
            output_filename=kde_file_name,
            xlim=(-0.003, 0.003),
            ylim=(0, 800),
            bins=1000,top=5
            )
        
    else:
        # Look at all patterns
        ignore = [26]      # Ignore some

        numbers_file = [item for item in numbers_file if item not in ignore]

        # Kernel density estimate
        ipt.plot_error_kdes(
            numpy_files,
            numbers_file,
            save_loc=kde_plots,
            output_filename=kde_file_name,
            xlim=(-0.005, 0.005),
            ylim=(0, 1350),
            bins=1000,top=15
        )
    

# Rank patterns (high level analysis)
if flags["High_level_analysis"]:

    # Read optimised excel sheet
    folder = 0
    error_column = 0
    batch = 11
    print(f'------------\n{batch}')

    if folder == 1:
        excel_path = rf"C:\Users\General User\nokop\pattern2\output\excel_docs\excel_{batch}"
    else:
        excel_path = r'output\excel_docs'

    op_metrics, omeas_error, op_param, onans, oindicators = ipt.read_spec_excel(excel_path, doc_num = 88)

    # Get top 5 RMSE, bias and std. dev error
    mask = (omeas_error[:,0] != 0)
    error = np.abs(omeas_error[mask,error_column])
        
    error_analys = True
    get_highest = False 

    if error_analys:
        original_indices = np.where(mask)[0]
        image_numbers = original_indices * 2
        
        # Create list of (image_number, error) tuples
        error_list = list(zip(image_numbers, error))
        
        # Sort by error value
        if get_highest:
            sorted_errors = sorted(error_list, key=lambda x: x[1], reverse=True)
            label = "Highest"
        else:
            sorted_errors = sorted(error_list, key=lambda x: x[1], reverse=False)
            label = "Lowest"
        
        # Remove duplicates based on error values
        seen = set()
        unique_errors = []
        
        for img_num, err_val in sorted_errors:
            if err_val not in seen:
                seen.add(err_val)
                unique_errors.append((img_num, err_val))
                if len(unique_errors) == 10: 
                    break
        
        print(f"\n{label} errors:")
        for i, (img_num, err_val) in enumerate(unique_errors, 1):
            print(f"{i}. Image {img_num} — error = {err_val:.8f}")

    else:
        metric_column = 5
        metric = np.abs(op_metrics[mask, metric_column])
        original_indices = np.where(mask)[0]
        image_numbers = original_indices * 2
        
        # Create list of (image_number, metric) tuples
        metric_list = list(zip(image_numbers, metric))
        
        # Sort by metric value
        if get_highest:
            sorted_metrics = sorted(metric_list, key=lambda x: x[1], reverse=True)
            label = "Highest"
        else:
            sorted_metrics = sorted(metric_list, key=lambda x: x[1], reverse=False)
            label = "Lowest"
        
        # Remove duplicates based on metric values
        seen = set()
        unique_metrics = []
        
        for img_num, met_val in sorted_metrics:
            if met_val not in seen:
                seen.add(met_val)
                unique_metrics.append((img_num, met_val))
                if len(unique_metrics) == 5:
                    break
        
        print(f"\n{label} 5 metric values:")
        for i, (img_num, met_val) in enumerate(unique_metrics, 1):
            print(f"{i}. Image {img_num} — metric = {met_val:.8f}")


    flaggg = {"Get_subsets": False}

    if flaggg["Get_subsets"]:
        # Get images
        # Save to
        image_subset_save_path = r"C:\Users\General User\nokop\pattern2\data\speckle_pattern_img\reference_im\subsets_trad_spec"
        if not os.path.exists(image_subset_save_path):
            os.makedirs(image_subset_save_path)

        file_numbers = [162,166,170,244]    


        for filenum in file_numbers:
                # Look at specfic patterns

                refere = r'C:\Users\General User\nokop\pattern2\data\speckle_pattern_img\reference_im\hold_ref_noised\ref_6'
                stri = ipt.get_image_strings(refere,parity='even')

                filenum = int(filenum/2)
                print(stri[filenum])
                to_image = os.path.join(refere,stri[filenum])
                mimage = ipt.readImage(to_image)

                first_part  = os.path.splitext(stri[filenum])[0]

                # Get image subset through slicing
                subsets = ipt.img_subsets(mimage, 75,150)
                subset = subsets[2,2,:,:]
                plt.imshow( subset, cmap = 'grey')


                to_sav = os.path.join(image_subset_save_path,f'{stri[filenum]}.png')
                plt.imsave(to_sav,subset,cmap='grey')
                plt.show(block=False)
                plt.close()




if flags["In-sample compare"]:
    # This section compares the IQR and RMSE results to the best result from
    # their respective samples
    
    analyse_error = True

    if analyse_error:
        percentile_matrix = np.full((10,13), np.nan, dtype=object)
        values_matrix = np.full((10,13), np.nan, dtype=object)

        percentile_matrix[1:,0] = ["RMSE","MSF","MIG",
                                   "Ef","MIOSD","Shan","PSA","SSI",
                                   "R"]
        values_matrix[1:,0] = percentile_matrix[1:,0]

        objectives = [0,1,2,3,4,5,6,7,8]
    else:
        percentile_matrix = np.full((9,13), np.nan, dtype=object)
        percentile_matrix[1:,0] = ["MSF","MIG","Ef","MIOSD","Shan","PSA","SSI","R_peak"]
        objectives = [1,2,3,4,5,6,7,8]
        error_cols = [0,4,1]

    percentile_matrix[0,1:] = ["RS","RCh","RP","RPc","RPb","RPs","S","Ch","P","Pc","Pbl","Ps"]
    values_matrix[0,1:] = ["RS","RCh","RP","RPc","RPb","RPs","S","Ch","P","Pc","Pbl","Ps"]

    # For every objective
    for inde,obj in enumerate(objectives):

        # For every pattern class
        for batch in range(12):
    
            # The even-numbered image prefix in the optimised batch for that objective
            image_prefix = obj * 24 + batch*2

            # Use the prefix to get the index in the excel sheet
            excel_vector_index = int(image_prefix / 2)
            # print(f"Excel vector: {excel_vector_index}")

            # Load optimised data using the determined index
            optimised_patterns_data = r'output\excel_docs'
            op_metrics, omeas_error, op_param, onans, oindicators = ipt.read_spec_excel(
                optimised_patterns_data, doc_num = 177,print_doc=False )

            # Load batch data
            excel_folder_address = rf'output\excel_docs\excel_{batch}'
            p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(
                excel_folder_address, doc_num=None,print_doc=False)

            # Filter the batch dataset
            mask = ((meas_error[:,0] != 0) 
                    & (~np.isnan(meas_error[:,0])) 
                    & (np.abs(np.nanmean(meas_error[:,0]) - meas_error[:,0]) <= 3*np.std(meas_error[:,0]))
                    )

            if analyse_error:
                # Compare optimised error to dataset. Get complimentary percentile score.
                error_column = 0
                opt_error = omeas_error[excel_vector_index,error_column]
                if opt_error == 0:
                    opt_error = 0.2
                # Get percentile
                percentile_in_sample = 100 - stats.percentileofscore(meas_error[mask,error_column], opt_error, kind='strict')
                # print(opt_error)
            else:
                opt_metric = op_metrics[excel_vector_index,inde]
                # Get percentile
                percentile_in_sample = stats.percentileofscore(p_metrics[mask,inde], opt_metric, kind='strict')


            
            # print(f"Percentile: {percentile_in_sample}")
            percentile_matrix[inde+1,batch+1] = float(np.round(percentile_in_sample,1))
            values_matrix[inde+1,batch+1] = opt_error


    df = pd.DataFrame(percentile_matrix[1:, :], columns=percentile_matrix[0, :])
    print(df.to_string(index=False))

    print(" ")
    df2 = pd.DataFrame(values_matrix[1:, :], columns=values_matrix[0, :])
    print(df2.to_string(index=False))


    
if flags["IQRvsRMSE"]:

# Paths
    save_dir = r"output\plots\November_3"
    os.makedirs(save_dir, exist_ok=True)

    # load data
    wait_time = 0.025

    patt_class = [0,1,2,3,4,5,6,7,8,9,10,11]
    # patt_class = [1,2,3,4,5,7,8,9,10,11]
    
    # Create a list of excel path strings (for each pattern class)
    excel_paths = [
        rf"output\excel_docs\excel_{i}" for i in patt_class
        ]

    # Initialise containers
    p_metrics_all = []
    meas_errors_all = []
    p_params_all = []
    nans_all = []
    indicators_all = []

    # Loop through paths and load data
    for path in excel_paths:
        p_metrics, meas_error, p_param, nans, indicators = ipt.read_spec_excel(path, doc_num=None)

        p_metrics_all.append(p_metrics)
        meas_errors_all.append(meas_error)
        p_params_all.append(p_param)
        nans_all.append(nans)
        indicators_all.append(indicators)
    
    # Convert lists to single big arrays
    p_metrics_all = np.concatenate(p_metrics_all, axis=0)
    meas_errors_all = np.concatenate(meas_errors_all, axis=0)
    p_params_all = np.concatenate(p_params_all, axis=0)
    nans_all = np.concatenate(nans_all, axis=0)
    indicators_all = np.concatenate(indicators_all, axis=0)

    msf = 203
    mig = 32
    ef = 0.48
    miosd = 17.29
    shn = 7.94
    psa = 12372769266.02177
    sssig = 35009.65
    rpeak = 3.64

    msfperc = stats.percentileofscore(p_metrics_all[:,0], msf, kind='strict')
    migperc = stats.percentileofscore(p_metrics_all[:,1], mig, kind='strict')
    efperc = stats.percentileofscore(p_metrics_all[:,2], ef, kind='strict')
    miosdperc = stats.percentileofscore(p_metrics_all[:,3], miosd, kind='strict')
    shnperc = stats.percentileofscore(p_metrics_all[:,4], shn, kind='strict')
    psaperc = stats.percentileofscore(p_metrics_all[:,5], psa, kind='strict')
    sssigperc = stats.percentileofscore(p_metrics_all[:,6], sssig, kind='strict')
    rpeakperc = stats.percentileofscore(p_metrics_all[:,7], rpeak, kind='strict')

    print(f'MSF = {msfperc:.2f}\nMIG = {migperc:.2f}\nEf = {efperc:.2f}\nMIOSD = {miosdperc:.2f}\nShannon = {shnperc:.2f}\nPSA = {psaperc:.2f}\nSSSIG = {sssigperc:.2f}\nR_peak = {rpeakperc:.2f}')


if flags['Miscellenious']:
    
    excel_path = rf'output\excel_docs'
    p_metrics, meas_error, _, _, _ = ipt.read_spec_excel(
        excel_path, doc_num=228
    )
    p_metrics1, meas_error1, _, _, _ = ipt.read_spec_excel(
        excel_path, doc_num=227
    )

    errors = meas_error[:, 0]
    mask = ((~np.isnan(errors)) 
            & (errors != 0)
            & (meas_error1[:, 0] != 0 )
            & (np.abs(errors - np.nanmean(errors)) <= 3 * np.nanstd(errors))
            ) 
    # & (np.abs(np.mean(meas_error) - 3 * np.std(meas_error[:,0])) < 0)
    plt.scatter(p_metrics[mask,0],meas_error[mask,0],alpha=0.5, label = "Analytical")
    plt.scatter(p_metrics1[mask,0],meas_error1[mask,0],  label = "Discrete")
    plt.legend()
    plt.grid(True)
    plt.show()

if flags['MetricPercentiles']:

    file_number = [1084,74,674,58,1094,246,1092,8,298,306,384,24]

    batch_R2 = np.full((8,12), np.nan)
    global_batch_R2 = np.full((8,12), np.nan)
    # -------------------------------------------------------
    # Load all metrics once for overall percentiles
    # -------------------------------------------------------
    Mmetrics = []
    Mmeas_error = []

    for j in range(12):
        excel_path = rf"C:\Users\General User\nokop\pattern1\output\excel_docs\excel_{j}"
        metrics, meas_error, _, _, _ = ipt.read_spec_excel(excel_path, doc_num=None)

        # Get all metrics, append lists
        Mmetrics.append(metrics)
        Mmeas_error.append(meas_error)

    # Stack lists to make one indexible object
    Mmetrics = np.vstack(Mmetrics)
    Mmeas_error = np.vstack(Mmeas_error)

    # -------------------------------------------------------
    # Main loop
    # -------------------------------------------------------
    for b in range(12):

        num = file_number[b] // 2
        excel_path = rf"C:\Users\General User\nokop\pattern1\output\excel_docs\excel_{b}"

        op_metrics, omeas_error, op_param, onans, oindicators = ipt.read_spec_excel(excel_path, doc_num=None)

        specific_entry = op_metrics[num, :]
        specific_entry_error = omeas_error[num, 0]

        print(f"Data loaded for batch {b}. Selected index = {num}. Error = {specific_entry_error}")

        # ---------------------------------------------------
        # Batch-specific percentiles
        # ---------------------------------------------------
        for row in range(8):
            batch_R2[row, b] = stats.percentileofscore(op_metrics[:, row], op_metrics[num, row])
            global_batch_R2[row, b] = stats.percentileofscore(Mmetrics[:,row], op_metrics[num,row])

        print(f"\nMetric R2 scores for batch {b}\n--------------------")
        print(f"msf = {batch_R2[0,b]}")
        print(f"mig = {batch_R2[1,b]}")
        print(f"ef = {batch_R2[2,b]}")
        print(f"miosd = {batch_R2[3,b]}")
        print(f"shannon = {batch_R2[4,b]}")
        print(f"psa = {batch_R2[5,b]}")
        print(f"rpeak = {batch_R2[6,b]}")
        print(f"sssig = {batch_R2[7,b]}")
        print("-------------------------------\n")

        # ---------------------------------------------------
        # Overall percentiles across all batches
        # ---------------------------------------------------
        msf_perc = stats.percentileofscore(Mmetrics[:,0], op_metrics[num,0])
        mig_perc = stats.percentileofscore(Mmetrics[:,1], op_metrics[num,1])
        ef_perc = stats.percentileofscore(Mmetrics[:,2], op_metrics[num,2])
        moisd_perc = stats.percentileofscore(Mmetrics[:,3], op_metrics[num,3])
        shannon_perc = stats.percentileofscore(Mmetrics[:,4], op_metrics[num,4])
        psa_perc = stats.percentileofscore(Mmetrics[:,5], op_metrics[num,5])
        rpeak_perc = stats.percentileofscore(Mmetrics[:,6], op_metrics[num,6])
        sssig_perc = stats.percentileofscore(Mmetrics[:,7], op_metrics[num,7])

        print("Overall percentiles\n---------------------------")
        print(f"msf = {msf_perc}")
        print(f"mig = {mig_perc}")
        print(f"ef = {ef_perc}")
        print(f"miosd = {moisd_perc}")
        print(f"shannon = {shannon_perc}")
        print(f"psa = {psa_perc}")
        print(f"rpeak = {rpeak_perc}")
        print(f"sssig = {sssig_perc}")
        print("-------------------------------\n")

    labels = [
        "msf",
        "mig",
        "ef",
        "miosd",
        "shannon",
        "psa",
        "rpeak",
        "sssig"
    ]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    pd.set_option("display.max_colwidth", None)

    df = pd.DataFrame(batch_R2, columns=[f"batch_{i}" for i in range(12)])
    df.insert(0, "Metric", labels)

    # Global evaluation
    dfg = pd.DataFrame(global_batch_R2, columns=[f"batch_{i}" for i in range(12)])
    dfg.insert(0, "Metric", labels)

    print("\n----------- Final batch_R2 ----------------\n")
    print(df)


    print("\n----------- Final global_batch_R2 ----------------\n")
    print(dfg)

    print(f"\n\nRow averages")
    average_intra = np.mean(batch_R2,axis=1)
    print(average_intra)
    print(f"\nGlobal")
    average_inter = np.mean(global_batch_R2,axis=1)
    print(average_inter)
