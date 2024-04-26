

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 2023

@author: emily

This script intends to plot results for several experiments conducted on different alpha values.
"""


## LIBRARIES
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import copy
from ml_spectroscopy.config import path_init
from ml_spectroscopy.DataPreprocessing_utils import image_reconstruct, image_deconstruct

## SET SEED FOR REPRODUCIBILITY
random.seed(100)


## Settings
data_name = 'GQlupb'
planet = 'GQlupB'
alpha_vals= [0,2,5,8,11,16,21,29,41,67,5000]
#alpha=2
bal=50
v=6
frame='simple'
len_folds = 8


plotname = 'test'
methods_ls = ['SNR', 'SNR_auto', 'ENET', 'RID', 'PCT', 'DNN', 'CNN1', 'CNN2']
color_ls = {'SNR': 'red', 'SNR_auto': 'brown', 'ENET': 'forestgreen', 'RID': 'lime', 'PCT': 'lightblue', 'DNN': 'blue',
            'CNN1': 'navy', 'CNN2': 'purple'}
title_ls = {'SNR': 'SNR', 'SNR_auto': 'SNR_auto', 'ENET': 'Elasticnet', 'RID': 'Ridge', 'PCT': 'Perceptron', 'DNN': 'DNN',
            'CNN1': 'CNN', 'CNN2': 'CNN'}





## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
#visual_path = subdir + "80_visualisation/"
#csv_res_path = subdir + "80_visualisation/"

# Directory to fetch results
dir_path = results_path + "export_CV/from_GPU_byfold/Res_00_090223_alphas_simple_T1200SG41/"  # Depending on the way we pick the files, can't we directly point to the right directory from the start?

#Directory to store results

#Directory to store results
visualisation_path = subdir + "80_visualisation/Results_per_alpha/Vis_00_090223_alphas_simple_T1200SG41/"
csv_path = subdir + "80_visualisation/Results_per_alpha/Vis_00_090223_alphas_simple_T1200SG41/"



planetlist = ['GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7']

# Initialize the values to be collected across methods
lr_precision_subdict = {key: None for key in alpha_vals}
lr_fpr_subdict = {key: None for key in alpha_vals}
lr_tpr_subdict = {key: None for key in alpha_vals}
lr_recall_subdict = {key: None for key in alpha_vals}


# Initialise the values to be collected across alpha values
lr_precision_dict = {key: None for key in alpha_vals}
lr_fpr_dict = {key: None for key in alpha_vals}
lr_tpr_dict = {key: None for key in alpha_vals}
lr_recall_dict = {key: None for key in alpha_vals}
a = 0



#initiate the loop
for alpha in alpha_vals:

    # What was the base template used for the experiemnts?
    template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}

    # And the base dataset corresponding to the base template?
    data1=pd.read_pickle(data_path+'data_4ml/v6_ccf_4ml_trim_robustness_simple/H2O_'+data_name+'_scale'+str(alpha)+'_bal'+str(bal)+'_temp1200.0_sg4.1_ccf_4ml_trim_norepetition_v6_simple.pkl')

    # NOTE: This is to build the data for the true values of Y train, test and validation. The X train and X valid are not needed. Maybe thos part can go outside the loop.
    i=5
    j=6
    data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
    data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
    data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

    X_train = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
    Y_train = data_train['H2O']

    X_valid = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
    Y_valid = data_valid['H2O']

    X_test = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
    Y_test = data_test['H2O']

    ls_data = ['results_GQlupb_data_0_alpha_'+str(alpha)+'_CV_testfold'+str(i)+'.pkl' for i in range(len_folds)]
    result_names=[ls_data[n][:-4] for n in range(len_folds)]

    keys = result_names
    ls_results = {key: None for key in keys}

    for i in range(0, len_folds):
        with open(dir_path + str(ls_data[i]), "rb") as f:
            ls_results[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1


    len_folds2 = len_folds
    ls_data2 = ['GA_results_GQlupb_data_0_alpha_'+str(alpha)+'_CV_testfold'+str(i)+'.pkl' for i in range(len_folds)]
    result_names2 = [ls_data2[n][:-4] for n in range(len_folds2)]

    keys2 = result_names2
    GA_results = {key: None for key in keys2}

    for i in range(0, len_folds):
        with open(dir_path + str(ls_data2[i]), "rb") as f:
            GA_results[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1

    for m in methods_ls:
        #For each fold:
        # get the results
        # get the indices of the results
        # Fill the gaps of the image with nans?
        # reconstruct the image with the results
        # Image should contain:
        # Map of accuracy (correctly classified is green , badly classified is red)
        # Map of True vs false positives?
        # Map of true vs false negatives ?
        # Could also have: Orange = negatives, true are dark, false are light; blue = true and false positives; true are dark, wrong are light


        for j in range(0, len_folds): # + 1):

            if j == 0:
                i = (len_folds - 1)
            else:
                i = j - 1

            data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
            data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
            data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

            X_train = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train = data_train['H2O']

            X_valid = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid = data_valid['H2O']

            X_test = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_test = data_test['H2O']

            ### quick test for the data reconstruction
            # get noise
            temp_noise = pd.read_pickle(data_path + 'csv_inputs/True_Spectrum_Noise/' + str(planetlist[j]) + '_Spectrum_noise_trim.pkl')
            ind2 = temp_noise.index
            ind1 = np.repeat(planetlist[j], len(temp_noise.index))
            arrays = [ind1, ind2]
            temp_noise.index = arrays
            noise_temp = temp_noise.loc[planetlist[j]]
            noise_temp_wl_shape = noise_temp.shape[1]
            # Get the data where the planet is indicate
            dtpath = data_path + "csv_inputs/True_Spectrum_Data"
            path0 = os.path.join(dtpath, str(planetlist[j]) + '_spectrum_dt.csv')  # Untrimmed data. therefore, take the WL range from the trimmed data-
            df_temp = pd.read_csv(path0)
            # rebuild the mask
            imsize = int(np.sqrt(len(df_temp['Planet'])))
            mask = np.reshape(np.array(df_temp['Planet']), (imsize, imsize))
            ##plt.imshow(mask)
            ##plt.show()
            # Create a cube for the mask: create a block first and then a cube
            mask_block = np.reshape(np.repeat(mask, noise_temp_wl_shape), (imsize * imsize, noise_temp_wl_shape))
            mask_cube = np.reshape(np.repeat(mask, noise_temp_wl_shape), (imsize, imsize, noise_temp_wl_shape))
            mask_cube_inv = np.empty((noise_temp_wl_shape, imsize, imsize))
            # revert the block to stack it first by wavelength.
            for w in range(noise_temp_wl_shape):
                mask_cube_inv[w, :, :] = mask_cube[:, :, w]
            mask_cube_inv_copy = copy.deepcopy(mask_cube_inv)
            mask_cube_inv[np.where(mask_cube_inv == 1)] = np.nan
            # Deconstruct a full image (before the trimming and collapsing of the data sets into simple noise spectra; we only use the spatial information to recover the locations of NA.
            PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct(mask_cube_inv)
            test = image_reconstruct(np.array(noise_temp), Planet_vec_shape[0], Planet_vec_shape[1],Planet_position_nan)
            plt.imshow(test[1, :, :])
            plt.title(str(planetlist[j]) + ', masking region')
            plt.show()
            ### End test


            if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                prob_Y = ls_results[j]['results'][m]['Y_pred_prob']
            elif m in ['SNR', 'SNR_auto']:
                prob_Y = ls_results[j]['results'][m]['SNR']
            else:
                prob_Y = ls_results[j]['results'][m]['Y_pred_prob'][:, 1]

            Y_pred = ls_results[j]['results'][m]['Y_pred']
            Y_acc = (np.array(Y_test == Y_pred)) * 1
            Y_CM = Y_test + 2*np.array(Y_pred)

            # Get the data
            dtpath0 = data_path + "csv_inputs/True_Spectrum_Data"
            noise_temp_wl_shape = 2 # here we do not need the wavelength dimension

            # Get the data where the planet is indicated
            path0 = os.path.join(dtpath0,str(planetlist[j]) + '_spectrum_dt.csv')  # Untrimmed data. therefore, take the WL range from the trimmed data-
            original_df = pd.read_csv(path0)
            # rebuild the mask
            imsize = int(np.sqrt(len(original_df['Planet'])))
            planet_mask = np.reshape(np.array(original_df['Planet']), (imsize, imsize))

            # Create a cube for the mask, create a block and then a cube
            mask_block = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape),(imsize * imsize, noise_temp_wl_shape))
            mask_cube = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape), (imsize, imsize, noise_temp_wl_shape))
            mask_cube_inv = np.empty((noise_temp_wl_shape, imsize, imsize))
            # revert the block to stack it first by wavelength.
            for w in range(noise_temp_wl_shape):
                mask_cube_inv[w, :, :] = mask_cube[:, :, w]
            mask_cube_inv_copy = copy.deepcopy(mask_cube_inv)
            mask_cube_inv[np.where(mask_cube_inv == 1)] = np.nan

            # Deconstruct a full image (here we only use two frames as the wavelength dimension is not of interest - but the function was built for more than one dimensio)
            PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct(mask_cube_inv[0:2,:,:])

            reconstruct_prediction = np.tile(Y_pred, 2).reshape(2, len(Y_pred)).T
            reconstruct_scores = np.tile(prob_Y, 2).reshape(2, len(prob_Y)).T
            reconstruct_accuracy = np.tile(Y_acc, 2).reshape(2, len(Y_acc)).T
            reconstruct_CM = np.tile(Y_CM, 2).reshape(2, len(Y_CM)).T

            img_prediction = image_reconstruct(reconstruct_prediction, Planet_vec_shape[0], Planet_vec_shape[1], Planet_position_nan)
            img_scores = image_reconstruct(reconstruct_scores, Planet_vec_shape[0], Planet_vec_shape[1], Planet_position_nan)
            img_accuracy = image_reconstruct(reconstruct_accuracy, Planet_vec_shape[0], Planet_vec_shape[1], Planet_position_nan)
            img_CM = image_reconstruct(reconstruct_CM, Planet_vec_shape[0], Planet_vec_shape[1], Planet_position_nan)

            # prediction
            plt.imshow(img_prediction[1, :, :])
            plt.title('Cube: ' +str(planetlist[j]) + ', method: ' +str(title_ls[m]) + '\n $\\alpha$ = ' +str(alpha)+ ', Prediction areas', cmap='plasma')
            plt.show()
            plt.savefig(visualisation_path + 'results_images/img_'+planetlist[j] + '_method_' +str(m) + '_alpha_' +str(alpha)+ '_Prediction_areas.pdf', bbox_inches='tight')
            plt.close()
            #plt.savefig(data_path + 'csv_inputs/CCF_realistic_fakeplanets/noise_images/' + str(
                        #planetlist[j]) + 'images_planet_masking_region.pdf')
            # score
            plt.imshow(img_scores[1, :, :])
            plt.title('Cube: ' +str(planetlist[j]) + ', method: ' +str(title_ls[m]) + '\n $\\alpha$ = ' +str(alpha)+ ', Scores')
            plt.show()
            plt.savefig(visualisation_path + 'results_images/img_'+planetlist[j] + '_method_' +str(m) + '_alpha_' +str(alpha)+ '_Scores.pdf', bbox_inches='tight')
            plt.close()                #plt.savefig(data_path + 'csv_inputs/CCF_realistic_fakeplanets/noise_images/' + str(
                        #planetlist[j]) + 'images_planet_masking_region.pdf')

                    # accuracy
            plt.imshow(img_accuracy[1, :, :], cmap='viridis') #RdYlGn
            plt.title('Cube: ' +str(planetlist[j]) + ', method: ' +str(title_ls[m]) + '\n $\\alpha$ = ' +str(alpha)+ ', Classification Accuracy Areas')
            plt.show()
            plt.savefig(visualisation_path + 'results_images/img_'+planetlist[j] + '_method_' +str(m) + '_alpha_' +str(alpha)+ '_Classification_Accuracy.pdf', bbox_inches='tight')
            plt.close()
                    #plt.savefig(data_path + 'csv_inputs/CCF_realistic_fakeplanets/noise_images/' + str(
                        #planetlist[j]) + 'images_planet_masking_region.pdf')

                    # Confusion matrix
            plt.imshow(img_CM[1, :, :], cmap='seismic_r')
            plt.title('Cube: ' +str(planetlist[j]) + ', method: ' +str(title_ls[m]) + '\n $\\alpha$ = ' +str(alpha)+ ', Confusion Matrix')
            plt.show()
            plt.savefig(visualisation_path + 'results_images/img_'+planetlist[j] + '_method_' +str(m) + '_alpha_' +str(alpha)+ '_Confusion_Matrix.pdf', bbox_inches='tight')
            plt.close()









#####
#            ax2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
#            ax1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
#            fig_acc, axes_acc = plt.subplots(nrows=3, ncols=3)
#            fig_acc.suptitle('Accuracy, $\\alpha=' + str(alpha) + '$', fontsize=14)
#
#            fig_CM, axes_CM = plt.subplots(nrows=3, ncols=3)
#            fig_CM.suptitle('Confusion Matrix, $\\alpha=' + str(alpha) + '$', fontsize=14)
#            axes_acc[ax1[j], ax2[j]].imshow(img_accuracy[1, :, :], cmap='RdYlGn')
#            axes_acc[ax1[j], ax2[j]].set_ylabel('[px]')
#            axes_acc[ax1[j], ax2[j]].set_xlabel('[px]')
#            axes_acc[ax1[j], ax2[j]].label_outer()
#           axes_acc[ax1[j], ax2[j]].set_title('Cube: ' +str(planetlist[j]) + ', method ' +str(m))

#        fig_acc.tight_layout()
#        #['TP', 'FP', 'FN', 'TN'], color = ['darkblue', 'royalblue', 'indianred', 'firebrick']
#        mylegends = [markers([0], [0], color='darkblue', lw=1),
#                     markers([0], [0], color='royalblue', lw=1),
#                     markers([0], [0], color='indianred', lw=1),
#                     markers([0], [0], color='firebrick', lw=1)]
#        fig_acc.legend(mylegends, ['TP', 'FP', 'FN', 'TN'],loc='lower left', bbox_to_anchor=(0.05, -0.4), fontsize=7)
        #fig_acc.savefig(
          #   visualisation_path + 'Global_Aggregated_PR_alpha_bal' + str(bal) + '_version' + str(v) + 'frame' + str(
           #     frame) + '.pdf')

