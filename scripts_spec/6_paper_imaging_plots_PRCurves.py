

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 2023

@author: emily

This script intends to plot results for several experiments conducted on different alpha values.
"""


## LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
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
alpha_vals= [6] #17 # [2,5,8,11,22,32,43,55]
#alpha=2
bal=50
v=6
frame='simple'
folds = [0,1,5] #12


plotname = 'test'
methods_ls = ['SNR', 'SNR_auto', 'ENET', 'RID', 'PCT', 'DNN', 'CNN1', 'CNN2']
color_ls = {'SNR': 'darkorange', 'PCT': 'mediumblue', 'CNN1': 'indigo', 'CNN2': 'darkorchid'}

title_ls = {'SNR': 'S/N', 'SNR_auto': 'S/N_auto', 'ENET': 'Elasticnet', 'RID': 'Ridge', 'PCT': 'Perceptron', 'DNN': 'DNN',
            'CNN1': 'CNN', 'CNN2': 'CNN'}
title_ls2 = {'SNR': 'S/N', 'SNR_auto': 'S/N_auto', 'ENET': 'Elasticnet', 'RID': 'Ridge', 'PCT': 'Perceptron', 'DNN': 'DNN',
            'CNN1': 'CNN1', 'CNN2': 'CNN2'}

title_ls3 = {'SNR': 'S/N', 'SNR_auto': 'S/N_auto', 'ENET': 'Elasticnet', 'RID': 'Ridge', 'PCT': 'PCT', 'DNN': 'DNN',
            'CNN1': 'CNN', 'CNN2': 'CNN'}

planetlist = ['GQlupb0','GQlupb1','GQlupb2','GQlupb3','GQlupb4','GQlupb5','GQlupb6','GQlupb7','PZTel_10','PZTel_11','PZTel_12','PZTel_13','PZTel_20','PZTel_21','PZTel_22','PZTel_23','PZTel_24','PZTel_25','PZTel_26']




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
dir_path = results_path + "export_CV/from_GPU_byfold/"  # Depending on the way we pick the files, can't we directly point to the right directory from the start?
#Directory to store results
visualisation_path = subdir + "80_visualisation/Realistic_fakeplanets/Vis_07_150523_alphas_advanced_realisticfake_gaussian_aperture/"
visualisation_path2 = subdir + "80_visualisation/Realistic_fakeplanets/Vis_07_150523_alphas_advanced_realisticfake_gaussian_aperture/"
csv_path = subdir + "80_visualisation/Realistic_fakeplanets/Vis_07_150523_alphas_advanced_realisticfake_gaussian_aperture/"




###################### Try with TPR cnstraint
#initiate the loop
for alpha in alpha_vals:

    # What was the base template used for the experiemnts?
    template_characteristics = {'Temp': 2900, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}


    tp_path = '/home/ipa/quanz/user_accounts/egarvin/Thesis/70_results/export_CV/from_GPU_byfold/'
    keys = folds
    ls_results_realistic_fake = {key: None for key in keys}
    for i in folds:
        with open(tp_path + 'results_realisticfakeplanets_data_0_alpha_'+str(alpha)+'_CV_testfold'+str(i)+'.pkl', "rb") as f:
            ls_results_realistic_fake[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1


        #CMdf = pd.read_csv(visualisation_path + "CM_" + str(m) + "_alpha" + str(alpha) + ".csv")

    data1 = pd.read_pickle(
        data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_H2O_crosscorr_data_alpha_' + str(
            alpha) + '_temp2800.0_sg4.1.pkl')

        # For each fold:
        # get the results
        # get the indices of the results
        # Fill the gaps of the image with nans?
        # reconstruct the image with the results
        # Image should contain:
        # Map of accuracy (correctly classified is green , badly classified is red)
        # Map of True vs false positives?
        # Map of true vs false negatives ?
        # Could also have: Orange = negatives, true are dark, false are light; blue = true and false positives; true are dark, wrong are light


    aggprecision = { "SNR":[],'PCT':[], 'CNN1':[], 'CNN2': []}
    aggrecall = { "SNR": [],'PCT': [], 'CNN1': [], 'CNN2': []}
    aggfpr = { "SNR": [],'PCT': [], 'CNN1': [], 'CNN2': []}
    aggtpr = { "SNR": [],'PCT': [], 'CNN1': [], 'CNN2': []}
    aggy_test_array = { "SNR": [],'PCT': [], 'CNN1': [], 'CNN2': []}
    aggscores_array = { "SNR": [],'PCT': [], 'CNN1': [], 'CNN2': []}

    for j in folds: # + 1):

        if j == 0:
            i = (len(planetlist) - 1)
        else:
            i = j - 1

        data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

        PR = {}
        ROC = {}
        for m in methods_ls:

            if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob']
                threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
                #Y_pred = np.array(prob_Y > 0.5) * 1

            elif m in ['SNR', 'SNR_auto']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['SNR']
                #Y_pred = np.array(prob_Y > 3) * 1
                threshold = [1, 2, 3, 4, 5]

            else:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob'][:, 1]
                #Y_pred = np.array(prob_Y > 0.5) * 1
                threshold = [0.1, 0.3, 0.5, 0.7, 0.9]

            Y_pred = ls_results_realistic_fake[j]['results'][m]['Y_pred']


            # dt_true = pd.read_pickle(data_path + "csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/injection_labels_set.pkl")
            # Y_true = dt_true.iloc[0:3041]
            # Y_pred[np.where(Y_true == 1)[0].tolist()]
            #
            # Positives_predictions = [Y_pred[index] for index in np.where(Y_true == 1)[0].tolist()]
            # Positives_scores = [prob_Y[index] for index in np.where(Y_true == 1)[0].tolist()]


            # Get the data
            dtpath0 = data_path + "csv_inputs/True_Spectrum_Data"
            noise_temp_wl_shape = 2  # here we do not need the wavelength dimension

            # Get the data where the planet is indicated
            path0 = os.path.join(dtpath0, str(planetlist[
                                                  j]) + '_spectrum_dt.csv')  # Untrimmed data. therefore, take the WL range from the trimmed data-
            original_df = pd.read_csv(path0)
            # rebuild the mask
            imsize = int(np.sqrt(len(original_df['Planet'])))
            planet_mask = np.reshape(np.array(original_df['Planet']), (imsize, imsize))

            # Create a cube for the mask, create a block and then a cube
            mask_block = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape),
                                    (imsize * imsize, noise_temp_wl_shape))
            mask_cube = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape),
                                   (imsize, imsize, noise_temp_wl_shape))
            mask_cube_inv = np.empty((noise_temp_wl_shape, imsize, imsize))
            # revert the block to stack it first by wavelength.
            for w in range(noise_temp_wl_shape):
                mask_cube_inv[w, :, :] = mask_cube[:, :, w]
            mask_cube_inv_copy = copy.deepcopy(mask_cube_inv)
            mask_cube_inv[np.where(mask_cube_inv == 1)] = np.nan

            # Deconstruct a full image (here we only use two frames as the wavelength dimension is not of interest - but the function was built for more than one dimensio)
            PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct(mask_cube_inv[0:2, :, :])

            reconstruct_prediction = np.tile(Y_pred, 2).reshape(2, len(Y_pred)).T
            reconstruct_scores = np.tile(prob_Y, 2).reshape(2, len(prob_Y)).T

            reconstruct_ccf = np.tile(data_test[0], 2).reshape(2, len(data_test[0])).T



            test_y = ls_results_realistic_fake[j]['y_test']
            y_test_array = np.array([arr[1] for arr in test_y])

            scores_array = np.array([arr[0] for arr in reconstruct_scores])



            img_prediction = image_reconstruct(reconstruct_prediction, Planet_vec_shape[0], Planet_vec_shape[1],
                                               Planet_position_nan)
            img_scores = image_reconstruct(reconstruct_scores, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            img_ccf = image_reconstruct(reconstruct_ccf, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            img_y = image_reconstruct(test_y, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            precision, recall, _ = precision_recall_curve(y_test_array,scores_array)
            PR[m] = (precision, recall)


            fpr, tpr, _ = roc_curve(y_test_array,scores_array)
            ROC[m] = (fpr, tpr, y_test_array, scores_array)

            methods_ls0 = ['SNR', 'PCT', 'CNN1', 'CNN2']
            if m in methods_ls0:
                aggprecision[m].extend(precision.tolist())
                aggrecall[m].extend(recall.tolist())
                aggfpr[m].extend(fpr.tolist())
                aggtpr[m].extend(tpr.tolist())
                aggy_test_array[m].extend(y_test_array.tolist())
                aggscores_array[m].extend(scores_array.tolist())





            ## prediction
            #plt.imshow(img_prediction[1, :, :])
            #plt.title(str(title_ls[m]), fontsize=18)
            #plt.xlabel('Pixels', fontsize=17)
            #plt.ylabel('Pixels', fontsize=17)
            #plt.savefig(visualisation_path + 'img_realisticfake_' + planetlist[j] + '_method_' + str(m) + '_alpha_' + str(
            #        alpha) + '_Base_Prediction_areas.pdf', bbox_inches='tight')
            #plt.show()
            #plt.close()

            if m in ['SNR']:
                # CCF
                plt.imshow(img_ccf[1, :, :], cmap='viridis')
                plt.title('Molecular Map', fontsize=18)
                clb=plt.colorbar()
                clb.set_label('Normalised CCF Values', fontsize=14)
                plt.xlabel('Pixels', fontsize=17)
                plt.ylabel('Pixels', fontsize=17)
                plt.savefig(visualisation_path + 'img_realisticfake_' + planetlist[j] + '_method_' + str(m) + '_alpha_' + str(
                        alpha) + '_CCF.pdf', bbox_inches='tight')
                #plt.show()
                plt.close()

            # score
            plt.imshow(img_scores[1, :, :], cmap='viridis')
            plt.title(str(title_ls[m]), fontsize=18)
            clb=plt.colorbar()
            if m in ['SNR', 'SNR_auto']:
                clb.set_label('Scores: S/N Values', fontsize=14)
            else:
                clb.set_label('Scores: Probabilities', fontsize=14)

                #Y_pred = np.array(prob_Y > 0.5) * 1
            plt.xlabel('Pixels', fontsize=17)
            plt.ylabel('Pixels', fontsize=17)
            plt.savefig(visualisation_path + 'img_realisticfake_' + planetlist[j] + '_method_' + str(m) + '_alpha_' + str(
                    alpha) + '_Scores.pdf', bbox_inches='tight')
            #plt.show()
            plt.close()

            #
            # # classification
            # tick_positions = [0, 1, 2, 3] #tick_positions
            # tick_labels = ["TN", "FN", "FP", "TP"]
            # for t in threshold:
            #     impred = img_scores[1, :, :] > t
            #     plt.imshow((1*img_y[1, :, :]+2*impred), cmap='viridis',  vmin=0, vmax=3)
            #     plt.title('Classification ' + str(title_ls[m] + ', T='+str(t)), fontsize=17)
            #     clb=plt.colorbar(ticks=tick_positions)
            #     clb.set_ticklabels(tick_labels)
            #     if m in ['SNR', 'SNR_auto']:
            #         clb.set_label('Confusion Matrix: S/N Values', fontsize=14)
            #     else:
            #         clb.set_label('Confusion Matrix: Probabilities', fontsize=14)
            #
            #         #Y_pred = np.array(prob_Y > 0.5) * 1
            #     plt.xlabel('Pixels', fontsize=17)
            #     plt.ylabel('Pixels', fontsize=17)
            #     plt.savefig(visualisation_path + 'img_realisticfake_' + planetlist[j] + '_method_' + str(m) + '_alpha_' + str(
            #             alpha) + '_class_'+str(t)+'.pdf', bbox_inches='tight')
            #     plt.show()
            #     plt.close()
            #
            #     # classification
            #     tick_positions = [0, 1, 2, 3]  # tick_positions
            #     tick_labels = ["TN", "FN", "FP", "TP"]



        methods_ls0 = ['SNR',  'PCT', 'CNN1', 'CNN2']
        color_ls0 = {'SNR': 'darkorange', 'PCT': 'mediumblue', 'CNN1': 'indigo', 'CNN2': 'darkorchid'}

        plt.figure()
        fig, ax = plt.subplots()
        baseline = sum(y_test_array)/len(y_test_array)
        ax.axhline(y=baseline, color='gray', linestyle='--', label = 'no skill')

        ax.set_ylabel('Precision', fontsize=17, color='black')
        ax.set_xlabel('Recall', fontsize=17, color='black')
        for m in methods_ls0:
            ax.plot(PR[m][1],PR[m][0], lw=1, color=color_ls0[m], label= str(title_ls2[m]) + ' AUC: ' + str(round(auc(PR[m][1],PR[m][0]), 3)))
        plt.tick_params(axis='both', which='major', labelsize=14, colors = "black")
        plt.title('P-R Curve', fontsize=17, color='black')
        plt.legend(labelcolor="black", framealpha=0.1, facecolor='gray')
        plt.savefig(visualisation_path + 'PRcurve_realisticfake_' + planetlist[j] + '_alpha_' + str(
                        alpha) +'.pdf', bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()


        plt.figure()
        fig, ax = plt.subplots()
        baseline = sum(y_test_array)/len(y_test_array)
        ax.plot([0,1], [0,1], color='gray', linestyle='--', linewidth=1, label = 'no skill')

        ax.set_ylabel('TPR', fontsize=17, color='black')
        ax.set_xlabel('FPR', fontsize=17, color='black')
        for m in methods_ls0:
            ax.plot(ROC[m][0],ROC[m][1], lw=1, color=color_ls0[m], label= str(title_ls2[m]) + ' AUC: ' + str(round(roc_auc_score(ROC[m][2],ROC[m][3]), 3)))
        plt.tick_params(axis='both', which='major', labelsize=14, colors = "black")
        plt.title('ROC Curve', fontsize=17, color='black')
        plt.legend(labelcolor="black", framealpha=0.1, facecolor='gray')
        plt.savefig(visualisation_path + 'ROCcurve_realisticfake_' + planetlist[j] + '_alpha_' + str(
                        alpha) +'.pdf', bbox_inches='tight', dpi=600)
        #plt.show()
        plt.close()


methods_ls0 = ['SNR',  'PCT', 'CNN1', 'CNN2']
color_ls0 = {'SNR': 'darkorange', 'PCT': 'mediumblue', 'CNN1': 'indigo', 'CNN2': 'darkorchid'}

plt.figure()
fig, ax = plt.subplots()
baseline = sum(aggy_test_array["SNR"])/len(aggy_test_array["SNR"])
ax.axhline(y=baseline, color='gray', linestyle='--', label = 'no skill')

ax.set_ylabel('Precision', fontsize=17, color='black')
ax.set_xlabel('Recall', fontsize=17, color='black')
for m in methods_ls0:
    aggprecision[m].sort()
    aggrecall[m].sort(reverse=True)
    ax.plot(aggrecall[m], aggprecision[m], lw=1, color=color_ls0[m], label= str(title_ls2[m]) + ' AUC: ' + str(round(auc(aggrecall[m],aggprecision[m]), 3)))
plt.tick_params(axis='both', which='major', labelsize=14, colors = "black")
plt.title('Aggregated P-R Curve', fontsize=17, color='black')
plt.legend(labelcolor="black", framealpha=0.1, facecolor='gray')
plt.savefig(visualisation_path + 'Aggregated_PRcurve_realisticfake_alpha_' + str(alpha) +'.pdf', bbox_inches='tight', dpi=600)
plt.show()
plt.close()


plt.figure()
fig, ax = plt.subplots()
ax.plot([0,1], [0,1], color='gray', linestyle='--', linewidth=1, label = 'no skill')

ax.set_ylabel('TPR', fontsize=17, color='black')
ax.set_xlabel('FPR', fontsize=17, color='black')
for m in methods_ls0:
    aggfpr[m].sort()
    aggtpr[m].sort()
    ax.plot(aggfpr[m],aggtpr[m], lw=1, color=color_ls0[m], label= str(title_ls2[m]) + ' AUC: ' + str(round(roc_auc_score(aggy_test_array[m],aggscores_array[m]), 3)))
plt.tick_params(axis='both', which='major', labelsize=14, colors = "black")
plt.title('Aggregated ROC Curve', fontsize=17, color='black')
plt.legend(labelcolor="black", framealpha=0.1, facecolor='gray')
plt.savefig(visualisation_path + 'Aggregated_ROCcurve_realisticfake_alpha_' + str(alpha) +'.pdf', bbox_inches='tight', dpi=600)
plt.show()
plt.close()



################################################################################################################################################
## For every method and every threshold show the threshold and score evolution
#initiate the loop
import matplotlib
m = ['SNR','PCT', 'CNN2']
num = {'SNR': 0, 'PCT': 1, 'CNN2': 2}
n = 1
for alpha in alpha_vals:

    # What was the base template used for the experiemnts?
    template_characteristics = {'Temp': 2900, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}


    tp_path = '/home/ipa/quanz/user_accounts/egarvin/Thesis/70_results/export_CV/from_GPU_byfold/'
    keys = folds
    ls_results_realistic_fake = {key: None for key in keys}
    for i in folds:
        with open(tp_path + 'results_realisticfakeplanets_data_0_alpha_'+str(alpha)+'_CV_testfold'+str(i)+'.pkl', "rb") as f:
            ls_results_realistic_fake[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1


        #CMdf = pd.read_csv(visualisation_path + "CM_" + str(m) + "_alpha" + str(alpha) + ".csv")

    data1 = pd.read_pickle(
        data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_H2O_crosscorr_data_alpha_' + str(
            alpha) + '_temp2800.0_sg4.1.pkl')

        # For each fold:
        # get the results
        # get the indices of the results
        # Fill the gaps of the image with nans?
        # reconstruct the image with the results
        # Image should contain:
        # Map of accuracy (correctly classified is green , badly classified is red)
        # Map of True vs false positives?
        # Map of true vs false negatives ?
        # Could also have: Orange = negatives, true are dark, false are light; blue = true and false positives; true are dark, wrong are light


    for j in folds: # + 1):

        if j == 0:
            i = (len(planetlist) - 1)
        else:
            i = j - 1

        data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

        PR = {}
        ROC = {}

        fig, axe = plt.subplots(3,6, figsize=(3 * 6, 3 * 3), layout='constrained')
        #cax0 = fig.add_axes([0.95, 0.15, 0.1, 0.8])
        for m in methods_ls:

            if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob']
                threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
                #Y_pred = np.array(prob_Y > 0.5) * 1

            elif m in ['SNR', 'SNR_auto']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['SNR']
                #Y_pred = np.array(prob_Y > 3) * 1
                threshold = [1, 2, 3, 4, 5]

            else:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob'][:, 1]
                #Y_pred = np.array(prob_Y > 0.5) * 1
                threshold = [0.1, 0.3, 0.5, 0.7, 0.9]

            Y_pred = ls_results_realistic_fake[j]['results'][m]['Y_pred']


            # dt_true = pd.read_pickle(data_path + "csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/injection_labels_set.pkl")
            # Y_true = dt_true.iloc[0:3041]
            # Y_pred[np.where(Y_true == 1)[0].tolist()]
            #
            # Positives_predictions = [Y_pred[index] for index in np.where(Y_true == 1)[0].tolist()]
            # Positives_scores = [prob_Y[index] for index in np.where(Y_true == 1)[0].tolist()]


            # Get the data
            dtpath0 = data_path + "csv_inputs/True_Spectrum_Data"
            noise_temp_wl_shape = 2  # here we do not need the wavelength dimension

            # Get the data where the planet is indicated
            path0 = os.path.join(dtpath0, str(planetlist[j]) + '_spectrum_dt.csv')  # Untrimmed data. therefore, take the WL range from the trimmed data-
            original_df = pd.read_csv(path0)
            # rebuild the mask
            imsize = int(np.sqrt(len(original_df['Planet'])))
            planet_mask = np.reshape(np.array(original_df['Planet']), (imsize, imsize))

            # Create a cube for the mask, create a block and then a cube
            mask_block = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape),
                                    (imsize * imsize, noise_temp_wl_shape))
            mask_cube = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape),
                                   (imsize, imsize, noise_temp_wl_shape))
            mask_cube_inv = np.empty((noise_temp_wl_shape, imsize, imsize))
            # revert the block to stack it first by wavelength.
            for w in range(noise_temp_wl_shape):
                mask_cube_inv[w, :, :] = mask_cube[:, :, w]
            mask_cube_inv_copy = copy.deepcopy(mask_cube_inv)
            mask_cube_inv[np.where(mask_cube_inv == 1)] = np.nan

            # Deconstruct a full image (here we only use two frames as the wavelength dimension is not of interest - but the function was built for more than one dimensio)
            PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct(mask_cube_inv[0:2, :, :])

            reconstruct_prediction = np.tile(Y_pred, 2).reshape(2, len(Y_pred)).T
            reconstruct_scores = np.tile(prob_Y, 2).reshape(2, len(prob_Y)).T

            reconstruct_ccf = np.tile(data_test[0], 2).reshape(2, len(data_test[0])).T


            test_y = ls_results_realistic_fake[j]['y_test']
            y_test_array = np.array([arr[1] for arr in test_y])

            scores_array = np.array([arr[0] for arr in reconstruct_scores])



            img_prediction = image_reconstruct(reconstruct_prediction, Planet_vec_shape[0], Planet_vec_shape[1],
                                               Planet_position_nan)
            img_scores = image_reconstruct(reconstruct_scores, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            img_ccf = image_reconstruct(reconstruct_ccf, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            img_y = image_reconstruct(test_y, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            precision, recall, _ = precision_recall_curve(y_test_array,scores_array)
            PR[m] = (precision, recall)

            fpr, tpr, _ = roc_curve(y_test_array,scores_array)
            ROC[m] = (fpr, tpr, y_test_array,scores_array)


            tick_positions = [0.5, 1.5, 2.5, 3.5] #tick_positions
            tick_labels = ["TN", "FN", "FP", "TP"]
            if m in ['SNR','PCT', 'CNN2']:
                t_num = 0
                # prediction
                #cax = axe[num[m],t_num].add_axes([axe[num[m],t_num].get_position().x1 + 0.01, axe[num[m],t_num].get_position().y0, 0.02, axe[num[m],t_num].get_position().height])

                img = axe[num[m],t_num].imshow(img_scores[1, :, :], cmap='viridis')
                clb=plt.colorbar(img, ax = axe[num[m],t_num])
                if m in ['SNR', 'SNR_auto']:
                    clb.set_label('S/N Values', fontsize=15)
                else:
                    clb.set_label('Probabilities', fontsize=15)

                clb.ax.tick_params(labelsize=14)

                axe[num[m],t_num].set_title('Scores, ' + str(title_ls[m]), fontsize=15)
                axe[num[m],t_num].set_xlabel('Pixels', fontsize=15)
                axe[num[m],t_num].set_ylabel('Pixels', fontsize=15)
                axe[num[m], t_num].tick_params(labelsize=14)
                axe[num[m], t_num].label_outer()


                cmap = matplotlib.cm.magma
                bounds = [0,1,2,3,4]
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

                for t in threshold:
                    t_num = t_num + 1
                    impred = img_scores[1, :, :] > t
                    img2 = axe[num[m],t_num].imshow((1 * img_y[1, :, :] + 2 * impred), cmap='magma', vmin=0, vmax=3)
                    axe[num[m],t_num].set_title('Classification ' + str(title_ls3[m] + ', T=' + str(t)), fontsize=14.5)

                    if t_num == (len(folds)) & num[m] == 1:
                        clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax = axe[:,5].ravel().tolist(), ticks=tick_positions)
                        clb.set_label('Confusion Matrix', fontsize=15)
                        clb.set_ticklabels(tick_labels)
                        clb.ax.tick_params(labelsize=14)

                    axe[num[m], t_num].tick_params(labelsize=14)
                    axe[num[m],t_num].set_xlabel('Pixels', fontsize=15)
                    axe[num[m],t_num].set_ylabel('Pixels', fontsize=15)
                    axe[num[m],t_num].label_outer()

        fig.suptitle('Test Case '+str(n), fontsize=16, fontweight="bold")
        #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        #plt.tight_layout()
        plt.savefig(visualisation_path + 'grid_realisticfake_' + planetlist[j] + '_all_alpha_' + str(alpha) + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        n=n+1



















#################################################################################
## After chosing which cases we want to use,


title_ls2 = {'SNR': 'S/N', 'SNR_auto': 'S/N_auto', 'ENET': 'Elasticnet', 'RID': 'Ridge', 'PCT': 'PCT', 'DNN': 'DNN',
            'CNN1': 'CNN1', 'CNN2': 'CNN2'}


m = ['SNR','PCT', 'CNN1', 'CNN2']
ni = 1
#initiate the loop
for alpha in alpha_vals:

    # What was the base template used for the experiemnts?
    template_characteristics = {'Temp': 2900, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}


    tp_path = '/home/ipa/quanz/user_accounts/egarvin/Thesis/70_results/export_CV/from_GPU_byfold/'
    keys = folds
    ls_results_realistic_fake = {key: None for key in keys}
    for i in folds:
        with open(tp_path + 'results_realisticfakeplanets_data_0_alpha_'+str(alpha)+'_CV_testfold'+str(i)+'.pkl', "rb") as f:
            ls_results_realistic_fake[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1


        #CMdf = pd.read_csv(visualisation_path + "CM_" + str(m) + "_alpha" + str(alpha) + ".csv")

    data1 = pd.read_pickle(
        data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_H2O_crosscorr_data_alpha_' + str(
            alpha) + '_temp2800.0_sg4.1.pkl')

        # For each fold:
        # get the results
        # get the indices of the results
        # Fill the gaps of the image with nans?
        # reconstruct the image with the results
        # Image should contain:
        # Map of accuracy (correctly classified is green , badly classified is red)
        # Map of True vs false positives?
        # Map of true vs false negatives ?
        # Could also have: Orange = negatives, true are dark, false are light; blue = true and false positives; true are dark, wrong are light
    n=0
    fig1, ax1 = plt.subplots(1, len(folds), figsize=(4.5 * len(folds), 4.5 * 1), layout='constrained')
    fig1.suptitle("P-R Curves", fontsize = 20, fontweight = "bold")

    for j in folds: # + 1):

        if j == 0:
            i = (len(planetlist) - 1)
        else:
            i = j - 1

        data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

        PR = {}
        ROC = {}
        for m in methods_ls:

            if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob']
                threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
                #Y_pred = np.array(prob_Y > 0.5) * 1

            elif m in ['SNR', 'SNR_auto']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['SNR']
                #Y_pred = np.array(prob_Y > 3) * 1
                threshold = [1, 2, 3, 4, 5]

            else:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob'][:, 1]
                #Y_pred = np.array(prob_Y > 0.5) * 1
                threshold = [0.1, 0.3, 0.5, 0.7, 0.9]

            Y_pred = ls_results_realistic_fake[j]['results'][m]['Y_pred']


            # Get the data
            dtpath0 = data_path + "csv_inputs/True_Spectrum_Data"
            noise_temp_wl_shape = 2  # here we do not need the wavelength dimension

            # Get the data where the planet is indicated
            path0 = os.path.join(dtpath0, str(planetlist[
                                                  j]) + '_spectrum_dt.csv')  # Untrimmed data. therefore, take the WL range from the trimmed data-
            original_df = pd.read_csv(path0)
            # rebuild the mask
            imsize = int(np.sqrt(len(original_df['Planet'])))
            planet_mask = np.reshape(np.array(original_df['Planet']), (imsize, imsize))

            # Create a cube for the mask, create a block and then a cube
            mask_block = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape),
                                    (imsize * imsize, noise_temp_wl_shape))
            mask_cube = np.reshape(np.repeat(planet_mask, noise_temp_wl_shape),
                                   (imsize, imsize, noise_temp_wl_shape))
            mask_cube_inv = np.empty((noise_temp_wl_shape, imsize, imsize))
            # revert the block to stack it first by wavelength.
            for w in range(noise_temp_wl_shape):
                mask_cube_inv[w, :, :] = mask_cube[:, :, w]
            mask_cube_inv_copy = copy.deepcopy(mask_cube_inv)
            mask_cube_inv[np.where(mask_cube_inv == 1)] = np.nan

            # Deconstruct a full image (here we only use two frames as the wavelength dimension is not of interest - but the function was built for more than one dimensio)
            PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct(mask_cube_inv[0:2, :, :])

            reconstruct_prediction = np.tile(Y_pred, 2).reshape(2, len(Y_pred)).T
            reconstruct_scores = np.tile(prob_Y, 2).reshape(2, len(prob_Y)).T

            reconstruct_ccf = np.tile(data_test[0], 2).reshape(2, len(data_test[0])).T



            test_y = ls_results_realistic_fake[j]['y_test']
            y_test_array = np.array([arr[1] for arr in test_y])

            scores_array = np.array([arr[0] for arr in reconstruct_scores])



            img_prediction = image_reconstruct(reconstruct_prediction, Planet_vec_shape[0], Planet_vec_shape[1],
                                               Planet_position_nan)
            img_scores = image_reconstruct(reconstruct_scores, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            img_ccf = image_reconstruct(reconstruct_ccf, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            img_y = image_reconstruct(test_y, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)

            precision, recall, _ = precision_recall_curve(y_test_array,scores_array)
            PR[m] = (precision, recall)

            fpr, tpr, _ = roc_curve(y_test_array,scores_array)
            ROC[m] = (fpr, tpr, y_test_array,scores_array)


        methods_ls0 = ['SNR', 'PCT', 'CNN1', 'CNN2']
        color_ls0 = {'SNR': 'darkorange', 'PCT': 'mediumblue', 'CNN1': 'indigo', 'CNN2': 'darkorchid'}

        #plt.figure()
        baseline = sum(y_test_array) / len(y_test_array)
        ax1[n].axhline(y=baseline, color='gray', linestyle='--', label='no skill')
        ax1[n].set_ylabel('Precision', fontsize=17, color='black')
        ax1[n].set_xlabel('Recall', fontsize=17, color='black')

        for m in methods_ls0:
            ax1[n].plot(PR[m][1], PR[m][0], lw=1, color=color_ls0[m],
                    label=str(title_ls2[m]) + ' AUC: ' + str(round(auc(PR[m][1], PR[m][0]), 3)))
        ax1[n].tick_params(axis='both', which='major', labelsize=14, colors="black")
        ax1[n].set_title('Case ' + str(ni), fontsize=17, color='black')
        ax1[n].legend(labelcolor="black", framealpha=0.1, facecolor='gray', loc='upper right')
        ax1[n].label_outer()
        plt.savefig(visualisation_path + 'PRcurve_case_realisticfake_alpha_' + str(
            alpha) + '.pdf', bbox_inches='tight', dpi=600)
        n = n + 1
        ni = ni + 1

    plt.show()
    plt.close()






            #
            # #plt.figure()
            # fig, ax = plt.subplots()
            # baseline = sum(y_test_array) / len(y_test_array)
            # ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='no skill')
            #
            # ax.set_ylabel('TPR', fontsize=17, color='black')
            # ax.set_xlabel('FPR', fontsize=17, color='black')
            #
            # for m in methods_ls0:
            #     ax.plot(ROC[m][0], ROC[m][1], lw=1, color=color_ls0[m],
            #             label=m + ' AUC: ' + str(round(roc_auc_score(ROC[m][2], ROC[m][3]), 3)))
            # plt.tick_params(axis='both', which='major', labelsize=14, colors="black")
            # plt.title('ROC Curve, case ' + str(n), fontsize=17, color='black')
            # plt.legend(labelcolor="black", framealpha=0.1, facecolor='gray')
            # #plt.savefig(visualisation_path + 'ROCcurve_case_realisticfake_' + planetlist[j] + '_alpha_' + str(
            # #    alpha) + '.pdf', bbox_inches='tight', dpi=600)
            # plt.show()
            # plt.close()
            #
            # n = n + 1