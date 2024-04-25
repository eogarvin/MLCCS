

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
import seaborn as sns
from functools import partial
from itertools import chain, repeat
from matplotlib import pyplot
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve, auc
from multiprocessing import Pool, freeze_support
import concurrent.futures
import random
import time
import pickle
import os
import copy
import sys
#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")

from ml_spectroscopy.crosscorrNormVec import crosscorrRV_vec
from ml_spectroscopy.config import path_init
from ml_spectroscopy.DataPreprocessing_utils import image_reconstruct, image_deconstruct
from ml_spectroscopy.plottings_utils_results import ROC_curve_customplot, ROC_curve_saveplt ,PR_curve_customplot, PR_curve_saveplt
from ml_spectroscopy.utility_functions import flatten, Average, grid_search0

## SET SEED FOR REPRODUCIBILITY
random.seed(100)


## Settings
data_name = 'GQlupb'
planet = 'GQlupB'
alpha_vals= [5] # [2,5,8,11,22,32,43,55]
#alpha=2
bal=50
v=6
frame='simple'
folds = [7,13,14,16]


plotname = 'test'
methods_ls = ['SNR', 'SNR_auto', 'ENET', 'RID', 'PCT', 'DNN', 'CNN1', 'CNN2']
color_ls = {'SNR': 'red', 'SNR_auto': 'brown', 'ENET': 'forestgreen', 'RID': 'lime', 'PCT': 'lightblue', 'DNN': 'blue',
            'CNN1': 'navy', 'CNN2': 'purple'}
title_ls = {'SNR': 'SNR', 'SNR_auto': 'SNR_auto', 'ENET': 'Elasticnet', 'RID': 'Ridge', 'PCT': 'Perceptron', 'DNN': 'DNN',
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
dir_path = results_path + "export_CV/from_GPU_byfold/Res_REAL_planets_241123/"  # Depending on the way we pick the files, can't we directly point to the right directory from the start?
#Directory to store results
visualisation_path = subdir + "80_visualisation/Realistic_fakeplanets/Vis_07_150523_REAL/"
csv_path = subdir + "80_visualisation/Realistic_fakeplanets/Vis_07_150523_REAL/"


###################### Try with TPR cnstraint

#initiate the loop
for alpha in alpha_vals:

    # What was the base template used for the experiemnts?
    template_characteristics = {'Temp': 2900, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}


    tp_path = '/home/ipa/quanz/user_accounts/egarvin/Thesis/70_results/export_CV/from_GPU_byfold/Res_REAL_planets_241123/'
    keys = folds
    ls_results_realistic_fake = {key: None for key in keys}
    for i in folds:
        with open(tp_path + 'results_REAL_data_0_alpha_'+str(alpha)+'_CV_testfold'+str(i)+'.pkl', "rb") as f:
            ls_results_realistic_fake[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1


        #CMdf = pd.read_csv(visualisation_path + "CM_" + str(m) + "_alpha" + str(alpha) + ".csv")
    data1 = pd.read_pickle(data_path + 'csv_inputs/CCF_True_Data_test/final_test_set/Real_Data_H2O_T2800.0_sg4.1.pkl')

    #data1 = pd.read_pickle(
    #    data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_H2O_crosscorr_data_alpha_' + str(
    #        alpha) + '_temp2800.0_sg4.1.pkl')

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


        for m in methods_ls:

            if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob']
                #Y_pred = np.array(prob_Y > 0.5) * 1

            elif m in ['SNR', 'SNR_auto']:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['SNR']
                #Y_pred = np.array(prob_Y > 3) * 1

            else:
                prob_Y = ls_results_realistic_fake[j]['results'][m]['Y_pred_prob'][:, 1]
                #Y_pred = np.array(prob_Y > 0.5) * 1

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


            dir_file_planet = data_path + 'True_HCI_data'
            dir_file_WR = data_path + 'wavelength_ranges'
            # If templates have more molecules, remember to adapt the number of dropped end columns in the function
            dir_file_mol_template = data_path + 'csv_inputs/Molecular_Templates_df2.csv'
            savedirccf = data_path + 'csv_inputs/True_CCF_data'
            savedirdata = data_path + 'csv_inputs/True_Spectrum_Data'

            hdu_list0 = fits.open(dir_file_planet + '/res_' + planetlist[j] + '.fits')
            hdu_list0.info()
            Planet_HCI = hdu_list0[0].data
            hdu_list0.close()
            Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down
            #Planet_WR = importWavelength_asList(dir_file_WR + '/WR_' + WRFilename, extension)

            # Transform the 3d cube into a 2d set of rows of spectrums and columns of wavelengths. NANS are removed but the info is stored in the last output
            PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct(Planet_HCI)

            # Deconstruct a full image (here we only use two frames as the wavelength dimension is not of interest - but the function was built for more than one dimensio)
            #PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct( )

            if planetlist[j][:2]  == 'PZ':
                size = 1795
                rv = 22
            elif planetlist[j][:2]  == 'GQ':
                size = 1958
                rv = 44

            reconstruct_prediction = np.tile(Y_pred, size).reshape(size, len(Y_pred)).T
            reconstruct_scores = np.tile(prob_Y, size).reshape(size, len(prob_Y)).T

            reconstruct_ccf = np.tile(data_test[0], size).reshape(size, len(data_test[0])).T

            img_prediction = image_reconstruct(reconstruct_prediction, Planet_vec_shape[0], Planet_vec_shape[1],
                                               Planet_position_nan)
            img_scores = image_reconstruct(reconstruct_scores, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)


            img_ccf = image_reconstruct(reconstruct_ccf, Planet_vec_shape[0], Planet_vec_shape[1],
                                           Planet_position_nan)



            # prediction
            plt.imshow(img_prediction[1, :, :])
            plt.title(str(title_ls[m]), fontsize=18)
            plt.xlabel('[px]', fontsize=17)
            plt.ylabel('[px]', fontsize=17)
            plt.savefig(visualisation_path + 'img_realisticfake_' + planetlist[j] + '_method_' + str(m) + '_alpha_' + str(
                    alpha) + '_Base_Prediction_areas.pdf', bbox_inches='tight')
            #plt.show()
            plt.close()


            # CCF
            plt.imshow(img_ccf[rv, :, :], cmap='viridis')
            plt.title('Molecular Map', fontsize=18)
            clb=plt.colorbar()
            clb.set_label('Normalised CCF Values', fontsize=14)
            plt.xlabel('[px]', fontsize=17)
            plt.ylabel('[px]', fontsize=17)
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
            plt.xlabel('[px]', fontsize=17)
            plt.ylabel('[px]', fontsize=17)
            plt.savefig(visualisation_path + 'img_realisticfake_' + planetlist[j] + '_method_' + str(m) + '_alpha_' + str(
                    alpha) + '_Scores.pdf', bbox_inches='tight')
            plt.show()
            plt.close()







