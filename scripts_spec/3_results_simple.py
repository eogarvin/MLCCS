## This file intends to write out and plot visualisations of the results from the output of the 2_main_simple.py file
## It will plot the results of each alpha level individually, for each CV fold. It will also plot the aggregated CV results for each alpha. Finally, it plots the aggregated results for several alpha levels of interest.
## The plots only show the ROC and PR curves for: SNR 1 and 2, Ridge regression, (elasticnet) and the neural networks + the CNNs.
## We also would like to show the distance in 2 distributions (mean or median) for the SNR, and maybe show the reconstructed image matrix, between the SNR and the CNN.
## Then, it creates the CSV files of the confusion matrices, which can be used to construct the tables directly in the latex file (or using R as an intermediary).
## Create clear sections and rewrite the functions if needed.


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
from itertools import chain, repeat
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve, auc
import random
import pickle
from ml_spectroscopy.config import path_init
from ml_spectroscopy.plottings_utils_results import ROC_curve_customplot, ROC_curve_saveplt ,PR_curve_customplot, PR_curve_saveplt
from ml_spectroscopy.utility_functions import flatten, Average, grid_search0

## SET SEED FOR REPRODUCIBILITY
random.seed(100)


## Settings
data_name = 'GQlupb'
planet = 'GQlupB'
alpha_vals=[0,5000] #[0,2,5,8,11,16,21,29,41,67,5000]
#alpha=2
bal=50
v=6
frame='simple'
len_folds = 8
alpha_vals_subset = alpha_vals[1:-1]


plotname = 'test'
methods_ls = ['SNR', 'SNR_auto', 'ENET', 'RID', 'PCT', 'DNN', 'CNN1', 'CNN2']
color_ls = {'SNR': 'red', 'SNR_auto': 'brown', 'ENET': 'forestgreen', 'RID': 'lime', 'PCT': 'lightblue', 'DNN': 'blue',
            'CNN1': 'navy', 'CNN2': 'purple'}





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
visualisation_path = subdir + "80_visualisation/Results_per_alpha/Vis_00_090223_alphas_simple_T1200SG41/"
csv_path = subdir + "80_visualisation/Results_per_alpha/Vis_00_090223_alphas_simple_T1200SG41/"


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
    data1=pd.read_pickle(data_path+'data_4ml/v'+str(v)+'_ccf_4ml_trim_robustness_simple/H2O_'+data_name+'_scale'+str(alpha)+'_bal'+str(bal)+'_temp1200.0_sg4.1_ccf_4ml_trim_norepetition_v'+str(v)+'_simple.pkl')

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

        # =============================================================================
        # # =============================================================================
        # # ROC curves
        # # =============================================================================
        # =============================================================================

        # methods: ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID', 'LAS', 'RF', 'XGB', 'ENET2']
        plt.style.use('seaborn')

        # Plot out the ROC curves
        ax2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ax1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        fig, axes = plt.subplots(nrows=3, ncols=3)
        fig.suptitle('ROC curves for $\\alpha=' + str(alpha) + '$', fontsize=14)

        for j in range(0, len_folds + 1):

            if j < len_folds:
                try:
                    if j == 0:
                        i = (len_folds - 1)
                    else:
                        i = j - 1

                    data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                        [(str(data1.index.levels[0][i]),)], axis=0)
                    data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                    data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                    X_train = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_train = data_train['H2O']

                    X_valid = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_valid = data_valid['H2O']

                    X_test = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_test = data_test['H2O']

                    df_pr, ax = ROC_curve_customplot(ls_results[j]['results'], axes, ax1[j], ax2[j], Y_test, j, methods_ls, color_ls)
                except KeyError:
                    pass
            else:
                axes[ax1[j], ax2[j]].axis('off')

        for j in range(0, len_folds + 1):
            axes[ax1[j], ax2[j]].label_outer()

            if j == 0:
                i = (len_folds - 1)
            else:
                i = j - 1

            if j == len_folds:

                axes[ax1[j], ax2[j]].axis('off')

                mylegends = [Line2D([0], [0], linestyle='--', color='gray', lw=1)]

                for k in range(0, len(methods_ls)):
                    mylegends.append(Line2D([0], [0], color=color_ls[methods_ls[k]], lw=1))

                new_methods_ls = [['No skill'], methods_ls]
                new_methods_ls = list(chain(*new_methods_ls))
                axes[ax1[j], ax2[j]].legend(mylegends, new_methods_ls, loc='lower left', bbox_to_anchor=(0.05, -0.4),
                                            fontsize=7)

        fig.savefig(
            visualisation_path + 'ROC_' + planet + '_alpha' + str(alpha) + '_bal' + str(bal) + '_combined_CV_' + str(
                plotname) + '_version' + str(v) + 'frame' + str(frame) + '.pdf')
        fig.savefig(
            visualisation_path + 'ROC_' + planet + '_alpha' + str(alpha) + '_bal' + str(bal) + '_combined_CV_' + str(
                plotname) + '_version' + str(v) + 'frame' + str(frame) + '.png')
        fig.show()

        # =============================================================================
        # Aggregated ROC curves
        # =============================================================================
        methods = ['SNR', 'SNR_auto', 'RID', 'ENET', 'DNN', 'PCT', 'CNN1', 'CNN2']

        plt.figure()
        plt.plot(np.array([0., 1.]), np.array([0., 1.]), linestyle='--', lw=1, color='gray')  # , label='No Skill')

        lr_fpr_subdict = {key: None for key in methods_ls}
        lr_tpr_subdict = {key: None for key in methods_ls}

        for m in methods:
            predictions_Y = []
            true_Y = []
            probability_Y = []

            for j in range(0, len_folds):
                Y = []
                Y_hat = []
                prob_Y = []

                if j < len_folds:
                    try:
                        if j == 0:
                            i = (len_folds - 1)
                        else:
                            i = j - 1

                        data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                            [(str(data1.index.levels[0][i]),)], axis=0)
                        data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                        data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                        X_train = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                        Y_train = data_train['H2O']

                        X_valid = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                        Y_valid = data_valid['H2O']

                        X_test = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                        Y_test = data_test['H2O']

                        Y = list(data_test['H2O'])
                        Y_hat = list(ls_results[j]['results'][m]['Y_pred'])

                        if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                            prob_Y = ls_results[j]['results'][m]['Y_pred_prob']

                        elif m in ['SNR', 'SNR_auto']:
                            prob_Y = ls_results[j]['results'][m]['SNR']
                        else:
                            prob_Y = ls_results[j]['results'][m]['Y_pred_prob'][:, 1]

                        predictions_Y.append(Y_hat)
                        true_Y.append(Y)
                        probability_Y.append(prob_Y)

                    except KeyError:
                        pass

            df_Y = pd.DataFrame(
                {'Y_true': flatten(true_Y), 'Y_pred': flatten(predictions_Y), 'Y_prob': flatten(probability_Y)})
            df_Y.to_csv(visualisation_path + "HatVSPred_" + str(m) + "_alpha" + str(alpha) + ".csv")

            lr_fpr_0, lr_tpr_0, _ = roc_curve(df_Y['Y_true'], df_Y['Y_prob'])
            auc_ROC = roc_auc_score(df_Y['Y_true'], df_Y['Y_prob'])

            plt.plot(lr_fpr_0, lr_tpr_0, lw=1, color=color_ls[m], label=m + ' AUC: ' + str(round(auc_ROC, 3)))

            # Save the LR precision and recall for the last plot
            lr_fpr_subdict[m] = lr_fpr_0
            lr_tpr_subdict[m] = lr_tpr_0

        lr_fpr_dict[alpha] = lr_fpr_subdict
        lr_tpr_dict[alpha] = lr_tpr_subdict

        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.title('Aggregated ROC Curve, alpha =' + str(alpha))
        plt.legend()
        plt.savefig(visualisation_path + 'Aggregated_ROC_alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.pdf')
        plt.savefig(visualisation_path + 'Aggregated_ROC_alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.png')
        plt.close()

    # =============================================================================
    # # =============================================================================
    # # PR curves
    # # =============================================================================
    # =============================================================================

    # methods: ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID']
    plt.style.use('seaborn')
    ax2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    ax1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle('PR curves for $\\alpha=' + str(alpha) + '$', fontsize=14)
    fig.plot([0, 1], [0.5, 0.5], linestyle='--', lw=1, color='gray')  # , label='No Skill')


    for j in range(0, len_folds + 1):

        if j < len_folds:
            try:
                if j == 0:
                    i = (len_folds - 1)
                else:
                    i = j - 1

                data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                    [(str(data1.index.levels[0][i]),)],
                    axis=0)
                data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                X_train = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_train = data_train['H2O']

                X_valid = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_valid = data_valid['H2O']

                X_test = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_test = data_test['H2O']

                df_pr, ax = PR_curve_customplot(ls_results[j]['results'], axes, ax1[j], ax2[j], Y_test, j, methods_ls, color_ls)

            except KeyError:
                pass
        else:
            axes[ax1[j], ax2[j]].axis('off')

    for j in range(0, len_folds + 1):
        axes[ax1[j], ax2[j]].label_outer()

        if j == 0:
            i = (len_folds - 1)
        else:
            i = j - 1

        if j == len_folds:

            axes[ax1[j], ax2[j]].axis('off')

            mylegends = [Line2D([0], [0], linestyle='--', color='gray', lw=1)]

            for k in range(0, len(methods_ls)):
                mylegends.append(Line2D([0], [0], color=color_ls[methods_ls[k]], lw=1))

            new_methods_ls = [['No skill'], methods_ls]
            new_methods_ls = list(chain(*new_methods_ls))
            axes[ax1[j], ax2[j]].legend(mylegends, new_methods_ls, loc='lower left', bbox_to_anchor=(0.05, -0.4),
                                        fontsize=7)

    fig.savefig(
        visualisation_path + 'PR_' + planet + '_alpha' + str(alpha) + '_bal' + str(bal) + '_combined_CV_' + str(
            plotname) + '_version' + str(v) + 'frame' + str(frame) + '.pdf')
    fig.savefig(
        visualisation_path + 'PR_' + planet + '_alpha' + str(alpha) + '_bal' + str(bal) + '_combined_CV_' + str(
            plotname) + '_version' + str(v) + 'frame' + str(frame) + '.png')
    fig.show()

    # =============================================================================
    # Aggregated PR curves
    # =============================================================================

    methods = ['SNR', 'SNR_auto', 'RID', 'ENET', 'DNN', 'PCT', 'CNN1', 'CNN2']

    plt.figure()
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', lw=1, color='gray')  # , label='No Skill')
    plt.plot(0, 0, marker='.', color='white')

    lr_precision_subdict = {key: None for key in methods_ls}
    lr_recall_subdict = {key: None for key in methods_ls}
    for m in methods:
        predictions_Y = []
        true_Y = []
        probability_Y = []

        for j in range(0, len_folds):
            Y = []
            Y_hat = []
            prob_Y = []

            if j < len_folds:
                try:
                    if j == 0:
                        i = (len_folds - 1)
                    else:
                        i = j - 1

                    data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                        [(str(data1.index.levels[0][i]),)], axis=0)
                    data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                    data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                    X_train = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_train = data_train['H2O']

                    X_valid = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_valid = data_valid['H2O']

                    X_test = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_test = data_test['H2O']

                    Y = list(data_test['H2O'])
                    Y_hat = list(ls_results[j]['results'][m]['Y_pred'])

                    if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                        prob_Y = ls_results[j]['results'][m]['Y_pred_prob']

                    elif m in ['SNR', 'SNR_auto']:
                        prob_Y = ls_results[j]['results'][m]['SNR']
                    else:
                        prob_Y = ls_results[j]['results'][m]['Y_pred_prob'][:, 1]

                    predictions_Y.append(Y_hat)
                    true_Y.append(Y)
                    probability_Y.append(prob_Y)

                except KeyError:
                    pass

        df_Y = pd.DataFrame(
            {'Y_true': flatten(true_Y), 'Y_pred': flatten(predictions_Y), 'Y_prob': flatten(probability_Y)})
        df_Y.to_csv(visualisation_path + "HatVSPred_PR_" + str(m) + "_alpha" + str(alpha) + ".csv")

        lr_precision, lr_recall, _ = precision_recall_curve(df_Y['Y_true'], df_Y['Y_prob'])
        lr_f1, lr_auc = f1_score(df_Y['Y_true'], df_Y['Y_pred']), auc(lr_recall, lr_precision)
        plt.plot(lr_recall, lr_precision, lw=1, color=color_ls[m], label=(m + ' AUC-PR: ' + str(round(lr_auc, 3))))

        lr_precision_subdict[m] = lr_precision
        lr_recall_subdict[m] = lr_recall
        # Save the LR precision and recall for the last plot
    lr_precision_dict[alpha] = lr_precision_subdict
    lr_recall_dict[alpha] = lr_recall_subdict

    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Aggregated PR Curve, alpha = ' + str(alpha))
    plt.legend(fontsize=8)
    plt.savefig(visualisation_path + 'Aggregated_PR_alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.pdf')
    plt.savefig(visualisation_path + 'Aggregated_PR_alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.png')
    plt.close()

# =============================================================================
# # ===========================================================================
# # Confusion Matrices
# # ===========================================================================
# =============================================================================

    methods0 = ['SNR3', 'SNR3_auto', 'SNR5', 'SNR5_auto', 'SNR', 'SNR_auto','RID', 'ENET', 'PCT', 'DNN', 'CNN1', 'CNN2']

    for m in methods0:
        CM = np.zeros((8, 4))

        if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2', 'PCT', 'DNN', 'CNN1', 'CNN2']:
            HP = np.zeros((8, len(list(ls_results[0]['hyperparameters'][m])) + 2))
        else:
            HP = np.zeros((8, 2))

        for j in range(0, len_folds):

            if j < len_folds:
                try:
                    if j == 0:
                        i = 7
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

                    if m == 'SNR3':
                        pred_snr3 = list(map(int, list(ls_results[j]['results']['SNR']['SNR'] >= 3)))
                        SNR_3_acc = sum(Y_test == pred_snr3) / len(Y_test)
                        CM[j, :] = np.array((confusion_matrix(Y_test, pred_snr3).ravel()))
                        HP[j, :] = [3, SNR_3_acc]


                    elif m == 'SNR3_auto':
                        pred_snr3_auto = list(map(int, list(ls_results[j]['results']['SNR_auto']['SNR'] >= 3)))
                        SNR_3_acc_auto = sum(Y_test == pred_snr3_auto) / len(Y_test)
                        CM[j, :] = np.array((confusion_matrix(Y_test, pred_snr3_auto).ravel()))
                        HP[j, :] = [3, SNR_3_acc_auto]

                    elif m == 'SNR5':
                        pred_snr5 = list(map(int, list(ls_results[j]['results']['SNR']['SNR'] >= 5)))
                        SNR_5_acc = sum(Y_test == pred_snr5) / len(Y_test)
                        CM[j, :] = np.array((confusion_matrix(data_test['H2O'], pred_snr5).ravel()))
                        HP[j, :] = [5, SNR_5_acc]

                    elif m == 'SNR5_auto':
                        pred_snr5_auto = list(map(int, list(ls_results[j]['results']['SNR_auto']['SNR'] >= 5)))
                        SNR_5_acc_auto = sum(Y_test == pred_snr5_auto) / len(Y_test)
                        CM[j, :] = np.array((confusion_matrix(Y_test, pred_snr5_auto).ravel()))
                        HP[j, :] = [5, SNR_5_acc_auto]

                    elif m in ['SNR', 'SNR_auto']:
                        accuracy_test = sum(Y_test == ls_results[j]['results'][m]['Y_pred']) / len(Y_test)
                        CM[j, :] = list(ls_results[j]['confusion matrix'][m])
                        HP[j, :] = [GA_results[j][m]['hyperparams'], accuracy_test]

                    else:
                        CM[j, :] = list(ls_results[j]['confusion matrix'][m])
                        lss = [list(ls_results[j]['hyperparameters'][m]), list(np.array((ls_results[j]['results'][m]['accuracy_train'],ls_results[j]['results'][m]['accuracy_valid_test'])))]
                        HP[j, :] = flatten(lss)


                except KeyError:
                    pass

        CMdf = pd.DataFrame(CM)
        CMdf.columns = ['tn', 'fp', 'fn', 'tp']
        CMdf.to_csv(visualisation_path + "CM_" + str(m) + "_alpha" + str(alpha) + ".csv")

        HPdf = pd.DataFrame(HP)
        HPdf.to_csv(visualisation_path + "HP_" + str(m) + "_alpha" + str(alpha) + ".csv")





# =============================================================================
# # =============================================================================
# # Distribution Plots
# # =============================================================================
# =============================================================================

        snry = pd.read_csv(visualisation_path + "HatVSPred_SNR_alpha"+str(alpha)+".csv")

        # matplotlib inline
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('seaborn-white')
        x1 = np.array(snry['Y_prob'][snry['Y_true'] == 1])
        x2 = np.array(snry['Y_prob'][snry['Y_true'] == 0])
        kwargs = dict(histtype='stepfilled', alpha=0.4, bins=60, density=True)
        plt.hist(x1, **kwargs, color='steelblue')
        plt.hist(x2, **kwargs, color='indianred')
        mylegends = [Line2D([0], [0], color='indianred', lw=1), Line2D([0], [0], color='steelblue', lw=1)]
        plt.legend(mylegends, ['H2O = 0, avg S/N = '+str(round(np.mean(x2),3)), 'H2O = 1, avg S/N = '+str(round(np.mean(x1),3))], fontsize=7)
        plt.title("Empirical probability of "+str(m)+" scores, for alpha = "+str(alpha))
        plt.savefig(visualisation_path + 'full_distribution_signals_alpha'+str(alpha)+'_bal'+str(bal)+'_version'+str(v)+'frame'+str(frame)+'.pdf')
        plt.savefig(visualisation_path + 'full_distribution_signals_alpha'+str(alpha)+'_bal'+str(bal)+'_version'+str(v)+'frame'+str(frame)+'.png')

        plt.style.use('seaborn-white')
        x1 = np.array(snry['Y_prob'][snry['Y_true'] == 1])
        x2 = np.array(snry['Y_prob'][snry['Y_true'] == 0])
        kwargs = dict(histtype='stepfilled', alpha=0.4, bins=60, density=False)
        plt.hist(x1, **kwargs, color='steelblue')
        plt.hist(x2, **kwargs, color='indianred')
        mylegends = [Line2D([0], [0], color='indianred', lw=1), Line2D([0], [0], color='steelblue', lw=1)]
        plt.legend(mylegends, ['H2O = 0, avg S/N = '+str(round(np.mean(x2),3)), 'H2O = 1, avg S/N = '+str(round(np.mean(x1),3))])
        plt.title("Frequency distribution of "+str(m)+" scores, for alpha = "+str(alpha))
        plt.savefig(visualisation_path + 'full_frequency_signals_alpha'+str(alpha)+'_bal'+str(bal)+'_version'+str(v)+'frame'+str(frame)+'.pdf')
        plt.savefig(visualisation_path + 'full_frequency_signals_alpha'+str(alpha)+'_bal'+str(bal)+'_version'+str(v)+'frame'+str(frame)+'.png')


# =============================================================================
# # =============================================================================
# # Aggregated ROC for alpha levels
# # =============================================================================
# =============================================================================
a=0
plt.style.use('seaborn')
ax2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
ax1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
fig_roc, axes_roc = plt.subplots(nrows=3, ncols=3)
fig_roc.suptitle('Aggregated ROC curves per alpha level', fontsize=14)
for alpha in alpha_vals_subset:
    # methods: ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID']
    #$\\alpha=' + str(alpha) + '$', fontsize=14)
    # create aggregated alpha curves
    axes_roc[ax1[a], ax2[a]].plot(np.array([0., 1.]), np.array([0., 1.]), linestyle='--', lw=1, color='gray')  # , label='No Skill')

    for m in methods_ls:
        #mylegends.append(Line2D([0], [0], color=color_ls[m], lw=1))
        ls_tpr_1 = lr_tpr_dict[alpha][m]
        ls_fpr_1 = lr_fpr_dict[alpha][m]

        axes_roc[ax1[a], ax2[a]].plot(ls_fpr_1, ls_tpr_1, lw=1, color=color_ls[m])


    axes_roc[ax1[a], ax2[a]].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    axes_roc[ax1[a], ax2[a]].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    axes_roc[ax1[a], ax2[a]].set_ylabel('True positive rate')
    axes_roc[ax1[a], ax2[a]].set_xlabel('False positive rate')
    axes_roc[ax1[a], ax2[a]].label_outer()
    axes_roc[ax1[a], ax2[a]].set_title('Alpha value: ' + str(alpha))
    #axes_roc[ax1[a], ax2[a]].legend(mylegends, methods_ls, loc='lower left', bbox_to_anchor=(0.05, -0.4), fontsize=7)
    a=a+1

#fig_roc.legend(mylegends, methods_ls, loc='lower left', bbox_to_anchor=(0.05, -0.4), fontsize=7)
fig_roc.tight_layout()
fig_roc.savefig(visualisation_path + 'Global_Aggregated_ROC_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.pdf')
fig_roc.savefig(visualisation_path + 'Global_Aggregated_ROC_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.png')
fig_roc.show()


# =============================================================================
# # =============================================================================
# # Aggregated PR for alpha levels
# # =============================================================================
# =============================================================================

a=0
plt.style.use('seaborn')
ax2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
ax1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
fig_pr, axes_pr = plt.subplots(nrows=3, ncols=3)
fig_pr.suptitle('Aggregated PR curves per alpha level', fontsize=14)
for alpha in alpha_vals_subset:
    # methods: ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID']
    #$\\alpha=' + str(alpha) + '$', fontsize=14)
    # create aggregated alpha curves
    axes_pr[ax1[a], ax2[a]].plot(np.array([0., -1.]), np.array([0., 1.]), linestyle='--', lw=1, color='gray')  # , label='No Skill')

    for m in methods_ls:
        #mylegends.append(Line2D([0], [0], color=color_ls[m], lw=1))
        lr_recall_1 = lr_recall_dict[alpha][m]
        lr_precision_1 = lr_precision_dict[alpha][m]
        axes_pr[ax1[a], ax2[a]].plot(lr_recall_1, lr_precision_1, lw=1, color=color_ls[m])

    axes_pr[ax1[a], ax2[a]].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    axes_pr[ax1[a], ax2[a]].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    axes_pr[ax1[a], ax2[a]].set_ylabel('Precision')
    axes_pr[ax1[a], ax2[a]].set_xlabel('Recall')
    axes_pr[ax1[a], ax2[a]].label_outer()
    axes_pr[ax1[a], ax2[a]].set_title('Alpha value: ' + str(alpha))
    a=a+1

#fig_pr.legend(mylegends, methods_ls, loc='lower left', bbox_to_anchor=(0.05, -0.4), fontsize=7)
fig_pr.tight_layout()
#fig_pr.legend()
fig_pr.savefig(visualisation_path + 'Global_Aggregated_PR_alpha_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.pdf')
fig_pr.savefig(visualisation_path + 'Global_Aggregated_PR_alpha_bal' + str(bal) + '_version' + str(v) + 'frame' + str(frame) + '.png')
fig_pr.show()





# =============================================================================
# # =============================================================================
# # Distribution Plots
# # =============================================================================
# =============================================================================
methods_ls = ['SNR', 'SNR_auto', 'ENET', 'RID', 'PCT', 'DNN', 'CNN1', 'CNN2']

for alpha in alpha_vals:
    for m in methods:
        scores = pd.read_csv(visualisation_path + "HatVSPred_"+str(m)+"_alpha"+str(alpha)+".csv")

        plt.style.use('seaborn-white')
        x1 = np.array(scores['Y_prob'][scores['Y_true'] == 1])
        x2 = np.array(scores['Y_prob'][scores['Y_true'] == 0])
        kwargs = dict(histtype='stepfilled', alpha=0.4, bins=60, density=False)
        plt.hist(x1, **kwargs, color='steelblue')
        plt.hist(x2, **kwargs, color='indianred')
        mylegends = [Line2D([0], [0], color='indianred', lw=1), Line2D([0], [0], color='steelblue', lw=1)]
        plt.legend(mylegends, ['H2O not in spectrum, avg S/N = ' + str(round(np.mean(x2), 3)),
                               'H2O in spectrum, avg S/N = ' + str(round(np.mean(x1), 3))])
        plt.title("Frequency distribution of "+str(m)+" scores, for alpha = " + str(alpha))
        plt.savefig(visualisation_path + 'frequency_signals_method'+str(m)+'alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(
            v) + 'frame' + str(frame) + '.pdf')
        plt.savefig(visualisation_path + 'frequency_signals_method'+str(m)+'_alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(
            v) + 'frame' + str(frame) + '.png')
        plt.close()
