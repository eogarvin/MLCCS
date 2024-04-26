# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 01:38:10 2021

@author: emily
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 22:07:53 2022

@author: emily

results
"""

## LIBRARIES

import pandas as pd
import numpy as np
import random
import sys
import pickle
import os
from matplotlib.lines import Line2D
from itertools import chain

# sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.crosscorrNormVec import crosscorrRV_vec
from ml_spectroscopy.config import path_init

import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import time
import concurrent.futures
import pandas as pd

from ml_spectroscopy.plottings_utils_results import ROC_curve_customplot, ROC_curve_saveplt, PR_curve_customplot, \
    PR_curve_saveplt
from ml_spectroscopy.utility_functions import flatten, Average, grid_search0

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

import seaborn as sns

## ACTIVE SUBDIR
subdir = path_init()
# subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
visual_path = subdir + "80_visualisation/"
csv_res_path = subdir + "80_visualisation/"

## SET SEED FOR REPRODUCIBILITY
random.seed(100)

## IMPORT DATA
## start with only trimmed data. Later can compare just the neural network between padded and trimmed data, using Planet_Signals[data.sum(axis=0)==0,:]
# Planet_Signals = pd.read_csv(data_path + "csv_inputs/Planet_Signals_df.csv", index_col=0)
# SETTINGS

data_name = 'GQlupb'
template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}

alpha = 10
beta=50
v=3
x = 0
planet = 'GQlupB'
#data1=pd.read_pickle(data_path+'data_4ml/v2_ccf_4ml_trim_robustness/H2O_'+data_name+'_scale'+str(alpha)+'_temp1200_sg4.1_ccf_4ml_trim_norepetition.pkl')
data1 = pd.read_pickle(data_path + 'data_4ml/v4_ccf_4ml_trim_robustness/H2O_' + data_name + '_scale' + str(
    alpha) + '_bal' + str(beta) + '_temp1200.0_sg2.9_co0.3_fe-0.3_ccf_4ml_trim_norepetition_v' + str(v) + '.pkl')


i = 5
j = 6
data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_train = data_train['H2O']

X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_valid = data_valid['H2O']

X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_test = data_test['H2O']

# import results from CV and store information as plots

dir_path = results_path + "export_CV/from_GPU_byfold/results/"
ls_data = os.listdir(dir_path)
len_folds = len(ls_data)
result_names = [ls_data[n][:-4] for n in range(len_folds)]

keys = result_names
ls_results = {key: None for key in keys}

for i in range(0, len_folds):
    with open(dir_path + str(ls_data[i]), "rb") as f:
        ls_results[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1

dir_path2 = results_path + "export_CV/from_GPU_byfold/GA_results/"
ls_data2 = os.listdir(dir_path2)
len_folds2 = len(ls_data2)
result_names2 = [ls_data2[n][:-4] for n in range(len_folds2)]

keys2 = result_names2
GA_results = {key: None for key in keys2}

for i in range(0, len_folds):
    with open(dir_path2 + str(ls_data2[i]), "rb") as f:
        GA_results[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1

methods_ls = ['SNR', 'SNR_auto', 'RF', 'XGB', 'LAS', 'ENET', 'RID', 'ENET2', 'PCT', 'DNN', 'CNN1', 'CNN2']
# methods_ls=['SNR', 'SNR_auto','CNN1','ENET','RID', 'XGB', 'ENET2']
plotname = 'test'
color_ls = {'SNR': 'red', 'SNR_auto': 'brown', 'PCT': 'lightblue', 'DNN': 'blue', 'CNN1': 'navy', 'CNN2': 'purple',
            'ENET': 'forestgreen', 'RID': 'lime', 'LAS': 'lightgreen', 'RF': 'yellow', 'XGB': 'orange',
            'ENET2': 'darkgreen'}

# =============================================================================
# # =============================================================================
# # ROC curves
# # =============================================================================
# =============================================================================


# methods: ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID', 'LAS', 'RF', 'XGB', 'ENET2']
plt.style.use('seaborn')

# Plot out the ROC curves
# ls_results[i]['Y_test']
alpha = 10
ax2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
ax1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.suptitle('ROC curves for $\\alpha=' + str(alpha) + '$', fontsize=14)

for j in range(0, len_folds + 1):

    if j < len_folds:
        try:

            if j == 0:
                i = 7
            else:
                i = j - 1

            data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)],
                                                                                     axis=0)
            data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
            data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

            X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train = data_train['H2O']

            X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid = data_valid['H2O']

            X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_test = data_test['H2O']

            df_pr, ax = ROC_curve_customplot(ls_results[j]['results'], axes, ax1[j], ax2[j], Y_test, i, methods_ls,
                                             color_ls, path=visual_path)

        except KeyError:
            pass
    else:
        axes[ax1[j], ax2[j]].axis('off')

for j in range(0, len_folds + 1):

    axes[ax1[j], ax2[j]].label_outer()

    if j == 0:
        i = 7
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

fig.savefig(visual_path + planet + '_final_plots_CV/ROC_' + planet + '_combined_CV_' + str(plotname) + '.pdf')
fig.savefig(visual_path + planet + '_final_plots_CV/ROC_' + planet + '_combined_CV_' + str(plotname) + '.png')

# res, f, ax1, ax2, Y_test, i, data_name='GQlupB', path, alpha=10


# =============================================================================
# Plots to save
# =============================================================================

for j in range(0, len_folds):
    try:
        if j == 0:
            i = 7
        else:
            i = j - 1

        data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)],
                                                                                 axis=0)
        data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
        data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

        X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_train = data_train['H2O']

        X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_valid = data_valid['H2O']

        X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_test = data_test['H2O']

        ROC_curve_saveplt(ls_results[j]['results'], Y_test, i, color_ls, path=visual_path)
    except KeyError:
        pass

# =============================================================================
# # =============================================================================
# # Precision Recall curves
# # =============================================================================
# =============================================================================


# =============================================================================
# ## Adaptive plots
# =============================================================================


# methods: ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID', 'LAS', 'RF', 'XGB', 'ENET2']
plt.style.use('seaborn')

# Plot out the ROC curves
# ls_results[i]['Y_test']
alpha = 10
ax2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
ax1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.suptitle('PR curves for $\\alpha=' + str(alpha) + '$', fontsize=14)

for j in range(0, len_folds + 1):

    if j < len_folds:
        try:

            if j == 0:
                i = 7
            else:
                i = j - 1

            data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)],
                                                                                     axis=0)
            data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
            data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

            X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train = data_train['H2O']

            X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid = data_valid['H2O']

            X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_test = data_test['H2O']

            df_pr, ax = PR_curve_customplot(ls_results[j]['results'], axes, ax1[j], ax2[j], Y_test, i, methods_ls,
                                            color_ls, path=visual_path)

        except KeyError:
            pass
    else:
        axes[ax1[j], ax2[j]].axis('off')

for j in range(0, len_folds + 1):

    axes[ax1[j], ax2[j]].label_outer()

    if j == 0:
        i = 7
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

fig.savefig(visual_path + planet + '_final_plots_CV/PR_' + planet + '_combined_CV_' + str(plotname) + '.pdf')
fig.savefig(visual_path + planet + '_final_plots_CV/PR_' + planet + '_combined_CV_' + str(plotname) + '.png')

# res, f, ax1, ax2, Y_test, i, data_name='GQlupB', path, alpha=10


# =============================================================================
# Saved plots
# =============================================================================


for j in range(0, len_folds):
    try:

        if j == 0:
            i = 7
        else:
            i = j - 1

        data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)],
                                                                                 axis=0)
        data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
        data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

        X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_train = data_train['H2O']

        X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_valid = data_valid['H2O']

        X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_test = data_test['H2O']

        PR_curve_saveplt(ls_results[j]['results'], Y_test, i, color_ls, path=visual_path)
    except KeyError:
        pass

# =============================================================================
# Aggregated ROC curves
# =============================================================================

methods = ['SNR', 'SNR_auto', 'RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2', 'DNN', 'PCT', 'CNN1', 'CNN2']

plt.figure()
plt.plot(np.array([0., 1.]), np.array([0., 1.]), linestyle='--', lw=1, color='gray')  # , label='No Skill')
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
                    i = 7
                else:
                    i = j - 1

                data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                    [(str(data1.index.levels[0][i]),)], axis=0)
                data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_train = data_train['H2O']

                X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_valid = data_valid['H2O']

                X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_test = data_test['H2O']

                Y = list(data_test['H2O'])
                Y_hat = list(ls_results[j]['results'][m]['Y_pred'])
                # ax.label_outer()

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

    df_Y = pd.DataFrame({'Y_true': flatten(true_Y), 'Y_pred': flatten(predictions_Y), 'Y_prob': flatten(probability_Y)})
    df_Y.to_csv(csv_res_path + planet + "_final_plots_CV/HatVSPred_" + str(m) + ".csv")

    sum(df_Y['Y_true'] == df_Y['Y_pred']) / len(df_Y['Y_true'])

    ns_probs_0 = [0 for _ in range(len(df_Y['Y_true']))]
    # predict probabilities
    lr_probs_0 = df_Y['Y_prob']
    testy_0 = df_Y['Y_true']
    # plot the roc curve for the model
    # yhat_0 = res[methods_ls[k]]['Y_pred']
    # ns_auc_0 = roc_auc_score(testy_0, ns_probs_0)
    lr_auc_0 = roc_auc_score(testy_0, lr_probs_0)

    ns_fpr_0, ns_tpr_0, _ = roc_curve(testy_0, ns_probs_0)
    lr_fpr_0, lr_tpr_0, _ = roc_curve(testy_0, lr_probs_0)

    auc_ROC = roc_auc_score(df_Y['Y_true'], df_Y['Y_prob'])
    lr_precision, lr_recall, _ = precision_recall_curve(df_Y['Y_true'], df_Y['Y_prob'])
    lr_f1, lr_auc = f1_score(df_Y['Y_true'], df_Y['Y_pred']), auc(lr_recall, lr_precision)

    testy_0 = df_Y['Y_true']
    ns_probs_0 = [0 for _ in range(len(testy_0))]
    # predict probabilities
    lr_probs_0 = df_Y['Y_prob']
    # plot the roc curve for the model
    # yhat_0 = res[methods_ls[k]]['Y_pred']
    # ns_auc_0 = roc_auc_score(testy_0, ns_probs_0)
    lr_auc_0 = roc_auc_score(testy_0, lr_probs_0)
    ns_fpr_0, ns_tpr_0, _ = roc_curve(testy_0, ns_probs_0)
    lr_fpr_0, lr_tpr_0, _ = roc_curve(testy_0, lr_probs_0)
    plt.plot(lr_fpr_0, lr_tpr_0, lw=1, color=color_ls[m], label=m + ' AUC: ' + str(round(auc_ROC, 3)))

    # axarr = f.add_subplot(3,3,i+1
    # plot the roc curve for the odel

plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title('Aggregated ROC Curve over all CV folds')
plt.legend()
plt.savefig(csv_res_path + planet + '_final_plots_CV/Aggregated_ROC.pdf')
plt.show()

# =============================================================================
# Aggregated PR curves
# =============================================================================

methods = ['SNR', 'SNR_auto', 'RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2', 'DNN', 'PCT', 'CNN1', 'CNN2']
# pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=1, color='gray')

plt.figure()
plt.plot([0, 1], [0.5, 0.5], linestyle='--', lw=1, color='gray')  # , label='No Skill')
plt.plot(0, 0, marker='.', color='white')

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
                    i = 7
                else:
                    i = j - 1

                data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                    [(str(data1.index.levels[0][i]),)], axis=0)
                data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_train = data_train['H2O']

                X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_valid = data_valid['H2O']

                X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_test = data_test['H2O']

                Y = list(data_test['H2O'])
                Y_hat = list(ls_results[j]['results'][m]['Y_pred'])
                # ax.label_outer()

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

    df_Y = pd.DataFrame({'Y_true': flatten(true_Y), 'Y_pred': flatten(predictions_Y), 'Y_prob': flatten(probability_Y)})
    df_Y.to_csv(csv_res_path + planet + "_final_plots_CV/HatVSPred_" + str(m) + ".csv")

    sum(df_Y['Y_true'] == df_Y['Y_pred']) / len(df_Y['Y_true'])

    ns_probs_0 = [0 for _ in range(len(df_Y['Y_true']))]
    # predict probabilities
    lr_probs_0 = df_Y['Y_prob']
    testy_0 = df_Y['Y_true']
    # plot the roc curve for the model
    # yhat_0 = res[methods_ls[k]]['Y_pred']
    # ns_auc_0 = roc_auc_score(testy_0, ns_probs_0)
    lr_auc_0 = roc_auc_score(testy_0, lr_probs_0)

    ns_fpr_0, ns_tpr_0, _ = roc_curve(testy_0, ns_probs_0)
    lr_fpr_0, lr_tpr_0, _ = roc_curve(testy_0, lr_probs_0)

    auc_ROC = roc_auc_score(df_Y['Y_true'], df_Y['Y_prob'])
    lr_precision, lr_recall, _ = precision_recall_curve(df_Y['Y_true'], df_Y['Y_prob'])
    lr_f1, lr_auc = f1_score(df_Y['Y_true'], df_Y['Y_pred']), auc(lr_recall, lr_precision)

    testy_0 = df_Y['Y_true']
    ns_probs_0 = [0 for _ in range(len(testy_0))]
    # predict probabilities
    lr_probs_0 = df_Y['Y_prob']
    # plot the roc curve for the model
    # yhat_0 = res[methods_ls[k]]['Y_pred']
    # ns_auc_0 = roc_auc_score(testy_0, ns_probs_0)
    lr_auc_0 = roc_auc_score(testy_0, lr_probs_0)
    ns_fpr_0, ns_tpr_0, _ = roc_curve(testy_0, ns_probs_0)
    lr_fpr_0, lr_tpr_0, _ = roc_curve(testy_0, lr_probs_0)
    plt.plot(lr_recall, lr_precision, lw=1, color=color_ls[m], label=(m + ' AUC: ' + str(round(lr_auc, 3))))

    # axarr = f.add_subplot(3,3,i+1
    # plot the roc curve for the odel

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Aggregated PR Curve over all CV folds')
plt.legend(fontsize=8)
plt.savefig(csv_res_path +planet + '_final_plots_CV/Aggregated_PR.pdf')
plt.show()

######################

methods = ['SNR', 'SNR_auto', 'XGB', 'RID', 'ENET2', 'PCT', 'CNN1']
methods_title = {'SNR': 'SNR', 'SNR_auto': 'ACF corrected SNR', 'XGB': 'Gradient tree boosting', 'RID': 'Ridge',
                 'ENET2': 'ElasticNet (2)', 'PCT': 'the Perceptron', 'CNN1': 'the CNN (1)'}

scores = np.zeros((len(methods), 6))
dfscores = pd.DataFrame(scores)
dfscores.columns = ['method', 'FDR_thresh', 'tp', 'fp', 'tn', 'fn']

it = 0

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
                    i = 7
                else:
                    i = j - 1

                data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                    [(str(data1.index.levels[0][i]),)], axis=0)
                data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                X_train = data_train.drop(['tempP', 'loggP', 'CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_train = data_train['H2O']

                X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_valid = data_valid['H2O']

                X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_test = data_test['H2O']

                Y = list(data_test['H2O'])
                Y_hat = list(ls_results[j]['results'][m]['Y_pred'])
                # ax.label_outer()

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

    df_Y = pd.DataFrame({'Y_true': flatten(true_Y), 'Y_pred': flatten(predictions_Y), 'Y_prob': flatten(probability_Y)})
    df_Y.to_csv(csv_res_path + planet + "_final_plots_CV/HatVSPred_" + str(m) + ".csv")

    sum(df_Y['Y_true'] == df_Y['Y_pred']) / len(df_Y['Y_true'])

    if m in ['SNR', 'SNR_auto']:
        hyperparams = np.arange(np.min(prob_Y), np.max(prob_Y), 0.001)
    else:
        hyperparams = np.arange(0, 1, 0.001)

    scores0 = grid_search0(hyperparams, df_Y['Y_prob'], df_Y['Y_true'])
    dfscores.iloc[it, :] = [m, scores0['optim_score'], scores0['tp'], scores0['fp'], scores0['tn'], scores0['fn']]

    g, axess = plt.subplots(nrows=2, ncols=2)
    g.suptitle(('Classification scores for ' + str(methods_title[m])), fontsize=14)

    sns.boxplot(x=df_Y['Y_true'], y=df_Y['Y_prob'], ax=axess[0, 0])
    axess[0, 0].set_title("Scores for " + str(m))

    axess[1, 0].hist(df_Y['Y_prob'])
    axess[1, 0].set_title('Prediction scores', fontsize=9)

    if m in ['SNR', 'SNR_auto']:
        axess[1, 0].set_xlim(-6, 6)
        axess[1, 0].axvline(x=GA_results[j][m]['hyperparams'][0], color='darkred',
                            label='Optimized Accuracy: T=' + str(round(GA_results[j][m]['hyperparams'][0], 2)))
        axess[1, 0].axvline(x=scores0['optim_score'], color='darkblue',
                            label='FDR=0.05: T*=' + str(round(scores0['optim_score'], 2)))
    else:
        axess[1, 0].set_xlim(0, 1)
        axess[1, 0].axvline(x=0.5, color='darkred', label='Optimized Accuracy: T=0.5')
        axess[1, 0].axvline(x=scores0['optim_score'], color='darkblue',
                            label='FDR=0.05: T*=' + str(round(scores0['optim_score'], 2)))

    axess[0, 1].hist(df_Y[df_Y['Y_true'] == 1]['Y_prob'])
    axess[0, 1].set_title('Scores: H20 = 1', fontsize=9)
    if m in ['SNR', 'SNR_auto']:
        axess[0, 1].set_xlim(-6, 6)
        axess[0, 1].axvline(x=GA_results[j][m]['hyperparams'][0], color='darkred',
                            label='Optimized Accuracy: T=' + str(round(GA_results[j][m]['hyperparams'][0], 2)))
        axess[0, 1].axvline(x=scores0['optim_score'], color='darkblue',
                            label='FDR=0.05: T*=' + str(round(scores0['optim_score'], 2)))
    else:
        axess[0, 1].set_xlim(0, 1)
        axess[0, 1].axvline(x=0.5, color='darkred', label='Optimized Accuracy: T=0.5')
        axess[0, 1].axvline(x=scores0['optim_score'], color='darkblue',
                            label='FDR=0.05: T*=' + str(round(scores0['optim_score'], 2)))
        # axess[0, 1].legend(fontsize=5)

    axess[1, 1].hist(df_Y[df_Y['Y_true'] == 0]['Y_prob'])
    axess[1, 1].set_title('Scores: H20 = 0', fontsize=9)
    if m in ['SNR', 'SNR_auto']:
        axess[1, 1].set_xlim(-6, 6)
        axess[1, 1].axvline(x=GA_results[j][m]['hyperparams'][0], color='darkred',
                            label='Optimized Accuracy: T=' + str(round(GA_results[j][m]['hyperparams'][0], 2)))
        axess[1, 1].axvline(x=scores0['optim_score'], color='darkblue',
                            label='FDR=0.05: T*=' + str(round(scores0['optim_score'], 2)))
    else:
        axess[1, 1].set_xlim(0, 1)
        axess[1, 1].axvline(x=0.5, color='darkred', label='Optimized Accuracy: T=0.5')
        axess[1, 1].axvline(x=scores0['optim_score'], color='darkblue',
                            label='FDR=0.05: T*' + str(round(scores0['optim_score'], 2)))

    g.tight_layout()
    mylegends = [Line2D([0], [0], color='darkred', lw=1), Line2D([0], [0], color='darkblue', lw=1)]

    if m in ['SNR', 'SNR_auto']:
        g.legend(mylegends, ['Optimal Accuracy: T=' + str(round(GA_results[j][m]['hyperparams'][0], 2)),
                             'FDR=0.05: T*=' + str(round(scores0['optim_score'], 2))], loc=(0.3835, 0.38),
                 prop=dict(size=8.2))
    else:
        g.legend(mylegends, ['Optimal Accuracy: T=0.5', 'FDR=0.05: T*=' + str(round(scores0['optim_score'], 2))],
                 loc=(0.3835, 0.38), prop=dict(size=8.2))

    g.savefig(csv_res_path +planet + '_final_plots_CV/Histogram_PIT_' + m + '.pdf')

    it = it + 1

dfscores.to_csv(csv_res_path + planet + "_final_plots_CV/dfscores.csv")

##### Hyperparameters and weights
# =============================================================================
#
#
# methods=['SNR','SNR_auto','RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']
#
# for m in methods:
#
#     CM = np.zeros((8,4))
#
#     if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:
#         HP = np.zeros((8, len(list(ls_results[0]['hyperparameters'][m]))+2))
#
#
#     for j in range(0,len_folds):
#
#
#         if j<len_folds:
#             try:
#                 if j==0:
#                     i=7
#                 else:
#                     i=j-1
#
#                 data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
#                 data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
#                 data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
#
#                 X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
#                 Y_train=data_train['H2O']
#
#                 X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
#                 Y_valid=data_valid['H2O']
#
#                 X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
#                 Y_test=data_test['H2O']
#
#
#                 CM[j,:]=list(ls_results[j]['confusion matrix'][m])
#
#                 if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:
#                     lss=[list(ls_results[j]['hyperparameters'][m]), list(np.array((ls_results[j]['results'][m]['accuracy_train'], ls_results[j]['results'][m]['accuracy_valid_test'])))]
#                     HP[j,:]=flatten(lss)
#
#             except KeyError:
#                 pass
#
#
#     CMdf=pd.DataFrame(CM)
#     CMdf.columns=['tn', 'fp', 'fn', 'tp']
#     CMdf.to_csv(csv_res_path+"csv_results/CM_"+str(m)+".csv")
#
#     if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:
#         HPdf=pd.DataFrame(HP)
#         HPdf.to_csv(csv_res_path+"csv_results/HP_"+str(m)+".csv")
#
#
#
#
# =============================================================================


methods0 = ['SNR3', 'SNR3_auto', 'SNR5', 'SNR5_auto', 'SNR', 'SNR_auto', 'RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2',
            'PCT', 'DNN', 'CNN1', 'CNN2']

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

                data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                    [(str(data1.index.levels[0][i]),)], axis=0)
                data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_train = data_train['H2O']

                X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_valid = data_valid['H2O']

                X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_test = data_test['H2O']

                if m == 'SNR3':
                    pred_snr3 = list(map(int, list(ls_results[j]['results']['SNR']['SNR'] >= 3)))
                    SNR_3_acc = sum(data_test['H2O'] == pred_snr3) / len(data_test['H2O'])
                    CM[j, :] = np.array((confusion_matrix(data_test['H2O'], pred_snr3).ravel()))
                    HP[j, :] = [3, SNR_3_acc]


                elif m == 'SNR3_auto':
                    pred_snr3_auto = list(map(int, list(ls_results[j]['results']['SNR_auto']['SNR'] >= 3)))
                    SNR_3_acc_auto = sum(data_test['H2O'] == pred_snr3_auto) / len(data_test['H2O'])
                    CM[j, :] = np.array((confusion_matrix(data_test['H2O'], pred_snr3_auto).ravel()))
                    HP[j, :] = [3, SNR_3_acc_auto]

                elif m == 'SNR5':
                    pred_snr5 = list(map(int, list(ls_results[j]['results']['SNR']['SNR'] >= 5)))
                    SNR_5_acc = sum(data_test['H2O'] == pred_snr5) / len(data_test['H2O'])
                    CM[j, :] = np.array((confusion_matrix(data_test['H2O'], pred_snr5).ravel()))
                    HP[j, :] = [5, SNR_5_acc]

                elif m == 'SNR5_auto':
                    pred_snr5_auto = list(map(int, list(ls_results[j]['results']['SNR_auto']['SNR'] >= 5)))
                    SNR_5_acc_auto = sum(data_test['H2O'] == pred_snr5_auto) / len(data_test['H2O'])
                    CM[j, :] = np.array((confusion_matrix(data_test['H2O'], pred_snr5_auto).ravel()))
                    HP[j, :] = [5, SNR_5_acc_auto]

                elif m in ['SNR', 'SNR_auto']:

                    accuracy_test = sum(data_test['H2O'] == ls_results[j]['results'][m]['Y_pred']) / len(
                        data_test['H2O'])
                    CM[j, :] = list(ls_results[j]['confusion matrix'][m])
                    HP[j, :] = [GA_results[j][m]['hyperparams'], accuracy_test]

                else:

                    CM[j, :] = list(ls_results[j]['confusion matrix'][m])
                    lss = [list(ls_results[j]['hyperparameters'][m]), list(np.array((ls_results[j]['results'][m][
                                                                                         'accuracy_train'],
                                                                                     ls_results[j]['results'][m][
                                                                                         'accuracy_valid_test'])))]
                    HP[j, :] = flatten(lss)

                    # if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:




            except KeyError:
                pass

    CMdf = pd.DataFrame(CM)
    CMdf.columns = ['tn', 'fp', 'fn', 'tp']
    CMdf.to_csv(csv_res_path + planet + "_final_plots_CV/CM_" + str(m) + ".csv")

    # if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:
    HPdf = pd.DataFrame(HP)
    HPdf.to_csv(csv_res_path + planet + "_final_plots_CV/HP_" + str(m) + ".csv")

# =============================================================================
# run times
# =============================================================================


methods = ['SNR', 'SNR_auto', 'RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2', 'PCT', 'DNN', 'CNN1', 'CNN2']
it = 0

RTtmp = np.zeros((len(methods), 4))
RTdf_GA = pd.DataFrame(RTtmp)
RTdf_GA.columns = ['method', 'min', 'max', 'mean']

RTdf_method = pd.DataFrame(RTtmp)
RTdf_method.columns = ['method', 'min', 'max', 'mean']

for m in methods:

    RT = np.zeros((8, 2))

    for j in range(0, len_folds):

        if j < len_folds:
            try:
                if j == 0:
                    i = 7
                else:
                    i = j - 1

                data_train = data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop(
                    [(str(data1.index.levels[0][i]),)], axis=0)
                data_valid = data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                data_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]

                X_train = data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_train = data_train['H2O']

                X_valid = data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_valid = data_valid['H2O']

                X_test = data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_test = data_test['H2O']

                RT[j, :] = [GA_results[j][m]['runtime_GA'], GA_results[j][m]['runtime_model']]


            except KeyError:
                pass


    RTdf_GA.iloc[it, 0] = m
    RTdf_GA.iloc[it, 1] = np.min(RT, axis=0)[0]
    RTdf_GA.iloc[it, 2] = np.max(RT, axis=0)[0]
    RTdf_GA.iloc[it, 3] = np.mean(RT, axis=0)[0]

    RTdf_method.iloc[it, 0] = m
    RTdf_method.iloc[it, 1] = np.min(RT, axis=0)[1]
    RTdf_method.iloc[it, 2] = np.max(RT, axis=0)[1]
    RTdf_method.iloc[it, 3] = np.mean(RT, axis=0)[1]

    it = it + 1

    RT = pd.DataFrame(RT)
    RT.to_csv(csv_res_path + planet + "_final_plots_CV/RT_" + str(m) + ".csv")

RTdf_GA.to_csv(csv_res_path + planet + "_final_plots_CV/RT_GA.csv")
RTdf_method.to_csv(csv_res_path + planet + "_final_plots_CV/RT_method.csv")

#sns.barplot(RTdf_GA['mean'])

plt.style.use('seaborn')
bars1 = plt.bar(x=RTdf_GA['method'], height=RTdf_GA['max'] / 60, width=0.7, color='lightgreen')
plt.title('Run Time of the optimization (in minutes)')
plt.savefig(csv_res_path + planet + "_final_plots_CV/model_GA_Runtime.pdf")

plt.style.use('seaborn')
bars1 = plt.bar(x=RTdf_method['method'], height=RTdf_method['max'] / 60, width=0.7, color='sandybrown')
plt.title('Run Time of the models (in minutes)')
plt.savefig(csv_res_path + planet + "_final_plots_CV/model_Runtime.pdf")

# =============================================================================
#
#
#
# plt.plot(ls_results[1]['results']['RF']['weights'])
# plt.plot(ls_results[1]['results']['LAS']['weights']['parameters'])
# plt.plot(ls_results[1]['results']['RID']['weights']['parameters'])
# plt.plot(ls_results[1]['results']['ENET']['weights']['parameters'])
# plt.plot(ls_results[1]['results']['ENET2']['weights']['parameters'])
# plt.plot(ls_results[1]['results']['PCT']['weights'][0][:,1])
# plt.plot(ls_results[1]['results']['CNN2']['weights'][4][:,1])
# =============================================================================


fg, axes = plt.subplots(4, 2)
ax01 = [0, 0, 1, 1, 2, 2, 3, 3]
ax02 = [0, 1, 0, 1, 0, 1, 0, 1]
for j in range(0, 8):
    axes[ax01[j], ax02[j]].plot(np.arange(-2000, 2000, 1), ls_results[j]['results']['RF']['weights'])
    axes[ax01[j], ax02[j]].label_outer()
    axes[ax01[j], ax02[j]].set_title("cv fold " + str(j), fontsize=9)
fg.suptitle("Random Forest Weights")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_RF.pdf")
fg.show()

fg, axes = plt.subplots(4, 2)
ax01 = [0, 0, 1, 1, 2, 2, 3, 3]
ax02 = [0, 1, 0, 1, 0, 1, 0, 1]
for j in range(0, 8):
    axes[ax01[j], ax02[j]].plot(np.arange(-2000, 2000, 1), ls_results[j]['results']['LAS']['weights']['parameters'])
    axes[ax01[j], ax02[j]].label_outer()
    axes[ax01[j], ax02[j]].set_title("cv fold " + str(j), fontsize=9)
fg.suptitle("Lasso Weights")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_LAS.pdf")
fg.show()

fg, axes = plt.subplots(4, 2)
ax01 = [0, 0, 1, 1, 2, 2, 3, 3]
ax02 = [0, 1, 0, 1, 0, 1, 0, 1]
for j in range(0, 8):
    axes[ax01[j], ax02[j]].plot(np.arange(-2000, 2000, 1), ls_results[j]['results']['RID']['weights']['parameters'])
    axes[ax01[j], ax02[j]].label_outer()
    axes[ax01[j], ax02[j]].set_title("cv fold " + str(j), fontsize=9)
fg.suptitle("Ridge Weights")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_RID.pdf")
fg.show()

fg, axes = plt.subplots(4, 2)
ax01 = [0, 0, 1, 1, 2, 2, 3, 3]
ax02 = [0, 1, 0, 1, 0, 1, 0, 1]
for j in range(0, 8):
    axes[ax01[j], ax02[j]].plot(np.arange(-2000, 2000, 1), ls_results[j]['results']['ENET']['weights']['parameters'])
    axes[ax01[j], ax02[j]].label_outer()
    axes[ax01[j], ax02[j]].set_title("cv fold " + str(j), fontsize=9)
fg.suptitle("ElasticNet Weights")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_ENET.pdf")
fg.show()

fg, axes = plt.subplots(4, 2)
ax01 = [0, 0, 1, 1, 2, 2, 3, 3]
ax02 = [0, 1, 0, 1, 0, 1, 0, 1]
for j in range(0, 8):
    axes[ax01[j], ax02[j]].plot(np.arange(-2000, 2000, 1), ls_results[j]['results']['ENET2']['weights']['parameters'])
    axes[ax01[j], ax02[j]].label_outer()
    axes[ax01[j], ax02[j]].set_title("cv fold " + str(j), fontsize=9)
fg.suptitle("ElasticNet(2) Weights")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_ENET2.pdf")
fg.show()

fg, axes = plt.subplots(4, 2)
ax01 = [0, 0, 1, 1, 2, 2, 3, 3]
ax02 = [0, 1, 0, 1, 0, 1, 0, 1]
for j in range(0, 8):
    axes[ax01[j], ax02[j]].plot(np.arange(-2000, 2000, 1), ls_results[j]['results']['PCT']['weights'][0][:, 1])
    axes[ax01[j], ax02[j]].label_outer()
    axes[ax01[j], ax02[j]].set_title("cv fold " + str(j), fontsize=9)
fg.suptitle("Perceptron Weights")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_PCT.pdf")
fg.show()

fg, axes = plt.subplots(4, 2)
ax01 = [0, 0, 1, 1, 2, 2, 3, 3]
ax02 = [0, 1, 0, 1, 0, 1, 0, 1]
for j in range(0, 8):
    axes[ax01[j], ax02[j]].plot(ls_results[j]['results']['CNN1']['weights'][4][:, 1])
    axes[ax01[j], ax02[j]].label_outer()
    axes[ax01[j], ax02[j]].set_title("cv fold " + str(j), fontsize=9)
fg.suptitle("Convolutional neural network (1) last layer weigths")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_CNN1.pdf")
fg.show()

color_ls = {'SNR': 'red', 'SNR_auto': 'brown', 'PCT': 'lightblue', 'DNN': 'blue', 'CNN1': 'navy', 'CNN2': 'purple',
            'ENET': 'forestgreen', 'RID': 'lime', 'LAS': 'lightgreen', 'RF': 'yellow', 'XGB': 'orange',
            'ENET2': 'darkgreen'}

fg, axes = plt.subplots(3, 2)
ax01 = [0, 0, 1, 1, 2, 2]
ax02 = [0, 1, 0, 1, 0, 1]

axes[ax01[0], ax02[0]].plot(np.arange(-2000, 2000, 1), ls_results[0]['results']['RF']['weights'], color=color_ls['RF'])
axes[ax01[0], ax02[0]].label_outer()
axes[ax01[0], ax02[0]].set_title("Weights Random Forest", fontsize=9)

axes[ax01[1], ax02[1]].plot(np.arange(-2000, 2000, 1), ls_results[0]['results']['LAS']['weights']['parameters'],
                            color=color_ls['LAS'])
axes[ax01[1], ax02[1]].label_outer()
axes[ax01[1], ax02[1]].set_title("Weights Lasso", fontsize=9)

axes[ax01[2], ax02[2]].plot(np.arange(-2000, 2000, 1), ls_results[0]['results']['RID']['weights']['parameters'],
                            color=color_ls['RID'])
axes[ax01[2], ax02[2]].label_outer()
axes[ax01[2], ax02[2]].set_title("Weights Ridge", fontsize=9)

axes[ax01[3], ax02[3]].plot(np.arange(-2000, 2000, 1), ls_results[7]['results']['ENET2']['weights']['parameters'],
                            color=color_ls['ENET'])
# axes[ax01[3], ax02[3]].label_outer()
axes[ax01[3], ax02[3]].set_title("Weights ElasticNet", fontsize=9)

axes[ax01[4], ax02[4]].plot(np.arange(-2000, 2000, 1), ls_results[2]['results']['PCT']['weights'][0][:, 1],
                            color=color_ls['PCT'])
axes[ax01[4], ax02[4]].label_outer()
axes[ax01[4], ax02[4]].set_title("Weights Perceptron", fontsize=9)

axes[ax01[5], ax02[5]].plot(ls_results[1]['results']['CNN1']['weights'][4][:, 1], color=color_ls['CNN1'])
axes[ax01[5], ax02[5]].label_outer()
axes[ax01[5], ax02[5]].set_title("Weights Conv. Neural Net", fontsize=9)

fg.suptitle("Model Weights")
fg.tight_layout()
fg.savefig(csv_res_path + planet + "_final_plots_CV/weights_models.pdf")
fg.show()

# =============================================================================
#
# fg, axes = plt.subplots(4,2)
# ax01 = [0,0,1,1,2,2,3,3]
# ax02 = [0,1,0,1,0,1,0,1]
# for j in range(0,8):
#     axes[ax01[j], ax02[j]].plot(ls_results[j]['results']['CNN2']['weights'][11][:,1])
# fg.tight_layout()
# fg.suptitle("Convolutional neural network (2) - last layer weights")
# fg.show()
#
# =============================================================================


snry = pd.read_csv(csv_res_path + planet + "_final_plots_CV/HatVSPred_SNR.csv")

from sklearn.neighbors import KernelDensity
import numpy as np

#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(snry['Y_prob'])

#snry['Y_prob'][snry['Y_true'] == 1]
#kde.score_samples(snry['Y_prob'][snry['Y_true'] == 1])
#hist = plt.hist(snry['Y_prob'][snry['Y_true'] == 1])

# matplotlib
#inline
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
x1 = np.array(snry['Y_prob'][snry['Y_true'] == 1])
x2 = np.array(snry['Y_prob'][snry['Y_true'] == 0])
kwargs = dict(histtype='stepfilled', alpha=0.3, bins=50)
plt.hist(x1, **kwargs, color='steelblue')
plt.hist(x2, **kwargs, color='indianred')
mylegends = [Line2D([0], [0], color='indianred', lw=1), Line2D([0], [0], color='steelblue', lw=1)]
plt.legend(mylegends, ['H2O not in spectrum', 'H2O in spectrum'])
plt.title("Distribution of signal-to-noise ratios in CCF at RV=0")
plt.savefig(csv_res_path + planet + "_final_plots_CV/distribution_signals.pdf")
plt.show()

##=================================================================================================================


## LIBRARIES

import pandas as pd
import numpy as np
import random
import sys
import pickle
import os
from matplotlib.lines import Line2D


#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.crosscorrNormVec import crosscorrRV_vec
from ml_spectroscopy.config import path_init

import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import time
import concurrent.futures
import pandas as pd

from ml_spectroscopy.plottings_utils import ROC_curve_plot, ROC_curve_saveplt, PR_curve_plot, PR_curve_saveplt



## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
visual_path = subdir + "80_visualisation/"



## SET SEED FOR REPRODUCIBILITY
random.seed(100)


## IMPORT DATA
## start with only trimmed data. Later can compare just the neural network between padded and trimmed data, using Planet_Signals[data.sum(axis=0)==0,:]
#Planet_Signals = pd.read_csv(data_path + "csv_inputs/Planet_Signals_df.csv", index_col=0)
# SETTINGS

data_name='GQlupb'
template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}

alpha=10
beta=50
v=3

planet='GQlupB'
#data1=pd.read_pickle(data_path+'data_4ml/v2_ccf_4ml_trim_robustness/H2O_'+data_name+'_scale'+str(alpha)+'_temp1200_sg4.1_ccf_4ml_trim_norepetition.pkl')
data1 = pd.read_pickle(data_path + 'data_4ml/v4_ccf_4ml_trim_robustness/H2O_' + data_name + '_scale' + str(
    alpha) + '_bal' + str(beta) + '_temp1200.0_sg2.9_co0.3_fe-0.3_ccf_4ml_trim_norepetition_v' + str(v) + '.pkl')

i=0
j=1
data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]


X_train=data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_train=data_train['H2O']


X_valid=data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_valid=data_valid['H2O']


X_test=data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_test=data_test['H2O']

        

# import results from CV and store information as plots 

dir_path = results_path + "CV_results/"
ls_data = os.listdir(dir_path)
len_folds=len(ls_data)
result_names=[ls_data[n][:-4] for n in range(len_folds)]

keys = result_names
ls_results = {key: None for key in keys}

for i in range(0,len_folds):
    with open(results_path+"CV_results/"+str(ls_data[i]), "rb") as f:
        ls_results[i] = pickle.load(f) # i is the validation number but the proper set is at i+1






#fig_size = [6.4, 4.8]
#f = pyplot.figure()


# Plot out the ROC curves
plt.style.use('seaborn')
#ls_results[i]['Y_test']
alpha=10
ax2=[0,1,2,0,1,2,0,1,2]
ax1=[0,0,0,1,1,1,2,2,2]
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.suptitle('ROC curves for $\\alpha='+str(alpha)+'$', fontsize=14)


for i in range(0,len_folds+1):
    
    if i<len_folds:
        try:
            if i==7:
                j=0
            else:
                j=i+1
            
            data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
            data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
            data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
            
            X_train=data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train=data_train['H2O']
            
            X_valid=data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid=data_valid['H2O']
            
            X_test=data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_test=data_test['H2O']
            
            df_roc, ax = ROC_curve_plot(ls_results[i], axes, ax1[i], ax2[i], Y_test, i, path=visual_path)
            #ax.label_outer()   
        except KeyError:
            pass  
    else:
        axes[ax1[i], ax2[i]].axis('off')
    
for i in range(0,len_folds+1):  
    axes[ax1[i], ax2[i]].label_outer() 
    
    
    if i==len_folds:
        axes[ax1[i], ax2[i]].axis('off')
        mylegends = [Line2D([0], [0], linestyle='--', color='gray', lw=1),
                     Line2D([0], [0], color='purple', lw=1),
                     Line2D([0], [0], color='blue', lw=1),
                     Line2D([0], [0], color='orange', lw=1),
                     Line2D([0], [0], color='red', lw=1),
                     Line2D([0], [0], color='brown', lw=1)]
        
        axes[ax1[i], ax2[i]].legend(mylegends, ['No skill', 'PCT', 'DNN', 'RF', 'SNR', 'SNR auto'], loc='lower left', bbox_to_anchor=(0.05,-0.1), fontsize=11) 

fig.savefig(visual_path+planet+'_plt_CV_results/ROC_'+planet+'_combined_CV2.pdf')
fig.savefig(visual_path+planet+'_plt_CV_results/ROC_'+planet+'_combined_CV2.png')




for i in range(0,len_folds):
    try: 
        if i==7:
            j=0
        else:
            j=i+1
        data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
        data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
        data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
        
        X_train=data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_train=data_train['H2O']
        
        X_valid=data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_valid=data_valid['H2O']
        
        X_test=data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_test=data_test['H2O']
            
        
        ROC_curve_saveplt(ls_results[i], Y_test, i, path=visual_path)
    except KeyError:
        pass




########### Precision Recall curves

# Plot out the ROC curves
plt.style.use('seaborn')
#ls_results[i]['Y_test']
alpha = 10
beta = 50
v = 3
ax2=[0,1,2,0,1,2,0,1,2]
ax1=[0,0,0,1,1,1,2,2,2]
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.suptitle('ROC curves for $\\alpha='+str(alpha)+'$', fontsize=14)


for i in range(0,len_folds+1):
    
    if i<len_folds:
        try:
        
            if i==7:
                j=0
            else:
                j=i+1
                
            data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
            data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
            data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
            
            X_train=data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train=data_train['H2O']
            
            X_valid=data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid=data_valid['H2O']
            
            X_test=data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_test=data_test['H2O']
            
            df_pr, ax = PR_curve_plot(ls_results[i], axes, ax1[i], ax2[i], Y_test, i, path=visual_path)
            
        except KeyError:
            pass  
    else:
        axes[ax1[i], ax2[i]].axis('off')


for i in range(0,len_folds+1):  
    axes[ax1[i], ax2[i]].label_outer() 
    
    if i==len_folds:
        axes[ax1[i], ax2[i]].axis('off')
        mylegends = [Line2D([0], [0], linestyle='--', color='gray', lw=1),
                     Line2D([0], [0], color='purple', lw=1),
                     Line2D([0], [0], color='blue', lw=1),
                     Line2D([0], [0], color='orange', lw=1),
                     Line2D([0], [0], color='red', lw=1),
                     Line2D([0], [0], color='brown', lw=1)]
        
        axes[ax1[i], ax2[i]].legend(mylegends, ['No skill', 'PCT', 'DNN', 'RF', 'SNR', 'SNR auto'], loc='lower left', bbox_to_anchor=(0.05,-0.1), fontsize=11) 

fig.savefig(visual_path+planet+'_plt_CV_results/PR_'+planet+'_combined_CV.pdf')
fig.savefig(visual_path+planet+'_plt_CV_results/PR_'+planet+'_combined_CV.png')





#res, f, ax1, ax2, Y_test, i, data_name='GQlupB', path, alpha=10





for i in range(0,len_folds):
    try: 
        
        if i==7:
            j=0
        else:
            j=i+1
            
        data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
        data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
        data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
        
        X_train=data_train.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_train=data_train['H2O']
        
        X_valid=data_valid.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_valid=data_valid['H2O']
        
        X_test=data_test.drop(['tempP', 'loggP','CO_ratio', 'Fe','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_test=data_test['H2O']
        
        PR_curve_saveplt(ls_results[i], Y_test, i, path=visual_path)
    except KeyError:
        pass

