# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 22:07:53 2022

@author: emily

results
"""

## LIBRARIES

import random
import pickle

# sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
# sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.config import path_init
from ml_spectroscopy.utility_functions import Average

import matplotlib.pyplot as plt

# from ml_spectroscopy.plottings_utils_results import ROC_curve_plot, ROC_curve_saveplt, PR_curve_plot, PR_curve_saveplt

from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

## ACTIVE SUBDIR
subdir = path_init()
# subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"  # Depending on the way we pick the files, can't we directly point to the right directory from the start?
visual_path = subdir + "80_visualisation/Results_per_alpha/Vis_00_090223_alphas_simple_T1200SG41/"
csv_res_path = subdir + "80_visualisation/Results_per_alpha/Vis_00_090223_alphas_simple_T1200SG41/"

## SET SEED FOR REPRODUCIBILITY
random.seed(100)

## Settings
data_name = 'GQlupb'
planet = 'GQlupB'
alpha_vals=[0,2,5,8,11,16,21,29,41,67,5000]
#alpha=2
bal=50
v=6
frame='simple'
len_folds = 8
alpha_vals_subset = alpha_vals[1:-1]


#plotname = 'test'
methods_ls = ['SNR', 'SNR_auto', 'RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2', 'PCT', 'DNN', 'CNN1', 'CNN2']
color_ls = {'SNR': 'red', 'SNR_auto': 'brown', 'PCT': 'lightblue', 'DNN': 'blue', 'CNN1': 'navy', 'CNN2': 'purple',
            'ENET': 'forestgreen', 'RID': 'lime', 'LAS': 'lightgreen', 'RF': 'yellow', 'XGB': 'orange',
            'ENET2': 'darkgreen'}


## IMPORT DATA
## start with only trimmed data. Later can compare just the neural network between padded and trimmed data, using Planet_Signals[data.sum(axis=0)==0,:]
# Planet_Signals = pd.read_csv(data_path + "csv_inputs/Planet_Signals_df.csv", index_col=0)
# SETTINGS

data_name = 'GQlupb'
planet = 'GQlupB'
dir_path = results_path + "export_CV/from_GPU_byfold/Res_00_090223_alphas_simple_T1200SG41/"
#methods = ['SNR', 'SNR_auto', 'RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2', 'PCT', 'DNN', 'CNN1', 'CNN2']
methods = ['SNR', 'SNR_auto', 'RID', 'ENET', 'DNN', 'PCT', 'CNN1', 'CNN2']

template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}

for alpha in alpha_vals:

    data1 = pd.read_pickle(
        data_path + 'data_4ml/v' + str(v) + '_ccf_4ml_trim_robustness_simple/H2O_' + data_name + '_scale' + str(
            alpha) + '_bal' + str(bal) + '_temp1200.0_sg4.1_ccf_4ml_trim_norepetition_v' + str(v) + '_simple.pkl')


    ls_data = ['results_GQlupb_data_0_alpha_' + str(alpha) + '_CV_testfold' + str(i) + '.pkl' for i in range(len_folds)]
    result_names = [ls_data[n][:-4] for n in range(len_folds)]
    keys = result_names
    ls_results = {key: None for key in keys}

    for i in range(0, len_folds):
        with open(dir_path + str(ls_data[i]), "rb") as f:
            ls_results[i] = pickle.load(f)  # i is the validation number but the proper set is at i+1


    accuracies_all = []
    ROC_all = []
    PR_all = []
    for m in methods:
        dt_temp = np.zeros((8, 5))
        accuracies_method = []
        ROC_method = []
        PR_method = []

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

                    X_train = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_train = data_train['H2O']

                    X_valid = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_valid = data_valid['H2O']

                    X_test = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                    Y_test = data_test['H2O']

                    if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                        accuracy_test = ls_results[j]['results'][m]['accuracy_valid_test']
                        accuracy_train = ls_results[i]['results'][m]['accuracy_train']
                        auc_ROC = roc_auc_score(data_test['H2O'], ls_results[j]['results'][m]['Y_pred_prob'])
                        lr_precision, lr_recall, _ = precision_recall_curve(data_test['H2O'],
                                                                            ls_results[j]['results'][m]['Y_pred_prob'])

                    elif m in ['SNR', 'SNR_auto']:
                        accuracy_test = sum(data_test['H2O'] == ls_results[j]['results'][m]['Y_pred']) / len(data_test['H2O'])
                        accuracy_train = np.nan
                        auc_ROC = roc_auc_score(data_test['H2O'], ls_results[j]['results'][m]['SNR'])
                        lr_precision, lr_recall, _ = precision_recall_curve(data_test['H2O'],
                                                                            ls_results[j]['results'][m]['SNR'])
                    else:
                        accuracy_test = ls_results[j]['results'][m]['accuracy_valid_test']
                        accuracy_train = ls_results[i]['results'][m]['accuracy_train']
                        auc_ROC = roc_auc_score(data_test['H2O'], ls_results[j]['results'][m]['Y_pred_prob'][:, 1])
                        lr_precision, lr_recall, _ = precision_recall_curve(data_test['H2O'],ls_results[j]['results'][m]['Y_pred_prob'][:,1])

                    lr_f1, lr_auc = f1_score(data_test['H2O'], ls_results[j]['results'][m]['Y_pred']), auc(lr_recall,lr_precision)
                    # ax.label_outer()

                    dt_temp[j, :] = [accuracy_train, accuracy_test, auc_ROC, lr_auc, lr_f1]

                    accuracies_method.append(accuracy_test)
                    ROC_method.append(auc_ROC)
                    PR_method.append(lr_auc)


                except KeyError:
                    pass

        dt = pd.DataFrame(dt_temp)
        dt.columns = ['acc_train', 'acc_test', 'auc_roc', 'auc_lr', 'f1']
        dt.to_csv(csv_res_path + "Summary_Table_alpha_"+str(alpha)+"_" + str(m) + ".csv")
        accuracies_all.append(Average(accuracies_method))
        ROC_all.append(Average(ROC_method))
        PR_all.append(Average(PR_method))

    ###############################################################################

    plt.style.use('seaborn')
    bars1 = plt.bar(x=methods, height=accuracies_all, width=0.7, color='indianred')
    plt.ylim([0.5, 1])
    plt.title('Model accuracy')
    plt.savefig(csv_res_path + 'Summary_Barplot_ACC_alpha'+str(alpha)+'.pdf')
    plt.close()

    # fig.suptitle('Areas under curves')

    bars2 = plt.bar(x=methods, height=ROC_all, width=0.7, color='purple')
    plt.ylim([0.5, 1])
    plt.title('Receiver Operating Characteristic AUC')
    plt.savefig(csv_res_path + 'Summary_Barplot_ROC_alpha'+str(alpha)+'.pdf')
    plt.close()


    bars3 = plt.bar(x=methods, height=PR_all, width=0.7, color='royalblue')
    bars3.ylim([0.5, 1])
    bars3.title('Precision-Recall AUC')
    bars3.savefig(csv_res_path + 'Summary_Barplot_PR_alpha'+str(alpha)+'.pdf')

