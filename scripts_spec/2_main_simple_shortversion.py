# -*- coding: utf-8 -*-
"""
Created on Mon Jan 3 14:20:58 2021

@author: emily

This code applies cross-validation
"""
## LIBRARIES
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import pickle
import time
import gc
import sys
from ml_spectroscopy.config import path_init, global_settings
from ml_spectroscopy.utility_functions import test_onCCF_rv0_SNR
from ml_spectroscopy.modelsML_utils import PCT_model, SNR_grid_search, CNN1_model, CNN2_model
from ml_spectroscopy.hpt_GA import genetic_algorithm, decode
from keras.utils.np_utils import to_categorical

# Set GPU and CV fold number
gpu=int(sys.argv[1])

if len(sys.argv)<3:
    j = gpu
elif len(sys.argv)==3:
    j=int(sys.argv[2])

print("gpu:"+str(gpu))
print("fold:"+str(j))

## ACTIVE SUBDIR
subdir = path_init()
gs = global_settings()
# subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

## PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
visual_path = subdir + "80_visualisation/"


## ENVIRONMENT VARIABLES

os.environ['TF_NUM_INTEROP_THREADS'] = "32"
os.environ['TF_NUM_INTRAOP_THREADS'] = "32"
os.environ['OMP_NUM_THREADS'] = "32"
os.environ['OPENBLAS_NUM_THREADS'] = "32"
os.environ['MKL_NUM_THREADS'] = "32"
os.environ['VECLIB_MAXIMUM_THREADS'] = "32"
os.environ['NUMEXPR_NUM_THREADS'] = "32"
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
## SET SEED
np.random.seed(100)

## SETTINGS (planet and main template)

# To adapt
mol = 'H2O'
version = 8 #6 base tests, 7 CO tests , 8 tests withgaussian noise, 9, multi tests
alpha_vec = [8] #8,11,16,21,29,41,67,5000 #2,5,11,21,67

# Do not change
data_name = 'GQlupb' # Name of the companion noise
x = 0  # first data set used for general analysis
bal = 50



if mol == 'H2O':
    template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0} # Baseline template characteristics (SNR)
elif mol == 'CO':
    template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 0, 'CO': 1} # Baseline template characteristics (SNR)





## CODE

for alpha in alpha_vec:
    # time it
    start_global = time.time()

    # valid fold:
    if j == 0:
        i = 7
    else:
        i = j - 1

    ## IMPORT DATA  (can be loops or parallellized tasks etc)

    data1 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp1200.0_sg4.1_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data2 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp1400.0_sg4.1_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data3 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp1600.0_sg4.1_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data4 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp2000.0_sg4.1_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data5 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp2500.0_sg4.1_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data6 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp1600.0_sg2.9_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data7 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp1600.0_sg3.5_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data8 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp1600.0_sg4.5_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')
    data9 = pd.read_pickle(data_path + 'data_4ml/v'+str(version)+'_ccf_4ml_trim_robustness_simple/' + mol + '_' + data_name + '_scale' + str(
        alpha) + '_bal' + str(bal) + '_temp1600.0_sg5.3_ccf_4ml_trim_norepetition_v'+str(version)+'_simple.pkl')

    datas = {0: data1, 1: data2, 2: data3, 3: data4, 4: data5, 5: data6, 6: data7, 7: data8, 8: data9}
    methods = ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID', 'LAS', 'RF', 'XGB', 'ENET2']
    len_index = len(data1[0].index.levels[0])

    X_train = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
    X_valid = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
    X_test = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}

    y_train = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
    y_valid = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
    y_test = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}

    for d in datas:
        data_train = datas[d].drop([(str(datas[d].index.levels[0][j]),)], axis=0).drop(
            [(str(datas[d].index.levels[0][i]),)], axis=0)
        data_valid = datas[d].loc[(str(datas[d].index.levels[0][i]), slice(None)), :]
        data_test = datas[d].loc[(str(datas[d].index.levels[0][j]), slice(None)), :]

        X_train[d] = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_train = data_train[mol]
        # y_train[d] = Y_train

        X_valid[d] = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_valid = data_valid[mol]
        # y_valid[d] = Y_valid

        X_test[d] = data_test.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_test = data_test[mol]
        # y_test[d] = Y_test

        y_train[d] = to_categorical(Y_train)
        y_valid[d] = to_categorical(Y_valid)
        y_test[d] = to_categorical(Y_test)

    # Data stack for CNN
    x_train = np.stack((np.array((X_train[0])), np.array((X_train[1])), np.array((X_train[2])), np.array((X_train[3])),
                        np.array((X_train[4])), np.array((X_train[5])), np.array((X_train[6])), np.array((X_train[7])),
                        np.array((X_train[8]))), axis=-1)
    x_valid = np.stack((np.array((X_valid[0])), np.array((X_valid[1])), np.array((X_valid[2])), np.array((X_valid[3])),
                        np.array((X_valid[4])), np.array((X_valid[5])), np.array((X_valid[6])), np.array((X_valid[7])),
                        np.array((X_valid[8]))), axis=-1)
    x_test = np.stack((np.array((X_test[0])), np.array((X_test[1])), np.array((X_test[2])), np.array((X_test[3])),
                       np.array((X_test[4])), np.array((X_test[5])), np.array((X_test[6])), np.array((X_test[7])),
                       np.array((X_test[8]))), axis=-1)

    if (y_train[0] == y_train[1]).all():
        y_train = y_train[0]

    if (y_valid[0] == y_valid[1]).all():
        y_valid = y_valid[0]

    if (y_test[0] == y_test[1]).all():
        y_test = y_test[0]


    del data1, data2, data3, data4, data5, data6, data7, data8, data9
    gc.collect()

    # Prepare the results storage:
    res_opt_method = {key: None for key in methods}
    CM_opt = {key: None for key in methods}
    hyperparam_optim = {key: None for key in methods[:-2]}
    optim_results = {key: None for key in methods}



    # set up optimizer:
    # define the total iterations
    n_iter = 10  # 20
    # bits per variable
    n_bits = 16
    # define the population size
    n_pop = 8  # 10
    # crossover rate
    r_cross = 0.9

    # =============================================================================
    # CNN 1
    # =============================================================================

    ## optim:
    start_o_CNN1 = time.time()
    # define range for input
    # bounds_DNN = [[16, 256], [100, 200], [0.0001, 0.01], [0.1,0.9], [0.1,0.9], [0.1,0.8], [1,5]]
    bounds_CNN1 = [[16, 64], [100, 200], [0.0001, 0.01], [0.1, 0.9], [2, 8], [2, 3]]
    # mutation rate
    r_mut_CNN1 = 1.0 / (float(n_bits) * len(bounds_CNN1))
    # perform the genetic algorithm search
    best_CNN1, score_CNN1, best_accuracies_valid_CNN1, best_accuracies_train_CNN1, track_generation_CNN1, track_hyperparams_CNN1 = genetic_algorithm(
        CNN1_model, bounds_CNN1, n_bits, n_iter, n_pop, r_cross, r_mut_CNN1, x_train, y_train, x_valid, y_valid)
    decoded_CNN1 = decode(bounds_CNN1, n_bits, best_CNN1)
    # end time
    end_o_CNN1 = time.time()

    # test model:
    start_m_CNN1 = time.time()
    res_opt_method['CNN1'] = CNN1_model(decoded_CNN1, x_train, y_train, x_test, y_test)
    end_m_CNN1 = time.time()

    CM_opt['CNN1'] = np.array((confusion_matrix(y_test[:, 1], res_opt_method['CNN1']['Y_pred']).ravel()))
    hyperparam_optim['CNN1'] = decoded_CNN1
    # optimization results



    optim_results['CNN1'] = {'best_accuracy_valid': best_accuracies_valid_CNN1,
                             'best_accuracy_train': best_accuracies_train_CNN1, 'generation': track_generation_CNN1,
                             'hyperparams': track_hyperparams_CNN1, 'runtime_GA':(end_o_CNN1-start_o_CNN1), 'runtime_model':(end_m_CNN1-start_m_CNN1)}
    print("CNN1 completed")

    # =============================================================================
    # CNN 2
    # =============================================================================

    # optim:
    start_o_CNN2 = time.time()
    # define range for input
    # bounds_DNN = [[16, 256], [100, 200], [0.0001, 0.01], [0.1,0.9], [0.1,0.9], [0.1,0.8], [1,5]]
    bounds_CNN2 = [[16, 64], [100, 200], [0.0001, 0.01], [0.1, 0.9], [2, 8], [2, 3], [0.1, 0.9], [0.1, 0.8], [0.1, 0.8],
                   [0.1, 5]]
    # mutation rate
    r_mut_CNN2 = 1.0 / (float(n_bits) * len(bounds_CNN2))
    # perform the genetic algorithm search
    best_CNN2, score_CNN2, best_accuracies_valid_CNN2, best_accuracies_train_CNN2, track_generation_CNN2, track_hyperparams_CNN2 = genetic_algorithm(
        CNN2_model, bounds_CNN2, n_bits, n_iter, n_pop, r_cross, r_mut_CNN2, x_train, y_train, x_valid, y_valid)
    decoded_CNN2 = decode(bounds_CNN2, n_bits, best_CNN2)
    # end time
    end_o_CNN2 = time.time()

    # test model:
    start_m_CNN2 = time.time()
    res_opt_method['CNN2'] = CNN2_model(decoded_CNN2, x_train, y_train, x_test, y_test)
    end_m_CNN2 = time.time()

    CM_opt['CNN2'] = np.array((confusion_matrix(y_test[:, 1], res_opt_method['CNN2']['Y_pred']).ravel()))
    hyperparam_optim['CNN2'] = decoded_CNN2
    # optimization results
    optim_results['CNN2'] = {'best_accuracy_valid': best_accuracies_valid_CNN2,
                             'best_accuracy_train': best_accuracies_train_CNN2, 'generation': track_generation_CNN2,
                             'hyperparams': track_hyperparams_CNN2, 'runtime_GA':(end_o_CNN2-start_o_CNN2), 'runtime_model':(end_m_CNN2-start_m_CNN2)}

    print("CNN2 completed")

    # =============================================================================
    #  Usual SNR
    # =============================================================================

    print("run SNR")

    #optim:
    start_o_SNR = time.time()
    hyperparams_SNR = np.arange(-10, 30, 0.2)
    SNR_best = SNR_grid_search(hyperparams_SNR, test_onCCF_rv0_SNR, X_train[x], y_train[:, 1])
    # end time
    end_o_SNR = time.time()
    # results:
    start_m_SNR = time.time()
    res_opt_method['SNR'] = test_onCCF_rv0_SNR(X_test[x], SNR_best['optimal_hyperparam'])
    end_m_SNR = time.time()

    CM_opt['SNR'] = np.array((confusion_matrix(y_test[:, 1], res_opt_method['SNR']['Y_pred']).ravel()))
    accuracy_SNR = sum(res_opt_method['SNR']['Y_pred'] == y_test[:, 1]) / len(y_test[:, 1])
    # optimization results
    optim_results['SNR'] = {'best_accuracy_valid': accuracy_SNR, 'hyperparams': SNR_best['optimal_hyperparam'], 'runtime_GA':(end_o_SNR-start_o_SNR), 'runtime_model':(end_m_SNR-start_m_SNR)}

    print("SNR completed")

    # =============================================================================
    # Neural network with 1 softmax layer (linear model)
    # =============================================================================
    # optim:
    start_o_PCT = time.time()
    # define range for input
    bounds_PCT = [[16, 256], [100, 200]]
    # mutation rate
    r_mut_PCT = 1.0 / (float(n_bits) * len(bounds_PCT))
    # perform the genetic algorithm search
    best_PCT, score_PCT, best_accuracies_valid_PCT, best_accuracies_train_PCT, track_generation_PCT, track_hyperparams_PCT = genetic_algorithm(
        PCT_model, bounds_PCT, n_bits, n_iter, n_pop, r_cross, r_mut_PCT, X_train[x], y_train, X_valid[x], y_valid)
    decoded_PCT = decode(bounds_PCT, n_bits, best_PCT)
    end_o_PCT = time.time()

    # test model:
    start_m_PCT = time.time()
    res_opt_method['PCT'] = PCT_model(decoded_PCT, X_train[x], y_train, X_test[x], y_test)
    end_m_PCT = time.time()

    CM_opt['PCT'] = np.array((confusion_matrix(y_test[:, 1], res_opt_method['PCT']['Y_pred']).ravel()))
    hyperparam_optim['PCT'] = decoded_PCT
    # optimization results
    optim_results['PCT'] = {'best_accuracy_valid': best_accuracies_valid_PCT,
                            'best_accuracy_train': best_accuracies_train_PCT, 'generation': track_generation_PCT,
                            'hyperparams': track_hyperparams_PCT, 'runtime_GA':(end_o_PCT-start_o_PCT), 'runtime_model':(end_m_PCT-start_m_PCT)}

    print("PCT completed")


    ## SAVE RESULTS

    # use a dictionary with keys using the right names.
    results = {'results': res_opt_method, 'confusion matrix': CM_opt, 'hyperparameters': hyperparam_optim,
               'y_test': list(y_test)}
    a_file = open(results_path + "export_CV/from_GPU_byfold/results_" + mol +"_" +str(data_name)+"_data_"+str(x)+"_alpha_"+str(alpha)+"_CV_testfold" + str(j) + ".pkl", "wb")
    pickle.dump(results, a_file)
    a_file.close()

    # export optimizaiton results
    a_file = open(results_path + "export_CV/from_GPU_byfold/GA_results_" + mol +"_" + str(data_name)+"_data_"+str(x)+"_alpha_"+str(alpha)+"_CV_testfold" + str(j) + ".pkl", "wb")
    pickle.dump(optim_results, a_file)
    a_file.close()

    ## OUT
    end_global = time.time()
    print('terminated cv ' + str(j) + ", Runtime:  " + str((end_global - start_global) / 60) + " minutes")


