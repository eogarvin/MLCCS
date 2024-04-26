# -*- coding: utf-8 -*-
"""
Created on Mon Jan 3 14:20:58 2021

@author: emily

Cross validation test fold 0
"""


## LIBRARIES
import pandas as pd
import numpy as np
import sys, os, pickle, time, gc
from sklearn.metrics import confusion_matrix
from ml_spectroscopy.config import path_init, global_settings
from ml_spectroscopy.utility_functions import test_onCCF_rv0_SNR
from ml_spectroscopy.modelsML_utils import  PCT_model,  SNR_grid_search, CNN1_model, CNN2_model
from ml_spectroscopy.hyperparam_tuning_GA import genetic_algorithm, decode
from keras.utils.np_utils import to_categorical



#define the molecule and working version
mol = "CO"
version = 7
alpha_set = ["alphaover6", "alphamin"]
#alpha_set = ["alphanone", "alphanone2", "alphaover2", "alphaover3", "alphaover4", "alphaover6", "alphamin"]

# Set GPU and CV fold number


gpu=int(sys.argv[1])

if len(sys.argv)<3:
    scaling = gpu
elif len(sys.argv)==3:
    scaling=int(sys.argv[2])


print("gpu:"+str(gpu))
print("scaling:"+alpha_set[scaling])
print("molecule:"+str(mol))


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
data_name = 'PZTelb'
template_characteristics = {'Temp': 2800, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}



#x = 0  # first data set for general analysis

# set test fold:
x = gs[0]  # first data set for general analysis
alpha_vec = [2] #[11,22] add 2,5,8 # [2,5,8,11,22,32,43,55]
#bal = 50

if mol == "H2O":
    a0 = 1
    b0 = 0
elif mol == "CO":
    a0 = 0
    b0 = 1

## CODE

for alpha in alpha_vec:
    # time it
    start_global = time.time()

    i=[0,1,2,3,4,5,8,9,10,15,16,17]
    j=[6,7,11,18]
    k1=[12,13,14]
    k2=[0,1,2] # For the test set, because we only have 7 datasets for PZ tel B in bad conditions. so 0,1,2, corresponds in fact to 12,13,14


    ## IMPORT DATA  (can be loops or parallellized tasks etc) - the method is trained with a constant alpha of 5
    data1 = pd.read_pickle(data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_'+str(mol)+'_crosscorr_data_alpha_' + str(alpha) + '_temp2800.0_sg4.1.pkl')
    data2 = pd.read_pickle(data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_'+str(mol)+'_crosscorr_data_alpha_' + str(alpha) + '_temp2500.0_sg4.1.pkl')
    data3 = pd.read_pickle(data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_'+str(mol)+'_crosscorr_data_alpha_' + str(alpha) + '_temp3100.0_sg4.1.pkl')
    data4 = pd.read_pickle(data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_'+str(mol)+'_crosscorr_data_alpha_' + str(alpha) + '_temp2700.0_sg3.7.pkl')
    data5 = pd.read_pickle(data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_'+str(mol)+'_crosscorr_data_alpha_' + str(alpha) + '_temp2900.0_sg4.3.pkl')

    # Need real PZ Tel B as test data

    datas = {0: data1, 1: data2, 2: data3, 3: data4, 4: data5}
    methods = ['SNR', 'PCT', 'CNN1', 'CNN2']
    len_index = len(data1[0].index.levels[0])

    del data1, data2, data3, data4, data5
    gc.collect()


    ## IMPORT DATA  (can be loops or parallellized tasks etc)

    ## IMPORT DATA  (can be loops or parallellized tasks etc)
    data1 = pd.read_pickle(data_path + 'csv_inputs/intro_plot_dataset/PZ_Tel_signals_'+str(alpha_set[scaling])+'_temp2800_logg_4.1_H2O_'+str(a0)+'_CO_'+str(b0)+'.pkl')
    # Other datasets for CNN
    data2 = pd.read_pickle(data_path + 'csv_inputs/intro_plot_dataset/PZ_Tel_signals_'+str(alpha_set[scaling])+'_temp2500_logg_4.1_H2O_'+str(a0)+'_CO_'+str(b0)+'.pkl')
    data3 = pd.read_pickle(data_path + 'csv_inputs/intro_plot_dataset/PZ_Tel_signals_'+str(alpha_set[scaling])+'_temp3100_logg_4.1_H2O_'+str(a0)+'_CO_'+str(b0)+'.pkl')
    data4 = pd.read_pickle(data_path + 'csv_inputs/intro_plot_dataset/PZ_Tel_signals_'+str(alpha_set[scaling])+'_temp2700_logg_3.7_H2O_'+str(a0)+'_CO_'+str(b0)+'.pkl')
    data5 = pd.read_pickle(data_path + 'csv_inputs/intro_plot_dataset/PZ_Tel_signals_'+str(alpha_set[scaling])+'_temp2900_logg_4.3_H2O_'+str(a0)+'_CO_'+str(b0)+'.pkl')
    # Other datasets for CNN

    datas_test = {0: data1, 1: data2, 2: data3, 3: data4, 4: data5}
    methods = ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID', 'LAS', 'RF', 'XGB', 'ENET2']
    #len_index = len(data1[0].index.levels[0])

    del data1, data2, data3, data4, data5
    gc.collect()

    # corresponds to the nu,ber of datasets
    X_train = {0: None, 1: None, 2: None, 3: None, 4: None}#, 5: None, 6: None, 7: None, 8: None}
    X_valid = {0: None, 1: None, 2: None, 3: None, 4: None}#, 5: None, 6: None, 7: None, 8: None}
    X_test = {0: None, 1: None, 2: None, 3: None, 4: None}#, 5: None, 6: None, 7: None, 8: None}

    y_train = {0: None, 1: None, 2: None, 3: None, 4: None}#, 5: None, 6: None, 7: None, 8: None}
    y_valid = {0: None, 1: None, 2: None, 3: None, 4: None}#, 5: None, 6: None, 7: None, 8: None}
    y_test = {0: None, 1: None, 2: None, 3: None, 4: None}#, 5: None, 6: None, 7: None, 8: None}

    for d in datas:
        data_train = datas[d].loc[datas[d].index.levels[0][i]]
        data_valid = datas[d].loc[datas[d].index.levels[0][j]]
        data_test = datas_test[d].loc[datas_test[d].index.levels[0][k2]]

        X_train[d] = data_train.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3'], axis=1)
        Y_train = data_train[mol]
        # y_train[d] = Y_train

        X_valid[d] = data_valid.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3'], axis=1)
        Y_valid = data_valid[mol]
        # y_valid[d] = Y_valid

        X_test[d] = data_test.drop(['planet'], axis=1)
        Y_test = data_test['planet']
        # y_test[d] = Y_test

        y_train[d] = to_categorical(Y_train)
        y_valid[d] = to_categorical(Y_valid)
        y_test[d] = to_categorical((Y_test, abs(Y_test-1)))[0]

    # Data stack for CNN
    x_train = np.stack((np.array((X_train[0])), np.array((X_train[1])), np.array((X_train[2])), np.array((X_train[3])), np.array((X_train[4]))), axis=-1)#, np.array((X_train[5])), np.array((X_train[6])), np.array((X_train[7])), np.array((X_train[8]))), axis=-1)#
    x_valid = np.stack((np.array((X_valid[0])), np.array((X_valid[1])), np.array((X_valid[2])), np.array((X_valid[3])), np.array((X_valid[4]))), axis=-1)#, np.array((X_valid[5])), np.array((X_valid[6])), np.array((X_valid[7])), np.array((X_valid[8]))), axis=-1)
    x_test = np.stack((np.array((X_test[0])), np.array((X_test[1])), np.array((X_test[2])), np.array((X_test[3])), np.array((X_test[4]))), axis=-1)#, np.array((X_test[5])), np.array((X_test[6])), np.array((X_test[7])), np.array((X_test[8]))), axis=-1)


    if (y_train[0] == y_train[1]).all():
        y_train = y_train[0]

    if (y_valid[0] == y_valid[1]).all():
        y_valid = y_valid[0]

    if (y_test[0] == y_test[1]).all():
        y_test = y_test[0]


    del datas, datas_test



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
    bounds_CNN2 = [[16, 64], [100, 200], [0.0001, 0.01], [0.1, 0.9], [2, 8], [2, 3], [0.1, 0.9], [0.1, 0.8], [0.1, 0.8],[0.1, 5]]
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

    # # =============================================================================
    # #  Autocorrelation correction SNR
    # # =============================================================================
    #
    # print("run SNR2")
    # # optim:
    # start_o_SNR2 = time.time()
    # templates = pd.read_csv(data_path + "csv_inputs/Molecular_Templates_df.csv", index_col=0)
    # template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (
    #         templates["loggP"] == template_characteristics['Surf_grav']) & (
    #                                  templates["H2O"] == template_characteristics['H2O']) & (
    #                                  templates["CO"] == template_characteristics['CO'])]
    # hyperparams_SNR_auto = np.arange(-10, 30, 0.2)
    # SNR_auto_best = SNR_auto_grid_search(hyperparams_SNR_auto, template, test_onCCF_rv0_SNR_autocorrel, X_train[x],
    #                                      y_train[:, 1])
    # end_o_SNR2 = time.time()
    #
    # # Results
    # start_m_SNR2 = time.time()
    # res_opt_method['SNR_auto'] = test_onCCF_rv0_SNR_autocorrel(X_test[x], template, SNR_auto_best['optimal_hyperparam'])
    # end_m_SNR2 = time.time()
    #
    # CM_opt['SNR_auto'] = np.array((confusion_matrix(y_test[:, 1], res_opt_method['SNR_auto']['Y_pred']).ravel()))
    # accuracy_SNR_auto = sum(res_opt_method['SNR_auto']['Y_pred'] == y_test[:, 1]) / len(y_test[:, 1])
    # # optimization results
    # optim_results['SNR_auto'] = {'best_accuracy_valid': accuracy_SNR_auto,
    #                              'hyperparams': SNR_auto_best['optimal_hyperparam'], 'runtime_GA':(end_o_SNR2-start_o_SNR2), 'runtime_model':(end_m_SNR2-start_m_SNR2)}
    #
    # print("SNR auto completed")

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
    #
    # # =============================================================================
    # # Neural network with dense layers
    # # =============================================================================
    #
    # # optim:
    # start_o_DNN = time.time()
    # # define range for input
    # # bounds_DNN = [[16, 256], [100, 200], [0.0001, 0.01], [0.1,0.9], [0.1,0.9], [0.1,0.8], [1,5]]
    # bounds_DNN = [[16, 64], [100, 200], [0.0001, 0.01], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.8], [0.1, 5]]
    # # mutation rate
    # r_mut_DNN = 1.0 / (float(n_bits) * len(bounds_DNN))
    # # perform the genetic algorithm search
    # best_DNN, score_DNN, best_accuracies_valid_DNN, best_accuracies_train_DNN, track_generation_DNN, track_hyperparams_DNN = genetic_algorithm(
    #     DNN_model, bounds_DNN, n_bits, n_iter, n_pop, r_cross, r_mut_DNN, X_train[x], y_train, X_valid[x], y_valid)
    # decoded_DNN = decode(bounds_DNN, n_bits, best_DNN)
    # end_o_DNN = time.time()
    #
    # # test model:
    # start_m_DNN = time.time()
    # res_opt_method['DNN'] = DNN_model(decoded_DNN, X_train[x], y_train, X_test[x], y_test)
    # end_m_DNN = time.time()
    #
    # CM_opt['DNN'] = np.array((confusion_matrix(y_test[:, 1], res_opt_method['DNN']['Y_pred']).ravel()))
    # hyperparam_optim['DNN'] = decoded_DNN
    # # optimization results
    # optim_results['DNN'] = {'best_accuracy_valid': best_accuracies_valid_DNN,
    #                         'best_accuracy_train': best_accuracies_train_DNN, 'generation': track_generation_DNN,
    #                         'hyperparams': track_hyperparams_DNN, 'runtime_GA':(end_o_DNN-start_o_DNN), 'runtime_model':(end_m_DNN-start_m_DNN)}
    #
    # print("DNN completed")
    #
    # # =============================================================================
    # # ElasticNet
    # # =============================================================================
    #
    # # optim:
    # start_o_ENET = time.time()
    # # define range for input
    # bounds_ENET = [[0, 0.5], [0.00001, 0.3], [1000, 3000]]
    # # mutation rate
    # r_mut_ENET = 1.0 / (float(n_bits) * len(bounds_ENET))
    # # perform the genetic algorithm search
    # best_ENET, score_ENET, best_accuracies_valid_ENET, best_accuracies_train_ENET, track_generation_ENET, track_hyperparams_ENET = genetic_algorithm(
    #     ElasticNet_model, bounds_ENET, n_bits, n_iter, n_pop, r_cross, r_mut_ENET, X_train[x], y_train, X_valid[x], y_valid)
    # decoded_ENET = decode(bounds_ENET, n_bits, best_ENET)
    # end_o_ENET = time.time()
    #
    # # test model:
    # start_m_ENET = time.time()
    # res_opt_method['ENET'] = ElasticNet_model(decoded_ENET, X_train[x], y_train, X_test[x], y_test)
    # end_m_ENET = time.time()
    #
    # CM_opt['ENET'] = (confusion_matrix(y_test[:, 1], res_opt_method['ENET']['Y_pred']).ravel())
    # hyperparam_optim['ENET'] = (decoded_ENET)
    # # optimization results
    # optim_results['ENET'] = {'best_accuracy_valid': best_accuracies_valid_ENET,
    #                          'best_accuracy_train': best_accuracies_train_ENET, 'generation': track_generation_ENET,
    #                          'hyperparams': track_hyperparams_ENET, 'runtime_GA':(end_o_ENET-start_o_ENET), 'runtime_model':(end_m_ENET-start_m_ENET)}
    #
    # print("ENET completed")
    #
    #
    #
    # # =============================================================================
    # # Ridge
    # # =============================================================================
    #
    # # optim:
    # start_o_RID = time.time()
    # # define range for input
    # bounds_RID = [[0.00001, 0.99], [1000, 3000]]
    # # mutation rate
    # r_mut_RID = 1.0 / (float(n_bits) * len(bounds_RID))
    # # perform the gRIDic algorithm search
    # best_RID, score_RID, best_accuracies_valid_RID, best_accuracies_train_RID, track_generation_RID, track_hyperparams_RID = genetic_algorithm(
    #     Ridge_model, bounds_RID, n_bits, n_iter, n_pop, r_cross, r_mut_RID, X_train[x], y_train, X_valid[x], y_valid)
    # decoded_RID = decode(bounds_RID, n_bits, best_RID)
    # end_o_RID = time.time()
    #
    # # test model:
    # start_m_RID = time.time()
    # res_opt_method['RID'] = Ridge_model(decoded_RID, X_train[x], y_train, X_test[x], y_test)
    # end_m_RID = time.time()
    #
    # CM_opt['RID'] = (confusion_matrix(y_test[:, 1], res_opt_method['RID']['Y_pred']).ravel())
    # hyperparam_optim['RID'] = (decoded_RID)
    # # optimization results
    # optim_results['RID'] = {'best_accuracy_valid': best_accuracies_valid_RID,
    #                          'best_accuracy_train': best_accuracies_train_RID, 'generation': track_generation_RID,
    #                          'hyperparams': track_hyperparams_RID, 'runtime_GA':(end_o_RID-start_o_RID), 'runtime_model':(end_m_RID-start_m_RID)}
    #
    # print("Ridge completed")




    ## SAVE RESULTS

    # use a dictionary with keys using the right names.
    results = {'results': res_opt_method, 'confusion matrix': CM_opt, 'hyperparameters': hyperparam_optim,
               'y_test': list(y_test)}
    a_file = open(results_path + "export_CV/from_GPU_byfold/Res_SIM_planets_220324_CO/results_SIM_"+str(mol)+"_data_alpha_"+str(alpha_set[scaling])+"_CV_testfold.pkl", "wb")
    pickle.dump(results, a_file)
    a_file.close()

    # export optimizaiton results
    a_file = open(results_path + "export_CV/from_GPU_byfold/Res_SIM_planets_220324_CO/GA_SIM_"+str(mol)+"_data_alpha_"+str(alpha_set[scaling])+"_CV_testfold.pkl", "wb")
    pickle.dump(optim_results, a_file)
    a_file.close()

    ## OUT
    end_global = time.time()
    print('terminated cv ' + str(k2) + ", Runtime:  " + str((end_global - start_global) / 60) + " minutes")


