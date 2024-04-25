# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 03:10:34 2021

Models for Machine learning , cross validation 

@author: emily
"""

## LIBRARIES 


import concurrent.futures
import pandas as pd
import numpy as np
import random
import os
import sys
import gc
import matplotlib.pyplot as plt
import math
from scipy.stats import t
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.crosscorrNormVec import crosscorrRV_vec
from ml_spectroscopy.config import path_init
from ml_spectroscopy.utility_functions import t_test_onCCF_max, t_test_onCCF_rv0, t_test_onCCF_rv0_onesided, test_onCCF_rv0_SNR


## ML 
import keras
from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, LeakyReLU
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.config import path_init
subdir = path_init()
visual_path = subdir + "80_visualisation/"

## models for cross validation


 
# =============================================================================
# grid search
# =============================================================================

def grid_search(hyperparams, metric, minimum=True):
    
    metric = np.array(metric)
    
    if minimum==True: # minimum
        opt_metric = np.min(metric)
    elif minimum==False: # maximum
        opt_metric = np.max(metric)

    pos_opt_metric = np.argwhere(metric == opt_metric)[0]

    hyperparam_optimal = hyperparams.iloc[pos_opt_metric,:]
    
    hyperparam_optimal=pd.Series(hyperparam_optimal.to_dict(orient='records'))
    
    return {'optimal_hyperparam': hyperparam_optimal, 'optimal_metric': opt_metric}
  


 
# =============================================================================
# grid search SNR
# =============================================================================

def SNR_grid_search(hyperparams, objective, X_train, Y_train):
    
    metric=np.empty((len(hyperparams)))
    
    for i in range(len(hyperparams)):
        Y_pred=objective(X_train, hyperparams[i])['Y_pred']
        metric[i]=sum(Y_pred==Y_train)/len(Y_train)
        
    opt_metric = np.max(metric)

    pos_opt_metric = np.argwhere(metric == opt_metric)[0]

    hyperparam_optimal = hyperparams[pos_opt_metric]
        
    return {'optimal_hyperparam': hyperparam_optimal, 'optimal_metric': opt_metric}


def SNR_grid_search_drv(hyperparams, objective, drv, X_train, Y_train):
    metric = np.empty((len(hyperparams)))

    for i in range(len(hyperparams)):
        Y_pred = objective(X_train, drv, hyperparams[i])['Y_pred']
        metric[i] = sum(Y_pred == Y_train) / len(Y_train)

    opt_metric = np.max(metric)

    pos_opt_metric = np.argwhere(metric == opt_metric)[0]

    hyperparam_optimal = hyperparams[pos_opt_metric]

    return {'optimal_hyperparam': hyperparam_optimal, 'optimal_metric': opt_metric}


def SNR_auto_grid_search(hyperparams, template, objective, X_train, Y_train):
    
    metric=np.empty((len(hyperparams)))
    
    for i in range(len(hyperparams)):
        Y_pred=objective(X_train, template, hyperparams[i])['Y_pred']
        metric[i]=sum(Y_pred==Y_train)/len(Y_train)
        
    opt_metric = np.max(metric)

    pos_opt_metric = np.argwhere(metric == opt_metric)[0]

    hyperparam_optimal = hyperparams[pos_opt_metric]
        
    return {'optimal_hyperparam': hyperparam_optimal, 'optimal_metric': opt_metric}

# =============================================================================
# simple ANN 
# =============================================================================


def PCT_model(hyperparams, X_train, y_train, X_valid_test, y_valid_test):

    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
    
    
    batch_size, epochs = hyperparams
    
    if batch_size<24:
        batch_size=16
    elif batch_size<48:
        batch_size=32
    elif batch_size<96:
        batch_size=64
    elif batch_size<150:
        batch_size=128
    else:
        batch_size=256

    #batch_size=math.ceil(batch_size / 2.) * 2
        
    epochs=round(epochs)
    
    # build model
    model_seq = Sequential()
    model_seq.add(Dense(2, input_shape=(X_train.shape[1],), activation='sigmoid'))

    model_seq.compile(optimizer='RMSprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    # fit model
    model_seq.summary()
    model_seq.fit(np.array((X_train)), y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    #validation_data=(X_valid, y_valid),
    
    # results
    prediction_probs = model_seq.predict(X_valid_test, batch_size=batch_size, verbose=0)   
    y_test_hat = np.argmax(prediction_probs, axis=1)

    # weights
    weights_PCT = model_seq.get_weights()

    # metrics
    scores_train = model_seq.evaluate(X_train, y_train, verbose=0)
    scores_valid_test = model_seq.evaluate(X_valid_test, y_valid_test, verbose=0)
    #misclass_error=np.mean(y_valid_test != y_test_hat)

    #metrics_scores=(model_seq.metrics_names[1], scores[1]*100)

    del model_seq
    gc.collect()
    
    return {'Y_pred': y_test_hat, 'Y_pred_prob': prediction_probs, 'accuracy_valid_test': scores_valid_test[1], 'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0], 'weights': weights_PCT}


# =============================================================================
# ANN with several layers, droput and learning rate
# =============================================================================


def DNN_model(hyperparameters, X_train, y_train, X_valid_test, y_valid_test):   #, batch_size=32, epochs=180, learning_rate=0.0008, momentum=0.82, leakyrelu_alpha=0.8, dropout=0.3, kernel_maxnorm=3):
    
    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
    

    batch_size, epochs, learning_rate, momentum, leakyrelu_alpha1, leakyrelu_alpha2, dropout, kernel_maxnorm = hyperparameters

    if batch_size<24:
        batch_size=16
    elif batch_size<48:
        batch_size=32
    elif batch_size<96:
        batch_size=64
    elif batch_size<150:
        batch_size=128
    else:
        batch_size=256

    #batch_size=math.ceil(batch_size / 2.) * 2
    #leakyrelu_alpha2 = leakyrelu_alpha1

    epochs=round(epochs)
    kernel_maxnorm=round(kernel_maxnorm)

    decay_rate = learning_rate / epochs
    
    # build model       
    model = Sequential()
    model.add(Dense(batch_size, input_dim=4000,activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model.add(Dropout(dropout))
    model.add(Dense(int(batch_size/2), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha2), kernel_constraint=maxnorm(kernel_maxnorm)))
    model.add(Dense(2, activation='sigmoid'))
       
    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
        
    # fit model
    model.fit(np.array((X_train)), y_train, validation_data=(X_valid_test, y_valid_test), batch_size=batch_size, epochs=epochs, verbose=1)

    # results
    prediction_probs = model.predict(X_valid_test, batch_size=batch_size, verbose=0)
    prediction_probs.shape
    y_test_hat = np.argmax(prediction_probs, axis=1)

    # weights
    weights_DNN = model.get_weights()

    # metrics
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    scores_valid_test = model.evaluate(X_valid_test, y_valid_test, verbose=0)
    #misclass_error=np.mean(y_valid_test != y_test_hat)

    del model
    gc.collect()
        
    return {'Y_pred': y_test_hat, 'Y_pred_prob': prediction_probs, 'accuracy_valid_test': scores_valid_test[1], 'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0], 'weights': weights_DNN}

# =============================================================================
# Temp
# =============================================================================

def DNN2_model(hyperparameters, X_train, y_train, X_valid_test,
              y_valid_test):  # , batch_size=32, epochs=180, learning_rate=0.0008, momentum=0.82, leakyrelu_alpha=0.8, dropout=0.3, kernel_maxnorm=3):

    # K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

    batch_size, epochs, learning_rate, momentum, leakyrelu_alpha1, leakyrelu_alpha2, dropout, kernel_maxnorm = hyperparameters

    if batch_size < 24:
        batch_size = 16
    elif batch_size < 48:
        batch_size = 32
    elif batch_size < 96:
        batch_size = 64
    elif batch_size < 150:
        batch_size = 128
    else:
        batch_size = 256

    # batch_size=math.ceil(batch_size / 2.) * 2
    # leakyrelu_alpha2 = leakyrelu_alpha1

    epochs = round(epochs)
    kernel_maxnorm = round(kernel_maxnorm)

    decay_rate = learning_rate / epochs

    # build model
    model = Sequential()
    model.add(Dense(batch_size, input_dim=4000, activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model.add(Dropout(dropout))
    model.add(Dense(int(batch_size / 2), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha2),
                    kernel_constraint=maxnorm(kernel_maxnorm)))
    model.add(Dropout(dropout))
    model.add(Dense(int(batch_size / 2), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha2),
                    kernel_constraint=maxnorm(kernel_maxnorm)))
    model.add(Dense(2, activation='sigmoid'))

    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # fit model
    model.fit(np.array((X_train)), y_train, validation_data=(X_valid_test, y_valid_test), batch_size=batch_size,
              epochs=epochs, verbose=1)

    # results
    prediction_probs = model.predict(X_valid_test, batch_size=batch_size, verbose=0)
    prediction_probs.shape
    y_test_hat = np.argmax(prediction_probs, axis=1)

    # weights
    weights_DNN = model.get_weights()

    # metrics
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    scores_valid_test = model.evaluate(X_valid_test, y_valid_test, verbose=0)
    # misclass_error=np.mean(y_valid_test != y_test_hat)

    del model
    gc.collect()

    return {'Y_pred': y_test_hat, 'Y_pred_prob': prediction_probs, 'accuracy_valid_test': scores_valid_test[1],
            'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0],
            'weights': weights_DNN}


# =============================================================================
# CNN only convolutions
# =============================================================================


def CNN1_model(hyperparameters, x_train, y_train, x_valid_test, y_valid_test):   #, batch_size=32, epochs=180, learning_rate=0.0008, momentum=0.82, leakyrelu_alpha=0.8, dropout=0.3, kernel_maxnorm=3):
    
    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
    

    batch_size, epochs, learning_rate, momentum, kernelsize, maxpooling = hyperparameters
    
    epochs=round(epochs)
    #kernel_maxnorm=round(kernel_maxnorm)

    if batch_size<24:
        batch_size=16
    elif batch_size<48:
        batch_size=32
    elif batch_size<96:
        batch_size=64
    elif batch_size<150:
        batch_size=128
    else:
        batch_size=256
        
    if kernelsize<4:
        kernelsize=3
    elif kernelsize<6:
        kernelsize=5
    else: 
        kernelsize=7
        
    
    verbose = 1
    decay_rate = learning_rate / epochs
    #momentum = 0.82
    
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model_CNN1 = Sequential()
    model_CNN1.add(Conv1D(filters=batch_size, kernel_size=kernelsize, input_shape=(n_timesteps, n_features)))
    model_CNN1.add(MaxPooling1D(pool_size=round(maxpooling)))
    model_CNN1.add(Conv1D(filters=(batch_size/2), kernel_size=kernelsize))
    model_CNN1.add(MaxPooling1D(pool_size=round(maxpooling)))
    model_CNN1.add(Flatten())
    model_CNN1.add(Dense(n_outputs, activation='sigmoid'))
    
    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    model_CNN1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model_CNN1.summary()
    print("batch: "+str(batch_size)+" epochs: "+str(epochs)+" LR: "+str(learning_rate)+" momentum: "+str(momentum)+" kernel: "+str(kernelsize)+" maxpool: "+str(maxpooling))

    # fit network
    #modfit_CNN1 = model_CNN1.fit(x_train, y_train, validation_data=(x_valid_test, y_valid_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
    modfit_CNN1 = model_CNN1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    #metrics_scores=(model_CNN.metrics_names[1], scores[1]*100)
    
    # results
    prediction_probsCNN1 = model_CNN1.predict(x_valid_test, batch_size=batch_size, verbose=1)
    y_test_hatCNN1 = np.argmax(prediction_probsCNN1, axis=1)

    # weights
    weights_CNN1 = model_CNN1.get_weights()

    # metrics
    scores_train = model_CNN1.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    scores_valid_test = model_CNN1.evaluate(x_valid_test, y_valid_test, batch_size=batch_size, verbose=0)

    del model_CNN1
    gc.collect()

    #batch_size=math.ceil(batch_size / 2.) * 2
       
    return {'Y_pred': y_test_hatCNN1, 'Y_pred_prob': prediction_probsCNN1, 'accuracy_valid_test': scores_valid_test[1], 'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0], 'weights': weights_CNN1}


# =============================================================================
# CNN with several layers, droput and learning rate
# =============================================================================


def CNN2_model(hyperparameters, x_train, y_train, x_valid_test, y_valid_test):   #, batch_size=32, epochs=180, learning_rate=0.0008, momentum=0.82, leakyrelu_alpha=0.8, dropout=0.3, kernel_maxnorm=3):
    
    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
    

    batch_size, epochs, learning_rate, momentum, kernelsize, maxpooling, leakyrelu_alpha1, dropout1, dropout2, kernel_maxnorm = hyperparameters
    
    epochs=round(epochs)
    kernel_maxnorm=round(kernel_maxnorm)

    if batch_size<24:
        batch_size=16
    elif batch_size<48:
        batch_size=32
    elif batch_size<96:
        batch_size=64
    elif batch_size<150:
        batch_size=128
    else:
        batch_size=256
        
    if kernelsize<4:
        kernelsize=3
    elif kernelsize<6:
        kernelsize=5
    else: 
        kernelsize=7
        
    
    verbose = 1
    decay_rate = learning_rate / epochs
    #momentum = 0.82
    
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model_CNN2 = Sequential()
    model_CNN2.add(Conv1D(filters=batch_size, kernel_size=kernelsize, input_shape=(n_timesteps,n_features), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model_CNN2.add(MaxPooling1D(pool_size=round(maxpooling)))
    model_CNN2.add(Conv1D(filters=(batch_size/2), kernel_size=kernelsize, activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model_CNN2.add(MaxPooling1D(pool_size=round(maxpooling)))
    model_CNN2.add(Flatten())
    model_CNN2.add(Dense((batch_size/2), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model_CNN2.add(Dropout(dropout1))
    model_CNN2.add(Dense((batch_size/2), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model_CNN2.add(Dropout(dropout2))
    model_CNN2.add(Dense((batch_size/4), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1), kernel_constraint=maxnorm(round(kernel_maxnorm))))
    model_CNN2.add(Dense(n_outputs, activation='sigmoid'))
    
    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    model_CNN2.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    model_CNN2.summary()

    # fit network
    #modfit_CNN2 = model_CNN2.fit(x_train, y_train, validation_data=(x_valid_test, y_valid_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
    modfit_CNN2 = model_CNN2.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    #metrics_scores=(model_CNN.metrics_names[1], scores[1]*100)
    
    # results
    prediction_probsCNN2 = model_CNN2.predict(x_valid_test, batch_size=batch_size, verbose=1)
    y_test_hatCNN2 = np.argmax(prediction_probsCNN2, axis=1)

    # weights
    weights_CNN2 = model_CNN2.get_weights()

    # metrics
    scores_train = model_CNN2.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    scores_valid_test = model_CNN2.evaluate(x_valid_test, y_valid_test, batch_size=batch_size, verbose=0)

    del model_CNN2
    gc.collect()
    #batch_size=math.ceil(batch_size / 2.) * 2
       
    return {'Y_pred': y_test_hatCNN2, 'Y_pred_prob': prediction_probsCNN2, 'accuracy_valid_test': scores_valid_test[1], 'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0], 'weights': weights_CNN2}

# =============================================================================
# Temp
# =============================================================================

def CNN3_model(hyperparameters, x_train, y_train, x_valid_test,
               y_valid_test):  # , batch_size=32, epochs=180, learning_rate=0.0008, momentum=0.82, leakyrelu_alpha=0.8, dropout=0.3, kernel_maxnorm=3):

    # K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

    batch_size, epochs, learning_rate, momentum, kernelsize, maxpooling, leakyrelu_alpha1, dropout1, dropout2, kernel_maxnorm = hyperparameters

    epochs = round(epochs)
    kernel_maxnorm = round(kernel_maxnorm)

    if batch_size < 24:
        batch_size = 16
    elif batch_size < 48:
        batch_size = 32
    elif batch_size < 96:
        batch_size = 64
    elif batch_size < 150:
        batch_size = 128
    else:
        batch_size = 256

    if kernelsize < 4:
        kernelsize = 3
    elif kernelsize < 6:
        kernelsize = 5
    else:
        kernelsize = 7

    verbose = 1
    decay_rate = learning_rate / epochs
    # momentum = 0.82

    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model_CNN2 = Sequential()
    model_CNN2.add(Conv1D(filters=batch_size, kernel_size=kernelsize, input_shape=(n_timesteps, n_features),
                          activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model_CNN2.add(MaxPooling1D(pool_size=round(maxpooling)))
    model_CNN2.add(Flatten())
    model_CNN2.add(Dense((batch_size / 2), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1)))
    model_CNN2.add(Dropout(dropout1))
    model_CNN2.add(Dense((batch_size / 2), activation=keras.layers.LeakyReLU(alpha=leakyrelu_alpha1),
                         kernel_constraint=maxnorm(round(kernel_maxnorm))))
    model_CNN2.add(Dense(n_outputs, activation='sigmoid'))

    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model_CNN2.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model_CNN2.summary()

    # fit network
    modfit_CNN2 = model_CNN2.fit(x_train, y_train, validation_data=(x_valid_test, y_valid_test), epochs=epochs,
                                 batch_size=batch_size, verbose=verbose)

    # metrics_scores=(model_CNN.metrics_names[1], scores[1]*100)

    # results
    prediction_probsCNN2 = model_CNN2.predict(x_valid_test, batch_size=batch_size, verbose=1)
    y_test_hatCNN2 = np.argmax(prediction_probsCNN2, axis=1)

    # weights
    weights_CNN2 = model_CNN2.get_weights()

    # metrics
    scores_train = model_CNN2.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    scores_valid_test = model_CNN2.evaluate(x_valid_test, y_valid_test, batch_size=batch_size, verbose=0)

    del model_CNN2
    gc.collect()
    # batch_size=math.ceil(batch_size / 2.) * 2

    return {'Y_pred': y_test_hatCNN2, 'Y_pred_prob': prediction_probsCNN2, 'accuracy_valid_test': scores_valid_test[1],
            'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0],
            'weights': weights_CNN2}


def CNN4_model(hyperparameters, x_train, y_train, x_valid_test, y_valid_test):
    # , batch_size=32, epochs=180, learning_rate=0.0008, momentum=0.82, leakyrelu_alpha=0.8, dropout=0.3, kernel_maxnorm=3):

    # K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

    batch_size, epochs, learning_rate, momentum, kernelsize, maxpooling = hyperparameters

    epochs = round(epochs)
    # kernel_maxnorm=round(kernel_maxnorm)

    if batch_size < 24:
        batch_size = 16
    elif batch_size < 48:
        batch_size = 32
    elif batch_size < 96:
        batch_size = 64
    elif batch_size < 150:
        batch_size = 128
    else:
        batch_size = 256

    if kernelsize < 4:
        kernelsize = 3
    elif kernelsize < 6:
        kernelsize = 5
    else:
        kernelsize = 7

    verbose = 1
    decay_rate = learning_rate / epochs
    # momentum = 0.82

    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model_CNN1 = Sequential()
    model_CNN1.add(Conv1D(filters=batch_size, kernel_size=kernelsize, input_shape=(n_timesteps, n_features)))
    model_CNN1.add(MaxPooling1D(pool_size=round(maxpooling)))
    model_CNN1.add(Flatten())
    model_CNN1.add(Dense(n_outputs, activation='sigmoid'))

    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model_CNN1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model_CNN1.summary()
    print("batch: " + str(batch_size) + " epochs: " + str(epochs) + " LR: " + str(learning_rate) + " momentum: " + str(
        momentum) + " kernel: " + str(kernelsize) + " maxpool: " + str(maxpooling))

    # fit network
    modfit_CNN1 = model_CNN1.fit(x_train, y_train, validation_data=(x_valid_test, y_valid_test), epochs=epochs,
                                 batch_size=batch_size, verbose=verbose)

    # metrics_scores=(model_CNN.metrics_names[1], scores[1]*100)

    # results
    prediction_probsCNN1 = model_CNN1.predict(x_valid_test, batch_size=batch_size, verbose=1)
    y_test_hatCNN1 = np.argmax(prediction_probsCNN1, axis=1)

    # weights
    weights_CNN1 = model_CNN1.get_weights()

    # metrics
    scores_train = model_CNN1.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    scores_valid_test = model_CNN1.evaluate(x_valid_test, y_valid_test, batch_size=batch_size, verbose=0)

    del model_CNN1
    gc.collect()

    # batch_size=math.ceil(batch_size / 2.) * 2

    return {'Y_pred': y_test_hatCNN1, 'Y_pred_prob': prediction_probsCNN1, 'accuracy_valid_test': scores_valid_test[1],
            'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0],
            'weights': weights_CNN1}


# =============================================================================
# ElasticNet
# =============================================================================


def ElasticNet_model(hyperparameters, X_train, y_train, X_valid_test, y_valid_test):


    #from sklearn.metrics import score
    l1_ratio, lambda0, max_iter0 = hyperparameters
    # L1 between 0 and 0.2 - actual was 0.8
    # alpha low also, between 0 and 0.2 actual was 0.00001
    enet = SGDClassifier(penalty='elasticnet', l1_ratio=l1_ratio, alpha=lambda0, loss='modified_huber', max_iter=max_iter0)
    #enet = LogisticRegression(penalty='elasticnet',C=(0.999), l1_ratio=0.9, max_iter=200, solver='saga')
    
    #enet = SGDClassifier(penalty='l1', alpha=0.00001, loss='log')
    fit_enet = enet.fit(X_train, y_train[:,1])
    
    ytrain_prob_enet = fit_enet.predict_proba(X_train)
    score_train_enet = sum(y_train[:,1]==np.argmax(ytrain_prob_enet, axis=1))/len(y_train[:,1])
    
    y_prob_enet = fit_enet.predict_proba(X_valid_test)
    y_pred_enet=np.argmax(y_prob_enet, axis=1)
    score_enet = sum(y_valid_test[:,1]==y_pred_enet)/len(y_valid_test)
    
    weights_enet={'intercept': fit_enet.intercept_ , 'parameters': fit_enet.coef_.flatten()}

    del enet
    gc.collect()
    
    return {'Y_pred': y_pred_enet, 'Y_pred_prob': y_prob_enet[:,1] , 'accuracy_valid_test': score_enet, 'accuracy_train': score_train_enet, 'weights': weights_enet}

# =============================================================================
# ElasticNet // Modified Huber
# =============================================================================

def ElasticNet_model2(hyperparameters, X_train, y_train, X_valid_test, y_valid_test):
    # from sklearn.metrics import score
    l1_ratio, lambda0, max_iter0 = hyperparameters
    # L1 between 0 and 0.2 - actual was 0.8
    # alpha low also, between 0 and 0.2 actual was 0.00001
    enet = SGDClassifier(penalty='elasticnet', l1_ratio=l1_ratio, alpha=lambda0, loss='modified_huber',
                         max_iter=max_iter0)
    # enet = LogisticRegression(penalty='elasticnet',C=(0.999), l1_ratio=0.9, max_iter=200, solver='saga')

    # enet = SGDClassifier(penalty='l1', alpha=0.00001, loss='log')
    fit_enet = enet.fit(X_train, y_train[:, 1])

    ytrain_prob_enet = fit_enet.predict_proba(X_train)
    score_train_enet = sum(y_train[:, 1] == np.argmax(ytrain_prob_enet, axis=1)) / len(y_train[:, 1])

    y_prob_enet = fit_enet.predict_proba(X_valid_test)
    y_pred_enet = np.argmax(y_prob_enet, axis=1)
    score_enet = sum(y_valid_test[:, 1] == y_pred_enet) / len(y_valid_test)

    weights_enet = {'intercept': fit_enet.intercept_, 'parameters': fit_enet.coef_.flatten()}

    del enet
    gc.collect()

    return {'Y_pred': y_pred_enet, 'Y_pred_prob': y_prob_enet[:, 1], 'accuracy_valid_test': score_enet,
            'accuracy_train': score_train_enet, 'weights': weights_enet}


# =============================================================================
# Ridge
# =============================================================================


def Ridge_model(hyperparameters, X_train, y_train, X_valid_test, y_valid_test):


    #from sklearn.metrics import score
    lambda0, max_iter0 = hyperparameters
    # L1 between 0 and 0.2 - actual was 0.8
    # alpha low also, between 0 and 0.2 actual was 0.00001
    rid = SGDClassifier(penalty='elasticnet', l1_ratio=0, alpha=lambda0, loss='modified_huber', max_iter=max_iter0)
    #rid = LogisticRegression(penalty='elasticnet',C=(0.999), l1_ratio=0.9, max_iter=200, solver='saga')
    
    #rid = SGDClassifier(penalty='l1', alpha=0.00001, loss='log')
    fit_rid = rid.fit(X_train, y_train[:,1])
    
    ytrain_prob_rid = fit_rid.predict_proba(X_train)
    score_train_rid = sum(y_train[:,1]==np.argmax(ytrain_prob_rid, axis=1))/len(y_train[:,1])
    
    y_prob_rid = fit_rid.predict_proba(X_valid_test)
    y_pred_rid=np.argmax(y_prob_rid, axis=1)
    score_rid = sum(y_valid_test[:,1]==y_pred_rid)/len(y_valid_test)
    
    weights_rid={'intercept': fit_rid.intercept_ , 'parameters': fit_rid.coef_.flatten()}

    del rid
    gc.collect()
    
    return {'Y_pred': y_pred_rid, 'Y_pred_prob': y_prob_rid[:,1] , 'accuracy_valid_test': score_rid, 'accuracy_train': score_train_rid, 'weights': weights_rid}





# =============================================================================
# Lasso
# =============================================================================


def Lasso_model(hyperparameters, X_train, y_train, X_valid_test, y_valid_test):


    #from sklearn.metrics import score
    lambda0, max_iter0 = hyperparameters
    # L1 between 0 and 0.2 - actual was 0.8
    # alpha low also, between 0 and 0.2 actual was 0.00001
    las = SGDClassifier(penalty='elasticnet', l1_ratio=1, alpha=lambda0, loss='modified_huber', max_iter=max_iter0)
    #las = LogisticRegression(penalty='elasticnet',C=(0.999), l1_ratio=0.9, max_iter=200, solver='saga')
    
    #las = SGDClassifier(penalty='l1', alpha=0.00001, loss='log')
    fit_las = las.fit(X_train, y_train[:,1])
    
    ytrain_prob_las = fit_las.predict_proba(X_train)
    score_train_las = sum(y_train[:,1]==np.argmax(ytrain_prob_las, axis=1))/len(y_train[:,1])
    
    y_prob_las = fit_las.predict_proba(X_valid_test)
    y_pred_las=np.argmax(y_prob_las, axis=1)
    score_las = sum(y_valid_test[:,1]==y_pred_las)/len(y_valid_test)
    
    weights_las={'intercept': fit_las.intercept_ , 'parameters': fit_las.coef_.flatten()}

    del las
    gc.collect()
    
    return {'Y_pred': y_pred_las, 'Y_pred_prob': y_prob_las[:,1] , 'accuracy_valid_test': score_las, 'accuracy_train': score_train_las, 'weights': weights_las}



# =============================================================================
# Gradient Tree Boosting
# =============================================================================

def GradientTreeBoosting_model(hyperparameters, X_train, y_train, X_valid_test, y_valid_test, plotimportance=False, path=visual_path+'GQLupB_final_plots_CV/testplot.pdf'):
    
    dtrain = xgb.DMatrix(X_train, label=y_train[:,1])
    dvalid_test = xgb.DMatrix(X_valid_test, label=y_valid_test[:,1])
    
    maxdepth, eta, subsample, num_round = hyperparameters
    
    param = {'max_depth': int(maxdepth),
             'eta': eta,
             'subsample': subsample,
             #'num_boost_round': 10,
             'objective': 'binary:logistic', 
             #'tree_method': 'gpu_hist'
             } # 3, 0.3
    
    param['nthread'] = 8
    param['eval_metric'] = 'error'
    evallist = [(dvalid_test, 'eval'), (dtrain, 'train')]
    
    #num_round = 100
    bst = xgb.train(param, dtrain, int(num_round), evallist)
    
    ypredXGB_prob_train = bst.predict(dtrain)
    ypredXGB_prob_train_temp = np.array((1-ypredXGB_prob_train, ypredXGB_prob_train)).T
    ypredXGB_train = np.argmax(ypredXGB_prob_train_temp, axis=1)
    score_train_XGB = sum(y_train[:,1]==ypredXGB_train)/len(y_train)

    
    ypredXGB_prob = bst.predict(dvalid_test)
    ypredXGB_prob_temp = np.array((1-ypredXGB_prob, ypredXGB_prob)).T
    ypredXGB = np.argmax(ypredXGB_prob_temp, axis=1)
    score_XGB = sum(y_valid_test[:,1]==ypredXGB)/len(y_valid_test)
    
    
    tn_rvXGB, fp_rvXGB, fn_rvXGB, tp_rvXGB = confusion_matrix(y_valid_test[:,1], ypredXGB).ravel()  
    np.array((tn_rvXGB, fp_rvXGB, fn_rvXGB, tp_rvXGB))/len(y_valid_test)
    
    if plotimportance==True:
        fig, ax = plt.subplots(figsize=(10,10))
        xgb.plot_importance(bst, max_num_features=40, height=0.5, ax=ax,importance_type='gain')
        plt.title('Feature Importance in terms of Gain')
        plt.savefig(path)
        plt.clf()
        #The average loss reduction gained when using this feature for splitting in trees.
        fig, ax = plt.subplots(figsize=(10,10))
        xgb.plot_importance(bst, max_num_features=40, height=0.5, ax=ax,importance_type='weight')
        plt.title('Feature Importance in terms of Weights')
        plt.savefig(path)
        plt.clf()


    del bst
    gc.collect()

    return {'Y_pred': ypredXGB, 'Y_pred_prob': ypredXGB_prob, 'accuracy_valid_test': score_XGB, 'accuracy_train': score_train_XGB}




# =============================================================================
# Random forest
# =============================================================================



def RandomForest_model(hyperparameters, X_train, y_train, X_valid_test, y_valid_test):
    
    nestimators, max_features, cost_complexity = hyperparameters
    
    nestimators, max_features = int(nestimators), int(max_features)
    
    
    rf_clf = RandomForestClassifier(n_estimators=nestimators, max_features=max_features, ccp_alpha=cost_complexity, n_jobs=8)
    
    # fit model
    rf_clf.fit(X_train, y_train)
    
    # results  
    probas = rf_clf.predict_proba(X_valid_test)
    Y_pred_RF = rf_clf.predict(X_valid_test)

    # metrics      
    accuracy_valid_test = rf_clf.score(X_valid_test, y_valid_test)
    accuracy_train = rf_clf.score(X_train, y_train)
    
    var_importance = rf_clf.feature_importances_

    del rf_clf
    gc.collect()
    
    return {'Y_pred': Y_pred_RF[:,1], 'Y_pred_prob': probas[1] , 'accuracy_valid_test': accuracy_valid_test, 'accuracy_train': accuracy_train, 'weights': var_importance}



# =============================================================================
# def GradientTreeBoosting_modelfigure(hyperparameters, X_train, y_train, X_valid_test, y_valid_test, path=savepath):
#     
#     dtrain = xgb.DMatrix(X_train, label=y_train[:,1])
#     dvalid_test = xgb.DMatrix(X_valid_test, label=y_valid_test[:,1])
#     
#     maxdepth, eta, subsample = hyperparameters
#     
#     param = {'max_depth': int(maxdepth),
#              'eta': eta,
#              'subsample': subsample,
#              #'num_boost_round': 10,
#              'objective': 'binary:logistic', 
#              #'tree_method': 'gpu_hist'
#              } # 3, 0.3
#     
#     param['nthread'] = 8
#     param['eval_metric'] = 'error'
#     evallist = [(dvalid_test, 'eval'), (dtrain, 'train')]
#     
#     num_round = 100
#     bst = xgb.train(param, dtrain, num_round, evallist)
#     
#     ypredXGB_prob_train = bst.predict(dtrain)
#     ypredXGB_prob_train_temp = np.array((1-ypredXGB_prob_train, ypredXGB_prob_train)).T
#     ypredXGB_train = np.argmax(ypredXGB_prob_train_temp, axis=1)
#     score_train_XGB = sum(y_train[:,1]==ypredXGB_train)/len(y_train)
# 
#     
#     ypredXGB_prob = bst.predict(dvalid_test)
#     ypredXGB_prob_temp = np.array((1-ypredXGB_prob, ypredXGB_prob)).T
#     ypredXGB = np.argmax(ypredXGB_prob_temp, axis=1)
#     score_XGB = sum(y_valid_test[:,1]==ypredXGB)/len(y_valid_test)
#     
#     
#     tn_rvXGB, fp_rvXGB, fn_rvXGB, tp_rvXGB = confusion_matrix(y_valid_test[:,1], ypredXGB).ravel()  
#     np.array((tn_rvXGB, fp_rvXGB, fn_rvXGB, tp_rvXGB))/len(y_valid_test)
# 
# 
#     fig, ax = plt.subplots(figsize=(10,10))
#     xgb.plot_importance(bst, max_num_features=40, height=0.5, ax=ax,importance_type='gain')
#     plt.title('Feature Importance in terms of Gain')
#     plt.savefig()
#     plt.show()
#     #The average loss reduction gained when using this feature for splitting in trees.
# 
#     fig, ax = plt.subplots(figsize=(10,10))
#     xgb.plot_importance(bst, max_num_features=40, height=0.5, ax=ax,importance_type='weight')
#     plt.title('Feature Importance in terms of Weights')
#     plt.show()
#     
#     
#     del bst
#     gc.collect()
# 
#     return {'Y_pred': ypredXGB, 'Y_pred_prob': ypredXGB_prob, 'accuracy_valid_test': score_XGB, 'accuracy_train': score_train_XGB}
# 
# =============================================================================


# =============================================================================
# XGB Random forest
# =============================================================================

def XGBRandomForest_model(hyperparameters, X_train, y_train, X_valid_test, y_valid_test):
    
    dtrain = xgb.DMatrix(X_train, label=y_train[:,1])
    dvalid_test = xgb.DMatrix(X_valid_test, label=y_valid_test[:,1])
    
    nestimators, prop_features, subsample, complexity = hyperparameters   
    
    param = {'booster': 'gbtree',
              'colsample_bynode': prop_features, # <1 (0.8)
              'learning_rate': 1, # =1 
              'max_depth': int(complexity),
              'num_parallel_tree': int(nestimators), # often 100
              'num_boost_round': 1, # Set to 1 if we do not want to boost the RF 
              'objective': 'binary:logistic',
              'subsample': subsample, # <1
              #'tree_method': 'gpu_hist'
              }
    
    param['nthread'] = 6
    param['eval_metric'] = 'error'
    evallist = [(dvalid_test, 'eval'), (dtrain, 'train')]
    
    num_round = 1
    bst = xgb.train(param, dtrain, num_round, evallist)
    
    ypredXGB_prob_train = bst.predict(dtrain)
    ypredXGB_prob_train_temp = np.array((1-ypredXGB_prob_train, ypredXGB_prob_train)).T
    ypredXGB_train = np.argmax(ypredXGB_prob_train_temp, axis=1)
    score_train_XGB = sum(y_train[:,1]==ypredXGB_train)/len(y_train)

    
    ypredXGB_prob = bst.predict(dvalid_test)
    ypredXGB_prob_temp = np.array((1-ypredXGB_prob, ypredXGB_prob)).T
    ypredXGB = np.argmax(ypredXGB_prob_temp, axis=1)
    score_XGB = sum(y_valid_test[:,1]==ypredXGB)/len(y_valid_test)
    
    
    tn_rvXGB, fp_rvXGB, fn_rvXGB, tp_rvXGB = confusion_matrix(y_valid_test[:,1], ypredXGB).ravel()  
    np.array((tn_rvXGB, fp_rvXGB, fn_rvXGB, tp_rvXGB))/len(y_valid_test)

    del bst
    gc.collect()

    return {'Y_pred': ypredXGB, 'Y_pred_prob': ypredXGB_prob , 'accuracy_valid_test': score_XGB, 'accuracy_train': score_train_XGB}


