# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:49:10 2021

@author: emily
"""

## LIBRARIES
import numpy as np 
import pandas as pd
from PyAstronomy import pyasl
from math import sqrt
from scipy.stats import t

# My modules
from ml_spectroscopy.config import path_init
from ml_spectroscopy.crosscorrNorm import crosscorrRVnorm
from sklearn.metrics import confusion_matrix


## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"


## Utility functions

################################################################################
## Create a small bootstrap to find the scaling values relative to the variance
################################################################################

def scale_miniboot(planet, noise, B, N, Replace=True):
    
    noise=pd.DataFrame(noise)
    planet=pd.DataFrame(planet)

    alphameans=np.zeros((B))
    alphasigmas=np.zeros((B))
    
    for i in range(0,B):
        noiseboot=noise.sample(N, replace=Replace)
        planetboot=planet.sample(N, replace=Replace)
        
        sigmanoise=noiseboot.std(axis='columns')
        sigmaplanet=planetboot.std(axis='columns')
        
        alpha0=np.array(sigmanoise)/np.array(sigmaplanet)
        alphameans[i]=np.mean(alpha0)
        alphasigmas[i]=np.std(alpha0)
        
        alpha=np.mean(alphameans)
      
    return alpha, alphameans, alphasigmas
 

#####################################################################################
## Function for line by line adjustment of alpha based on the signal to noise ratio
#####################################################################################


def scale_SNR(planet, noise, M, N0, N, step):
    
    minsize=min(planet.shape[0], noise.shape[0])
    alpha=np.arange(N0,N,step)
    beta0=pd.DataFrame(np.arange(0,minsize))
    beta=np.array(beta0.sample(M, replace=False))
    SNR=np.zeros((len(beta),len(alpha)))
    
    for j in range(0,M):
        it=0
        for i in alpha:
            planetarray=np.array(planet.iloc[int(beta[[j]]),:])
            noisearray=np.array(noise.iloc[int(beta[[j]]),:])
            planetwl=pd.to_numeric(planet.columns)
            noisewl=pd.to_numeric(noise.columns)
            
            combination=i*planetarray+noisearray
            
            rv1, cc1 = pyasl.crosscorrRV(noisewl, noisearray, planetwl, planetarray, -2000., 2000., 2000./2000., skipedge=70, mode='doppler')
            rv2, cc2 = pyasl.crosscorrRV(noisewl, combination, planetwl, planetarray, -2000., 2000., 2000./2000., skipedge=70, mode='doppler')
            
            SNR[j,it]=np.max(cc2)/np.std(cc1)
            it=it+1

            
    return SNR, alpha


#####################################################################################
## t tests on average of data series 
#####################################################################################


def t_test_onSeriesMean(data):
    sqrtn = sqrt(len(data.mean(axis=0).index))
    xbar = data.mean(axis=0).mean()
    sigmabar = data.mean(axis=0).std()
    #alpha = 0.05
    df = len(data.mean(axis=0).index) - 1
    tstat = (xbar - 0) / (sigmabar / sqrtn)
    #cv = t.ppf(1.0 - alpha, df)
    p = (1 - t.cdf(abs(tstat), df)) * 2
    return {'Pval': p, 't-stat': tstat}




def t_test_onCCF_max(data, alpha):
    sqrtn = np.sqrt(len(data.columns))
    xbar = data.mean(axis=1)
    sigmabar = data.std(axis=1)
    maxtotest = data.max(axis=1)
    #alpha = 0.05
    df = len(data.columns) - 1
    tstat = (maxtotest - xbar) / (sigmabar / sqrtn)
    #cv = t.ppf(1.0 - alpha, df)
    p = (1 - t.cdf(abs(tstat), df)) * 2
    
    ypred=[]
    for i in range(0,len(p)): 
        if p[i]<=alpha:
            yt=1
        elif p[i]>alpha:
            yt=0
        ypred.append(yt)
        
    return {'Y_pred': ypred,'Pval': p, 't-stat': tstat}


# =============================================================================
# 
# def t_test_onCCF_rv0(data, alpha=0.05):
#     
#     sqrtn = np.sqrt(len(data.columns))
#     xbar = data.mean(axis=1)
#     sigmabar = data.std(axis=1)
#     #alpha = 0.05
#     df = len(data.columns) - 1
#     tstat = (xbar - data[0]) / (sigmabar / sqrtn)
#     #cv = t.ppf(1.0 - alpha, df)
#     p = (1 - t.cdf(abs(tstat), df)) * 2
#     
#     ypred=[]
#     for i in range(0,len(p)): 
#         if p[i]<=alpha:
#             yt=1
#         elif p[i]>alpha:
#             yt=0
#         ypred.append(yt)
#         
#     return {'Y_pred': ypred,'Pval': p, 't-stat': tstat}
# 
# 
# =============================================================================




def t_test_onCCF_rv0(data, alpha):
    
    sqrtn = np.sqrt(len(data.columns))
    xbar = data.mean(axis=1)
    sigmabar = data.std(axis=1)
    #alpha = 0.05
    df = len(data.columns) - 2
    tstat = (data[0] - xbar) / (sigmabar / sqrtn)
    cv = t.ppf((1.0 - alpha/2), df)
    
    p = (1 - t.cdf(abs(tstat), df)) * 2

    
    ypred=[]
    for i in range(0,len(p)): 
        if p[i]<alpha:
            yt=1
        elif p[i]>=alpha:
            yt=0
        ypred.append(yt)
        
    ypredt=[]
    for i in range(0,len(p)): 
        if abs(tstat[i])<=cv:
            yt=0
        elif abs(tstat[i])>cv:
            yt=1
        ypredt.append(yt)
        
    return {'Y_predp': ypred,'Y_predt': ypredt, 'Pval': p, 't-stat': tstat}





def t_test_onCCF_rv0_onesided(data, alpha):
    
    sqrtn = np.sqrt(len(data.columns))
    xbar = data.mean(axis=1)
    sigmabar = data.std(axis=1)
    #alpha = 0.05
    df = len(data.columns) - 1
    tstat = (data[0] - xbar) / (sigmabar / sqrtn)
    cv = t.ppf(1.0 - alpha, df)
    
    p = (1-t.cdf(tstat, df))
    #p = (t.sf(tstat, df)) 
    
    ypred=[]
    for i in range(0,len(p)): 
        if p[i]<alpha:
            yt=1
        elif p[i]>=alpha:
            yt=0
        ypred.append(yt)
        
    ypredt=[]
    for i in range(0,len(p)): 
        if tstat[i]<=cv:
            yt=0
        elif (tstat[i])>cv:
            yt=1
        ypredt.append(yt)
        
    return {'Y_predp': ypred,'Y_predt': ypredt, 'Pval': p, 't-stat': tstat}





## SNR function
def test_onCCF_rv0_SNR(data, snr):
    sigmabar = data.drop(data[range(-200, 200)], axis=1).std(axis=1)
    stat = data[0] / sigmabar

    ypredstat = []
    for i in range(0, len(stat)):
        if stat[i] <= snr:
            yt = 0
        elif (stat[i]) > snr:
            yt = 1

        ypredstat.append(yt)

    return {'Y_pred': ypredstat, 'SNR': stat}


## SNR function with lower RV steps
def test_onCCF_rv0_SNR_drv(data, drv, snr):
    sigmabar = data.drop(data[range(-200, 200, drv)], axis=1).std(axis=1)
    stat = data[0] / sigmabar

    ypredstat = []
    for i in range(0, len(stat)):
        if stat[i] <= snr:
            yt = 0
        elif (stat[i]) > snr:
            yt = 1

        ypredstat.append(yt)

    return {'Y_pred': ypredstat, 'SNR': stat}


## SNR function
def test_onCCF_rv0_SNR_CO(data, snr):
    sigmabar = data.drop(data[range(-750, 750)], axis=1).std(axis=1)
    stat = data[0] / sigmabar

    ypredstat = []
    for i in range(0, len(stat)):
        if stat[i] <= snr:
            yt = 0
        elif (stat[i]) > snr:
            yt = 1

        ypredstat.append(yt)

    return {'Y_pred': ypredstat, 'SNR': stat}

def test_onCCF_rv0_SNR_autocorrel(data,template,snr):

    TempCol = template.columns.get_loc("tempP")
    tf = template.drop(template.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()
    
    df = tf
    dw = tw
    
    cc1, rv1 = crosscorrRVnorm(dw, df, tw, tf, -2000, 2000, 1, mode="doppler", skipedge=100, edgeTapering=None)
    cc0=pd.DataFrame(np.reshape(cc1, [1,4000]), columns=rv1)
    ratio=np.array((abs(data[0])))/np.array((cc0[0])) # good
    cc=np.tile(np.array((cc0)), (len(ratio),1))*np.transpose([ratio] * cc0.shape[1])
    cc=pd.DataFrame(cc, columns=rv1)
 
    sigmabar = (np.array(data.drop(data[range(-200,200)], axis=1))-np.array(cc.drop(cc.columns[range(-200,200)], axis=1))).std(axis=1)
    mubar = (np.array(data.drop(data[range(-200,200)], axis=1))-np.array(cc.drop(cc.columns[range(-200,200)], axis=1))).mean(axis=1)

    #sigmabar = data.drop(data[range(-200,200)], axis=1).std(axis=1)
    
    sqrtn = np.sqrt(len(data.drop(data[range(-200,200)], axis=1).columns))
    #xbar = X_test.mean(axis=1)
    #sigmabar = X_test.drop(data[range(-200,200)], axis=1).std(axis=1)
    #alpha = 0.05
    #stat = (data[0]) / (sigmabar / sqrtn)
    #stat = (data[0]-mubar) / sigmabar
    stat = data[0] / sigmabar
    
    
    ypredstat=[]
    for i in range(0,len(stat)): 
        if stat[i]<=snr:
            yt=0
        elif (stat[i])>snr:
            yt=1
            
        ypredstat.append(yt)

    return {'Y_pred': ypredstat, 'SNR': stat}





# =============================================================================
# get the keys from a dictionary
# =============================================================================

def DictKeystoList(dictionary_object):
    ls = []
    for s in dictionary_object.keys():
        ls.append(s)          
    return ls


# =============================================================================
# flatten a list
# =============================================================================


def flatten(t):
    return [item for sublist in t for item in sublist]


# =============================================================================
# average a list

# =============================================================================

def Average(lst):
    return sum(lst) / len(lst)



# =============================================================================
# grid search threshold
# =============================================================================


def grid_search0(hyperparams, prob_Y, data_test):
    
    store=np.zeros((len(hyperparams)))
    store[:]=np.nan
    
    for i in range(len(hyperparams)):
        newpred=list(map(int,list(prob_Y>=hyperparams[i])))   
        store[i]=confusion_matrix(data_test, newpred).ravel()[1]/(confusion_matrix(data_test, newpred).ravel()[1]+confusion_matrix(data_test, newpred).ravel()[3])
    
    try:
        optim=int(np.argwhere(store < 0.05)[0])
        optimal_hyperparam = hyperparams[optim]
        optim_pred = list(map(int, list(prob_Y >= hyperparams[optim])))
        tn, fp, fn, tp = confusion_matrix(data_test, optim_pred).ravel()

    except IndexError:
        optimal_hyperparam = np.nan
        tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan

    
    return {'optim_score': optimal_hyperparam, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
