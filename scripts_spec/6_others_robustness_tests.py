# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:15:38 2022

@author: emily
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


#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
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

from ml_spectroscopy.plottings_utils_results import ROC_curve_customplot, ROC_curve_saveplt ,PR_curve_customplot, PR_curve_saveplt
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
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
visual_path = subdir + "80_visualisation/"
csv_res_path= subdir + "90_sweave_template/"



## SET SEED FOR REPRODUCIBILITY
random.seed(100)


## IMPORT DATA
## start with only trimmed data. Later can compare just the neural network between padded and trimmed data, using Planet_Signals[data.sum(axis=0)==0,:]
#Planet_Signals = pd.read_csv(data_path + "csv_inputs/Planet_Signals_df.csv", index_col=0)
# SETTINGS

data_name='GQlupb'
template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}

alpha=10
x=6
planet='GQlupB'
data1=pd.read_pickle(data_path+'data_4ml/v2_ccf_4ml_trim_robustness/H2O_'+data_name+'_scale'+str(alpha)+'_temp1200_sg4.1_ccf_4ml_trim_norepetition.pkl')


i=5
j=6
data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]


X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_train=data_train['H2O']


X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_valid=data_valid['H2O']


X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
Y_test=data_test['H2O']


# import results from CV and store information as plots 

dir_path = results_path + "export_CV/from_GPU_byfold/150122/results/"
ls_data = os.listdir(dir_path)
len_folds=len(ls_data)
result_names=[ls_data[n][:-4] for n in range(len_folds)]

keys = result_names
ls_results = {key: None for key in keys}

for i in range(0,len_folds):
    with open(dir_path+str(ls_data[i]), "rb") as f:
        ls_results[i] = pickle.load(f) # i is the validation number but the proper set is at i+1
# =============================================================================
# 
# 
# dir_path2 = results_path + "export_CV/from_GPU_byfold/GA_results/"
# ls_data2 = os.listdir(dir_path2)
# len_folds2=len(ls_data2)
# result_names2=[ls_data2[n][:-4] for n in range(len_folds2)]
# 
# keys2 = result_names2
# GA_results = {key: None for key in keys2}
# 
# for i in range(0,len_folds):
#     with open(dir_path2+str(ls_data2[i]), "rb") as f:
#         GA_results[i] = pickle.load(f) # i is the validation number but the proper set is at i+1
# 
# =============================================================================
methods_ls= ['SNR', 'SNR_auto',  'RF', 'XGB', 'LAS', 'ENET', 'RID', 'ENET2', 'PCT', 'DNN', 'CNN1', 'CNN2']
#methods_ls=['SNR', 'SNR_auto','CNN1','ENET','RID', 'XGB', 'ENET2']
plotname='test'
color_ls={'SNR':'red', 'SNR_auto': 'brown', 'PCT':'lightblue', 'DNN': 'blue', 'CNN1':'navy', 'CNN2': 'purple', 'ENET':'forestgreen', 'RID':'lime', 'LAS':'lightgreen', 'RF':'yellow', 'XGB':'orange', 'ENET2':'darkgreen'}


# =============================================================================
# # =============================================================================
# # ROC curves
# # =============================================================================
# =============================================================================



# methods: ['SNR', 'SNR_auto', 'PCT', 'DNN', 'CNN1', 'CNN2', 'ENET', 'RID', 'LAS', 'RF', 'XGB', 'ENET2']
plt.style.use('seaborn')


# Plot out the ROC curves
#ls_results[i]['Y_test']
alpha=10
ax2=[0,1,2,0,1,2,0,1,2]
ax1=[0,0,0,1,1,1,2,2,2]
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.suptitle('ROC curves for $\\alpha='+str(alpha)+'$', fontsize=14)


for j in range(0,len_folds+1):
    
    if j<len_folds:
        try:
        
            if j==0:
                i=7
            else:
                i=j-1 
                
            data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
            data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
            data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
            
            X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train=data_train['H2O']
            
            X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid=data_valid['H2O']
            
            X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_test=data_test['H2O']
            
            df_pr, ax = ROC_curve_customplot(ls_results[j]['results'], axes, ax1[j], ax2[j], Y_test, i, methods_ls, color_ls, path=visual_path)
            
        except KeyError:
            pass  
    else:
        axes[ax1[j], ax2[j]].axis('off')


for j in range(0,len_folds+1):  
       
    axes[ax1[j], ax2[j]].label_outer() 
    
    if j==0:
        i=7
    else:
        i=j-1     
        
    if j==len_folds:
        axes[ax1[j], ax2[j]].axis('off')
        
        mylegends = [Line2D([0], [0], linestyle='--', color='gray', lw=1)]
        
        for k in range(0,len(methods_ls)):
            mylegends.append(Line2D([0], [0], color=color_ls[methods_ls[k]], lw=1))

        new_methods_ls=[['No skill'], methods_ls]
        new_methods_ls = list(chain(*new_methods_ls))
        axes[ax1[j], ax2[j]].legend(mylegends, new_methods_ls, loc='lower left', bbox_to_anchor=(0.05,-0.4), fontsize=7) 

fig.savefig(visual_path+planet+'_final_plots_CV/ROC_'+planet+'_combined_CV_'+str(plotname)+'robustness.pdf')
fig.savefig(visual_path+planet+'_final_plots_CV/ROC_'+planet+'_combined_CV_'+str(plotname)+'robustness.png')

#res, f, ax1, ax2, Y_test, i, data_name='GQlupB', path, alpha=10






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
#ls_results[i]['Y_test']
alpha=10
ax2=[0,1,2,0,1,2,0,1,2]
ax1=[0,0,0,1,1,1,2,2,2]
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.suptitle('PR curves for $\\alpha='+str(alpha)+'$, on template', fontsize=14)


for j in range(0,len_folds+1):
    
    if j<len_folds:
        try:
        
            if j==0:
                i=7
            else:
                i=j-1 
                
            data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
            data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
            data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
            
            X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train=data_train['H2O']
            
            X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid=data_valid['H2O']
            
            X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_test=data_test['H2O']
            
            df_pr, ax = PR_curve_customplot(ls_results[j]['results'], axes, ax1[j], ax2[j], Y_test, i, methods_ls, color_ls, path=visual_path)
            
        except KeyError:
            pass  
    else:
        axes[ax1[j], ax2[j]].axis('off')


for j in range(0,len_folds+1):  
       
    axes[ax1[j], ax2[j]].label_outer() 
    
    if j==0:
        i=7
    else:
        i=j-1     
        
    if j==len_folds:
        axes[ax1[j], ax2[j]].axis('off')
        
        mylegends = [Line2D([0], [0], linestyle='--', color='gray', lw=1)]
        
        for k in range(0,len(methods_ls)):
            mylegends.append(Line2D([0], [0], color=color_ls[methods_ls[k]], lw=1))

        new_methods_ls=[['No skill'], methods_ls]
        new_methods_ls = list(chain(*new_methods_ls))
        axes[ax1[j], ax2[j]].legend(mylegends, new_methods_ls, loc='lower left', bbox_to_anchor=(0.05,-0.4), fontsize=7) 

fig.savefig(visual_path+planet+'_final_plots_CV/PR_'+planet+'_combined_CV_'+str(plotname)+'robustness.pdf')
fig.savefig(visual_path+planet+'_final_plots_CV/PR_'+planet+'_combined_CV_'+str(plotname)+'robustness.png')

#res, f, ax1, ax2, Y_test, i, data_name='GQlupB', path, alpha=10




# =============================================================================
# Aggregated ROC curves 
# =============================================================================

methods=['SNR','SNR_auto','RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2', 'DNN', 'PCT', 'CNN1', 'CNN2']

plt.figure()
plt.plot(np.array([0., 1.]),np.array([0., 1.]), linestyle='--', lw=1, color='gray')#, label='No Skill')
for m in methods:
    
    predictions_Y = []
    true_Y = []
    probability_Y = []
    
    for j in range(0,len_folds):
        Y = []
        Y_hat = []
        prob_Y = []
        
        if j<len_folds:
            try:
                if j==0:
                    i=7
                else:
                    i=j-1
                
                data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
                data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
                data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
                
                X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_train=data_train['H2O']
                
                X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_valid=data_valid['H2O']
                
                X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
                Y_test=data_test['H2O']
                              
                
                Y=list(data_test['H2O'])
                Y_hat=list(ls_results[j]['results'][m]['Y_pred'])
                #ax.label_outer()  
                
                
                if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                    prob_Y=ls_results[j]['results'][m]['Y_pred_prob']  
                    
                elif m in ['SNR', 'SNR_auto']:
                    prob_Y = ls_results[j]['results'][m]['SNR']
                else:
                    prob_Y = ls_results[j]['results'][m]['Y_pred_prob'][:,1]          
                
                predictions_Y.append(Y_hat)
                true_Y.append(Y)
                probability_Y.append(prob_Y)


            except KeyError:
                pass
            
           
    df_Y=pd.DataFrame({'Y_true': flatten(true_Y),'Y_pred': flatten(predictions_Y), 'Y_prob': flatten(probability_Y)})
    df_Y.to_csv(csv_res_path+"csv_results/HatVSPred_"+str(m)+".csv")
    
    sum(df_Y['Y_true'] == df_Y['Y_pred'])/len(df_Y['Y_true'])

    
    
    ns_probs_0 = [0 for _ in range(len(df_Y['Y_true']))]
         # predict probabilities
    lr_probs_0 = df_Y['Y_prob']
    testy_0=df_Y['Y_true'] 
        # plot the roc curve for the model
        #yhat_0 = res[methods_ls[k]]['Y_pred']
        #ns_auc_0 = roc_auc_score(testy_0, ns_probs_0)
    lr_auc_0 = roc_auc_score(testy_0, lr_probs_0)
        
   
    ns_fpr_0, ns_tpr_0, _ = roc_curve(testy_0, ns_probs_0)
    lr_fpr_0, lr_tpr_0, _ = roc_curve(testy_0, lr_probs_0)
  
    auc_ROC=roc_auc_score(df_Y['Y_true'], df_Y['Y_prob'])
    lr_precision, lr_recall, _ = precision_recall_curve(df_Y['Y_true'], df_Y['Y_prob'])
    lr_f1, lr_auc = f1_score(df_Y['Y_true'], df_Y['Y_pred']), auc(lr_recall, lr_precision)  
         
     
     
    testy_0=df_Y['Y_true']
    ns_probs_0 = [0 for _ in range(len(testy_0))]
        # predict probabilities
    lr_probs_0 = df_Y['Y_prob']
         # plot the roc curve for the model
         #yhat_0 = res[methods_ls[k]]['Y_pred']
         #ns_auc_0 = roc_auc_score(testy_0, ns_probs_0)
    lr_auc_0 = roc_auc_score(testy_0, lr_probs_0) 
    ns_fpr_0, ns_tpr_0, _ = roc_curve(testy_0, ns_probs_0)
    lr_fpr_0, lr_tpr_0, _ = roc_curve(testy_0, lr_probs_0)
    plt.plot(lr_fpr_0, lr_tpr_0, lw=1, color=color_ls[m], label=m+' AUC: '+ str(round(auc_ROC,3)))        

     #axarr = f.add_subplot(3,3,i+1
     # plot the roc curve for the odel


plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title('Aggregated ROC Curve over all CV folds, Template 5')
plt.legend()
plt.savefig('C:/Users/emily/Documents/ML_spectroscopy_thesis/90_sweave_template/Aggregated_ROC_robustness.pdf')
plt.show() 

  
