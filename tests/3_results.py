# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 01:38:10 2021

@author: emily
"""

## LIBRARIES

import random
import pickle
import os
from matplotlib.lines import Line2D


#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.config import path_init

import matplotlib.pyplot as plt
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
            
            X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train=data_train['H2O']
            
            X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid=data_valid['H2O']
            
            X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
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

fig.savefig(visual_path+planet+'_plt_CV_results/ROC_'+planet+'_combined_CV.pdf')
fig.savefig(visual_path+planet+'_plt_CV_results/ROC_'+planet+'_combined_CV.png')






for i in range(0,len_folds):
    try: 
        if i==7:
            j=0
        else:
            j=i+1
        data_train=data1.drop([(str(data1.index.levels[0][j]),)], axis=0).drop([(str(data1.index.levels[0][i]),)], axis=0)
        data_valid=data1.loc[(str(data1.index.levels[0][i]), slice(None)), :]
        data_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), :]
        
        X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_train=data_train['H2O']
        
        X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_valid=data_valid['H2O']
        
        X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_test=data_test['H2O']
            
        
        ROC_curve_saveplt(ls_results[i], Y_test, i, path=visual_path)
    except KeyError:
        pass




########### Precision Recall curves

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
            
            X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_train=data_train['H2O']
            
            X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
            Y_valid=data_valid['H2O']
            
            X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
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
        
        X_train=data_train.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_train=data_train['H2O']
        
        X_valid=data_valid.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_valid=data_valid['H2O']
        
        X_test=data_test.drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
        Y_test=data_test['H2O']
        
        PR_curve_saveplt(ls_results[i], Y_test, i, path=visual_path)
    except KeyError:
        pass

