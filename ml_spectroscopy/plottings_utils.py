# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:33:00 2021

utility function file to create functions for plotting of the results. 

@author: emily
"""


#######################################################

# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

import seaborn as sns




def ROC_curve_plot(res, f, ax1, ax2, Y_test, i, path, data_name='GQlupB', alpha=10):

            
    testy0=Y_test
    
    ns_probs0 = [0 for _ in range(len(testy0))]
    # predict probabilities
    lr_probs0 = res['SNR']['SNR']
    # calculate scores
    ns_auc0 = roc_auc_score(testy0, ns_probs0)
    lr_auc0 = roc_auc_score(testy0, lr_probs0)
    # summarize scores
    
    # calculate roc curves
    ns_fpr0, ns_tpr0, _ = roc_curve(testy0, ns_probs0)
    lr_fpr0, lr_tpr0, _ = roc_curve(testy0, lr_probs0)
    # plot the roc curve for the model
    
    
    # corrected SNR
    testy4=Y_test
    ns_probs4 = [0 for _ in range(len(testy4))]
    # predict probabilities
    lr_probs4 = res['SNR_auto']['SNR']
    # calculate scores
    ns_auc4 = roc_auc_score(testy4, ns_probs4)
    lr_auc4 = roc_auc_score(testy4, lr_probs4)
    # summarize scores
    
    # calculate roc curves
    ns_fpr4, ns_tpr4, _ = roc_curve(testy4, ns_probs4)
    lr_fpr4, lr_tpr4, _ = roc_curve(testy4, lr_probs4)
    # plot the roc curve for the model
    
    
    # RF
    testy1=Y_test
    ns_probs1 = [0 for _ in range(len(testy1))]
    # predict probabilities
    lr_probs1 = res['RF']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc1 = roc_auc_score(testy0, ns_probs1)
    lr_auc1 = roc_auc_score(testy0, lr_probs1)
    # summarize scores
    # calculate roc curves
    ns_fpr1, ns_tpr1, _ = roc_curve(testy1, ns_probs1)
    lr_fpr1, lr_tpr1, _ = roc_curve(testy1, lr_probs1)
    
    
    # NNET2
    testy3=Y_test
    ns_probs3 = [0 for _ in range(len(testy3))]
    # predict probabilities
    lr_probs3 = res['NNET2']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc3 = roc_auc_score(testy3, ns_probs3)
    lr_auc3 = roc_auc_score(testy3, lr_probs3)
    # summarize scores
    # calculate roc curves
    ns_fpr3, ns_tpr3, _ = roc_curve(testy3, ns_probs3)
    lr_fpr3, lr_tpr3, _ = roc_curve(testy3, lr_probs3)
    
    # NNET1
    testy=Y_test
    ns_probs = [0 for _ in range(len(testy))]
    # predict probabilities
    # keep probabilities for the positive outcome only
    lr_probs = res['NNET']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    
    
    
    # save a full version of the plot
    #saveplot_ROC(path, data_name, alpha, i, ns_fpr, ns_tpr, lr_fpr, lr_tpr, lr_fpr3, lr_tpr3, lr_fpr1, lr_tpr1, lr_fpr0, lr_tpr0, lr_fpr4, lr_tpr4,ns_auc0,lr_auc,lr_auc3,lr_auc1,lr_auc0,lr_auc4)

# =============================================================================
#     # CNN
#     testy=Y_test
#     ns_probsCNN = [0 for _ in range(len(testy))]
#     # predict probabilities
#     lr_probsCNN = prediction_probsCNN
#     # keep probabilities for the positive outcome only
#     lr_probsCNN = lr_probsCNN[:, 1]
#     # calculate scores
#     ns_aucCNN = roc_auc_score(testy, ns_probsCNN)
#     lr_aucCNN = roc_auc_score(testy, lr_probsCNN)
#     # summarize scores
#     # calculate roc curves
#     ns_fprCNN, ns_tprCNN, _ = roc_curve(testy, ns_probsCNN)
#     lr_fprCNN, lr_tprCNN, _ = roc_curve(testy, lr_probsCNN)
# =============================================================================
    if i == 7: 
        j = 0
    else: 
        j = i + 1
    #axarr = f.add_subplot(3,3,i+1)
    # plot the roc curve for the model
    f[ax1, ax2].plot(ns_fpr, ns_tpr, linestyle='--', lw=1, color='gray')#, label='No Skill')
    f[ax1, ax2].plot(lr_fpr, lr_tpr, lw=1, color='purple')#, label='NNET')
    f[ax1, ax2].plot(lr_fpr3, lr_tpr3, lw=1, color='blue')#, label='NNET2')
    #pyplot.plot(lr_fprCNN, lr_tprCNN, marker='.', label='CNN ROC AUC=%.3f' % (lr_aucCNN))
    f[ax1, ax2].plot(lr_fpr1, lr_tpr1, lw=1, color='orange')#, label='RF')
    f[ax1, ax2].plot(lr_fpr0, lr_tpr0,lw=1, color='red')#, label='SNR')
    f[ax1, ax2].plot(lr_fpr4, lr_tpr4, lw=1, color='brown')#, label='SNR_auto')
    f[ax1, ax2].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].set_ylabel('True positive rate')
    f[ax1, ax2].set_xlabel('False positive rate')
    f[ax1, ax2].set_title('Cross-validation test fold: '+str(j))

    #f.legend(loc="lower left", fontsize = 'x-small')
    
    #import seaborn as sns
    #sns.lineplot(x=lr_fpr, y=lr_tpr, lw=2, ax=f[ax1, ax2])
    
    # fit the model and plot errors.
    print('No Skill: ROC AUC=%.3f' % (ns_auc0))
    print('PCT: ROC AUC=%.3f' % (lr_auc))
    print('DNN: ROC AUC=%.3f' % (lr_auc3))
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))
    print('RF: ROC AUC=%.3f' % (lr_auc1))
    print('SNR: ROC AUC=%.3f' % (lr_auc0))
    print('SNR_auto: ROC AUC=%.3f' % (lr_auc4))
    
    # fit the model and plot errors.
    df_roc=pd.Series({'No Skill': [ns_auc0], 'PCT': [lr_auc], 'DNN':[lr_auc3], 'RF':[lr_auc1], 'SNR':[lr_auc0], 'SNR_auto':[lr_auc4]})
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return df_roc, f[ax1, ax2]









def ROC_curve_saveplt(res, Y_test, i, path, data_name='GQlupB', alpha=10):

            
    testy0=Y_test
    
    ns_probs0 = [0 for _ in range(len(testy0))]
    # predict probabilities
    lr_probs0 = res['SNR']['SNR']
    # calculate scores
    ns_auc0 = roc_auc_score(testy0, ns_probs0)
    lr_auc0 = roc_auc_score(testy0, lr_probs0)
    # summarize scores   
    # calculate roc curves
    ns_fpr0, ns_tpr0, _ = roc_curve(testy0, ns_probs0)
    lr_fpr0, lr_tpr0, _ = roc_curve(testy0, lr_probs0)
    # plot the roc curve for the model
    
    
    # corrected SNR
    testy4=Y_test
    ns_probs4 = [0 for _ in range(len(testy4))]
    # predict probabilities
    lr_probs4 = res['SNR_auto']['SNR']
    # calculate scores
    ns_auc4 = roc_auc_score(testy4, ns_probs4)
    lr_auc4 = roc_auc_score(testy4, lr_probs4)
    # summarize scores
    
    # calculate roc curves
    ns_fpr4, ns_tpr4, _ = roc_curve(testy4, ns_probs4)
    lr_fpr4, lr_tpr4, _ = roc_curve(testy4, lr_probs4)
    # plot the roc curve for the model
    
    
    # RF
    testy1=Y_test
    ns_probs1 = [0 for _ in range(len(testy1))]
    # predict probabilities
    lr_probs1 = res['RF']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc1 = roc_auc_score(testy0, ns_probs1)
    lr_auc1 = roc_auc_score(testy0, lr_probs1)
    # summarize scores
    # calculate roc curves
    ns_fpr1, ns_tpr1, _ = roc_curve(testy1, ns_probs1)
    lr_fpr1, lr_tpr1, _ = roc_curve(testy1, lr_probs1)
    
    
    # NNET2
    testy3=Y_test
    ns_probs3 = [0 for _ in range(len(testy3))]
    # predict probabilities
    lr_probs3 = res['NNET2']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc3 = roc_auc_score(testy3, ns_probs3)
    lr_auc3 = roc_auc_score(testy3, lr_probs3)
    # summarize scores
    # calculate roc curves
    ns_fpr3, ns_tpr3, _ = roc_curve(testy3, ns_probs3)
    lr_fpr3, lr_tpr3, _ = roc_curve(testy3, lr_probs3)
    
    # NNET1
    testy=Y_test
    ns_probs = [0 for _ in range(len(testy))]
    # predict probabilities
    # keep probabilities for the positive outcome only
    lr_probs = res['NNET']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)  
    
    if i == 7: 
        j = 0
    else: 
        j = i + 1
        
        
    pyplot.figure()
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill AUC=%.3f' % (ns_auc0), color='gray')
    pyplot.plot(lr_fpr, lr_tpr, lw=2, label='PCT AUC=%.3f' % (lr_auc), color='purple')
    pyplot.plot(lr_fpr3, lr_tpr3,lw=2, label='DNN AUC=%.3f' % (lr_auc3), color='blue')
    #pyplot.plot(lr_fprCNN, lr_tprCNN, marker='.', label='CNN ROC AUC=%.3f' % (lr_aucCNN))
    pyplot.plot(lr_fpr1, lr_tpr1, lw=2, label='RF AUC=%.3f' % (lr_auc1), color='orange')
    pyplot.plot(lr_fpr0, lr_tpr0, lw=2, label='SNR AUC=%.3f' % (lr_auc0), color='red')
    pyplot.plot(lr_fpr4, lr_tpr4, lw=2, label='SNR_auto AUC=%.3f' % (lr_auc4), color='brown')    
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.title(str(data_name)+': ROC curves for $\\alpha=$'+str(alpha)+', CV fold: '+str(j))
    # show the legend
    #f.legend(prop={'size':2)
    # show the plot
    pyplot.savefig(path+data_name+'_plt_CV_results/ROC_'+data_name+'_CV_fold'+str(j)+'.pdf')
    pyplot.savefig(path+data_name+'_plt_CV_results/ROC'+data_name+'_CV_fold'+str(j)+'.png')
    pyplot.show()

    
    #sns.lineplot(x=lr_fpr, y=lr_tpr, lw=2, ax=f[ax1, ax2])
    
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return






def PR_curve_plot(res, f, ax1, ax2, Y_test, i, path, data_name='GQlupB', alpha=10):

    testy0=Y_test   
    ns_probs0 = [0 for _ in range(len(testy0))]
    # predict probabilities
    lr_probs0 = res['SNR']['SNR']
    # calculate scores
    ns_auc0 = roc_auc_score(testy0, ns_probs0)
    lr_auc0 = roc_auc_score(testy0, lr_probs0)
    # summarize scores   
    # calculate roc curves
    ns_fpr0, ns_tpr0, _ = roc_curve(testy0, ns_probs0)
    lr_fpr0, lr_tpr0, _ = roc_curve(testy0, lr_probs0)
    # plot the roc curve for the model
    
    
    testy1=Y_test
    ns_probs0 = [0 for _ in range(len(testy1))]
    # predict probabilitiespp
    lr_probs0 =  res['SNR']['SNR']
    # plot the roc curve for the model
    yhat0 = res['SNR']['Y_pred']
    lr_precision0, lr_recall0, _ = precision_recall_curve(testy1, lr_probs0)
    lr_f10, lr_auc0 = f1_score(testy1, yhat0), auc(lr_recall0, lr_precision0)   
    
    
    testy1=Y_test
    ns_probs01 = [0 for _ in range(len(testy1))]
    # predict probabilitiespp
    lr_probs01 =  res['SNR_auto']['SNR']
    # plot the roc curve for the model
    yhat01 = res['SNR_auto']['Y_pred']
    lr_precision01, lr_recall01, _ = precision_recall_curve(testy1, lr_probs01)
    lr_f101, lr_auc01 = f1_score(testy1, yhat01), auc(lr_recall01, lr_precision01)  
    
    
    testy1=Y_test
    ns_probs1 = [0 for _ in range(len(testy1))]
    # predict probabilities
    lr_probs1 = res['RF']['Y_pred_prob'][:, 1]    
    # plot the roc curve for the model
    yhat1 = res['RF']['Y_pred']
    lr_precision1, lr_recall1, _ = precision_recall_curve(testy1, lr_probs1)
    lr_f11, lr_auc1 = f1_score(testy1, yhat1), auc(lr_recall1, lr_precision1)  
    
    
    testy3=Y_test
    ns_probs3 = [0 for _ in range(len(testy1))]
    # predict probabilities
    lr_probs3 = res['NNET2']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhatANN2 = res['NNET2']['Y_pred']
    lr_precision3, lr_recall3, _ = precision_recall_curve(testy3, lr_probs3)
    lr_f13, lr_auc3 = f1_score(testy3, yhatANN2), auc(lr_recall3, lr_precision3)
    
    
    testy=Y_test
    # predict probabilities
    lr_probs = res['NNET']['Y_pred_prob'][:, 1]
    # calculate scores
    # keep probabilities for the positive outcome only
    # predict class values
    yhatANN = res['NNET']['Y_pred']
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhatANN), auc(lr_recall, lr_precision)
    # plot the roc curve for the model
    
    
# =============================================================================
#     ## CNN
#     testy=Y_test
#     # predict probabilities
#     lr_probsCNN = prediction_probsCNN[:, 1]
#     # calculate scores
#     # predict class values
#     yhatCNN = np.argmax(prediction_probsCNN, axis=1)
#     lr_precisionCNN, lr_recallCNN, _ = precision_recall_curve(testy, lr_probsCNN)
#     lr_f1CNN, lr_aucCNN = f1_score(testy, yhatCNN), auc(lr_recallCNN, lr_precisionCNN)
#        
# =============================================================================
    
    
    # save a full version of the plot
    #saveplot_ROC(path, data_name, alpha, i, ns_fpr, ns_tpr, lr_fpr, lr_tpr, lr_fpr3, lr_tpr3, lr_fpr1, lr_tpr1, lr_fpr0, lr_tpr0, lr_fpr4, lr_tpr4,ns_auc0,lr_auc,lr_auc3,lr_auc1,lr_auc0,lr_auc4)


    if i == 7: 
        j = 0
    else: 
        j = i + 1
        
        
        
    no_skill = len(testy[testy==1]) / len(testy)
    f[ax1, ax2].plot([0, 1], [no_skill, no_skill], linestyle='--', lw=1, color='gray')
    f[ax1, ax2].plot(lr_recall, lr_precision, lw=1, color='purple') #NNET
    f[ax1, ax2].plot(lr_recall3, lr_precision3, lw=1, color='blue') #NNET2
    #f[ax1, ax2].plot(lr_recallCNN, lr_precisionCNN, marker='.', label='CNN')
    f[ax1, ax2].plot(lr_recall1, lr_precision1, lw=1, color='orange') # RF
    f[ax1, ax2].plot(lr_recall0, lr_precision0, lw=1, color='red') # SNR
    f[ax1, ax2].plot(lr_recall01, lr_precision01, lw=1, color='brown') # SNR_auto
    f[ax1, ax2].plot(0, 0, marker='.', color='white')
    #axarr = f.add_subplot(3,3,i+1)
    # plot the roc curve for the model
    f[ax1, ax2].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].set_ylabel('Precision')
    f[ax1, ax2].set_xlabel('Recall')
    f[ax1, ax2].set_title('Cross-validation test fold: '+str(j))
    

    #f.legend(loc="lower left", fontsize = 'x-small')
    
    #import seaborn as sns
    #sns.lineplot(x=lr_fpr, y=lr_tpr, lw=2, ax=f[ax1, ax2])
    
    # fit the model and plot errors.
    print('No Skill: PR AUC=%.3f' % (ns_auc0))
    print('PCT: PR f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    print('DNN: PR f1=%.3f auc=%.3f' % (lr_f13, lr_auc3))
    #print('CNN: PR f1=%.3f auc=%.3f' % (lr_f1CNN, lr_aucCNN))
    print('RF: PR  f1=%.3f auc=%.3f' % (lr_f11, lr_auc1))
    print('SNR: PR  f1=%.3f auc=%.3f' % (lr_f10, lr_auc0))
    print('SNR_auto: PR  f1=%.3f auc=%.3f' % (lr_f101, lr_auc01))
    
    # fit the model and plot errors.
    df_pr=pd.Series({'No Skill': [ns_auc0], 'NNET': [(lr_f1, lr_auc)], 'NNET2':[(lr_f13, lr_auc3)], 'RF':[(lr_f11, lr_auc1)], 'SNR':[(lr_f10, lr_auc0)], 'SNR_auto':[(lr_f101, lr_auc01)]})
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return df_pr, f[ax1, ax2]








def PR_curve_saveplt(res, Y_test, i, path, data_name='GQlupB', alpha=10):

 
    testy0=Y_test   
    ns_probs0 = [0 for _ in range(len(testy0))]
    # predict probabilities
    lr_probs0 = res['SNR']['SNR']
    # calculate scores
    ns_auc0 = roc_auc_score(testy0, ns_probs0)
    lr_auc0 = roc_auc_score(testy0, lr_probs0)
    # summarize scores   
    # calculate roc curves
    ns_fpr0, ns_tpr0, _ = roc_curve(testy0, ns_probs0)
    lr_fpr0, lr_tpr0, _ = roc_curve(testy0, lr_probs0)
    # plot the roc curve for the model
    
    
    testy1=Y_test
    ns_probs0 = [0 for _ in range(len(testy1))]
    # predict probabilitiespp
    lr_probs0 =  res['SNR']['SNR']
    # plot the roc curve for the model
    yhat0 = res['SNR']['Y_pred']
    lr_precision0, lr_recall0, _ = precision_recall_curve(testy1, lr_probs0)
    lr_f10, lr_auc0 = f1_score(testy1, yhat0), auc(lr_recall0, lr_precision0)   
    
    
    testy1=Y_test
    ns_probs01 = [0 for _ in range(len(testy1))]
    # predict probabilitiespp
    lr_probs01 =  res['SNR_auto']['SNR']
    # plot the roc curve for the model
    yhat01 = res['SNR_auto']['Y_pred']
    lr_precision01, lr_recall01, _ = precision_recall_curve(testy1, lr_probs01)
    lr_f101, lr_auc01 = f1_score(testy1, yhat01), auc(lr_recall01, lr_precision01)  
    
    
    testy1=Y_test
    ns_probs1 = [0 for _ in range(len(testy1))]
    # predict probabilities
    lr_probs1 = res['RF']['Y_pred_prob'][:, 1]    
    # plot the roc curve for the model
    yhat1 = res['RF']['Y_pred']
    lr_precision1, lr_recall1, _ = precision_recall_curve(testy1, lr_probs1)
    lr_f11, lr_auc1 = f1_score(testy1, yhat1), auc(lr_recall1, lr_precision1)  
    
    
    testy3=Y_test
    ns_probs3 = [0 for _ in range(len(testy1))]
    # predict probabilities
    lr_probs3 = res['NNET2']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhatANN2 = res['NNET2']['Y_pred']
    lr_precision3, lr_recall3, _ = precision_recall_curve(testy3, lr_probs3)
    lr_f13, lr_auc3 = f1_score(testy3, yhatANN2), auc(lr_recall3, lr_precision3)
    
    
    testy=Y_test
    # predict probabilities
    lr_probs = res['NNET']['Y_pred_prob'][:, 1]
    # calculate scores
    # keep probabilities for the positive outcome only
    # predict class values
    yhatANN = res['NNET']['Y_pred']
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhatANN), auc(lr_recall, lr_precision)
    # plot the roc curve for the model
    
    
# =============================================================================
#     ## CNN
#     testy=Y_test
#     # predict probabilities
#     lr_probsCNN = prediction_probsCNN[:, 1]
#     # calculate scores
#     # predict class values
#     yhatCNN = np.argmax(prediction_probsCNN, axis=1)
#     lr_precisionCNN, lr_recallCNN, _ = precision_recall_curve(testy, lr_probsCNN)
#     lr_f1CNN, lr_aucCNN = f1_score(testy, yhatCNN), auc(lr_recallCNN, lr_precisionCNN)
#        
# =============================================================================
    
    
    # save a full version of the plot
    #saveplot_ROC(path, data_name, alpha, i, ns_fpr, ns_tpr, lr_fpr, lr_tpr, lr_fpr3, lr_tpr3, lr_fpr1, lr_tpr1, lr_fpr0, lr_tpr0, lr_fpr4, lr_tpr4,ns_auc0,lr_auc,lr_auc3,lr_auc1,lr_auc0,lr_auc4)


    if i == 7: 
        j = 0
    else: 
        j = i + 1
        

    # summarize scores
    # plot the precision-recall curves
    pyplot.figure()
    no_skill = len(testy[testy==1]) / len(testy)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill: PR AUC=%.3f' % (ns_auc0), color='gray')
    pyplot.plot(lr_recall, lr_precision, lw=2, label='PCT: PR f1=%.3f auc=%.3f' % (lr_f1, lr_auc), color='purple')
    pyplot.plot(lr_recall3, lr_precision3,lw=2, label='DNN: PR f1=%.3f auc=%.3f' % (lr_f13, lr_auc3), color='blue')
    #pyplot.plot(lr_recallCNN, lr_precisionCNN, marker='.', label='CNN')
    pyplot.plot(lr_recall1, lr_precision1, lw=2, label='RF: PR  f1=%.3f auc=%.3f' % (lr_f11, lr_auc1), color='orange')
    pyplot.plot(lr_recall0, lr_precision0, lw=2, label='SNR: PR  f1=%.3f auc=%.3f' % (lr_f10, lr_auc0), color='red')
    pyplot.plot(lr_recall01, lr_precision01, lw=2, label='SNR_auto: PR  f1=%.3f auc=%.3f' % (lr_f101, lr_auc01), color='brown')
    pyplot.plot(0, 0, marker='.', color='white')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.title(str(data_name)+': PR curves for $\\alpha=$'+str(alpha)+', CV fold: '+str(j))
    # show the legend
    pyplot.legend()
    #f.legend(prop={'size':2)
    # show the plot
    pyplot.savefig(path+data_name+'_plt_CV_results/PR_'+data_name+'_CV_fold'+str(j)+'.pdf')
    pyplot.savefig(path+data_name+'_plt_CV_results/PR_'+data_name+'_CV_fold'+str(j)+'.png')
    # show the legend
    # show the plot
    pyplot.show()    
    #sns.lineplot(x=lr_fpr, y=lr_tpr, lw=2, ax=f[ax1, ax2])  
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))
    return


    

# precision-recall curve and f1



# summarize scores
# plot the precision-recall curves



# show the plot
pyplot.show()


# alpha =100
#ANN 1 : array([0.64938551, 0.00254273, 0.02161322, 0.32645854])
#SNR: array([0.42477751, 0.22715073, 0.00904082, 0.33903094])
# ANN 2:  array([0.641051  , 0.01087724, 0.01709281, 0.33097895])



