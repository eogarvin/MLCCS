# -*- coding: utf-8 -*-
"""
Created on Tue Jan 1 15:33:00 2022

utility function file to create functions for plotting of the results. 

@author: emily
"""


#######################################################

# roc curve and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot






def ROC_curve_customplot(res, f, ax1, ax2, Y_test, j, methods_ls, color_ls):

    testy0=Y_test
    
    ns_probs0 = [0 for _ in range(len(testy0))]
    # predict probabilities
    lr_probs0 = res['SNR']['SNR']
    # calculate scores
    ns_auc0 = roc_auc_score(testy0, ns_probs0)
    #lr_auc0 = roc_auc_score(testy0, lr_probs0)
    # summarize scores
    
    # calculate roc curves
    ns_fpr0, ns_tpr0, _ = roc_curve(testy0, ns_probs0)
    lr_fpr0, lr_tpr0, _ = roc_curve(testy0, lr_probs0)
    # plot the roc curve for the model
        
    
    #no_skill = len(Y_test[Y_test==1]) / len(Y_test)
    keys=methods_ls
    ls_mtd = {key: None for key in keys}

    for k in range(0,len(methods_ls)):

        testy_0=Y_test
        ns_probs_0 = [0 for _ in range(len(testy_0))]
        # predict probabilities
        if methods_ls[k] in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
            lr_probs_0 = res[methods_ls[k]]['Y_pred_prob']
            
        elif methods_ls[k] in ['SNR', 'SNR_auto']:
            lr_probs_0 = res[methods_ls[k]]['SNR']
            
        else:
            lr_probs_0 = res[methods_ls[k]]['Y_pred_prob'][:, 1]
        # plot the roc curve for the model
        lr_auc_0 = roc_auc_score(testy_0, lr_probs_0)
        ns_fpr_0, ns_tpr_0, _ = roc_curve(testy_0, ns_probs_0)
        lr_fpr_0, lr_tpr_0, _ = roc_curve(testy_0, lr_probs_0)
        
        f[ax1, ax2].plot(lr_fpr_0, lr_tpr_0, lw=1, color=color_ls[methods_ls[k]])
        ls_mtd[methods_ls[k]]=[lr_auc_0]

    #axarr = f.add_subplot(3,3,i+1)
    # plot the roc curve for the model
    f[ax1, ax2].plot(ns_fpr0, ns_tpr0, linestyle='--', lw=1, color='gray')#, label='No Skill')
    f[ax1, ax2].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].set_ylabel('True positive rate')
    f[ax1, ax2].set_xlabel('False positive rate')
    f[ax1, ax2].set_title('CV test fold: '+str(j))

    ls_mtd['noskill']=[(ns_auc0)]
    
    # store AUCs
    df_roc=pd.Series(ls_mtd)  

    return df_roc, f[ax1, ax2]



# =============================================================================
# save plot one by one  
# =============================================================================

def ROC_curve_saveplt(res, Y_test, j, color_ls,  path, data_name='GQlupB', alpha=10):

            
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
    testy_SNR=Y_test
    ns_probs_SNR = [0 for _ in range(len(testy_SNR))]
    # predict probabilities
    lr_probs_SNR = res['SNR']['SNR']
    # calculate scores
    ns_auc_SNR = roc_auc_score(testy_SNR, ns_probs_SNR)
    lr_auc_SNR = roc_auc_score(testy_SNR, lr_probs_SNR)
    # summarize scores
    
    # calculate roc curves
    ns_fpr_SNR, ns_tpr_SNR, _ = roc_curve(testy_SNR, ns_probs_SNR)
    lr_fpr_SNR, lr_tpr_SNR, _ = roc_curve(testy_SNR, lr_probs_SNR)
    # plot the roc curve for the model
    
    
    # corrected SNRauto
    testy_SNRauto=Y_test
    ns_probs_SNRauto = [0 for _ in range(len(testy_SNRauto))]
    # predict probabilities
    lr_probs_SNRauto = res['SNR_auto']['SNR']
    # calculate scores
    ns_auc_SNRauto = roc_auc_score(testy_SNRauto, ns_probs_SNRauto)
    lr_auc_SNRauto = roc_auc_score(testy_SNRauto, lr_probs_SNRauto)
    # summarize scores
    
    # calculate roc curves
    ns_fpr_SNRauto, ns_tpr_SNRauto, _ = roc_curve(testy_SNRauto, ns_probs_SNRauto)
    lr_fpr_SNRauto, lr_tpr_SNRauto, _ = roc_curve(testy_SNRauto, lr_probs_SNRauto)
    # plot the roc curve for the model
    
    
    # DNN
    testy_CNN2=Y_test
    ns_probs_CNN2 = [0 for _ in range(len(testy_CNN2))]
    # predict probabilities
    lr_probs_CNN2 = res['CNN2']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc_CNN2 = roc_auc_score(testy_CNN2, ns_probs_CNN2)
    lr_auc_CNN2 = roc_auc_score(testy_CNN2, lr_probs_CNN2)
    # summarize scores
    # calculate roc curves
    ns_fpr_CNN2, ns_tpr_CNN2, _ = roc_curve(testy_CNN2, ns_probs_CNN2)
    lr_fpr_CNN2, lr_tpr_CNN2, _ = roc_curve(testy_CNN2, lr_probs_CNN2)
    
    
    # DNN
    testy_CNN1=Y_test
    ns_probs_CNN1 = [0 for _ in range(len(testy_CNN1))]
    # predict probabilities
    lr_probs_CNN1 = res['CNN1']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc_CNN1 = roc_auc_score(testy_CNN1, ns_probs_CNN1)
    lr_auc_CNN1 = roc_auc_score(testy_CNN1, lr_probs_CNN1)
    # summarize scores
    # calculate roc curves
    ns_fpr_CNN1, ns_tpr_CNN1, _ = roc_curve(testy_CNN1, ns_probs_CNN1)
    lr_fpr_CNN1, lr_tpr_CNN1, _ = roc_curve(testy_CNN1, lr_probs_CNN1)
    
    
    # DNN
    testy_DNN=Y_test
    ns_probs_DNN = [0 for _ in range(len(testy_DNN))]
    # predict probabilities
    lr_probs_DNN = res['DNN']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc_DNN = roc_auc_score(testy_DNN, ns_probs_DNN)
    lr_auc_DNN = roc_auc_score(testy_DNN, lr_probs_DNN)
    # summarize scores
    # calculate roc curves
    ns_fpr_DNN, ns_tpr_DNN, _ = roc_curve(testy_DNN, ns_probs_DNN)
    lr_fpr_DNN, lr_tpr_DNN, _ = roc_curve(testy_DNN, lr_probs_DNN)
 
    
    # DNN
    testy_PCT=Y_test
    ns_probs_PCT = [0 for _ in range(len(testy_PCT))]
    # predict probabilities
    lr_probs_PCT = res['PCT']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc_PCT = roc_auc_score(testy_PCT, ns_probs_PCT)
    lr_auc_PCT = roc_auc_score(testy_PCT, lr_probs_PCT)
    # summarize scores
    # calculate roc curves
    ns_fpr_PCT, ns_tpr_PCT, _ = roc_curve(testy_PCT, ns_probs_PCT)
    lr_fpr_PCT, lr_tpr_PCT, _ = roc_curve(testy_PCT, lr_probs_PCT)   
    
    # DNN
    testy_LAS=Y_test
    ns_probs_LAS = [0 for _ in range(len(testy_LAS))]
    # predict probabilities
    lr_probs_LAS = res['LAS']['Y_pred_prob']
    # calculate scores
    ns_auc_LAS = roc_auc_score(testy_LAS, ns_probs_LAS)
    lr_auc_LAS = roc_auc_score(testy_LAS, lr_probs_LAS)
    # summarize scores
    # calculate roc curves
    ns_fpr_LAS, ns_tpr_LAS, _ = roc_curve(testy_LAS, ns_probs_LAS)
    lr_fpr_LAS, lr_tpr_LAS, _ = roc_curve(testy_LAS, lr_probs_LAS)
    
    
    # DNN
    testy_RID=Y_test
    ns_probs_RID = [0 for _ in range(len(testy_RID))]
    # predict probabilities
    lr_probs_RID = res['RID']['Y_pred_prob']
    # calculate scores
    ns_auc_RID = roc_auc_score(testy_RID, ns_probs_RID)
    lr_auc_RID = roc_auc_score(testy_RID, lr_probs_RID)
    # summarize scores
    # calculate roc curves
    ns_fpr_RID, ns_tpr_RID, _ = roc_curve(testy_RID, ns_probs_RID)
    lr_fpr_RID, lr_tpr_RID, _ = roc_curve(testy_RID, lr_probs_RID)
    
    
    
    # DNN
    testy_ENET=Y_test
    ns_probs_ENET = [0 for _ in range(len(testy_ENET))]
    # predict probabilities
    lr_probs_ENET = res['ENET']['Y_pred_prob']
    # calculate scores
    ns_auc_ENET = roc_auc_score(testy_ENET, ns_probs_ENET)
    lr_auc_ENET = roc_auc_score(testy_ENET, lr_probs_ENET)
    # summarize scores
    # calculate roc curves
    ns_fpr_ENET, ns_tpr_ENET, _ = roc_curve(testy_ENET, ns_probs_ENET)
    lr_fpr_ENET, lr_tpr_ENET, _ = roc_curve(testy_ENET, lr_probs_ENET)
    
        
    # DNN
    testy_ENET2=Y_test
    ns_probs_ENET2 = [0 for _ in range(len(testy_ENET2))]
    # predict probabilities
    lr_probs_ENET2 = res['ENET2']['Y_pred_prob']
    # calculate scores
    ns_auc_ENET2 = roc_auc_score(testy_ENET2, ns_probs_ENET2)
    lr_auc_ENET2 = roc_auc_score(testy_ENET2, lr_probs_ENET2)
    # summarize scores
    # calculate roc curves
    ns_fpr_ENET2, ns_tpr_ENET2, _ = roc_curve(testy_ENET2, ns_probs_ENET2)
    lr_fpr_ENET2, lr_tpr_ENET2, _ = roc_curve(testy_ENET2, lr_probs_ENET2)
    
        
    # DNN
    testy_RF=Y_test
    ns_probs_RF = [0 for _ in range(len(testy_RF))]
    # predict probabilities
    lr_probs_RF = res['RF']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc_RF = roc_auc_score(testy_RF, ns_probs_RF)
    lr_auc_RF = roc_auc_score(testy_RF, lr_probs_RF)
    # summarize scores
    # calculate roc curves
    ns_fpr_RF, ns_tpr_RF, _ = roc_curve(testy_RF, ns_probs_RF)
    lr_fpr_RF, lr_tpr_RF, _ = roc_curve(testy_RF, lr_probs_RF)
    
        
    # DNN
    testy_XGB=Y_test
    ns_probs_XGB = [0 for _ in range(len(testy_XGB))]
    # predict probabilities
    lr_probs_XGB = res['XGB']['Y_pred_prob']
    # calculate scores
    ns_auc_XGB = roc_auc_score(testy_XGB, ns_probs_XGB)
    lr_auc_XGB = roc_auc_score(testy_XGB, lr_probs_XGB)
    # summarize scores
    # calculate roc curves
    ns_fpr_XGB, ns_tpr_XGB, _ = roc_curve(testy_XGB, ns_probs_XGB)
    lr_fpr_XGB, lr_tpr_XGB, _ = roc_curve(testy_XGB, lr_probs_XGB)
        
        
    pyplot.figure()
    pyplot.plot(ns_fpr0, ns_tpr0, linestyle='--', label='No Skill AUC=%.3f' % (ns_auc0), color='gray')
    pyplot.plot(lr_fpr_PCT, lr_tpr_PCT, label='PCT AUC=%.3f' % (lr_auc_PCT), color=color_ls['PCT'])
    pyplot.plot(lr_fpr_DNN, lr_tpr_DNN, label='DNN AUC=%.3f' % (lr_auc_DNN), color=color_ls['DNN'])
    pyplot.plot(lr_fpr_CNN1, lr_tpr_CNN1, label='CNN1 ROC AUC=%.3f' % (lr_auc_CNN1), color=color_ls['CNN1'])
    pyplot.plot(lr_fpr_CNN2, lr_tpr_CNN2, label='CNN2 ROC AUC=%.3f' % (lr_auc_CNN2), color=color_ls['CNN2'])
    pyplot.plot(lr_fpr_LAS, lr_tpr_LAS, label='LAS ROC AUC=%.3f' % (lr_auc_LAS), color=color_ls['LAS'])
    pyplot.plot(lr_fpr_RID, lr_tpr_RID, label='RID ROC AUC=%.3f' % (lr_auc_RID), color=color_ls['RID'])
    pyplot.plot(lr_fpr_ENET, lr_tpr_ENET, label='ENET ROC AUC=%.3f' % (lr_auc_ENET), color=color_ls['ENET'])
    pyplot.plot(lr_fpr_RF, lr_tpr_RF, label='RF ROC AUC=%.3f' % (lr_auc_RF), color=color_ls['RF'])
    pyplot.plot(lr_fpr_XGB, lr_tpr_XGB, label='XGB ROC AUC=%.3f' % (lr_auc_XGB), color=color_ls['XGB'])
    pyplot.plot(lr_fpr_SNR, lr_tpr_SNR, label='SNR ROC AUC=%.3f' % (lr_auc_SNR), color=color_ls['SNR'])
    pyplot.plot(lr_fpr_SNRauto, lr_tpr_SNRauto, label='SNR_auto ROC AUC=%.3f' % (lr_auc_SNRauto), color=color_ls['SNR_auto'])


    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.title(str(data_name)+': ROC curves for $\\alpha=$'+str(alpha)+', CV fold: '+str(j))
    # show the legend
    pyplot.legend(fontsize=8.5)
    # show the plot
    pyplot.savefig(path+data_name+'_final_plots_CV/ROC_'+data_name+'_CV_fold'+str(j)+'.pdf')
    pyplot.savefig(path+data_name+'_final_plots_CV/ROC'+data_name+'_CV_fold'+str(j)+'.png')
    pyplot.show()

    
    #sns.lineplot(x=lr_fpr, y=lr_tpr, lw=2, ax=f[ax1, ax2])
    
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return






# =============================================================================
# Nice customizable plot 
# =============================================================================

def PR_curve_customplot(res, f, ax1, ax2, Y_test, j, methods_ls, color_ls):
   

    # Set the no skill model
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


    # plot the roc curve for the model

    no_skill = len(Y_test[Y_test==1]) / len(Y_test)
    keys = methods_ls
    keys=methods_ls
    ls_mtd = {key: None for key in keys}



    # Retrieve the precisions and recalls for every model
    for k in range(0,len(methods_ls)):

        testy_0=Y_test
        ns_probs_0 = [0 for _ in range(len(testy_0))]
        # predict probabilities
        if methods_ls[k] in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
            lr_probs_0 = res[methods_ls[k]]['Y_pred_prob']

        elif methods_ls[k] in ['SNR', 'SNR_auto']:
            lr_probs_0 =  res[methods_ls[k]]['SNR']

        else:
            lr_probs_0 = res[methods_ls[k]]['Y_pred_prob'][:, 1]
        # plot the roc curve for the model
        yhat_0 = res[methods_ls[k]]['Y_pred']
        lr_precision_0, lr_recall_0, _ = precision_recall_curve(testy_0, lr_probs_0)
        lr_f1_0, lr_auc_0 = f1_score(testy_0, yhat_0), auc(lr_recall_0, lr_precision_0)

        f[ax1, ax2].plot(lr_recall_0, lr_precision_0, lw=1, color=color_ls[methods_ls[k]]) #PCT

        ls_mtd[methods_ls[k]]=[(lr_f1_0, lr_auc_0)]

    f[ax1, ax2].plot([0, 1], [no_skill, no_skill], linestyle='--', lw=1, color='gray')
    f[ax1, ax2].plot(0, 0, marker='.', color='white')
    #axarr = f.add_subplot(3,3,i+1)
    # plot the roc curve for the model
    f[ax1, ax2].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].set_ylabel('Precision')
    f[ax1, ax2].set_xlabel('Recall')
    f[ax1, ax2].set_title('CV test fold: '+str(j))



    ls_mtd['noskill']=[(ns_auc0, ns_auc0)]
    # fit the model and plot errors.
    df_pr=pd.Series(ls_mtd)
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return df_pr, f[ax1, ax2]




# =============================================================================
# Save plots one by one
# =============================================================================
def PR_curve_saveplt(res, Y_test, j, color_ls, path, data_name='GQlupB', alpha=10):

    # Retrieve the precisions and recalls for every model
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
     
    testy_SNR=Y_test
    ns_probs_SNR = [0 for _ in range(len(testy_SNR))]
    # predict probabilitiespp
    lr_probs_SNR =  res['SNR']['SNR']
    # plot the roc curve for the model
    yhat_SNR = res['SNR']['Y_pred']
    lr_precision_SNR, lr_recall_SNR, _ = precision_recall_curve(testy_SNR, lr_probs_SNR)
    lr_f1_SNR, lr_auc_SNR = f1_score(testy_SNR, yhat_SNR), auc(lr_recall_SNR, lr_precision_SNR)  
 
    testy_SNRauto=Y_test
    ns_probs_SNRauto = [0 for _ in range(len(testy_SNRauto))]
    # predict probabilitiespp
    lr_probs_SNRauto =  res['SNR_auto']['SNR']
    # plot the roc curve for the model
    yhat_SNRauto = res['SNR_auto']['Y_pred']
    lr_precision_SNRauto, lr_recall_SNRauto, _ = precision_recall_curve(testy_SNRauto, lr_probs_SNRauto)
    lr_f1_SNRauto, lr_auc_SNRauto = f1_score(testy_SNRauto, yhat_SNRauto), auc(lr_recall_SNRauto, lr_precision_SNRauto)  
 
    testy_PCT=Y_test
    ns_probs_PCT = [0 for _ in range(len(testy_PCT))]
    # predict probabilities
    lr_probs_PCT = res['PCT']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_PCT = res['PCT']['Y_pred']
    lr_precision_PCT, lr_recall_PCT, _ = precision_recall_curve(testy_PCT, lr_probs_PCT)
    lr_f1_PCT, lr_auc_PCT = f1_score(testy_PCT, yhat_PCT), auc(lr_recall_PCT, lr_precision_PCT) 
    
    testy_DNN=Y_test
    ns_probs_DNN = [0 for _ in range(len(testy_DNN))]
    # predict probabilities
    lr_probs_DNN = res['DNN']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_DNN = res['DNN']['Y_pred']
    lr_precision_DNN, lr_recall_DNN, _ = precision_recall_curve(testy_DNN, lr_probs_DNN)
    lr_f1_DNN, lr_auc_DNN = f1_score(testy_DNN, yhat_DNN), auc(lr_recall_DNN, lr_precision_DNN)
     
    testy_CNN1=Y_test
    ns_probs_CNN1 = [0 for _ in range(len(testy_CNN1))]
    # predict probabilities
    lr_probs_CNN1 = res['CNN1']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_CNN1 = res['CNN1']['Y_pred']
    lr_precision_CNN1, lr_recall_CNN1, _ = precision_recall_curve(testy_CNN1, lr_probs_CNN1)
    lr_f1_CNN1, lr_auc_CNN1 = f1_score(testy_CNN1, yhat_CNN1), auc(lr_recall_CNN1, lr_precision_CNN1)
   
    testy_CNN2=Y_test
    ns_probs_CNN2 = [0 for _ in range(len(testy_CNN2))]
    # predict probabilities
    lr_probs_CNN2 = res['CNN2']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_CNN2 = res['CNN2']['Y_pred']
    lr_precision_CNN2, lr_recall_CNN2, _ = precision_recall_curve(testy_CNN2, lr_probs_CNN2)
    lr_f1_CNN2, lr_auc_CNN2 = f1_score(testy_CNN2, yhat_CNN2), auc(lr_recall_CNN2, lr_precision_CNN2)
 
    testy_ENET=Y_test
    ns_probs_ENET = [0 for _ in range(len(testy_ENET))]
    # predict probabilities
    lr_probs_ENET = res['ENET']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_ENET = res['ENET']['Y_pred']
    lr_precision_ENET, lr_recall_ENET, _ = precision_recall_curve(testy_ENET, lr_probs_ENET)
    lr_f1_ENET, lr_auc_ENET = f1_score(testy_ENET, yhat_ENET), auc(lr_recall_ENET, lr_precision_ENET)
   
    testy_RID=Y_test
    ns_probs_RID = [0 for _ in range(len(testy_RID))]
    # predict probabilities
    lr_probs_RID = res['RID']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_RID = res['RID']['Y_pred']
    lr_precision_RID, lr_recall_RID, _ = precision_recall_curve(testy_RID, lr_probs_RID)
    lr_f1_RID, lr_auc_RID = f1_score(testy_RID, yhat_RID), auc(lr_recall_RID, lr_precision_RID)
   
    testy_LAS=Y_test
    ns_probs_LAS = [0 for _ in range(len(testy_LAS))]
    # predict probabilities
    lr_probs_LAS = res['LAS']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_LAS = res['LAS']['Y_pred']
    lr_precision_LAS, lr_recall_LAS, _ = precision_recall_curve(testy_LAS, lr_probs_LAS)
    lr_f1_LAS, lr_auc_LAS = f1_score(testy_LAS, yhat_LAS), auc(lr_recall_LAS, lr_precision_LAS)
 
    testy_ENET2=Y_test
    ns_probs_ENET2 = [0 for _ in range(len(testy_ENET2))]
    # predict probabilities
    lr_probs_ENET2 = res['ENET2']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_ENET2 = res['ENET2']['Y_pred']
    lr_precision_ENET2, lr_recall_ENET2, _ = precision_recall_curve(testy_ENET2, lr_probs_ENET2)
    lr_f1_ENET2, lr_auc_ENET2 = f1_score(testy_ENET2, yhat_ENET2), auc(lr_recall_ENET2, lr_precision_ENET2)
   
    testy_RF=Y_test
    ns_probs_RF = [0 for _ in range(len(testy_RF))]
    # predict probabilities
    lr_probs_RF = res['RF']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_RF = res['RF']['Y_pred']
    lr_precision_RF, lr_recall_RF, _ = precision_recall_curve(testy_RF, lr_probs_RF)
    lr_f1_RF, lr_auc_RF = f1_score(testy_RF, yhat_RF), auc(lr_recall_RF, lr_precision_RF)
   
    testy_XGB=Y_test
    ns_probs_XGB = [0 for _ in range(len(testy_XGB))]
    # predict probabilities
    lr_probs_XGB = res['XGB']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_XGB = res['XGB']['Y_pred']
    lr_precision_XGB, lr_recall_XGB, _ = precision_recall_curve(testy_XGB, lr_probs_XGB)
    lr_f1_XGB, lr_auc_XGB = f1_score(testy_XGB, yhat_XGB), auc(lr_recall_XGB, lr_precision_XGB)


    # summarize scores
    # plot the precision-recall curves
    pyplot.figure()
    no_skill = len(Y_test[Y_test==1]) / len(Y_test)

    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=1, color='gray')
    pyplot.plot(lr_recall_CNN2, lr_precision_CNN2, label='CNN2: PR f1=%.3f auc=%.3f' % (lr_f1_CNN2, lr_auc_CNN2), color=color_ls['CNN2'])
    pyplot.plot(lr_recall_CNN1, lr_precision_CNN1, label='CNN1: PR f1=%.3f auc=%.3f' % (lr_f1_CNN1, lr_auc_CNN1), color=color_ls['CNN1'])
    pyplot.plot(lr_recall_DNN, lr_precision_DNN,  label='DNN: PR f1=%.3f auc=%.3f' % (lr_f1_DNN, lr_auc_DNN), color=color_ls['DNN']) #DNN
    pyplot.plot(lr_recall_PCT, lr_precision_PCT, label='PCT: PR f1=%.3f auc=%.3f' % (lr_f1_PCT, lr_auc_PCT), color=color_ls['PCT']) #PCT
    pyplot.plot(lr_recall_LAS, lr_precision_LAS, label='LAS: PR f1=%.3f auc=%.3f' % (lr_f1_LAS, lr_auc_LAS), color=color_ls['LAS'])
    pyplot.plot(lr_recall_RID, lr_precision_RID, label='RID: PR f1=%.3f auc=%.3f' % (lr_f1_RID, lr_auc_RID), color=color_ls['RID'])
    pyplot.plot(lr_recall_ENET, lr_precision_ENET, label='ENET: PR f1=%.3f auc=%.3f' % (lr_f1_ENET, lr_auc_ENET), color=color_ls['ENET'])
    pyplot.plot(lr_recall_RF, lr_precision_RF, label='RF: PR  f1=%.3f auc=%.3f' % (lr_f1_RF, lr_auc_RF), color=color_ls['RF']) # RF
    pyplot.plot(lr_recall_XGB, lr_precision_XGB, label='XGB: PR f1=%.3f auc=%.3f' % (lr_f1_XGB, lr_auc_XGB), color=color_ls['XGB'])
    pyplot.plot(lr_recall_SNR, lr_precision_SNR, label='SNR: PR  f1=%.3f auc=%.3f' % (lr_f1_SNR, lr_auc_SNR), color=color_ls['SNR']) # SNR
    pyplot.plot(lr_recall_SNRauto, lr_precision_SNRauto, label='SNR_auto: PR  f1=%.3f auc=%.3f' % (lr_f1_SNRauto, lr_auc_SNRauto), color=color_ls['SNR_auto']) # SNR_auto
    
    pyplot.plot(0, 0, marker='.', color='white')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.title(str(data_name)+': PR curves for $\\alpha=$'+str(alpha)+', CV fold: '+str(j))

    pyplot.legend(loc='lower right',fontsize=8.5, bbox_to_anchor=(0.5,-0.01))
    
    pyplot.savefig(path+'/PR_'+data_name+'_CV_fold'+str(j)+'.pdf')
    pyplot.savefig(path+'/PR_'+data_name+'_CV_fold'+str(j)+'.png')

    pyplot.show()
    
    return










# =============================================================================
# Of no use anymore:     
# =============================================================================


def PR_curve_plot_old(res, f, ax1, ax2, Y_test, i, path, data_name='GQlupB', alpha=10):

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
    lr_probs3 = res['DNN']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhatANN2 = res['DNN']['Y_pred']
    lr_precision3, lr_recall3, _ = precision_recall_curve(testy3, lr_probs3)
    lr_f13, lr_auc3 = f1_score(testy3, yhatANN2), auc(lr_recall3, lr_precision3)
    
    
    testy=Y_test
    # predict probabilities
    lr_probs = res['PCT']['Y_pred_prob'][:, 1]
    # calculate scores
    # keep probabilities for the positive outcome only
    # predict class values
    yhatANN = res['PCT']['Y_pred']
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
    f[ax1, ax2].plot(lr_recall, lr_precision, lw=1, color='purple') #PCT
    f[ax1, ax2].plot(lr_recall3, lr_precision3, lw=1, color='blue') #DNN
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
    f[ax1, ax2].set_title('CV test fold: '+str(j))
    

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
    df_pr=pd.Series({'No Skill': [ns_auc0], 'PCT': [(lr_f1, lr_auc)], 'DNN':[(lr_f13, lr_auc3)], 'RF':[(lr_f11, lr_auc1)], 'SNR':[(lr_f10, lr_auc0)], 'SNR_auto':[(lr_f101, lr_auc01)]})
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return df_pr, f[ax1, ax2]


def PR_curve_plot_all_old(res, f, ax1, ax2, Y_test, i, path, data_name='GQlupB', alpha=10):

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
     
    testy_SNR=Y_test
    ns_probs_SNR = [0 for _ in range(len(testy_SNR))]
    # predict probabilitiespp
    lr_probs_SNR =  res['SNR']['SNR']
    # plot the roc curve for the model
    yhat_SNR = res['SNR']['Y_pred']
    lr_precision_SNR, lr_recall_SNR, _ = precision_recall_curve(testy_SNR, lr_probs_SNR)
    lr_f1_SNR, lr_auc_SNR = f1_score(testy_SNR, yhat_SNR), auc(lr_recall_SNR, lr_precision_SNR)  
 
    testy_SNRauto=Y_test
    ns_probs_SNRauto = [0 for _ in range(len(testy_SNRauto))]
    # predict probabilitiespp
    lr_probs_SNRauto =  res['SNR_auto']['SNR']
    # plot the roc curve for the model
    yhat_SNRauto = res['SNR_auto']['Y_pred']
    lr_precision_SNRauto, lr_recall_SNRauto, _ = precision_recall_curve(testy_SNRauto, lr_probs_SNRauto)
    lr_f1_SNRauto, lr_auc_SNRauto = f1_score(testy_SNRauto, yhat_SNRauto), auc(lr_recall_SNRauto, lr_precision_SNRauto)  
 
    testy_PCT=Y_test
    ns_probs_PCT = [0 for _ in range(len(testy_PCT))]
    # predict probabilities
    lr_probs_PCT = res['PCT']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_PCT = res['PCT']['Y_pred']
    lr_precision_PCT, lr_recall_PCT, _ = precision_recall_curve(testy_PCT, lr_probs_PCT)
    lr_f1_PCT, lr_auc_PCT = f1_score(testy_PCT, yhat_PCT), auc(lr_recall_PCT, lr_precision_PCT) 
    
    testy_DNN=Y_test
    ns_probs_DNN = [0 for _ in range(len(testy_DNN))]
    # predict probabilities
    lr_probs_DNN = res['DNN']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_DNN = res['DNN']['Y_pred']
    lr_precision_DNN, lr_recall_DNN, _ = precision_recall_curve(testy_DNN, lr_probs_DNN)
    lr_f1_DNN, lr_auc_DNN = f1_score(testy_DNN, yhat_DNN), auc(lr_recall_DNN, lr_precision_DNN)
     
    testy_CNN1=Y_test
    ns_probs_CNN1 = [0 for _ in range(len(testy_CNN1))]
    # predict probabilities
    lr_probs_CNN1 = res['CNN1']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_CNN1 = res['CNN1']['Y_pred']
    lr_precision_CNN1, lr_recall_CNN1, _ = precision_recall_curve(testy_CNN1, lr_probs_CNN1)
    lr_f1_CNN1, lr_auc_CNN1 = f1_score(testy_CNN1, yhat_CNN1), auc(lr_recall_CNN1, lr_precision_CNN1)
   
    testy_CNN2=Y_test
    ns_probs_CNN2 = [0 for _ in range(len(testy_CNN2))]
    # predict probabilities
    lr_probs_CNN2 = res['CNN2']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_CNN2 = res['CNN2']['Y_pred']
    lr_precision_CNN2, lr_recall_CNN2, _ = precision_recall_curve(testy_CNN2, lr_probs_CNN2)
    lr_f1_CNN2, lr_auc_CNN2 = f1_score(testy_CNN2, yhat_CNN2), auc(lr_recall_CNN2, lr_precision_CNN2)
 
    testy_ENET=Y_test
    ns_probs_ENET = [0 for _ in range(len(testy_ENET))]
    # predict probabilities
    lr_probs_ENET = res['ENET']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_ENET = res['ENET']['Y_pred']
    lr_precision_ENET, lr_recall_ENET, _ = precision_recall_curve(testy_ENET, lr_probs_ENET)
    lr_f1_ENET, lr_auc_ENET = f1_score(testy_ENET, yhat_ENET), auc(lr_recall_ENET, lr_precision_ENET)
   
    testy_RID=Y_test
    ns_probs_RID = [0 for _ in range(len(testy_RID))]
    # predict probabilities
    lr_probs_RID = res['RID']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_RID = res['RID']['Y_pred']
    lr_precision_RID, lr_recall_RID, _ = precision_recall_curve(testy_RID, lr_probs_RID)
    lr_f1_RID, lr_auc_RID = f1_score(testy_RID, yhat_RID), auc(lr_recall_RID, lr_precision_RID)
   
    testy_LAS=Y_test
    ns_probs_LAS = [0 for _ in range(len(testy_LAS))]
    # predict probabilities
    lr_probs_LAS = res['LAS']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_LAS = res['LAS']['Y_pred']
    lr_precision_LAS, lr_recall_LAS, _ = precision_recall_curve(testy_LAS, lr_probs_LAS)
    lr_f1_LAS, lr_auc_LAS = f1_score(testy_LAS, yhat_LAS), auc(lr_recall_LAS, lr_precision_LAS)
 
    testy_ENET2=Y_test
    ns_probs_ENET2 = [0 for _ in range(len(testy_ENET2))]
    # predict probabilities
    lr_probs_ENET2 = res['ENET2']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_ENET2 = res['ENET2']['Y_pred']
    lr_precision_ENET2, lr_recall_ENET2, _ = precision_recall_curve(testy_ENET2, lr_probs_ENET2)
    lr_f1_ENET2, lr_auc_ENET2 = f1_score(testy_ENET2, yhat_ENET2), auc(lr_recall_ENET2, lr_precision_ENET2)
   
    testy_RF=Y_test
    ns_probs_RF = [0 for _ in range(len(testy_RF))]
    # predict probabilities
    lr_probs_RF = res['RF']['Y_pred_prob'][:, 1]
    # plot the roc curve for the model
    yhat_RF = res['RF']['Y_pred']
    lr_precision_RF, lr_recall_RF, _ = precision_recall_curve(testy_RF, lr_probs_RF)
    lr_f1_RF, lr_auc_RF = f1_score(testy_RF, yhat_RF), auc(lr_recall_RF, lr_precision_RF)
   
    testy_XGB=Y_test
    ns_probs_XGB = [0 for _ in range(len(testy_XGB))]
    # predict probabilities
    lr_probs_XGB = res['XGB']['Y_pred_prob']
    # plot the roc curve for the model
    yhat_XGB = res['XGB']['Y_pred']
    lr_precision_XGB, lr_recall_XGB, _ = precision_recall_curve(testy_XGB, lr_probs_XGB)
    lr_f1_XGB, lr_auc_XGB = f1_score(testy_XGB, yhat_XGB), auc(lr_recall_XGB, lr_precision_XGB)
 
    
    

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
        
        
        
    no_skill = len(Y_test[Y_test==1]) / len(Y_test)
    f[ax1, ax2].plot([0, 1], [no_skill, no_skill], linestyle='--', lw=1, color='gray')
    f[ax1, ax2].plot(lr_recall_PCT, lr_precision_PCT, lw=1, color='lightblue') #PCT
    f[ax1, ax2].plot(lr_recall_DNN, lr_precision_DNN, lw=1, color='blue') #DNN
    f[ax1, ax2].plot(lr_recall_CNN1, lr_precision_CNN1, lw=1, color='purple')
    f[ax1, ax2].plot(lr_recall_CNN2, lr_precision_CNN2, lw=1, color='darkmagenta')
    f[ax1, ax2].plot(lr_recall_RID, lr_precision_RID, lw=1, color='lime')
    f[ax1, ax2].plot(lr_recall_LAS, lr_precision_LAS, lw=1, color='lightgreen')
    f[ax1, ax2].plot(lr_recall_ENET, lr_precision_ENET, lw=1, color='chartreuse')
    f[ax1, ax2].plot(lr_recall_RF, lr_precision_RF, lw=1, color='yellow') # RF
    f[ax1, ax2].plot(lr_recall_XGB, lr_precision_XGB, lw=1, color='orange')
    f[ax1, ax2].plot(lr_recall_SNR, lr_precision_SNR, lw=1, color='red') # SNR
    f[ax1, ax2].plot(lr_recall_SNRauto, lr_precision_SNRauto, lw=1, color='brown') # SNR_auto
    f[ax1, ax2].plot(0, 0, marker='.', color='white')
    #axarr = f.add_subplot(3,3,i+1)
    # plot the roc curve for the model
    f[ax1, ax2].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].set_ylabel('Precision')
    f[ax1, ax2].set_xlabel('Recall')
    f[ax1, ax2].set_title('CV test fold: '+str(j))
    

    #f.legend(loc="lower left", fontsize = 'x-small')
    
    #import seaborn as sns
    #sns.lineplot(x=lr_fpr, y=lr_tpr, lw=2, ax=f[ax1, ax2])
    
    # fit the model and plot errors.
    print('No Skill: PR AUC=%.3f' % (ns_auc0))
    print('PCT: PR f1=%.3f auc=%.3f' % (lr_f1_PCT, lr_auc_PCT))
    print('DNN: PR f1=%.3f auc=%.3f' % (lr_f1_DNN, lr_auc_DNN))
    print('CNN: PR f1=%.3f auc=%.3f' % (lr_f1_CNN1, lr_auc_CNN1))
    print('CNN: PR f1=%.3f auc=%.3f' % (lr_f1_CNN2, lr_auc_CNN2))
    print('CNN: PR f1=%.3f auc=%.3f' % (lr_f1_RID, lr_auc_RID))
    print('CNN: PR f1=%.3f auc=%.3f' % (lr_f1_LAS, lr_auc_LAS))
    print('CNN: PR f1=%.3f auc=%.3f' % (lr_f1_ENET, lr_auc_ENET))
    print('CNN: PR f1=%.3f auc=%.3f' % (lr_f1_XGB, lr_auc_XGB))
    print('RF: PR  f1=%.3f auc=%.3f' % (lr_f1_RF, lr_auc_RF))
    print('SNR: PR  f1=%.3f auc=%.3f' % (lr_f1_SNR, lr_auc_SNR))
    print('SNR_auto: PR  f1=%.3f auc=%.3f' % (lr_f1_SNRauto, lr_auc_SNRauto))
    
    # fit the model and plot errors.
    df_pr=pd.Series({'No Skill': [ns_auc0], 'PCT': [(lr_f1_PCT, lr_auc_PCT)], 'DNN':[(lr_f1_DNN, lr_auc_DNN)], 'CNN1':[(lr_f1_CNN1, lr_auc_CNN1)], 'CNN2':[(lr_f1_CNN2, lr_auc_CNN2)], 'LAS':[(lr_f1_LAS, lr_auc_LAS)], 'RID':[(lr_f1_RID, lr_auc_RID)], 'ENET':[(lr_f1_ENET, lr_auc_ENET)], 'ENET2':[(lr_f1_ENET2, lr_auc_ENET2)], 'XGB':[(lr_f1_XGB, lr_auc_XGB)],  'RF':[(lr_f1_RF, lr_auc_RF)], 'SNR':[(lr_f1_SNR, lr_auc_SNR)], 'SNR_auto':[(lr_f1_SNRauto, lr_auc_SNRauto)]})
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return df_pr, f[ax1, ax2]


def ROC_curve_saveplt_old(res, Y_test, i, path, data_name='GQlupB', alpha=10):

            
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
    
    
    # DNN
    testy3=Y_test
    ns_probs3 = [0 for _ in range(len(testy3))]
    # predict probabilities
    lr_probs3 = res['DNN']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc3 = roc_auc_score(testy3, ns_probs3)
    lr_auc3 = roc_auc_score(testy3, lr_probs3)
    # summarize scores
    # calculate roc curves
    ns_fpr3, ns_tpr3, _ = roc_curve(testy3, ns_probs3)
    lr_fpr3, lr_tpr3, _ = roc_curve(testy3, lr_probs3)
    
    # PCT1
    testy=Y_test
    ns_probs = [0 for _ in range(len(testy))]
    # predict probabilities
    # keep probabilities for the positive outcome only
    lr_probs = res['PCT']['Y_pred_prob'][:, 1]
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
    pyplot.savefig(path+data_name+'_final_plots_CV/ROC_'+data_name+'_CV_fold'+str(j)+'.pdf')
    pyplot.savefig(path+data_name+'_final_plots_CV/ROC'+data_name+'_CV_fold'+str(j)+'.png')
    pyplot.show()

    
    #sns.lineplot(x=lr_fpr, y=lr_tpr, lw=2, ax=f[ax1, ax2])
    
    #print('CNN: ROC AUC=%.3f' % (lr_aucCNN))à
    return


def ROC_curve_plot_old(res, f, ax1, ax2, Y_test, i, path, data_name='GQlupB', alpha=10):

            
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
    
    
    # DNN
    testy3=Y_test
    ns_probs3 = [0 for _ in range(len(testy3))]
    # predict probabilities
    lr_probs3 = res['DNN']['Y_pred_prob'][:, 1]
    # calculate scores
    ns_auc3 = roc_auc_score(testy3, ns_probs3)
    lr_auc3 = roc_auc_score(testy3, lr_probs3)
    # summarize scores
    # calculate roc curves
    ns_fpr3, ns_tpr3, _ = roc_curve(testy3, ns_probs3)
    lr_fpr3, lr_tpr3, _ = roc_curve(testy3, lr_probs3)
    
    # PCT1
    testy=Y_test
    ns_probs = [0 for _ in range(len(testy))]
    # predict probabilities
    # keep probabilities for the positive outcome only
    lr_probs = res['PCT']['Y_pred_prob'][:, 1]
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
    f[ax1, ax2].plot(lr_fpr, lr_tpr, lw=1, color='purple')#, label='PCT')
    f[ax1, ax2].plot(lr_fpr3, lr_tpr3, lw=1, color='blue')#, label='DNN')
    #pyplot.plot(lr_fprCNN, lr_tprCNN, marker='.', label='CNN ROC AUC=%.3f' % (lr_aucCNN))
    f[ax1, ax2].plot(lr_fpr1, lr_tpr1, lw=1, color='orange')#, label='RF')
    f[ax1, ax2].plot(lr_fpr0, lr_tpr0,lw=1, color='red')#, label='SNR')
    f[ax1, ax2].plot(lr_fpr4, lr_tpr4, lw=1, color='brown')#, label='SNR_auto')
    f[ax1, ax2].xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    f[ax1, ax2].set_ylabel('True positive rate')
    f[ax1, ax2].set_xlabel('False positive rate')
    f[ax1, ax2].set_title('CV test fold: '+str(j))

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

