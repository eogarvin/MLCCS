# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 22:07:53 2022

@author: emily

results
"""







##############################################################################
# 
##############################################################################



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
x=0
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

dir_path = results_path + "export_CV/from_GPU_byfold/results/"
ls_data = os.listdir(dir_path)
len_folds=len(ls_data)
result_names=[ls_data[n][:-4] for n in range(len_folds)]

keys = result_names
ls_results = {key: None for key in keys}

for i in range(0,len_folds):
    with open(dir_path+str(ls_data[i]), "rb") as f:
        ls_results[i] = pickle.load(f) # i is the validation number but the proper set is at i+1


dir_path2 = results_path + "export_CV/from_GPU_byfold/GA_results/"
ls_data2 = os.listdir(dir_path2)
len_folds2=len(ls_data2)
result_names2=[ls_data2[n][:-4] for n in range(len_folds2)]

keys2 = result_names2
GA_results = {key: None for key in keys2}

for i in range(0,len_folds):
    with open(dir_path2+str(ls_data2[i]), "rb") as f:
        GA_results[i] = pickle.load(f) # i is the validation number but the proper set is at i+1

methods_ls= ['SNR', 'SNR_auto',  'RF', 'XGB', 'LAS', 'ENET', 'RID', 'ENET2', 'PCT', 'DNN', 'CNN1', 'CNN2']
#methods_ls=['SNR', 'SNR_auto','CNN1','ENET','RID', 'XGB', 'ENET2']
plotname='test'
color_ls={'SNR':'red', 'SNR_auto': 'brown', 'PCT':'lightblue', 'DNN': 'blue', 'CNN1':'navy', 'CNN2': 'purple', 'ENET':'forestgreen', 'RID':'lime', 'LAS':'lightgreen', 'RF':'yellow', 'XGB':'orange', 'ENET2':'darkgreen'}





# =============================================================================
# # =============================================================================
# # ROC curves
# # =============================================================================
# =============================================================================


# =============================================================================
# Aggregated ROC curves 
# =============================================================================

methods=['SNR_auto','RID','PCT', 'CNN1']
color_ls={'SNR':'red', 'SNR_auto': 'brown', 'PCT':'navy', 'DNN': 'blue', 'CNN1':'purple', 'CNN2': 'purple', 'ENET':'forestgreen', 'RID':'lightblue', 'LAS':'lightgreen', 'RF':'yellow', 'XGB':'orange', 'ENET2':'darkgreen'}
method_title_ls={'SNR_auto': 'Signal-to-noise Ratio', 'PCT':'Neural Network', 'DNN': 'blue', 'CNN1':'Convolutional Neural Network', 'CNN2': 'purple', 'ENET':'forestgreen', 'RID':'Ridge regression', 'LAS':'lightgreen', 'RF':'yellow', 'XGB':'orange', 'ENET2':'darkgreen'}


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
    plt.plot(lr_fpr_0, lr_tpr_0, lw=2, color=color_ls[m], label=method_title_ls[m]+' AUC: '+ str(round(auc_ROC,3)))        

     #axarr = f.add_subplot(3,3,i+1
     # plot the roc curve for the odel


plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title('Aggregated ROC Curves over all CV folds')
plt.legend()
plt.savefig(visual_path+planet+'_poster_plots/Aggregated_ROC.png', dpi=800, bbox_inches='tight')
plt.show() 

 

##############################################################################
#  Molecular mapping examples
##############################################################################


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


# -*- coding: utf-8 -*-
import csv
import numpy as np
import time
import pandas as pd
import pickle
from photutils import CircularAperture
import matplotlib.pyplot as plt
from astropy.io import fits
from functools import reduce
from itertools import chain
import scipy as sp
import os
from ml_spectroscopy.crosscorrNormVec import crosscorrRV_vec
from ml_spectroscopy.config import path_init
from ml_spectroscopy.DataPreprocessing_utils import importWavelength_asList, image_deconstruct, image_reconstruct


## PATHS

## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plots_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
visual_path = subdir + "80_visualisation/"
csv_res_path= subdir + "90_sweave_template/"



## SET SEED FOR REPRODUCIBILITY
random.seed(100)


os.environ['TF_NUM_INTEROP_THREADS'] = "16"
os.environ['TF_NUM_INTRAOP_THREADS'] = "16"
os.environ['OMP_NUM_THREADS']="16"
os.environ['OPENBLAS_NUM_THREADS']="16"
os.environ['MKL_NUM_THREADS']="16"
#os.environ['VECLIB_MAXIMUM_THREADS']="16"
#os.environ['NUMEXPR_NUM_THREADS']="16"
os.environ['CUDA_VISIBLE_DEVICES']=""


def spot_planet_incube3(planetFilename, WRFilename, WR_extension, template_characteristics, rv=0, RV=True, MAX=False):
    filename = planetFilename
    extension = WR_extension

    dir_file_planet = data_path + 'True_HCI_data'
    dir_file_WR = data_path + 'wavelength_ranges'
    # If templates have more molecules, remember to adapt the number of dropped end columns in the function
    dir_file_mol_template = data_path + "csv_inputs/Molecular_Templates_df.csv"
    # aperture 
    # Where to save data sets
    savedirccf = data_path + "csv_inputs/True_CCF_data"
    savedirdata = data_path + "csv_inputs/True_Spectrum_Data"
    # plot location
    dirplot = plots_path + 'Data_preprocessing_output_figures/'
    dirplot2 = plots_path + 'Data_preprocessing_output_RVs/'
    hdu_list0 = fits.open(dir_file_planet + '/res_' + filename + '.fits')
    hdu_list0.info()
    Planet_HCI = hdu_list0[0].data
    hdu_list0.close()
    Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down
    Planet_WR = importWavelength_asList(dir_file_WR + '/WR_' + WRFilename, extension)

    # Transform the 3d cube into a 2d set of rows of spectrums and columns of wavelengths. NANS are removed but the info is stored in the last output
    PlanetHCI_nanrm, PlanetHCI_vec_shape, PlanetHCI_position_nan = image_deconstruct(Planet_HCI)

    # Check the reconstruction function works fine with this image by directly reconstructing into the cube
    Planet_HCI_reconstructed = image_reconstruct(PlanetHCI_nanrm, PlanetHCI_vec_shape[0], PlanetHCI_vec_shape[1],
                                                 PlanetHCI_position_nan)

    # Import template   
    MT_df = pd.read_csv(dir_file_mol_template, index_col=0)
    template_planetHCI = MT_df[MT_df["tempP"] == template_characteristics['Temp']][MT_df["loggP"] == template_characteristics['Surf_grav']][
        MT_df["H2O"] == template_characteristics['H2O']][MT_df["CO"] == template_characteristics['CO']]

    # if the selection is not well specified (e.g some precisions were not given on additional molecules)
    if template_planetHCI.shape[0] > 1:
        template_planetHCI = template_planetHCI.head(1)
        pd.DataFrame(template_planetHCI)
        print("Warning: Several templates available for this request, 1st template was selected to continue tehe task. Not all elements from the template have been defined, please check for temperature, surface gravity, or any additional molecule request which may be forgotten")

    # Drop columns from temperature to end
    TempCol = template_planetHCI.columns.get_loc("tempP")
    tf = template_planetHCI.drop(template_planetHCI.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()

    # Run the ccf
    planetHCI_rv, planetHCI_cc = crosscorrRV_vec(Planet_WR, PlanetHCI_nanrm, tw, tf, -2000, 2000, 1, mode='doppler',
                                                 normalized=True)

    # Reconstruct the ccf image
    PlanetHCI_reconstructed_ccf = image_reconstruct(planetHCI_cc, int(PlanetHCI_vec_shape[0]),
                                                    int(planetHCI_cc.shape[1]), PlanetHCI_position_nan)

    # Output CSVs 

    if MAX == True:
        ### Take the maximum points of all the spatial positions and spot the brightest point (assuming it is the planet)
        # Spot the max ccf to find the planet. !!! This function should be improved for the following cases: we have many planets, or we have a ccf peak outlier, or ccf of the planet is not super bright and an outlier rules it out
        find_planetHCI = np.max(PlanetHCI_reconstructed_ccf, axis=0)
        # Plot it out
        plt.imshow(find_planetHCI, cmap=plt.cm.viridis)
        plt.title("Mapping of H2O for GQ Lup B ", fontsize=14)
        cbar=plt.colorbar()             
        cbar.set_label("Maximal values of the normalized cross-correlation")
        plt.ylabel("[Pixels]")
        plt.xlabel("[Pixels]")
        plt.savefig(visual_path+planet+'_poster_plots/GQlupB_molmap_max.png', dpi=800, bbox_inches='tight')
        plt.show()
        
    if RV == True:
        find_planetHCI_RV = PlanetHCI_reconstructed_ccf[2050, :, :]
        plt.imshow(find_planetHCI_RV, cmap=plt.cm.viridis)
        plt.title("Mapping of H2O for GQ Lup B", fontsize=14)
        cbar=plt.colorbar()             
        cbar.set_label("Normalized cross-correlation values at the companion location", fontsize=8)
        cbar.ax.tick_params(labelsize=8) 
        plt.ylabel("[Pixels]")
        plt.xlabel("[Pixels]")
        plt.savefig(visual_path+planet+'_poster_plots/GQlupB_molmap_RV.png', dpi=800, bbox_inches='tight')
        plt.show() 
    return


### Run the function 

dir_path_data = data_path + "True_HCI_data"
ls_data = os.listdir(dir_path_data)

ls_planetFilename = []
for i in range(0, len(ls_data)):
    ls_planetFilename.append(ls_data[i][4:][:-5])

# print(ls_planetFilename)
t = time.process_time()
ls_aperturesize = {'BetaPicb': 3, 'GQlupb': 5.5, 'PZTel': 5.5, 'ROXs42B': 4, 'PDS70': 5.5}
ls_WR_extension = {'BetaPicb': 'txt', 'GQlupb': 'txt', 'PZTel': 'txt', 'ROXs42B': 'txt', 'PDS70': 'txt'}

template_characteristics_BP = {'Temp': 1700, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}
template_characteristics_GQ = {'Temp': 2700, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}
template_characteristics_ROX = {'Temp': 2200, 'Surf_grav': 3.9, 'H2O': 1, 'CO': 0}
template_characteristics_PZ = {'Temp': 3000, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}
template_characteristics_PD = {'Temp': 1200, 'Surf_grav': 3.1, 'H2O': 1, 'CO': 0}

ls_template_characteristics = {'BetaPicb': template_characteristics_BP, 'GQlupb': template_characteristics_GQ,
                               'PZTel': template_characteristics_PZ, 'ROXs42B': template_characteristics_ROX,
                               'PDS70': template_characteristics_PD}


ls_data='res_GQlupb3.fits'

planetFilename = 'GQlupb3'
if planetFilename[:2] == 'GQ':
        aperturesize = ls_aperturesize[planetFilename[:6]]
        WR_filename = planetFilename[:6]
        WR_extension = ls_WR_extension[planetFilename[:6]]
        template_characteristics = ls_template_characteristics[planetFilename[:6]]
        rv=0

spot_planet_incube3(planetFilename, WR_filename, WR_extension, template_characteristics, rv, RV=True, MAX=True)

 

  

# =============================================================================
# CM results to CV
# =============================================================================    

methods0=['SNR3', 'SNR3_auto', 'SNR5', 'SNR5_auto' ,'SNR','SNR_auto','RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']

for m in methods0:
    
    CM = np.zeros((8,4))
    
    if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:
        HP = np.zeros((8, len(list(ls_results[0]['hyperparameters'][m]))+2))
    else:
        HP = np.zeros((8,2))
    
    
    for j in range(0,len_folds):

        
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
                              
                if m == 'SNR3':
                    pred_snr3=list(map(int,list(ls_results[j]['results']['SNR']['SNR']>=3)))
                    SNR_3_acc=sum(data_test['H2O']==pred_snr3)/len(data_test['H2O'])
                    CM[j,:]=np.array((confusion_matrix(data_test['H2O'], pred_snr3).ravel()))
                    HP[j,:]=[3, SNR_3_acc]
             
                    
                elif m == 'SNR3_auto':
                    pred_snr3_auto=list(map(int,list(ls_results[j]['results']['SNR_auto']['SNR']>=3)))
                    SNR_3_acc_auto=sum(data_test['H2O']==pred_snr3_auto)/len(data_test['H2O'])
                    CM[j,:]=np.array((confusion_matrix(data_test['H2O'], pred_snr3_auto).ravel()))
                    HP[j,:]=[3, SNR_3_acc_auto]

                elif m == 'SNR5':
                    pred_snr5=list(map(int,list(ls_results[j]['results']['SNR']['SNR']>=5)))
                    SNR_5_acc=sum(data_test['H2O']==pred_snr5)/len(data_test['H2O'])
                    CM[j,:]=np.array((confusion_matrix(data_test['H2O'], pred_snr5).ravel()))
                    HP[j,:]=[5, SNR_5_acc]
                    
                elif m == 'SNR5_auto':
                    pred_snr5_auto=list(map(int,list(ls_results[j]['results']['SNR_auto']['SNR']>=5)))
                    SNR_5_acc_auto=sum(data_test['H2O']==pred_snr5_auto)/len(data_test['H2O'])
                    CM[j,:]=np.array((confusion_matrix(data_test['H2O'], pred_snr5_auto).ravel()))
                    HP[j,:]=[5, SNR_5_acc_auto]
                    
                elif m in ['SNR','SNR_auto']:

                        accuracy_test =sum(data_test['H2O']==ls_results[j]['results'][m]['Y_pred'])/len(data_test['H2O']) 
                        CM[j,:]=list(ls_results[j]['confusion matrix'][m])
                        HP[j,:]=[GA_results[j][m]['hyperparams'], accuracy_test]      
                               
                else:
                    
                    CM[j,:]=list(ls_results[j]['confusion matrix'][m])
                    lss=[list(ls_results[j]['hyperparameters'][m]), list(np.array((ls_results[j]['results'][m]['accuracy_train'], ls_results[j]['results'][m]['accuracy_valid_test'])))]
                    HP[j,:]=flatten(lss)

                    #if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:

                        
              

            except KeyError:
                pass
            
           
    CMdf=pd.DataFrame(CM)
    CMdf.columns=['tn', 'fp', 'fn', 'tp']
    CMdf.to_csv(visual_path+planet+'_poster_plots/CM_'+str(m)+'.csv')
    
    #if m in ['RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']:
    HPdf=pd.DataFrame(HP)
    HPdf.to_csv(visual_path+planet+'_poster_plots/csv_results/HP_'+str(m)+'.csv')   




# =============================================================================
# run times
# =============================================================================


methods=['SNR','SNR_auto','RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']
it=0

RTtmp=np.zeros((len(methods),4))
RTdf_GA=pd.DataFrame(RTtmp)
RTdf_GA.columns=['method', 'min', 'max', 'mean']

RTdf_method=pd.DataFrame(RTtmp)
RTdf_method.columns=['method', 'min', 'max', 'mean']


for m in methods:
    
    RT = np.zeros((8, 2))
    
    
    for j in range(0,len_folds):

        
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
                                  
                RT[j,:]=[GA_results[j][m]['runtime_GA'], GA_results[j][m]['runtime_model']]

                    

            except KeyError:
                pass
            
    RTdf_GA.iloc[it,0]=m
    RTdf_GA.iloc[it,1]=np.min(RT, axis=0)[0]
    RTdf_GA.iloc[it,2]=np.max(RT, axis=0)[0]
    RTdf_GA.iloc[it,3]=np.mean(RT, axis=0)[0]
    
    
    RTdf_method.iloc[it,0]=m
    RTdf_method.iloc[it,1]=np.min(RT, axis=0)[1]
    RTdf_method.iloc[it,2]=np.max(RT, axis=0)[1]
    RTdf_method.iloc[it,3]=np.mean(RT, axis=0)[1]
    
    it=it+1
            

    #RTdf.to_csv(visual_path+planet+"_poster_plots/csv_results/RT_"+str(m)+".csv") 
RTdf_GA.to_csv(visual_path+planet+"_poster_plots/csv_results/RT_GA.csv")   
RTdf_method.to_csv(visual_path+planet+"_poster_plots/csv_results/RT_method.csv")   


#barplot(RTdf_GA)  


plt.style.use('seaborn')
bars1 = plt.bar(x=RTdf_GA['method'], height=RTdf_GA['max']/60, width=0.7, color='lightgreen')
plt.title('Run Time of the optimization (in minutes)')
plt.savefig(csv_res_path+'model_GA_Runtime.pdf') 

plt.style.use('seaborn')
bars1 = plt.bar(x=RTdf_method['method'], height=RTdf_method['max']/60, width=0.7, color='sandybrown')
plt.title('Run Time of the models (in minutes)')
plt.savefig(csv_res_path+'model_Runtime.pdf')     

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

# =============================================================================
# Neural Networks are able to learn patterns from the data
# =============================================================================

plt.style.use('seaborn')

fg, axes = plt.subplots(2,1)
ax01 = [0,0]
#ax02 = [0,1]
axes[0].plot(np.arange(-2000,2000,1), ls_results[j]['results']['PCT']['weights'][0][:,1], color='darkblue')
axes[1].plot(ls_results[j]['results']['CNN1']['weights'][4][:,1], color='indigo')
axes[1].set_ylabel("weights", fontsize=9)  
axes[1].set_xlabel("last layer neurons", fontsize=9)  

axes[0].set_title("Perceptron", fontsize=9)  
axes[1].set_title("Convolutional neural network", fontsize=9) 
axes[0].set_ylabel("weights", fontsize=9)  
axes[0].set_xlabel("radial velocities or input neurons", fontsize=9)  

fg.suptitle("Neural Networks weigths", fontsize=15)
fg.tight_layout()
fg.savefig(visual_path+planet+'_poster_plots/weights_neur.png', dpi=800, bbox_inches='tight')
fg.show()

 ##############################3
plt.style.use('seaborn')

fg, axes = plt.subplots(1,2)
ax01 = [0,0]
#ax02 = [0,1]
axes[0].plot(np.arange(-2000,2000,1), ls_results[j]['results']['PCT']['weights'][0][:,1], color='darkblue')
axes[1].plot(ls_results[j]['results']['CNN1']['weights'][4][:,1], color='indigo')
axes[1].set_ylabel("weights", fontsize=9)  
axes[1].set_xlabel("last layer neurons", fontsize=9)  
axes[1].label_outer()

axes[0].set_title("Perceptron", fontsize=9)  
axes[1].set_title("Convolutional neural network", fontsize=9) 
axes[0].set_ylabel("weights", fontsize=9)  
axes[0].set_xlabel("radial velocities or input neurons", fontsize=9)  
axes[0].label_outer()

fg.suptitle("Neural Networks weigths", fontsize=15)
fg.tight_layout()
fg.savefig(visual_path+planet+'_poster_plots/weights_flip_neur.png', dpi=800, bbox_inches='tight')
fg.show()

# =============================================================================
# Distribution of signals
# =============================================================================

snry=pd.read_csv(csv_res_path+"csv_results/HatVSPred_SNR.csv")
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
x1=np.array(snry['Y_prob'][snry['Y_true']==1])
x2=np.array(snry['Y_prob'][snry['Y_true']==0])
kwargs = dict(histtype='stepfilled', alpha=0.5, bins=50)
plt.hist(x1, **kwargs,  color='steelblue')
plt.hist(x2, **kwargs,  color='indianred')
mylegends=[Line2D([0], [0], color='indianred', lw=1), Line2D([0], [0], color='steelblue', lw=1)]
plt.legend(mylegends, ['SNRs for spectra without H2O', 'SNRs for spectra containing H2O'])
plt.title("Distribution of signal-to-noise ratios of the cross-correlated spaxels")
plt.xlabel("Signal-to-noise ratio (SNR) at a given radial velocity")
plt.ylabel("Frequency (number of occurences)")
plt.savefig(visual_path+planet+'_poster_plots/distribution_signals.png', dpi=800, bbox_inches='tight')
plt.show()









# =============================================================================
# Example H2O signals
# =============================================================================


import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, LeakyReLU
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD
# from keras.optimizers import gradient_descent_v2

import xgboost as xgb
from xgboost import plot_tree

from sklearn.feature_selection import SelectFromModel

# sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")

from ml_spectroscopy.config import path_init
from ml_spectroscopy.utility_functions import test_onCCF_rv0_SNR, test_onCCF_rv0_SNR_autocorrel, t_test_onCCF_max, t_test_onCCF_rv0, t_test_onCCF_rv0_onesided


from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression



## ACTIVE SUBDIR
subdir = path_init()
# subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
visual_path = subdir + "80_visualisation/"

# Environment


os.environ['TF_NUM_INTEROP_THREADS'] = "16"
os.environ['TF_NUM_INTRAOP_THREADS'] = "16"
os.environ['OMP_NUM_THREADS' ] ="16"
os.environ['OPENBLAS_NUM_THREADS'] ="16"
os.environ['MKL_NUM_THREADS' ] ="16"
os.environ['VECLIB_MAXIMUM_THREADS' ] ="16"
os.environ['NUMEXPR_NUM_THREADS' ] ="16"
os.environ['CUDA_VISIBLE_DEVICES' ] =""

## SET SEED
np.random.seed(100)


## SETTINGS

data_name ='GQlupb'
template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}
alpha = 10

# validation fold
i = 6
# test fold:

if i == 7:
    j = 0
else:
    j = i + 1

x = 0  # first data set


data4 = pd.read_csv(data_path + 'csv_inputs/True_CCF_data/GQlupb3_crosscorr_dt.csv', index_col=0)
data8 = pd.read_csv(data_path + 'csv_inputs/True_CCF_data/GQlupb7_crosscorr_dt.csv', index_col=0)


plt.style.use('seaborn')
#plt.plot(data8[data8['0']==1].drop('0', axis=1).iloc[30,:])
plt.plot(np.arange(-2000,2000,1), data4[data4['0']==1].drop('0', axis=1).iloc[47,:], color="dodgerblue")
#plt.plot(np.arange(-2000,2000,1), data4[data4['0']==1].drop('0', axis=1).iloc[1,:])
plt.title("Example of an H2O signal in a cross-correlated spaxel of GQLup B")
plt.xlabel("Radial velocity span [Km/S]")
plt.ylabel("Normalized cross-correlation function")
plt.savefig(visual_path+planet+'_poster_plots/H2O_signal_gqlupb_royal.png', dpi=800, bbox_inches='tight')
plt.show()


# Simulated H2O signals

plt.style.use('seaborn')
plt.figure(figsize=(8,4))
data100=pd.read_pickle(data_path+'data_4ml/v2_ccf_4ml_trim_robustness/H2O_'+data_name+'_scale'+str(10)+'_temp1200_sg4.1_ccf_4ml_trim_norepetition.pkl')
dt_tmp=data100.loc[data100['subclass']=='molSignal'].drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
plt.plot(dt_tmp.columns, dt_tmp.iloc[1],color='mediumblue')
plt.title('Weak cross-correlation signals for H2O on simulated data')
plt.xlabel("Radial velocity span [Km/S]")
plt.ylabel("Normalized cross-correlation function")
plt.savefig(visual_path+planet+'_poster_plots/weak_signal_long2.pdf',bbox_inches='tight')
    


plt.style.use('seaborn')
plt.figure(figsize=(8,4))
data100=pd.read_pickle(data_path+'data_4ml/v2_ccf_4ml_trim_robustness/H2O_'+data_name+'_scale'+str(10)+'_temp1200_sg4.1_ccf_4ml_trim_norepetition.pkl')
dt_tmp=data100.loc[data100['subclass']=='molSignal'].drop(['tempP', 'loggP','H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
plt.plot(dt_tmp.columns, dt_tmp.iloc[1],color='mediumblue')
plt.title('Weak cross-correlation signals for H2O on simulated data')
plt.xlabel("Radial velocity span [Km/S]")
plt.ylabel("Normalized cross-correlation function")
plt.savefig(visual_path+planet+'_poster_plots/weak_signal_long2.png', dpi=900, bbox_inches='tight')
    
    
#==========================================================================
# Accuracy
#==========================================================================


## LIBRARIES

import pandas as pd
import numpy as np
import random
import sys
import pickle
import os
from matplotlib.lines import Line2D


#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.crosscorrNormVec import crosscorrRV_vec
from ml_spectroscopy.config import path_init
from ml_spectroscopy.utility_functions import flatten, Average


import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import time
import concurrent.futures
import pandas as pd

#from ml_spectroscopy.plottings_utils_results import ROC_curve_plot, ROC_curve_saveplt, PR_curve_plot, PR_curve_saveplt

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
planet='GQlupB'
data1=pd.read_pickle(data_path+'data_4ml/v2_ccf_4ml_trim_robustness/H2O_'+data_name+'_scale'+str(alpha)+'_temp1200_sg4.1_ccf_4ml_trim_norepetition.pkl')




dir_path = results_path + "export_CV/from_GPU_byfold/results/"
ls_data = os.listdir(dir_path)
len_folds=len(ls_data)
result_names=[ls_data[n][:-4] for n in range(len_folds)]

keys = result_names
ls_results = {key: None for key in keys}

for i in range(0,len_folds):
    with open(dir_path+str(ls_data[i]), "rb") as f:
        ls_results[i] = pickle.load(f) # i is the validation number but the proper set is at i+1




methods=['SNR','SNR_auto','RF', 'XGB', 'LAS', 'RID', 'ENET', 'ENET2','PCT', 'DNN', 'CNN1', 'CNN2']

accuracies_all=[]
ROC_all=[]
PR_all=[]
for m in methods:
    dt_temp=np.zeros((8,5))
    accuracies_method=[]
    ROC_method=[]
    PR_method=[]

    for j in range(0,len_folds):
        
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
                

                
                if m in ['ENET', 'LAS', 'RID', 'ENET2', 'XGB']:
                    accuracy_test = ls_results[j]['results'][m]['accuracy_valid_test']
                    accuracy_train=ls_results[i]['results'][m]['accuracy_train']
                    auc_ROC=roc_auc_score(data_test['H2O'],ls_results[j]['results'][m]['Y_pred_prob'])    
                    lr_precision, lr_recall, _ = precision_recall_curve(data_test['H2O'],ls_results[j]['results'][m]['Y_pred_prob'])
                    
                elif m in ['SNR', 'SNR_auto']:
                    accuracy_test =sum(data_test['H2O']==ls_results[j]['results'][m]['Y_pred'])/len(data_test['H2O'])
                    accuracy_train=np.nan
                    auc_ROC=roc_auc_score(data_test['H2O'],ls_results[j]['results'][m]['SNR'])
                    lr_precision, lr_recall, _ = precision_recall_curve(data_test['H2O'],ls_results[j]['results'][m]['SNR'])
                else:
                    accuracy_test = ls_results[j]['results'][m]['accuracy_valid_test']
                    accuracy_train=ls_results[i]['results'][m]['accuracy_train']
                    auc_ROC=roc_auc_score(data_test['H2O'],ls_results[j]['results'][m]['Y_pred_prob'][:,1])
                    lr_precision, lr_recall, _ = precision_recall_curve(data_test['H2O'],ls_results[j]['results'][m]['Y_pred_prob'][:,1])
                
                
                lr_f1, lr_auc = f1_score(data_test['H2O'], ls_results[j]['results'][m]['Y_pred']), auc(lr_recall, lr_precision)  
                #ax.label_outer()  
                
                dt_temp[j,:]=[accuracy_train, accuracy_test, auc_ROC, lr_auc, lr_f1]
                
                accuracies_method.append(accuracy_test)
                ROC_method.append(auc_ROC)
                PR_method.append(lr_auc)


            except KeyError:
                pass
            
            
    dt=pd.DataFrame(dt_temp)
    dt.columns=['acc_train', 'acc_test', 'auc_roc', 'auc_lr', 'f1']
    dt.to_csv(csv_res_path+"csv_results/results_"+str(m)+".csv")
    accuracies_all.append(Average(accuracies_method))
    ROC_all.append(Average(ROC_method))
    PR_all.append(Average(PR_method))


################
###############3
#################3
###############3
##################33
plt.style.use('seaborn')
y_pos=np.arange(len([methods[1],methods[8],methods[10]]))
y_pos=np.array([0.5, 0.8, 1.1])
plt.figure(figsize=(30,5))
plt.barh(y_pos, [ROC_all[1],ROC_all[8],ROC_all[10]], color=['lightsteelblue','darkblue', 'indigo'], height=0.2)
plt.xlim([0.5, 1])
plt.yticks(y_pos, labels=['Signal-to-noise ratio','Perceptron','Convolutional Neural net.'], fontsize=30)
plt.xticks(np.arange(0.5,1.05,0.05), fontsize=25)
plt.title('Model performance (Areas under Receiver Operating Characteristic curve)', fontsize=30)
plt.savefig(visual_path+planet+'_poster_plots/model_roc.png', dpi=800, bbox_inches='tight')




######################

methods=['SNR','SNR_auto','XGB','RID','ENET2','PCT','CNN1']
methods_title={'SNR':'SNR','SNR_auto':'ACF corrected SNR', 'XGB':'Gradient tree boosting','RID': 'Ridge', 'ENET2': 'ElasticNet (2)','PCT':'the Perceptron','CNN1':'the CNN (1)'}


scores=np.zeros((len(methods), 6))
dfscores=pd.DataFrame(scores)
dfscores.columns=['method', 'FDR_thresh', 'tp', 'fp', 'tn', 'fn']


it=0

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
    
    
    if m in ['SNR', 'SNR_auto']:
        hyperparams=np.arange(np.min(prob_Y),np.max(prob_Y), 0.001)
    else:
        hyperparams=np.arange(0,1, 0.001)
    
    
    scores0 = grid_search0(hyperparams,df_Y['Y_prob'],df_Y['Y_true'])    
    dfscores.iloc[it,:]=[m,scores0['optim_score'], scores0['tp'], scores0['fp'],  scores0['tn'], scores0['fn']]

    g, axess = plt.subplots(nrows=2, ncols=2)
    g.suptitle(('Classification scores for '+str(methods_title[m])), fontsize=14)
    
    sns.boxplot(x=df_Y['Y_true'], y=df_Y['Y_prob'], ax=axess[0,0])
    axess[0,0].set_title("Scores for "+str(m))
    
    axess[1, 0].hist(df_Y['Y_prob'])
    axess[1, 0].set_title('Prediction scores', fontsize=9)
    
    if m in ['SNR', 'SNR_auto']:
        axess[1, 0].set_xlim(-6,6)
        axess[1, 0].axvline(x=GA_results[j][m]['hyperparams'][0], color='darkred', label='Optimized Accuracy: T='+str(round(GA_results[j][m]['hyperparams'][0],2)))
        axess[1, 0].axvline(x=scores0['optim_score'], color='darkblue', label='FDR=0.05: T*='+str(round(scores0['optim_score'],2)) )
    else:
        axess[1, 0].set_xlim(0,1)
        axess[1, 0].axvline(x=0.5, color='darkred', label='Optimized Accuracy: T=0.5')
        axess[1, 0].axvline(x=scores0['optim_score'], color='darkblue', label='FDR=0.05: T*='+str(round(scores0['optim_score'],2)) )
     
    
    axess[0, 1].hist(df_Y[df_Y['Y_true']==1]['Y_prob'])
    axess[0, 1].set_title('Scores: H20 = 1', fontsize=9)
    if m in ['SNR', 'SNR_auto']:
        axess[0, 1].set_xlim(-6,6)
        axess[0, 1].axvline(x=GA_results[j][m]['hyperparams'][0], color='darkred', label='Optimized Accuracy: T='+str(round(GA_results[j][m]['hyperparams'][0],2)))
        axess[0, 1].axvline(x=scores0['optim_score'], color='darkblue', label='FDR=0.05: T*='+str(round(scores0['optim_score'],2)) )
    else:
        axess[0, 1].set_xlim(0,1)
        axess[0, 1].axvline(x=0.5, color='darkred', label='Optimized Accuracy: T=0.5')
        axess[0, 1].axvline(x=scores0['optim_score'], color='darkblue', label='FDR=0.05: T*='+str(round(scores0['optim_score'],2)) )
        #axess[0, 1].legend(fontsize=5)

          
    axess[1, 1].hist(df_Y[df_Y['Y_true']==0]['Y_prob'])
    axess[1, 1].set_title('Scores: H20 = 0', fontsize=9)
    if m in ['SNR', 'SNR_auto']:
        axess[1, 1].set_xlim(-6,6)
        axess[1, 1].axvline(x=GA_results[j][m]['hyperparams'][0], color='darkred', label='Optimized Accuracy: T='+str(round(GA_results[j][m]['hyperparams'][0],2)))
        axess[1, 1].axvline(x=scores0['optim_score'], color='darkblue', label='FDR=0.05: T*='+str(round(scores0['optim_score'],2)) )
    else:
        axess[1, 1].set_xlim(0,1)
        axess[1, 1].axvline(x=0.5, color='darkred', label='Optimized Accuracy: T=0.5')
        axess[1, 1].axvline(x=scores0['optim_score'], color='darkblue', label='FDR=0.05: T*'+str(round(scores0['optim_score'],2)) )
        
    g.tight_layout()
    mylegends=[Line2D([0], [0], color='darkred', lw=1), Line2D([0], [0], color='darkblue', lw=1)]
    
    if m in ['SNR', 'SNR_auto']:
        g.legend(mylegends, ['Optimal Accuracy: T='+str(round(GA_results[j][m]['hyperparams'][0],2)), 'FDR=0.05: T*='+str(round(scores0['optim_score'],2))], loc= (0.3835, 0.38),  prop=dict(size=8.2))
    else:
        g.legend(mylegends, ['Optimal Accuracy: T=0.5', 'FDR=0.05: T*='+str(round(scores0['optim_score'],2))], loc= (0.3835, 0.38),  prop=dict(size=8.2))

    g.savefig(visual_path+planet+'_poster_plots/Histogram_PIT_'+m+'.pdf')

    it=it+1
        

dfscores.to_csv(visual_path+planet+'_poster_plots/dfscores.csv') 
