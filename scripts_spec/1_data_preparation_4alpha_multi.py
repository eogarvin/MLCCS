
#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.config import path_init
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from ml_spectroscopy.ccf_class import CCFSignalNoise
from ml_spectroscopy.config import global_settings as gs
import time
import sys


# if run in screen
v=int(sys.argv[1])


if len(sys.argv)<3:
    mol_def="H2O"
    nmol=1
elif len(sys.argv)==3:
    mol_def=str(sys.argv[2])
    nmol = 1
elif len(sys.argv)>3:
    mol_def = [str(sys.argv[2])]
    for m in range(3,len(sys.argv)):
        mol_def.append(str(sys.argv[m]))
    nmol=len(mol_def)



print("molecule:"+str(mol_def))
print("version:"+str(v))



## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"


planet = "GQlupb"
#mol = "H2O"
x, a, beta, vsn = gs()
version = v

# environment variables
maxcore=64

os.environ['OMP_NUM_THREADS']=str(maxcore)
os.environ['OPENBLAS_NUM_THREADS']=str(maxcore)
os.environ['MKL_NUM_THREADS']=str(maxcore)
os.environ['VECLIB_MAXIMUM_THREADS']=str(maxcore)
os.environ['NUMEXPR_NUM_THREADS']=str(maxcore)
os.environ['CUDA_VISIBLE_DEVICES']=""

## Define SNR



## Read some data
data_file_templates = os.path.join(data_path, "csv_inputs/Molecular_Templates_df.csv")
templates = pd.read_csv(data_file_templates, index_col=0)


if nmol == 1:
    data_signal_file = os.path.join(data_path,"Signal4alpha_bal50_mol"+str(mol_def)+"_simple.pkl")
    data_noise_file = os.path.join(data_path,"Noise4alpha_bal50_mol"+str(mol_def)+"_simple.pkl")

elif nmol>1:
    data_signal_file = os.path.join(data_path,"Signal4alpha_bal50_mol_multi_"+'_'.join(mol_def)+"_simple.pkl")
    data_noise_file = os.path.join(data_path,"Noise4alpha_bal50_mol_multi_"+'_'.join(mol_def)+"_simple.pkl")

noise_raw = pd.read_pickle(data_noise_file)
signal_raw = pd.read_pickle(data_signal_file)


# Now each of those datasets have to be (1) cross correlated with each molecule, and (2) combined to form a total dataset with various alpha values
for mol in mol_def:

    if mol == "H2O":
        val_H2O = 1
        val_CO = 0
    elif mol == "CO":
        val_H2O = 0
        val_CO = 1


    # get data
    TempCol_dt= signal_raw.columns.get_loc("tempP")
    df_signal0 = signal_raw.drop(signal_raw.columns[TempCol_dt:], axis=1)

    dw = pd.to_numeric(df_signal0.columns)
    y_meta = signal_raw[signal_raw.columns[TempCol_dt:]]
    df_signal = np.array(df_signal0)

    df_noise = np.array(noise_raw)
    index_noise = noise_raw.index

    # Do the calculations for different alpha values
    #interp_alpha = np.arange(4, 11, 1)

    ### Prepare the rest of the data sets
    Teff=[1200, 1400, 1600, 2000, 2500]
    SG=[2.9, 3.5, 4.1, 4.5, 5.3]
    nparam = 2
    ATSMP = list(map(tuple, np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T))

    # load the data (or more preferably, re-use the data of the previous analysis)
    # Also re-use the alphas, just change the templates

    execution_time = []
    for atsmp in tqdm(ATSMP):
        template_characteristics = {'Temp': atsmp[0],
                                    'Surf_grav': atsmp[1],
                                    'H2O': val_H2O,
                                    'CO': val_CO}

        template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]

        TempCol = template.columns.get_loc("tempP")
        tf = template.drop(template.columns[TempCol:], axis=1)
        tw = pd.to_numeric(tf.columns)
        tf = np.array(tf).flatten()

        # CCF over signal and noise separately (for the predefined template)
        start_time = time.time()
        ccf_tmp = CCFSignalNoise(signal=df_signal,
                             noise=df_noise,
                             dw=dw,
                             tw=tw,
                             tf=tf,
                             rvmin=-2000, rvmax=2000, drv=1,
                             mode="doppler", skipedge=0,
                             num_processes=maxcore)
        end_time = time.time()
        execution_time = [execution_time, (end_time - start_time)]

        ## To output the datasets of interest:
        for alpha_star in range(4,11):
            tmp_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=round(alpha_star))
            tmp_data_frame = pd.DataFrame(tmp_cc_signal_and_noise)
            tmp_data_frame.columns = drvs
            tmp_data_frame.index = index_noise
            y_meta.index = index_noise
            tmp_data_frame_tosave = tmp_data_frame.join(y_meta)
            tmp_data_frame_tosave.to_pickle(
                data_path + "data_4ml/v"+str(version)+"_ccf_4ml_trim_robustness_simple/" + str(mol) + "_" + str(planet) + "_scale" + str(
                    round(alpha_star)) + "_bal" + str(beta) + "_temp" + str(template_characteristics['Temp']) + "_sg" + str(
                    template_characteristics['Surf_grav']) + "_ccf_4ml_trim_norepetition_v"+str(version)+"_simple.pkl")
            #labels = np.array(y_meta[str(mol)] == 1.0)

    execution_time = pd.Series(execution_time)
    execution_time.to_pickle(data_path + "data_4ml/v"+str(version)+"_ccf_4ml_trim_robustness_simple/Time_it_"+ str(mol) + "_" + str(planet) + "_scale" + str(
                    round(alpha_star)) + "_bal" + str(beta)+"_version"+str(version)+".pkl")