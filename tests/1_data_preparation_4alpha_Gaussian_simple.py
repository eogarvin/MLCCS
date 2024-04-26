import random

#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.config import path_init
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_spectroscopy.ccf_class import CCFSignalNoise
from ml_spectroscopy.config import global_settings as gs
from sklearn import metrics
import scipy.interpolate as interpolate
import time
import sys
import random

# if run in screen
v=int(sys.argv[1])

if len(sys.argv)<3:
    mol="H2O"
elif len(sys.argv)==3:
    mol=str(sys.argv[2])


print("molecule:"+mol)
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
b = beta

# environment variables
maxcore=24

os.environ['OMP_NUM_THREADS']=str(maxcore)
os.environ['OPENBLAS_NUM_THREADS']=str(maxcore)
os.environ['MKL_NUM_THREADS']=str(maxcore)
os.environ['VECLIB_MAXIMUM_THREADS']=str(maxcore)
os.environ['NUMEXPR_NUM_THREADS']=str(maxcore)

#set seed
random.seed(100)
np.random.seed(100)
## Read some data
data_file_templates = os.path.join(data_path, "csv_inputs/Molecular_Templates_df.csv")
templates = pd.read_csv(data_file_templates, index_col=0)


if mol == "H2O":
    val_H2O = 1
    val_CO = 0
elif mol == "CO":
    val_H2O = 0
    val_CO = 1


template_characteristics = {'Temp': 1200,
                            'Surf_grav': 4.1,
                            'H2O': val_H2O,
                            'CO': val_CO}

template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]
TempCol = template.columns.get_loc("tempP")
tf = template.drop(template.columns[TempCol:], axis=1)
tw = pd.to_numeric(tf.columns)
tf = np.array(tf).flatten()

########################################################################################
## Creating the gaussian noise. We did it once, comment out these lines; we leave them commented out for reproducibility.
################################################################################################
#
#
# data_signal_file = os.path.join(data_path,
#                                 "Signal4alpha_bal50_mol"+str(mol)+"_simple.pkl")
#
# data_noise_file = os.path.join(data_path,
#                                "Noise4alpha_bal50_mol"+str(mol)+"_simple.pkl")
#
#
#
# noise_raw = pd.read_pickle(data_noise_file)
# signal_raw = pd.read_pickle(data_signal_file)
#
# means_all = np.mean(noise_raw, axis = 1)
# sigma_all = np.std(noise_raw, axis = 1)
#
# # We repeat the mean of each noise instance 1443 times ( the width of the spectrum)
# mean_shape = np.repeat(np.array(means_all), 1436).reshape((24327,1436)) #, index=noise_raw.index, columns = noise_raw.columns)
# std_shape = np.repeat(np.array(sigma_all), 1436).reshape((24327,1436)) #, index=noise_raw.index, columns = noise_raw.columns)
#
# gaussian_noise = np.random.normal(mean_shape, std_shape, (24327,1436))
#
#
# gaussian_noise4alpha = pd.DataFrame(gaussian_noise, index=noise_raw.index, columns = noise_raw.columns)
# gaussian_noise4alpha.to_pickle(data_path + "GaussianNoise4alpha_bal" + str(b) + "_mol" + mol + "_simple.pkl")
# del noise_raw

data_signal_file = os.path.join(data_path,
                                 "Signal4alpha_bal50_mol"+str(mol)+"_simple.pkl")
data_noise_file = os.path.join(data_path,
                               "GaussianNoise4alpha_bal50_mol"+str(mol)+"_simple.pkl")


noise_gauss = pd.read_pickle(data_noise_file)
signal_raw = pd.read_pickle(data_signal_file)

TempCol_dt= signal_raw.columns.get_loc("tempP")
df_signal = signal_raw.drop(signal_raw.columns[TempCol_dt:], axis=1)

dw = pd.to_numeric(df_signal.columns)
y_meta = signal_raw[signal_raw.columns[TempCol_dt:]]
df_signal = np.array(df_signal)

df_noise = np.array(noise_gauss)
index_noise = noise_gauss.index



# Test the class function for different alpha values: test one alpha

# CCF over signal and noise separately (for the predefined template)
ccf_tmp = CCFSignalNoise(signal=df_signal,
                         noise=df_noise,
                         dw=dw,
                         tw=tw,
                         tf=tf,
                         rvmin=-2000, rvmax=2000, drv=1,
                         mode="doppler", skipedge=0,
                         num_processes=maxcore)



new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=100.0)

tmp_data_frame = pd.DataFrame(new_cc_signal_and_noise)
tmp_data_frame.columns = drvs
tmp_data_frame.index = index_noise

y_meta.index = index_noise
tmp_data_frame_tosave = tmp_data_frame.join(y_meta)
labels = np.array(y_meta[str(mol)] == 1.0)


# Run the experiement over a range of alpha values


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


# results of an SNR with a threshold of 3
results = test_onCCF_rv0_SNR(tmp_data_frame, 3.0)
scores_snr = np.array(results["SNR"])

metrics.roc_auc_score(labels, scores_snr)



# Compute for different alpha values

alpha_values = np.linspace(0, 5000, 1000)


auc_results = []
aucPR_results = []
f1_results = []
accuracy_results = []

for tmp_alpha in tqdm(alpha_values):
    tmp_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=tmp_alpha)
    tmp_data_frame = pd.DataFrame(tmp_cc_signal_and_noise)
    tmp_data_frame.columns = drvs

    tmp_results = test_onCCF_rv0_SNR(tmp_data_frame, 3.0)
    tmp_scores_snr = np.array(tmp_results["SNR"])

    auc_results.append(metrics.roc_auc_score(labels, tmp_scores_snr))

    lr_precision_PCT, lr_recall_PCT, _ = metrics.precision_recall_curve(labels, tmp_scores_snr)
    aucPR_results.append(metrics.auc(lr_recall_PCT, lr_precision_PCT))

    f1_results.append(metrics.f1_score(labels, tmp_results["Y_pred"]))

    accuracy_results.append(metrics.accuracy_score(labels, tmp_results["Y_pred"]))

plt.style.use('seaborn')
plt.figure(figsize=(12, 8))
plt.plot(alpha_values, auc_results, label="classical SNR ROC AUC")
plt.plot(alpha_values, aucPR_results, label="classical SNR PR AUC")
#plt.plot(alpha_values, f1_results, label="classical SNR F1")
plt.plot(alpha_values, accuracy_results, label="classical SNR Accuracy")
plt.legend()
plt.xlabel("Alpha value")
plt.ylabel("AUC")
plt.title("Performance of SNR as a decreasing function of noise level")
plt.savefig(data_path + "data_4ml/v"+str(version)+"_ccf_4ml_trim_robustness_simple/v"+str(version)+"_ccf_4ml_trim_alphavals_simple/SNR_per_alpha/"+str(mol)+"_"+str(planet)+"_bal"+str(beta)+"_temp"+str(template_characteristics['Temp'])+"_sg"+str(template_characteristics['Surf_grav'])+"_co"+"_ccf_4ml_trim_norepetition_v"+str(version)+"_simple.pdf")
plt.show()
# for which alpha values do we have an AUC ROC of 0.5, 0.55, 0.6, 0.65, 0.7, max?


auc_results0 = np.array(auc_results)

test = interpolate.interp1d(auc_results0, alpha_values)
interp_alpha = test((np.min(auc_results0), 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, np.max(auc_results0)))

alphavals4rocauc = interpolate.barycentric_interpolate(auc_results0, alpha_values, np.array((0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,np.max(auc_results0))))
# alphavals4rocauc = interpolate.barycentric_interpolate(auc_results0, alpha_values, np.array((0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,np.max(auc_results0))))


## To putput the datasets of interest:
for alpha_star in tqdm(interp_alpha):
    tmp_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=round(alpha_star))
    tmp_data_frame = pd.DataFrame(tmp_cc_signal_and_noise)
    tmp_data_frame.columns = drvs
    tmp_data_frame.index = index_noise
    y_meta.index = index_noise
    tmp_data_frame_tosave = tmp_data_frame.join(y_meta)
    tmp_data_frame_tosave.to_pickle(data_path + "data_4ml/v"+str(version)+"_ccf_4ml_trim_robustness_simple/v"+str(version)+"_ccf_4ml_trim_alphavals_simple/"+str(mol)+"_"+str(planet)+"_scale"+str(round(alpha_star))+"_bal"+str(beta)+"_temp"+str(template_characteristics['Temp'])+"_sg"+str(template_characteristics['Surf_grav'])+"_ccf_4ml_trim_norepetition_v"+str(version)+"_simple.pkl")
    labels = np.array(y_meta[str(mol)] == 1.0)




### Prepare the rest of the data sets


Teff=[1200, 1400, 1600, 2000, 2500] #1600 2500
SG=[2.9, 3.5, 4.1, 4.5, 5.3]
nparam = 2
ATSMP = list(map(tuple, np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T))

# load the data (or more preferably, re-use the data of the previous analysis)
# Also re-use the alphas, just change the templates

#data_noise_file = os.path.join(data_path, "Noise4alpha_bal50_molH2O.pkl")
#noise_raw = pd.read_pickle(data_noise_file)
#df_noise = np.array(noise_raw)
#index_noise = noise_raw.index

#data_signal_file = os.path.join(data_path, "Signal4alpha_bal50_molH2O.pkl")
#signal_raw = pd.read_pickle(data_signal_file)
#TempCol_dt= signal_raw.columns.get_loc("tempP")
#df_signal = signal_raw.drop(signal_raw.columns[TempCol_dt:], axis=1)
#dw = pd.to_numeric(df_signal.columns)
#y_meta = signal_raw[signal_raw.columns[TempCol_dt:]]
#df_signal = np.array(df_signal)
#

execution_time = []
for atsmp in tqdm(ATSMP):
    template_characteristics = {'Temp': atsmp[0],
                                'Surf_grav': atsmp[1],
                                'H2O': template_characteristics['H2O'],
                                'CO': template_characteristics['CO']}

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

    tmp_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=8)
    tmp_data_frame = pd.DataFrame(tmp_cc_signal_and_noise)
    tmp_data_frame.columns = drvs
    tmp_data_frame.index = index_noise
    y_meta.index = index_noise
    tmp_data_frame_tosave = tmp_data_frame.join(y_meta)
    tmp_data_frame_tosave.to_pickle(
        data_path + "data_4ml/v" + str(version) + "_ccf_4ml_trim_robustness_simple/" + str(mol) + "_" + str(
            planet) + "_scale" + str(
            round(alpha_star)) + "_bal" + str(beta) + "_temp" + str(template_characteristics['Temp']) + "_sg" + str(
            template_characteristics['Surf_grav']) + "_ccf_4ml_trim_norepetition_v" + str(version) + "_simple.pkl")

    ## To output the datasets of interest:
    for alpha_star in interp_alpha:
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
