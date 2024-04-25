# -*- coding: utf-8 -*-
"""
Created on Mon May 2 12:31:25 2022

@author: emily garvin
"""

# Function for data cubes decomposition and planet spotting, as well as data generation and templates+ spectrum binding
# In this version B, the goal is to correctly deconstruct the cubes, using the shift to sport the planet with the right center and remove it accoring to the shifting function.
# Here the goal is to get


## ACTIVE SUBDIR


## LIBRARIES
import time
import numpy as np
import pandas as pd
import sys, os
import random
from itertools import compress

# sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
# sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.DataPreprocessing_utils import spectrum_dataset_4ML_norep, spot_planet_incube, planetSignal_preprocessing, templates_preprocessing, trim_data_WR, padd_data_WR
from ml_spectroscopy.config import path_init

subdir = path_init()

## PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
# Each section can be run independently but later sections might depend on data existence from the previous sections


# =============================================================================
# =============================================================================
# Prepare the noise data
# =============================================================================
# =============================================================================


# =============================================================================
# Preprocess the data cubes to find the planet
# =============================================================================

# Later: add CO as outputs to the function, generalize in case we have ccf outliers or weak ccf which make us miss the planet, generalize to several planets
mol = 'CO'

if mol == "H2O":
    a0 = 1
    b0 = 0
elif mol == "CO":
    a0 = 0
    b0 = 1


dir_path_data = data_path + "True_HCI_data"
ls_data = os.listdir(dir_path_data)

ls_planetFilename = []
for i in range(0, len(ls_data)):
    ls_planetFilename.append(ls_data[i][4:][:-5])
#ls_planetFilename = ['PZTel_10', 'PZTel_11', 'PZTel_12', 'PZTel_13', 'PZTel_20', 'PZTel_21', 'PZTel_22', 'PZTel_23',
#                     'PZTel_24', 'PZTel_25', 'PZTel_26']
ls_planetFilename = ['GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7']
# print(ls_planetFilename)
t = time.process_time()
ls_aperturesize = {'BetaPicb': 3, 'GQlupb': 5.5, 'PZTel': 5.5, 'ROXs42B': 4, 'PDS70': 5.5}
ls_WR_extension = {'BetaPicb': 'txt', 'GQlupb': 'txt', 'PZTel': 'txt', 'ROXs42B': 'txt', 'PDS70': 'txt'}

template_characteristics_BP = {'Temp': 1700, 'Surf_grav': 4.1, 'H2O': a0, 'CO': b0}
template_characteristics_GQ = {'Temp': 2700, 'Surf_grav': 4.1, 'H2O': a0, 'CO': b0}
template_characteristics_ROX = {'Temp': 2200, 'Surf_grav': 3.9, 'H2O': a0, 'CO': b0}
template_characteristics_PZ = {'Temp': 2800, 'Surf_grav': 4.1, 'H2O': a0, 'CO': b0}
template_characteristics_PD = {'Temp': 1200, 'Surf_grav': 3.1, 'H2O': a0, 'CO': b0}

ls_template_characteristics = {'BetaPicb': template_characteristics_BP, 'GQlupb': template_characteristics_GQ,
                               'PZTel': template_characteristics_PZ, 'ROXs42B': template_characteristics_ROX,
                               'PDS70': template_characteristics_PD}

for i in range(0, len(ls_planetFilename)):
    planetFilename = ls_planetFilename[i]

    if planetFilename[:2] == 'Be':
        aperturesize = ls_aperturesize[planetFilename[:8]]
        WR_filename = planetFilename[:8]
        WR_extension = ls_WR_extension[planetFilename[:8]]
        template_characteristics = ls_template_characteristics[planetFilename[:8]]
        spot_planet_incube(planetFilename, WR_filename, WR_extension, aperturesize, template_characteristics,
                           centering_data=False, mol = mol)

    elif planetFilename[:2] == 'GQ':
        aperturesize = ls_aperturesize[planetFilename[:6]]
        WR_filename = planetFilename[:6]
        WR_extension = ls_WR_extension[planetFilename[:6]]
        template_characteristics = ls_template_characteristics[planetFilename[:6]]
        spot_planet_incube(planetFilename, WR_filename, WR_extension, aperturesize, template_characteristics,
                           centering_data=True, mol = mol)

    elif planetFilename[:2] == 'PD':
        aperturesize = ls_aperturesize[planetFilename[:5]]
        WR_filename = planetFilename[:5]
        WR_extension = ls_WR_extension[planetFilename[:5]]
        template_characteristics = ls_template_characteristics[planetFilename[:5]]
        spot_planet_incube(planetFilename, WR_filename, WR_extension, aperturesize, template_characteristics,
                           centering_data=False, mol = mol)

    elif planetFilename[:2] == 'PZ':
        aperturesize = ls_aperturesize[planetFilename[:5]]
        WR_filename = planetFilename[:7]
        WR_extension = ls_WR_extension[planetFilename[:5]]  # to account for 1st and 2nd cube
        template_characteristics = ls_template_characteristics[planetFilename[:5]]
        spot_planet_incube(planetFilename, WR_filename, WR_extension, aperturesize, template_characteristics,
                           centering_data=True, mol = mol)

    elif planetFilename[:2] == 'RO':
        aperturesize = ls_aperturesize[planetFilename[:7]]
        WR_filename = planetFilename[:7]
        WR_extension = ls_WR_extension[planetFilename[:7]]
        template_characteristics = ls_template_characteristics[planetFilename[:7]]
        spot_planet_incube(planetFilename, WR_filename, WR_extension, aperturesize, template_characteristics,
                           centering_data=True, mol = mol)

elapsed_time = time.process_time() - t
print(elapsed_time)

######### now we have found the planet and excluded it from the data to keep only noise


# =============================================================================
# Create spectrums from true noise -> trimmed data
# =============================================================================

dir_path_WR = data_path + "wavelength_ranges/"
savedir_spec_true = data_path + "csv_inputs/True_Spectrum_Noise/"
ls_data = os.listdir(dir_path_WR)
ls_data = ['WR_BetaPicb.txt', 'WR_GQlupb.txt', 'WR_PZTel_1.txt', 'WR_PZTel_2.txt', 'WR_ROXs42B.txt']

trim_data_WR(ls_data, dir_path_WR, savedir_spec_true, include_synthetic=False)

# =============================================================================
# Create spectrums from true noise -> padded data
# =============================================================================

dir_path_WR = data_path + "wavelength_ranges/"
savedir_spec_true = data_path + "csv_inputs/True_Spectrum_Noise/"
ls_data = ['WR_BetaPicb.txt', 'WR_GQlupb.txt', 'WR_PZTel_1.txt', 'WR_PZTel_2.txt', 'WR_ROXs42B.txt']

padd_data_WR(ls_data, savedir_spec_true)

# =============================================================================
# =============================================================================
# Prepare the spectrums and templates
# =============================================================================
# =============================================================================


# =============================================================================
# Bind the simulated planet signals into a CSV
# =============================================================================

dir_pathP = data_path + "synthetic_data/planet_signals2"
mol_cols = ['H2O', 'CO', 'CH4', 'NH3']
nfiles = 198000
file_wavelength_range = data_path + "synthetic_data/wavelength_names_header"
dir_save = data_path + "csv_inputs"

planetSignal_df = planetSignal_preprocessing(dir_pathP, nfiles, mol_cols, file_wavelength_range, dir_save, addCO=True,
                                             addFE=True)

# =============================================================================
# Bind the templates into a CSV
# =============================================================================

dir_pathT = data_path + "synthetic_data/templates2"
molecules = ['H2O', 'CO']
nfiles = 26400  # 768
file_wavelength_range = data_path + "synthetic_data/wavelength_names_header"
dir_save = data_path + "csv_inputs"

molecularTemplates_df = templates_preprocessing(dir_pathT, nfiles, molecules, file_wavelength_range, dir_save,
                                                addCO=True, addFE=True)


# =============================================================================
# From the trimmed data, generate spectra for ML, with injected planets
# Version 2: without noise or signal repetition / robust version - noise is shuffled and molecule's values appear only once
# =============================================================================
random.seed(100)

Planet_Signals = pd.read_csv(data_path + "csv_inputs/Planet_Signals_df2.csv", index_col=0)
# for each planet, for each alpha and for each molecule.
datasets = ['BetaPicb', 'GQlupb', 'PZTel_1', 'PZTel_2', 'ROXs42B']
subsets = np.array(('BetaPicb', 'GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7',
                    'PZTel_10', 'PZTel_11', 'PZTel_12', 'PZTel_13', 'PZTel_20', 'PZTel_21', 'PZTel_22', 'PZTel_23',
                    'PZTel_24', 'PZTel_25', 'PZTel_26', 'ROXs42B0', 'ROXs42B1', 'ROXs42B2'))

mol = ['H2O', 'CO']
alpha = [1, 10, 50]
bal = [50, 30, 20]
s = 1  # (for the seed)
# alpha = [1, 10, 100, 500, 1000]

for m in mol:
    for i in range(0, len(datasets)):
        for b in bal:
            for a in alpha:

                datasetsi = datasets[i]
                subloc = tuple(
                    [sub[0:len(datasetsi)] == datasetsi for sub in subsets])  # which list elements have the planet name
                sublist = list(compress(subsets, subloc))  # extract the list elements which are True

                keys = sublist
                ls_data = {key: None for key in keys}
                for j in range(0, len(sublist)):
                    ls_data[sublist[j]] = pd.read_pickle(data_path + "csv_inputs/True_Spectrum_Noise/" + str(
                        sublist[j]) + "_Spectrum_noise_trim.pkl")  # Use the trimmed version of the spectrum noise

                ## Function to use the data, simulated planets with a molecule and an alpha value.
                data_init = pd.concat(ls_data)
                # planet_signals0 = Planet_Signals

                dt_temp = spectrum_dataset_4ML_norep(data_init, Planet_Signals, alpha=a, mol=m, balancing=b)
                dt_temp.to_pickle(data_path + "data_4ml/v4_spectrums_4ml_trim_robustness/" + str(m) + "_" + str(
                    datasetsi) + "_b" + str(b)
                                  + "_" + str(a) + "_spectrums_4ml_trim_norepetition.pkl")
                del dt_temp

            s = s + 1  # The seed stays the same for the different alphas but changes for different balancing or noise data

# tmp= pd.read_pickle(data_path + "data_4ml/v2_spectrums_4ml_trim_norepeat/"+str(m)+"_"+str(datasetsi)+"_"+str(a)+"_spectrums_4ml_trim_norepetition.pkl")


# =============================================================================
# =============================================================================
# Bind the data sets
# =============================================================================
# =============================================================================

datasets = ['BetaPicb', 'GQlupb', 'PZTel_1', 'PZTel_2', 'ROXs42B']
ls_data = ['WR_BetaPicb.txt', 'WR_GQlupb.txt', 'WR_PZTel_1.txt', 'WR_PZTel_2.txt', 'WR_ROXs42B.txt']

for m in ['H2O', 'CO']:
    for a in [1, 10, 50]:
        for b in [50, 30, 20]:
            ls = list()
            for i in range(0, len(datasets)):
                datasetsi = datasets[i]

                data_temp = pd.read_pickle(
                    data_path + "data_4ml/v4_spectrums_4ml_trim_robustness/" + str(m) + "_" + str(
                        datasetsi) + "_b" + str(b) + "_" + str(a) + "_spectrums_4ml_trim_norepetition.pkl")
                ls.append(data_temp)

            data_stack = pd.concat(ls)

            data_stack.to_pickle(
                data_path + "data_4ml/v4_spectrums_4ml_trim_robustness/" + str(m) + "_Allplanets_b" + str(
                    b) + "_" + str(a) + "_spectrums_4ml_trim_norepetition.pkl")


