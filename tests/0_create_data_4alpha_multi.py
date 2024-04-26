
## LIBRARIES
import numpy as np
import pandas as pd
import sys
from itertools import compress
#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.DataPreprocessing_utils import split_dataframe
from ml_spectroscopy.config import path_init



mol_def=str(sys.argv[1])
if NameError:
    mol_def = "H2O"


if mol_def == "H2O":
    s = 100
elif mol_def == "CO":
    s = 200
elif mol_def == "NH3":
    s = 300
elif mol_def == "CH4":
    s = 400

if len(sys.argv) == 3:
    mol_def = [mol_def, sys.argv[2]]
elif len(sys.argv) > 3:
    mol_def = [str(sys.argv[1])]
    for m in range(2,len(sys.argv)):
        mol_def.append(str(sys.argv[m]))


print("molecule(s): "+str(mol_def))

subdir=path_init()

## PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
# Each section can be run independently but later sections might depend on data existence from the previous sections


# =============================================================================
# Transform trimmed or padded noise data into a spectrum with injected planets
# Here we make use of the noise and signals only once (no sampling with replacement)
# =============================================================================

# can also take the chance to make one data set with separated temperatures?
def spectrum_dataset_4alpha_norep_multi(data0, planet_signals0, mol, balancing=50, seed=100):
    import random
    random.seed(seed)

    save_double_index = data0.index
    save_double_index.levels[0]
    length_levels = len(save_double_index.levels[0])

    # Join a column with the searched molecule
    # zerodt = pd.DataFrame({mol: np.zeros((data0.shape[0]))})
    # zerodt.index = save_double_index
    # data1 = data0.join(zerodt)

    planet_signals0_trim = pd.concat([planet_signals0[list(data0.columns)],
                                      planet_signals0[['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3']]],
                                     axis=1)


    # Separate between a set with the molecule and the other without - shuffling and random picking is to be made here
    cols = planet_signals0_trim.columns[planet_signals0_trim.columns.isin(mol)]
    data_mol = planet_signals0_trim[(planet_signals0_trim[cols] == 1).any(axis=1)]
    data_mol = data_mol.sample(data_mol.shape[0], replace=False)
    data_mol_split = split_dataframe(data_mol, length_levels)

    data_nomol = planet_signals0_trim[(planet_signals0_trim[cols] == 0).all(axis=1)]
    data_nomol = data_nomol.sample(data_nomol.shape[0], replace=False)
    data_nomol_split = split_dataframe(data_nomol, length_levels)

    # Shuffle the data to decorrelate the spatial positions
    data_signal = pd.DataFrame([])
    data_noise = pd.DataFrame([])

    for ind in range(0, int(length_levels)):

        data_temp0 = data0.loc[(save_double_index.levels[0][ind], slice(None)), :]

        # Decorrelate spatial positions
        data_temp = data_temp0.sample(data_temp0.shape[0], replace=False)
        ind_save = data_temp.index

        # Randomly pick molecular signals

        if balancing == 50:
            MOL_to_input0 = data_mol_split[ind].sample(round(data_temp.shape[0] / 2), replace=False)
            noMOL_to_input0 = data_nomol_split[ind].sample(round(data_temp.shape[0] / 4), replace=False)
        elif balancing == 30:
            MOL_to_input0 = data_mol_split[ind].sample(round(data_temp.shape[0] / 3.33333), replace=False)
            noMOL_to_input0 = data_nomol_split[ind].sample(round(data_temp.shape[0] / 3.33333), replace=False)
        elif balancing == 20:
            MOL_to_input0 = data_mol_split[ind].sample(round(data_temp.shape[0] / 5), replace=False)
            noMOL_to_input0 = data_nomol_split[ind].sample(round(data_temp.shape[0] / 5), replace=False)

        # prepare the datasets to scale - dataset of same shape and size of the noise
        MOL_to_input = MOL_to_input0.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3'], axis=1)
        noMOL_to_input = noMOL_to_input0.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3'], axis=1)
        PureNoise_to_input = pd.DataFrame(
            np.zeros((data_temp.shape[0] - (MOL_to_input.shape[0] + noMOL_to_input.shape[0]), noMOL_to_input.shape[1])))
        PureNoise_to_input.columns = noMOL_to_input.columns
        dataset_to_input = pd.concat([MOL_to_input, noMOL_to_input, PureNoise_to_input], axis=0)
        dataset_to_input.index = ind_save  # we index it as the noise

        # store info to be used after the scaling
        MOL_keepinfo = MOL_to_input0[['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3']]
        zerodt_data1 = pd.DataFrame({'subclass': np.zeros((MOL_keepinfo.shape[0]))})
        zerodt_data1[
        :] = 'molSignal'  # Allows to keep track of how the data was splitted and if molecules really correspond
        zerodt_data1.index = MOL_keepinfo.index
        MOL_keepinfo_dt = pd.concat([MOL_keepinfo, zerodt_data1], axis=1)

        noMOL_keepinfo = noMOL_to_input0[['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3']]
        zerodt_data2 = pd.DataFrame({'subclass': np.zeros((noMOL_keepinfo.shape[0]))})
        zerodt_data2[:] = 'molNoise'
        zerodt_data2.index = noMOL_keepinfo.index
        noMOL_keepinfo_dt = pd.concat([noMOL_keepinfo, zerodt_data2], axis=1)

        PureNoise_keepinfo = pd.DataFrame(
            np.zeros((data_temp.shape[0] - (MOL_keepinfo.shape[0] + noMOL_keepinfo.shape[0]), noMOL_keepinfo.shape[1])))
        PureNoise_keepinfo.columns = noMOL_keepinfo.columns
        PureNoise_keepinfo[['tempP', 'loggP']] = np.nan
        zerodt_data3 = pd.DataFrame({'subclass': np.zeros((PureNoise_keepinfo.shape[0]))})
        zerodt_data3[:] = 'pureNoise'
        zerodt_data3.index = PureNoise_keepinfo.index
        PureNoise_keepinfo_dt = pd.concat([PureNoise_keepinfo, zerodt_data3], axis=1)

        PureNoise_keepinfo_dt.columns = noMOL_keepinfo_dt.columns

        # Create the data set of signals
        dataset_keepinfo = pd.concat([MOL_keepinfo_dt, noMOL_keepinfo_dt, PureNoise_keepinfo_dt], axis=0)
        dataset_keepinfo.index = ind_save  # Index it as noise data

        tmp_noise = data_temp
        tmp_signals = pd.concat([dataset_to_input, dataset_keepinfo], axis=1)

        # shuffle the signals
        tmp_signals = tmp_signals.sample(tmp_signals.shape[0], replace=False)

        data_signal = pd.concat([data_signal, tmp_signals], axis=0)
        data_noise = pd.concat([data_noise, tmp_noise], axis=0)

    return data_noise, data_signal


# =============================================================================
# Version 2: without noise or signal repetition / robust version - noise is shuffled and H20 values appear only once
# =============================================================================


Planet_Signals = pd.read_csv(data_path + "csv_inputs/Planet_Signals_df.csv", index_col=0)
# for each planet, for each alpha and for each molecule.
datasets = ['GQlupb']
subsets = np.array(('GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7'))

mol = mol_def  # ['H2O', 'CO']
bal = [50]  # [50,30,20]
 # (for the seed)
# alpha = [1, 10, 100, 500, 1000]
s=0

for i in range(0, len(datasets)):
    for b in bal:

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

        noise, signal = spectrum_dataset_4alpha_norep_multi(data_init, Planet_Signals, mol=mol, balancing=b, seed = s)

        noise.to_pickle(data_path + "Noise4alpha_bal" + str(b) + "_mol_multi_"+'_'.join(mol_def)+"_simple.pkl")
        signal.to_pickle(data_path + "Signal4alpha_bal" + str(b) + "_mol_multi_"+'_'.join(mol_def)+"_simple.pkl")


