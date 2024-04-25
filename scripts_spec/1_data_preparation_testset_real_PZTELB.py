
#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
from ml_spectroscopy.config import path_init
import os
import numpy as np
import pandas as pd
from ml_spectroscopy.crosscorrNormVec import crosscorrRV_vec


import random

## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"


planet = "GQlupb"
mol = "H2O"
#  x, a, beta, vsn = gs()
#environment variables
maxcore=32

os.environ['OMP_NUM_THREADS']=str(maxcore)
os.environ['OPENBLAS_NUM_THREADS']=str(maxcore)
os.environ['MKL_NUM_THREADS']=str(maxcore)
os.environ['VECLIB_MAXIMUM_THREADS']=str(maxcore)
os.environ['NUMEXPR_NUM_THREADS']=str(maxcore)
os.environ['CUDA_VISIBLE_DEVICES']=""

np.random.seed(100)

## Read some data
data_file_templates = os.path.join(data_path, "csv_inputs/Molecular_Templates_df.csv")
templates = pd.read_csv(data_file_templates, index_col=0)
data1 = pd.read_pickle(
    data_path + 'Signal4alpha_bal50_molH2O_simple.pkl')
data1=data1.drop(['tempP', 'loggP', 'H2O', 'CO', 'CH4', 'NH3', 'subclass'], axis=1)
columns_save=data1.columns

################### Part 1 - Evaluate gaussianity of the signal ##################
###****************************************************************************###
### import the datasets with planets to fit a gaussian blob to them.           ###
###****************************************************************************###


planetlist = ['GQlupb0','GQlupb1','GQlupb2','GQlupb3','GQlupb4','GQlupb5','GQlupb6','GQlupb7','PZTel_10','PZTel_11','PZTel_12','PZTel_13','PZTel_20','PZTel_21','PZTel_22','PZTel_23','PZTel_24','PZTel_25','PZTel_26']
planettitle = ['GQ Lup B, cube 0','GQ Lup B, cube 1','GQ Lup B, cube 2','GQ Lup B, cube 3','GQ Lup B, cube 4','GQ Lup B, cube 5','GQ Lup B, cube 6','GQ Lup B, cube 7','PZ Tel B (1), cube 0','PZ Tel B (1), cube 1','PZ Tel B (1), cube 2','PZ Tel B (1), cube 3','PZ Tel B (2), cube 0','PZ Tel B (2), cube 1','PZ Tel B (2), cube 2','PZ Tel B (2), cube 3','PZ Tel B (2), cube 4','PZ Tel B (2), cube 5','PZ Tel B (2), cube 6']

#planetlist = ['PZTel_20','PZTel_21','PZTel_22','PZTel_23','PZTel_24','PZTel_25','PZTel_26']
#planettitle = ['PZ Tel B (2), cube 0','PZ Tel B (2), cube 1','PZ Tel B (2), cube 2','PZ Tel B (2), cube 3','PZ Tel B (2), cube 4','PZ Tel B (2), cube 5','PZ Tel B (2), cube 6']


dtpthtest = data_path + "csv_inputs/True_CCF_data"
path_planet0 = subdir + "30_data/Data_Planets/"
dir_file_planet = data_path + 'True_HCI_data'

# get the columns where the planets are saved
real_data_frame = pd.DataFrame()
for i in range(19):
    plnts0 = pd.read_csv(data_path + 'csv_inputs/True_Spectrum_Data/' + str(planetlist[i]) + '_spectrum_dt.csv', index_col=0)
    plnts1=plnts0[list(columns_save)]
    #plnts_csv['Planet'].column = "H2O"
    plnts = plnts1.join(plnts0['Planet'])
    #real_data_frame0 = pd.DataFrame(plnts, columns = plnts.columns)
    ind1 = plnts.index
    ind0 = np.repeat(planetlist[i], len(ind1))
    arr = [ind0,ind1]
    plnts.index = arr
    real_data_frame = pd.concat([real_data_frame, plnts])

real_data_frame = real_data_frame.rename(columns={'Planet': 'H2O'})
pd.to_pickle(real_data_frame, data_path + 'csv_inputs/CCF_True_Data_test/trimmed_data_all/Real_Data_all.pkl')




#planets=real_data_frame.loc["GQlupb0":"GQlupb1"]

def prepare_planet(atsmp, planets=real_data_frame, templates=templates):
    #atsmp = ATSMP[0]

    template_characteristics = {'Temp': atsmp[0],
                                'Surf_grav': atsmp[1],
                                'H2O': 1,
                                'CO': 0}

    planet_wr = [float(numeric_string) for numeric_string in planets.drop("H2O", axis=1).columns]

    # PlanetHCI_nanrm, PlanetHCI_vec_shape, PlanetHCI_position_nan = image_deconstruct(Planet_HCI)

    template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]
    TempCol = template.columns.get_loc("tempP")
    tf = template.drop(template.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()

    planetHCI_rv, planetHCI_cc = crosscorrRV_vec(planet_wr, np.array(planets.drop("H2O", axis=1)), tw, tf, -2000, 2000, 1, mode='doppler', normalized=True)

    #PlanetHCI_reconstructed_ccf = image_reconstruct(planetHCI_cc, int(PlanetHCI_vec_shape[0]), int(planetHCI_cc.shape[1]), PlanetHCI_position_nan)

    result_data = pd.DataFrame(planetHCI_cc, columns=planetHCI_rv)
    result_data.index = planets.index
    result_data = result_data.join(planets["H2O"])
    pd.to_pickle(result_data, data_path + 'csv_inputs/CCF_True_Data_test/final_test_set/Real_Data_H2O_T'+str(template_characteristics['Temp'])+'_sg'+str(template_characteristics['Surf_grav'])+'.pkl')

    return result_data











###************************************************************************************************************###
### Advanced train_valid and test set: Select planets similar to GQ Lup B and PZ Tel B to inject in the frames ###
###************************************************************************************************************###


### MULTIPROCESSING apply for other templates - train and validation data
from multiprocessing import Pool

Teff=[2300, 2500, 2700, 2800, 2900, 3100] #1600 2500
SG=[3.7, 3.9, 4.1, 4.3]
nparam = 2
#ATSMP = list(map(tuple, np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T))
#ATSMP = list(map(tuple, np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T))

# for the alpha multiprocessing
nprocess = len(Teff)*len(SG)
parameters = list(zip(np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T))
#parameters = list(zip(np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T * nprocess))

# evaluate all alpha values
pool = Pool(nprocess)
#pool.daemon = False
results = pool.starmap(prepare_planet, parameters)
pool.close()




#
# hdu_list0 = fits.open(dir_file_planet + '/res_' + planetlist[16] + '.fits')
# hdu_list0.info()
# Planet_HCI = hdu_list0[0].data
# hdu_list0.close()
# Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down
# Planet_WR = importWavelength_asList(data_path + 'wavelength_ranges/WR_' + planetlist[16], 'txt')
# PlanetHCI_nanrm, PlanetHCI_vec_shape, PlanetHCI_position_nan = image_deconstruct(Planet_HCI)
#
# planetHCI_rv, planetHCI_cc = crosscorrRV_vec(desired_array, PlanetHCI_nanrm, tw, tf, -2000, 2000, 1, mode='doppler', normalized=True)
#
# PlanetHCI_reconstructed_ccf = image_reconstruct(planetHCI_cc, int(PlanetHCI_vec_shape[0]), int(planetHCI_cc.shape[1]), PlanetHCI_position_nan)


# Transform the 3d cube into a 2d set of rows of spectrums and columns of wavelengths. NANS are removed but the info is stored in the last output
