
#
from ml_spectroscopy.config import path_init
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_spectroscopy.ccf_class import CCFSignalNoise
from ml_spectroscopy.config import global_settings as gs
from astropy.io import fits
import gc
from photutils import CircularAperture
from astropy.modeling import models, fitting
from ml_spectroscopy.DataPreprocessing_utils import importWavelength_asList, image_reconstruct, image_deconstruct
import copy
import sys

import random

## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"

#define the molecule
mol=str(sys.argv[1])

if NameError:
    mol = "H2O"

print("molecule: "+str(mol))

if mol == "H2O":
    s = 100
elif mol == "CO":
    s = 200
elif mol == "NH3":
    s = 300
elif mol == "CH4":
    s = 400


planet = "GQlupb"
x, a, beta, vsn = gs()


if mol == "H2O":
    a0 = 1
    b0 = 0
elif mol == "CO":
    a0 = 0
    b0 = 1


# environment variables
maxcore=32

os.environ['OMP_NUM_THREADS']=str(maxcore)
os.environ['OPENBLAS_NUM_THREADS']=str(maxcore)
os.environ['MKL_NUM_THREADS']=str(maxcore)
os.environ['VECLIB_MAXIMUM_THREADS']=str(maxcore)
os.environ['NUMEXPR_NUM_THREADS']=str(maxcore)
os.environ['CUDA_VISIBLE_DEVICES']=""

np.random.seed(100)


################### Part 1 - Evaluate gaussianity of the signal ##################
###****************************************************************************###
### import the datasets with planets to fit a gaussian blob to them.           ###
###****************************************************************************###


planetlist1 = ['PZTel_10','PZTel_11','PZTel_12','PZTel_13']
planettitle = ['PZ Tel B (1), cube 0','PZ Tel B (1), cube 1','PZ Tel B (1), cube 2','PZ Tel B (1), cube 3','PZ Tel B (2), cube 0','PZ Tel B (2), cube 1','PZ Tel B (2), cube 2','PZ Tel B (2), cube 3']


dtpthtest = data_path + "csv_inputs/True_CCF_data"
path_planet0 = subdir + "30_data/Data_Planets/"
dir_file_planet = data_path + 'True_HCI_data'

pztel_stdx = []
pztel_stdy = []
pztel_mean_snr = []

dt_planet_crosscorr = pd.DataFrame()
for i in range(4):

    # Open the Original planet
    hdu_list0 = fits.open(dir_file_planet + '/res_' + planetlist1[i] + '.fits')
    hdu_list0.info()
    Planet_HCI = hdu_list0[0].data
    hdu_list0.close()
    Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down

    # Open the cross-correlated datasets
    temp_planet_crosscorr0 = pd.read_csv(data_path + 'csv_inputs/True_CCF_data/'+str(mol)+'/'+str(planetlist1[i])+'_crosscorr_dt.csv', index_col= 0)
    temp_planet_crosscorr = temp_planet_crosscorr0.drop('0', axis=1)
    ind_2=temp_planet_crosscorr.index
    ind_1=np.repeat(planetlist1[i], len(temp_planet_crosscorr.index))
    arrays = [ind_1,ind_2]
    temp_planet_crosscorr.index = arrays
    frames = [dt_planet_crosscorr, temp_planet_crosscorr]
    dt_planet_crosscorr = pd.concat(frames)

    # Open the centering cubes to get the positions
    if planetlist1[i] in ['GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7']:
        cube_centering = importWavelength_asList(path_planet0 + "GQlupb/Centering_cubes"+str(planetlist1[i][-1:]))
        RV=2044
    elif planetlist1[i] in ['PZTel_10', 'PZTel_11', 'PZTel_12', 'PZTel_13']:
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_1/Centering_cubes"+str(planetlist1[i][-1:]))
        RV=2022
    elif planetlist1[i] in ['PZTel_20', 'PZTel_21', 'PZTel_22', 'PZTel_23', 'PZTel_24', 'PZTel_25', 'PZTel_26']:
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes"+str(planetlist1[i][-1:]))
        RV=2077

    # Reconstruct the CCF image using the original image pixel locations
    PlanetHCI_nanrm, PlanetHCI_vec_shape, PlanetHCI_position_nan = image_deconstruct(Planet_HCI)
    Planet_HCI_reconstructed_ccf = image_reconstruct(np.array(temp_planet_crosscorr), PlanetHCI_vec_shape[0], 4000, PlanetHCI_position_nan)

    # Define the center of the image and append the centering information to sport the planet
    planet_position = [len(Planet_HCI_reconstructed_ccf[0]) / 2, len(Planet_HCI_reconstructed_ccf[2]) / 2]
    positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])] #(positions (x,y))

    #Get the average in an aperture
    aperturesize = 3.5
    aperture_circ = CircularAperture(positions, r=aperturesize)
    aperture_masks00 = aperture_circ.to_mask(method='center')
    aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
    # aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]

    plt.imshow(Planet_HCI_reconstructed_ccf[2022,:,:])
    plt.show()

    mask_aperture_image00 = copy.deepcopy(aperture_image00)
    mask_aperture_image00[aperture_image00==0] = np.nan

    SNR_img = Planet_HCI_reconstructed_ccf[2022,:,:]/np.std(np.concatenate((Planet_HCI_reconstructed_ccf[0:1822,:,:],Planet_HCI_reconstructed_ccf[2222:4000,:,:]), axis=0), axis=0)
    mean_SNR = np.nanmean(SNR_img*mask_aperture_image00)
    pztel_mean_snr.append(mean_SNR)

    plt.imshow(Planet_HCI_reconstructed_ccf[2022, :, :] * aperture_image00)
    plt.show()

    # Get gaussian
    x,y = np.meshgrid(np.arange(56), np.arange(56))
    gaussian = models.Gaussian2D(amplitude=1, x_mean=round(positions[0]), y_mean=round(positions[1]), x_stddev=1, y_stddev=1)

    # Define auxiliary function for tieing the standard deviations
    def tie_stddev(gaussian):
        return gaussian.y_stddev
    gaussian.x_stddev.tied = tie_stddev

    fit_p = fitting.LevMarLSQFitter()
    gaussian_model = fit_p(gaussian, x, y, Planet_HCI_reconstructed_ccf[RV,:,:])

    pztel_stdx.append(gaussian_model.x_stddev.value)
    pztel_stdy.append(gaussian_model.y_stddev.value)

    del temp_planet_crosscorr, frames
    gc.collect()





################### Part 2 - Evaluate the intensity of the signal ################
###****************************************************************************###
### Here we want to get the average SNR of the PZTel2 image.                   ###
###****************************************************************************###

planetlist2 = ['PZTel_20','PZTel_21','PZTel_22','PZTel_23']
planettitle = ['PZ Tel B (1), cube 0','PZ Tel B (1), cube 1','PZ Tel B (1), cube 2','PZ Tel B (1), cube 3','PZ Tel B (2), cube 0','PZ Tel B (2), cube 1','PZ Tel B (2), cube 2','PZ Tel B (2), cube 3']


dtpthtest = data_path + "csv_inputs/True_CCF_data"
path_planet0 = subdir + "30_data/Data_Planets/"
dir_file_planet = data_path + 'True_HCI_data'


pztel_mean_snr_2 = []

dt_planet_crosscorr = pd.DataFrame()
for i in range(4):

    # Open the Original planet
    hdu_list0 = fits.open(dir_file_planet + '/res_' + planetlist2[i] + '.fits')
    hdu_list0.info()
    Planet_HCI = hdu_list0[0].data
    hdu_list0.close()
    Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down

    # Open the cross-correlated datasets
    temp_planet_crosscorr0 = pd.read_csv(data_path + 'csv_inputs/True_CCF_data/'+str(mol)+'/'+str(planetlist2[i])+'_crosscorr_dt.csv', index_col= 0)
    temp_planet_crosscorr = temp_planet_crosscorr0.drop('0', axis=1)
    ind_2=temp_planet_crosscorr.index
    ind_1=np.repeat(planetlist1[i], len(temp_planet_crosscorr.index))
    arrays = [ind_1,ind_2]
    temp_planet_crosscorr.index = arrays
    frames = [dt_planet_crosscorr, temp_planet_crosscorr]
    dt_planet_crosscorr = pd.concat(frames)

    # Open the centering cubes to get the positions
    if planetlist1[i] in ['GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7']:
        cube_centering = importWavelength_asList(path_planet0 + "GQlupb/Centering_cubes"+str(planetlist2[i][-1:]))
        RV=2044
    elif planetlist1[i] in ['PZTel_10', 'PZTel_11', 'PZTel_12', 'PZTel_13']:
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_1/Centering_cubes"+str(planetlist2[i][-1:]))
        RV=2022
    elif planetlist1[i] in ['PZTel_20', 'PZTel_21', 'PZTel_22', 'PZTel_23', 'PZTel_24', 'PZTel_25', 'PZTel_26']:
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes"+str(planetlist2[i][-1:]))
        RV=2077

    # Reconstruct the CCF image using the original image pixel locations
    PlanetHCI_nanrm, PlanetHCI_vec_shape, PlanetHCI_position_nan = image_deconstruct(Planet_HCI)
    Planet_HCI_reconstructed_ccf = image_reconstruct(np.array(temp_planet_crosscorr), PlanetHCI_vec_shape[0], 4000, PlanetHCI_position_nan)

    # Define the center of the image and append the centering information to sport the planet
    planet_position = [len(Planet_HCI_reconstructed_ccf[0]) / 2, len(Planet_HCI_reconstructed_ccf[2]) / 2]
    positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])] #(positions (x,y))

    #Get the average in an aperture
    aperturesize = 3.5
    aperture_circ = CircularAperture(positions, r=aperturesize)
    aperture_masks00 = aperture_circ.to_mask(method='center')
    aperture_image00 = aperture_masks00.to_image(shape=(56, 56))

    plt.imshow(Planet_HCI_reconstructed_ccf[2077,:,:])
    plt.show()

    mask_aperture_image00 = copy.deepcopy(aperture_image00)
    mask_aperture_image00[aperture_image00==0] = np.nan

    SNR_img = Planet_HCI_reconstructed_ccf[2077,:,:]/np.std(np.concatenate((Planet_HCI_reconstructed_ccf[0:1877,:,:],Planet_HCI_reconstructed_ccf[2277:4000,:,:]), axis=0), axis=0)
    mean_SNR = np.nanmean(SNR_img*mask_aperture_image00)
    pztel_mean_snr_2.append(mean_SNR)

    plt.imshow(Planet_HCI_reconstructed_ccf[2077, :, :] * aperture_image00)
    plt.show()

    del temp_planet_crosscorr, frames
    gc.collect()



meta_data=pd.DataFrame([pztel_mean_snr, pztel_mean_snr_2, pztel_stdx, pztel_stdy])
meta_data.index = ["mean_snr", "mean_snr2", "stdx", "stdy"]
meta_data.columns = planetlist1

pd.to_pickle(meta_data, data_path + "csv_inputs/intro_plot_dataset/meta_data_newRV.pkl")






################### Part 1 - Insert the signal into the data     #################
###****************************************************************************###
### In this secton we want to insert the signal with the recovered gaussianity ###
###****************************************************************************###

# In each of these data sets there is a hole -> Due to data preprocessing to get the noise (c.f. 0_(b)_cubes_preprocessing)
# We want to be able to reconstruct the image despite the hole
# Then, propose a new aperture to insert the planet.
planetlist1 = ['PZTel_10','PZTel_11','PZTel_12','PZTel_13']

planetlist2 = ['PZTel_20','PZTel_21','PZTel_22','PZTel_23'] #,'PZTel_24','PZTel_25','PZTel_26']
dt_noise = pd.read_pickle(data_path + 'csv_inputs/CCF_True_Data_test/trimmed_data_all/Real_Data_all.pkl')

dtpathtest = data_path + "csv_inputs/intro_plot_dataset"
listofnanpositions = []
injection_regions_frames = {}
injection_data_set = pd.DataFrame()
injection_labels_set = pd.DataFrame()
buffer = 12

#template_characteristics = {'Temp': 2800, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}
ls_WR_extension = {'BetaPicb': 'txt', 'GQlupb': 'txt', 'PZTel': 'txt', 'ROXs42B': 'txt', 'PDS70': 'txt'}
dir_path_WR = data_path + "wavelength_ranges/"
hdu_list0 = fits.open(data_path + 'True_HCI_data/res_PZTel_10.fits')
hdu_list0.info()
Planet_HCI = hdu_list0[0].data
hdu_list0.close()
Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down
Planet_WR = importWavelength_asList(dir_path_WR + 'WR_PZTel_2')
PlanetHCI_nanrm, PlanetHCI_vec_shape, PlanetHCI_position_nan = image_deconstruct(Planet_HCI)
Planet_HCI_reconstructed = image_reconstruct(PlanetHCI_nanrm, PlanetHCI_vec_shape[0], PlanetHCI_vec_shape[1],PlanetHCI_position_nan)




dt_signals = pd.read_csv(data_path + 'csv_inputs/Planet_Signals_df.csv', index_col=0)
signal = dt_signals.loc[(dt_signals['tempP'] == 2800) & (dt_signals["loggP"] == 4.1) & (dt_signals["H2O"] == 1) & (dt_signals["CO"] == 1) & (dt_signals["CH4"] == 0.0) & (dt_signals["NH3"] == 0.0)]


dt_noise_to_use = dt_noise.loc[planetlist2].drop('H2O', axis=1)
pd.to_pickle(dt_noise.loc[planetlist2], data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_noise.pkl")

injection_data_set = pd.DataFrame()
Labels = pd.Series
for i in range(4):
    ### First, get the signal of the cross-correlated data; This will allow to estimat ethe gaussian decay.
    dir_path_WR = data_path + "wavelength_ranges/"
    hdu_list0 = fits.open(data_path + 'True_HCI_data/res_PZTel_10.fits')
    hdu_list0.info()
    Planet_HCI = hdu_list0[0].data
    hdu_list0.close()
    Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down
    Planet_WR = importWavelength_asList(dir_path_WR + 'WR_PZTel_2')

    # select a planet
    noise_temp = dt_noise.loc[planetlist2[i]].drop('H2O', axis=1)
    noise_temp_wl_shape = noise_temp.shape[1]

    #WR = [str(i) for i in Planet_WR]
    signal_trim = signal[noise_temp.columns]

    signal_as_array = np.array(signal_trim)
    signal_cube = np.tile(signal_as_array, 56*56).reshape((56, 56, noise_temp_wl_shape)).T
    # should give a flat image


    ## prepare the fake spectrum:
    aperturesize = 3.5 # average of gaussian blob decays are 3.5
    # Get the position of the aperture.
    path_planet0 = subdir + "30_data/Data_Planets/"
    cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
    planet_position = [56/ 2, 56/ 2]
    positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

    aperture_circ = CircularAperture(positions, r=aperturesize)
    aperture_masks0 = aperture_circ.to_mask(method='center')
    aperture_image0 = aperture_masks0.to_image(shape=(56, 56))
    labels0 = pd.Series(aperture_image0.flatten(), name='planet')

    ## 2. GAUSSIAN BLOB

    # Use the center and aperture size to construct a gaussian blob which will serve for the test set:
    # Get gaussian
    x,y = np.meshgrid(np.arange(56), np.arange(56))
    gaussian = models.Gaussian2D(amplitude=1, x_mean=round(positions[0]), y_mean=round(positions[1]), x_stddev=pztel_stdx[i], y_stddev=pztel_stdy[i])

    # Create an injection data cube with a gaussian aperture
    injection_gaussian_cubes=np.empty((noise_temp_wl_shape,56,56))
    # revert the block to stack it first by wavelength. --> stacked frames = cube
    for k in range(noise_temp_wl_shape):
        injection_gaussian_cubes[k,:,:]=gaussian(x,y)[:,:]
    # use the copy of the previous masked regions of the true planet, to indicate where to add nans (this way the injection matrix is collapsed into a data set in the exact same way as the new data)

    #signals * gaussian decay:
    signals_to_inject = injection_gaussian_cubes * signal_cube
    injection_data_gaussian, injection_gaussian_cubes_shape, injection_gaussian_cubes_position_nan = image_deconstruct(signals_to_inject)
    # name indices for the 'mask' injection data frame
    injection_data_frame = pd.DataFrame(injection_data_gaussian)
    ind1 = dt_noise_to_use.loc[planetlist2[i]].index
    ind0 = np.repeat(planetlist2[i], len(ind1))
    arr = [ind0,ind1]
    injection_data_frame.index = arr
    injection_data_frame.columns = noise_temp.columns

    labels0.index = arr
    injection_data_frame = injection_data_frame.join(labels0)

    df1 = [injection_data_set, injection_data_frame]
    injection_data_set = pd.concat(df1)

pd.to_pickle(injection_data_set, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_H2O_CO.pkl")



# Multiply the planet grid with the mask to remove the empty places.

###************************************************###
### Cross-correlate and try different alpha values ###
###************************************************###
# We simply overlay the PZTelB basis without removing
# the signal parts; we treat the base image of PZTelb
# as noise and we add the fake signal on top.


data_file_templates = os.path.join(data_path, "csv_inputs/Molecular_Templates_df.csv")
templates = pd.read_csv(data_file_templates, index_col=0)


# Base template

for t in range(5):

    template_characteristics_0 = {'Temp': 2800, 'Surf_grav': 4.1, 'H2O': a0, 'CO': b0}
    template_characteristics_1 = {'Temp': 2500, 'Surf_grav': 4.1, 'H2O': a0, 'CO': b0}
    template_characteristics_2 = {'Temp': 3100, 'Surf_grav': 4.1, 'H2O': a0, 'CO': b0}
    template_characteristics_3 = {'Temp': 2700, 'Surf_grav': 3.7, 'H2O': a0, 'CO': b0}
    template_characteristics_4 = {'Temp': 2900, 'Surf_grav': 4.3, 'H2O': a0, 'CO': b0}
    ls_template_characteristics = [template_characteristics_0, template_characteristics_1, template_characteristics_2, template_characteristics_3, template_characteristics_4]

    template = templates.loc[(templates['tempP'] == ls_template_characteristics[t]['Temp']) & (templates["loggP"] == ls_template_characteristics[t]['Surf_grav']) & (templates["H2O"] == ls_template_characteristics[t]['H2O']) & (templates["CO"] == ls_template_characteristics[t]['CO'])]
    TempCol = template.columns.get_loc("tempP")
    tf = template.drop(template.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()

    dw = pd.to_numeric(injection_data_set.drop('planet', axis=1).columns)
    df_signal = np.array(injection_data_set.drop('planet', axis=1))

    df_noise = np.array(dt_noise_to_use)
    index_noise = dt_noise_to_use.index


    # CCF over signal and noise separately (for the predefined template)
    ccf_tmp = CCFSignalNoise(signal=df_signal,
                             noise=df_noise,
                             dw=dw,
                             tw=tw,
                             tf=tf,
                             rvmin=-2000, rvmax=2000, drv=1,
                             mode="doppler", skipedge=0,
                             num_processes=36)
    # Base
    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=121.23)
    res00, res01, res02, res03 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=116.75)
    res10, res11, res12, res13 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=113.85)
    res20, res21, res22, res23 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=100.85)
    res30, res31, res32, res33 = np.split(new_cc_signal_and_noise, 4, axis=0)


    res = [res00, res11, res22, res33]
    mean_SNR_ccf_all = []
    # Get the average in an aperture
    for i in range(4):
        Planet_HCI_reconstructed = image_reconstruct(res[i], PlanetHCI_vec_shape[0], len(drvs), PlanetHCI_position_nan)

        aperturesize = 3.5
        path_planet0 = subdir + "30_data/Data_Planets/"
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
        planet_position = [56 / 2, 56 / 2]
        positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

        aperture_circ = CircularAperture(positions, r=aperturesize)
        aperture_masks00 = aperture_circ.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
        # aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]
        mask_aperture_image00 = copy.deepcopy(aperture_image00)
        mask_aperture_image00[aperture_image00 == 0] = np.nan

        SNR_img = Planet_HCI_reconstructed[2000, :, :] / np.std(np.concatenate((Planet_HCI_reconstructed[0:1800, :, :], Planet_HCI_reconstructed[2200:4000, :, :]), axis=0), axis=0)
        mean_SNR_ccf = np.nanmean(SNR_img * mask_aperture_image00)
        mean_SNR_ccf_all.append(mean_SNR_ccf)

    print(mean_SNR_ccf_all)
    print(meta_data)

    df1 = pd.DataFrame(np.concatenate([res00, res11, res22, res33]))
    df1.columns = drvs
    df1.index = injection_data_set.index
    df1 = df1.join(injection_data_set['planet'])

    pd.to_pickle(df1, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphanone_temp"+str(ls_template_characteristics[t]['Temp'])+"_logg_"+str(ls_template_characteristics[t]['Surf_grav'])+"_H2O_"+str(a0)+"_CO_"+str(b0)+".pkl")

    if t == 0:
        meta_dt = pd.DataFrame([[121.23,116.75,113.85,100.85],mean_SNR_ccf_all])
        meta_dt.columns = planetlist2
        meta_dt.index = ['alpha', 'mean_snr']
        pd.to_pickle(meta_dt, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphanone_H2O_"+str(a0)+"_CO_"+str(b0)+"_meta.pkl")



### none bad conditions

    # alphabadconditions
    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=52.46)
    res00, res01, res02, res03 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=40.455)
    res10, res11, res12, res13 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=45.575)
    res20, res21, res22, res23 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=44.18)
    res30, res31, res32, res33 = np.split(new_cc_signal_and_noise, 4, axis=0)

    res = [res00, res11, res22, res33]
    mean_SNR_ccf_all = []
    # Get the average in an aperture
    for i in range(4):
        Planet_HCI_reconstructed = image_reconstruct(res[i], PlanetHCI_vec_shape[0], len(drvs), PlanetHCI_position_nan)
        #plt.imshow(Planet_HCI_reconstructed[2000, :, :])
        #plt.show()

        aperturesize = 3.5
        path_planet0 = subdir + "30_data/Data_Planets/"
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
        planet_position = [56 / 2, 56 / 2]
        positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

        aperture_circ = CircularAperture(positions, r=aperturesize)
        aperture_masks00 = aperture_circ.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
        # aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]
        mask_aperture_image00 = copy.deepcopy(aperture_image00)
        mask_aperture_image00[aperture_image00 == 0] = np.nan

        SNR_img = Planet_HCI_reconstructed[2000, :, :] / np.std(np.concatenate((Planet_HCI_reconstructed[0:1800, :, :], Planet_HCI_reconstructed[2200:4000, :, :]), axis=0), axis=0)
        mean_SNR_ccf = np.nanmean(SNR_img * mask_aperture_image00)
        mean_SNR_ccf_all.append(mean_SNR_ccf)
        #aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]

    print(mean_SNR_ccf_all)
    print(meta_data)


    df2 = pd.DataFrame(np.concatenate([res00, res11, res22, res33]))
    df2.columns = drvs
    df2.index = injection_data_set.index
    df2 = df2.join(injection_data_set['planet'])
    pd.to_pickle(df2, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphanone2_temp"+str(ls_template_characteristics[t]['Temp'])+"_logg_"+str(ls_template_characteristics[t]['Surf_grav'])+"_H2O_"+str(a0)+"_CO_"+str(b0)+".pkl")

    if t==0:
        meta_dt = pd.DataFrame([[52.46,40.455,45.575,44.18],mean_SNR_ccf_all])
        meta_dt.columns = planetlist2
        meta_dt.index = ['alpha', 'mean_snr']
        pd.to_pickle(meta_dt, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphanone2_H2O_"+str(a0)+"_CO_"+str(b0)+"_meta.pkl")


    # alphaover2
    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=121.23/2)
    res00, res01, res02, res03 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=116.75/2)
    res10, res11, res12, res13 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=113.85/2)
    res20, res21, res22, res23 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=100.85/2)
    res30, res31, res32, res33 = np.split(new_cc_signal_and_noise, 4, axis=0)

    res = [res00, res11, res22, res33]
    mean_SNR_ccf_all = []
    # Get the average in an aperture
    for i in range(4):
        Planet_HCI_reconstructed = image_reconstruct(res[i], PlanetHCI_vec_shape[0], len(drvs), PlanetHCI_position_nan)

        aperturesize = 3.5
        path_planet0 = subdir + "30_data/Data_Planets/"
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
        planet_position = [56 / 2, 56 / 2]
        positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

        aperture_circ = CircularAperture(positions, r=aperturesize)
        aperture_masks00 = aperture_circ.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
        mask_aperture_image00 = copy.deepcopy(aperture_image00)
        mask_aperture_image00[aperture_image00 == 0] = np.nan

        SNR_img = Planet_HCI_reconstructed[2000, :, :] / np.std(np.concatenate((Planet_HCI_reconstructed[0:1800, :, :], Planet_HCI_reconstructed[2200:4000, :, :]), axis=0), axis=0)
        mean_SNR_ccf = np.nanmean(SNR_img * mask_aperture_image00)
        mean_SNR_ccf_all.append(mean_SNR_ccf)


    print(mean_SNR_ccf_all)
    print(meta_data)


    df2 = pd.DataFrame(np.concatenate([res00, res11, res22, res33]))
    df2.columns = drvs
    df2.index = injection_data_set.index
    df2 = df2.join(injection_data_set['planet'])
    pd.to_pickle(df2, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover2_temp"+str(ls_template_characteristics[t]['Temp'])+"_logg_"+str(ls_template_characteristics[t]['Surf_grav'])+"_H2O_"+str(a0)+"_CO_"+str(b0)+".pkl")

    if t==0:
        meta_dt = pd.DataFrame([[121.23/2,116.75/2,113.85/2,100.85/2],mean_SNR_ccf_all])
        meta_dt.columns = planetlist2
        meta_dt.index = ['alpha', 'mean_snr']
        pd.to_pickle(meta_dt, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover2_H2O_"+str(a0)+"_CO_"+str(b0)+"_meta.pkl")


    # alphaover3
    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=121.23/3)
    res00, res01, res02, res03 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=116.75/3)
    res10, res11, res12, res13 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=113.85/3)
    res20, res21, res22, res23 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=100.85/3)
    res30, res31, res32, res33 = np.split(new_cc_signal_and_noise, 4, axis=0)

    res = [res00, res11, res22, res33]
    mean_SNR_ccf_all = []
    # Get the average in an aperture
    for i in range(4):
        Planet_HCI_reconstructed = image_reconstruct(res[i], PlanetHCI_vec_shape[0], len(drvs), PlanetHCI_position_nan)
        #plt.imshow(Planet_HCI_reconstructed[2000, :, :])
        #plt.show()

        aperturesize = 3.5
        path_planet0 = subdir + "30_data/Data_Planets/"
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
        planet_position = [56 / 2, 56 / 2]
        positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

        aperture_circ = CircularAperture(positions, r=aperturesize)
        aperture_masks00 = aperture_circ.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
        # aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]
        mask_aperture_image00 = copy.deepcopy(aperture_image00)
        mask_aperture_image00[aperture_image00 == 0] = np.nan

        SNR_img = Planet_HCI_reconstructed[2000, :, :] / np.std(np.concatenate((Planet_HCI_reconstructed[0:1800, :, :], Planet_HCI_reconstructed[2200:4000, :, :]), axis=0), axis=0)
        mean_SNR_ccf = np.nanmean(SNR_img * mask_aperture_image00)
        mean_SNR_ccf_all.append(mean_SNR_ccf)
        #aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]


    print(mean_SNR_ccf_all)
    print(meta_data)


    df2 = pd.DataFrame(np.concatenate([res00, res11, res22, res33]))
    df2.columns = drvs
    df2.index = injection_data_set.index
    df2 = df2.join(injection_data_set['planet'])
    pd.to_pickle(df2, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover3_temp"+str(ls_template_characteristics[t]['Temp'])+"_logg_"+str(ls_template_characteristics[t]['Surf_grav'])+"_H2O_"+str(a0)+"_CO_"+str(b0)+".pkl")

    if t==0:
        meta_dt = pd.DataFrame([[121.23/3,116.75/3,113.85/3,100.85/3],mean_SNR_ccf_all])
        meta_dt.columns = planetlist2
        meta_dt.index = ['alpha', 'mean_snr']
        pd.to_pickle(meta_dt, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover3_H2O_"+str(a0)+"_CO_"+str(b0)+"_meta.pkl")



    # alphaover4
    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=121.23/4)
    res00, res01, res02, res03 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=116.75/4)
    res10, res11, res12, res13 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=113.85/4)
    res20, res21, res22, res23 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=100.85/4)
    res30, res31, res32, res33 = np.split(new_cc_signal_and_noise, 4, axis=0)

    res = [res00, res11, res22, res33]
    mean_SNR_ccf_all = []
    # Get the average in an aperture
    for i in range(4):
        Planet_HCI_reconstructed = image_reconstruct(res[i], PlanetHCI_vec_shape[0], len(drvs), PlanetHCI_position_nan)
        #plt.imshow(Planet_HCI_reconstructed[2000, :, :])
        #plt.show()

        aperturesize = 3.5
        path_planet0 = subdir + "30_data/Data_Planets/"
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
        planet_position = [56 / 2, 56 / 2]
        positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

        aperture_circ = CircularAperture(positions, r=aperturesize)
        aperture_masks00 = aperture_circ.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
        mask_aperture_image00 = copy.deepcopy(aperture_image00)
        mask_aperture_image00[aperture_image00 == 0] = np.nan

        SNR_img = Planet_HCI_reconstructed[2000, :, :] / np.std(np.concatenate((Planet_HCI_reconstructed[0:1800, :, :], Planet_HCI_reconstructed[2200:4000, :, :]), axis=0), axis=0)
        mean_SNR_ccf = np.nanmean(SNR_img * mask_aperture_image00)
        mean_SNR_ccf_all.append(mean_SNR_ccf)


    print(mean_SNR_ccf_all)
    print(meta_data)


    df2 = pd.DataFrame(np.concatenate([res00, res11, res22, res33]))
    df2.columns = drvs
    df2.index = injection_data_set.index
    df2 = df2.join(injection_data_set['planet'])
    pd.to_pickle(df2, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover4_temp"+str(ls_template_characteristics[t]['Temp'])+"_logg_"+str(ls_template_characteristics[t]['Surf_grav'])+"_H2O_"+str(a0)+"_CO_"+str(b0)+".pkl")

    if t==0:
        meta_dt = pd.DataFrame([[121.23/4,116.75/4,113.85/4,100.85/4],mean_SNR_ccf_all])
        meta_dt.columns = planetlist2
        meta_dt.index = ['alpha', 'mean_snr']
        pd.to_pickle(meta_dt, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover4_H2O_"+str(a0)+"_CO_"+str(b0)+"_meta.pkl")



    # alphaover6
    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=121.23/6)
    res00, res01, res02, res03 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=116.75/6)
    res10, res11, res12, res13 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=113.85/6)
    res20, res21, res22, res23 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=100.85/6)
    res30, res31, res32, res33 = np.split(new_cc_signal_and_noise, 4, axis=0)

    res = [res00, res11, res22, res33]
    mean_SNR_ccf_all = []
    # Get the average in an aperture
    for i in range(4):
        Planet_HCI_reconstructed = image_reconstruct(res[i], PlanetHCI_vec_shape[0], len(drvs), PlanetHCI_position_nan)

        aperturesize = 3.5
        path_planet0 = subdir + "30_data/Data_Planets/"
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
        planet_position = [56 / 2, 56 / 2]
        positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

        aperture_circ = CircularAperture(positions, r=aperturesize)
        aperture_masks00 = aperture_circ.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
        # aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]
        mask_aperture_image00 = copy.deepcopy(aperture_image00)
        mask_aperture_image00[aperture_image00 == 0] = np.nan

        SNR_img = Planet_HCI_reconstructed[2000, :, :] / np.std(np.concatenate((Planet_HCI_reconstructed[0:1800, :, :], Planet_HCI_reconstructed[2200:4000, :, :]), axis=0), axis=0)
        mean_SNR_ccf = np.nanmean(SNR_img * mask_aperture_image00)
        mean_SNR_ccf_all.append(mean_SNR_ccf)

    print(mean_SNR_ccf_all)
    print(meta_data)

    df4 = pd.DataFrame(np.concatenate([res00, res11, res22, res33]))
    df4.columns = drvs
    df4.index = injection_data_set.index
    df4 = df4.join(injection_data_set['planet'])
    pd.to_pickle(df4, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover6_temp"+str(ls_template_characteristics[t]['Temp'])+"_logg_"+str(ls_template_characteristics[t]['Surf_grav'])+"_H2O_"+str(a0)+"_CO_"+str(b0)+".pkl")

    if t==0:
        meta_dt = pd.DataFrame([[121.23/6,116.75/6,113.85/6,100.85/6],mean_SNR_ccf_all])
        meta_dt.columns = planetlist2
        meta_dt.index = ['alpha', 'mean_snr']
        pd.to_pickle(meta_dt, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphaover6_H2O_"+str(a0)+"_CO_"+str(b0)+"_meta.pkl")



    # alphaover_exact
    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=16.06)
    res00, res01, res02, res03 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=6.83)
    res10, res11, res12, res13 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=25.54)
    res20, res21, res22, res23 = np.split(new_cc_signal_and_noise, 4, axis=0)

    new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=11.07)
    res30, res31, res32, res33 = np.split(new_cc_signal_and_noise, 4, axis=0)

    res = [res00, res11, res22, res33]
    mean_SNR_ccf_all = []
    # Get the average in an aperture
    for i in range(4):
        Planet_HCI_reconstructed = image_reconstruct(res[i], PlanetHCI_vec_shape[0], len(drvs), PlanetHCI_position_nan)

        aperturesize = 3.5
        path_planet0 = subdir + "30_data/Data_Planets/"
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes" + planetlist2[i][7:])
        planet_position = [56 / 2, 56 / 2]
        positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])]

        aperture_circ = CircularAperture(positions, r=aperturesize)
        aperture_masks00 = aperture_circ.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=(56, 56))
        # aperture_data00 = aperture_image00 * Planet_HCI_reconstructed[2022,:,:]
        mask_aperture_image00 = copy.deepcopy(aperture_image00)
        mask_aperture_image00[aperture_image00 == 0] = np.nan

        SNR_img = Planet_HCI_reconstructed[2000, :, :] / np.std(np.concatenate((Planet_HCI_reconstructed[0:1800, :, :], Planet_HCI_reconstructed[2200:4000, :, :]), axis=0), axis=0)
        mean_SNR_ccf = np.nanmean(SNR_img * mask_aperture_image00)
        mean_SNR_ccf_all.append(mean_SNR_ccf)

    print(mean_SNR_ccf_all)
    print(pztel_mean_snr_2)


    df6 = pd.DataFrame(np.concatenate([res00, res11, res22, res33]))
    df6.columns = drvs
    df6.index = injection_data_set.index
    df6 = df6.join(injection_data_set['planet'])
    pd.to_pickle(df6, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphamin_temp"+str(ls_template_characteristics[t]['Temp'])+"_logg_"+str(ls_template_characteristics[t]['Surf_grav'])+"_H2O_"+str(a0)+"_CO_"+str(b0)+".pkl")

    if t==0:
        meta_dt = pd.DataFrame([[16.06,6.83,25.54,11.07],mean_SNR_ccf_all])
        meta_dt.columns = planetlist2
        meta_dt.index = ['alpha', 'mean_snr']
        pd.to_pickle(meta_dt, data_path + "csv_inputs/intro_plot_dataset/PZ_Tel_signals_alphamin_H2O_"+str(a0)+"_CO_"+str(b0)+"_meta.pkl")


