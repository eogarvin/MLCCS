"""
Created on Mon Jan 3 14:20:58 2021

@authors: Emily Garvin
"""
from ml_spectroscopy.config import path_init
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_spectroscopy.ccf_class import CCFSignalNoise
from ml_spectroscopy.config import global_settings as gs
from sklearn import metrics
from astropy.io import fits
import gc
from ml_spectroscopy.utility_functions import test_onCCF_rv0_SNR
from photutils import CircularAperture
import copy
from astropy.modeling import models, fitting
from ml_spectroscopy.CreatePlanetsGrid_class import CreatePlanetsGrid
from ml_spectroscopy.DataPreprocessing_utils import importWavelength_asList, image_reconstruct, image_deconstruct
from multiprocessing import Pool
import sys
import time


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
    mol = "CO"

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

np.random.seed(s)

## Read some data
data_file_templates = os.path.join(data_path, "csv_inputs/Molecular_Templates_df.csv")
templates = pd.read_csv(data_file_templates, index_col=0)

template_characteristics = {'Temp': 1200,
                            'Surf_grav': 4.1,
                            'H2O': a0,
                            'CO': b0}

template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]

TempCol = template.columns.get_loc("tempP")
tf = template.drop(template.columns[TempCol:], axis=1)
tw = pd.to_numeric(tf.columns)
tf = np.array(tf).flatten()


################### Part 1 - Evaluate gaussianity of the signal ##################
###****************************************************************************###
### import the datasets with planets to fit a gaussian blob to them.           ###
###****************************************************************************###


planetlist = ['GQlupb0','GQlupb1','GQlupb2','GQlupb3','GQlupb4','GQlupb5','GQlupb6','GQlupb7','PZTel_10','PZTel_11','PZTel_12','PZTel_13','PZTel_20','PZTel_21','PZTel_22','PZTel_23','PZTel_24','PZTel_25','PZTel_26']
planettitle = ['GQ Lup B, cube 0','GQ Lup B, cube 1','GQ Lup B, cube 2','GQ Lup B, cube 3','GQ Lup B, cube 4','GQ Lup B, cube 5','GQ Lup B, cube 6','GQ Lup B, cube 7','PZ Tel B (1), cube 0','PZ Tel B (1), cube 1','PZ Tel B (1), cube 2','PZ Tel B (1), cube 3','PZ Tel B (2), cube 0','PZ Tel B (2), cube 1','PZ Tel B (2), cube 2','PZ Tel B (2), cube 3','PZ Tel B (2), cube 4','PZ Tel B (2), cube 5','PZ Tel B (2), cube 6']


dtpthtest = data_path + "csv_inputs/True_CCF_data"
path_planet0 = subdir + "30_data/Data_Planets/"
dir_file_planet = data_path + 'True_HCI_data'

gqlupb_stdx = []
gqlupb_stdy = []
pztel_stdx = []
pztel_stdy = []

dt_planet_crosscorr = pd.DataFrame()
for i in range(19):

    # Open the Original planet
    hdu_list0 = fits.open(dir_file_planet + '/res_' + planetlist[i] + '.fits')
    hdu_list0.info()
    Planet_HCI = hdu_list0[0].data
    hdu_list0.close()
    Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down

    # Open the cross-correlated datasets' + filename + '_crosscorr_dt'+str(mol)+'.csv'
    temp_planet_crosscorr0 = pd.read_csv(data_path + 'csv_inputs/True_CCF_data/'+str(mol)+'/'+str(planetlist[i])+'_crosscorr_dt.csv', index_col= 0)
    temp_planet_crosscorr = temp_planet_crosscorr0.drop('0', axis=1)
    ind_2=temp_planet_crosscorr.index
    ind_1=np.repeat(planetlist[i], len(temp_planet_crosscorr.index))
    arrays = [ind_1,ind_2]
    temp_planet_crosscorr.index = arrays
    frames = [dt_planet_crosscorr, temp_planet_crosscorr]
    dt_planet_crosscorr = pd.concat(frames)

    # Open the centering cubes to get the positions
    if planetlist[i] in ['GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7']:
        cube_centering = importWavelength_asList(path_planet0 + "GQlupb/Centering_cubes"+str(planetlist[i][-1:]))
        RV=2044
    elif planetlist[i] in ['PZTel_10', 'PZTel_11', 'PZTel_12', 'PZTel_13']:
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_1/Centering_cubes"+str(planetlist[i][-1:]))
        RV=2022
    elif planetlist[i] in ['PZTel_20', 'PZTel_21', 'PZTel_22', 'PZTel_23', 'PZTel_24', 'PZTel_25', 'PZTel_26']:
        cube_centering = importWavelength_asList(path_planet0 + "PZTel_2/Centering_cubes"+str(planetlist[i][-1:]))
        RV=2077


    # Reconstruct the CCF image using the original image pixel locations
    PlanetHCI_nanrm, PlanetHCI_vec_shape, PlanetHCI_position_nan = image_deconstruct(Planet_HCI)
    Planet_HCI_reconstructed_ccf = image_reconstruct(np.array(temp_planet_crosscorr), PlanetHCI_vec_shape[0], 4000, PlanetHCI_position_nan)

    # Define the center of the image and append the centering information to sport the planet
    planet_position = [len(Planet_HCI_reconstructed_ccf[0]) / 2, len(Planet_HCI_reconstructed_ccf[2]) / 2]
    positions = [planet_position[0] + (cube_centering[0]), planet_position[1] - (cube_centering[2])] #(positions (x,y))

    # Get gaussian
    x,y = np.meshgrid(np.arange(56), np.arange(56))
    gaussian = models.Gaussian2D(amplitude=1, x_mean=round(positions[0]), y_mean=round(positions[1]), x_stddev=1.5, y_stddev=1.5)

    # Define auxiliary function for tieing the standard deviations
    def tie_stddev(gaussian):
        return gaussian.y_stddev
    gaussian.x_stddev.tied = tie_stddev

    fit_p = fitting.LevMarLSQFitter()
    gaussian_model = fit_p(gaussian, x, y, Planet_HCI_reconstructed_ccf[RV,:,:])

    if i <= 7:
        gqlupb_stdx.append(gaussian_model.x_stddev.value)
        gqlupb_stdy.append(gaussian_model.y_stddev.value)
    elif i > 7 & i <= 11:
        pztel_stdx.append(gaussian_model.x_stddev.value)
        pztel_stdy.append(gaussian_model.y_stddev.value)

    del temp_planet_crosscorr, frames
    gc.collect()



########################### Part 2 - inject the planet ###########################
###****************************************************************************###
### import the datasets and store them in a single big multi-index data frame. ###
###****************************************************************************###
planetlist = ['GQlupb0','GQlupb1','GQlupb2','GQlupb3','GQlupb4','GQlupb5','GQlupb6','GQlupb7','PZTel_10','PZTel_11','PZTel_12','PZTel_13','PZTel_20','PZTel_21','PZTel_22','PZTel_23','PZTel_24','PZTel_25','PZTel_26']
dtpthtest = data_path + "csv_inputs/True_Spectrum_Data"

dt_noise = pd.DataFrame()
for i in range(19):
    temp_noise = pd.read_pickle(data_path + 'csv_inputs/True_Spectrum_Noise/'+str(planetlist[i])+'_Spectrum_noise_trim.pkl')
    ind2=temp_noise.index
    ind1=np.repeat(planetlist[i], len(temp_noise.index))
    arrays = [ind1,ind2]
    temp_noise.index = arrays
    frames = [dt_noise, temp_noise]
    dt_noise = pd.concat(frames)
    del temp_noise, frames
    gc.collect()

pd.to_pickle(dt_noise, data_path+'csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/noise_cubes_GQlupbx_PZTel1x_PZTel2x.pkl')


###*****************************************************************************************###
### Recover the masked area of the planet and create an aperture for fake planet insertion  ###
###*****************************************************************************************###

# In each of these data sets there is a hole -> Due to data preprocessing to get the noise (c.f. 0_(b)_cubes_preprocessing)
# We want to be able to reconstruct the image despite the hole
# Then, propose a new aperture to insert the planet.

planetlist = ['GQlupb0','GQlupb1','GQlupb2','GQlupb3','GQlupb4','GQlupb5','GQlupb6','GQlupb7','PZTel_10','PZTel_11','PZTel_12','PZTel_13','PZTel_20','PZTel_21','PZTel_22','PZTel_23','PZTel_24','PZTel_25','PZTel_26']
dtpathtest = data_path + "csv_inputs/True_Spectrum_Data"
listofnanpositions = []
injection_regions_frames = {}
gaussian_parameters_frames = {}
injection_data_set = pd.DataFrame()
injection_labels_set = pd.DataFrame()
buffer = 12

injection_data_set_gauss_aperture = pd.DataFrame()
injection_data_set_gauss_decay = pd.DataFrame()


for i in range(19):

    # select a planet
    noise_temp = dt_noise.loc[planetlist[i]]
    noise_temp_wl_shape = noise_temp.shape[1]

    # Get the data where the planet is indicated
    path0 = os.path.join(dtpathtest,
                         str(planetlist[i])+'_spectrum_dt.csv') # Untrimmed data. therefore, take the WL range from the trimmed data-
    df_temp = pd.read_csv(path0)
    # rebuild the mask
    imsize = int(np.sqrt(len(df_temp['Planet'])))
    mask = np.reshape(np.array(df_temp['Planet']), (imsize,imsize))
    ##plt.imshow(mask)
    ##plt.show()

    # Create a cube for the mask, create a block and then a cube
    mask_block = np.reshape(np.repeat(mask, noise_temp_wl_shape), (imsize*imsize, noise_temp_wl_shape))
    mask_cube = np.reshape(np.repeat(mask, noise_temp_wl_shape), (imsize,imsize,noise_temp_wl_shape))
    mask_cube_inv = np.empty((noise_temp_wl_shape,imsize,imsize))
    # revert the block to stack it first by wavelength.
    for j in range(noise_temp_wl_shape):
        mask_cube_inv[j,:,:]=mask_cube[:,:,j]
    mask_cube_inv_copy = copy.deepcopy(mask_cube_inv)
    mask_cube_inv[np.where(mask_cube_inv==1)]=np.nan

    # Deconstruct a full image (before the trimming and collapsing of the data sets into simple noise spectra; we only use the spatial information to recover the locations of NA.
    PlanetHCI_nanrm, Planet_vec_shape, Planet_position_nan = image_deconstruct(mask_cube_inv)
    #test = image_reconstruct(PlanetHCI_nanrm, PlanetHCI_vec_shape[0], PlanetHCI_vec_shape[1], PlanetHCI_position_nan)
    #plt.imshow(test[1000,:,:])
    #plt.show()

    # Use the spatial locations of the NA values (masked out planet) from the previous deconstruction, and reconstruct our planet
    test = image_reconstruct(np.array(dt_noise.loc[planetlist[i]]), Planet_vec_shape[0], Planet_vec_shape[1], Planet_position_nan)
    plt.imshow(test[1000,:,:])
    plt.title(str(planettitle[i]) +', masking region', fontsize=18)
    plt.xlabel('[px]', fontsize=17)
    plt.ylabel('[px]', fontsize=17)
    plt.savefig(data_path+'csv_inputs/CCF_realistic_fakeplanets/noise_images/'+str(planetlist[i])+'images_planet_masking_region'+str(mol)+'.pdf', bbox_inches='tight')
    if planetlist[i] in ['GQlupb0', 'PZTel_10', 'PZTel_25']:
        plt.show()
    plt.close()

    ## MASKING

    # Set a random aperture in which to inject the planet. First define the size and then the center in a randomly picked region (excluding 6 pixels from the edge, excluding the masked region and min 1 pixel around it)
    y_mask = np.median(np.argwhere(np.isnan(test[1000, :, :])), 0)[0]
    x_mask = np.median(np.argwhere(np.isnan(test[1000, :, :])), 0)[1]


    if planetlist[i] in ['GQlupb0', 'GQlupb1', 'GQlupb2', 'GQlupb3', 'GQlupb4', 'GQlupb5', 'GQlupb6', 'GQlupb7']:
        buffer = 10
        aperturesize = np.random.choice((3.5,4))
    elif planetlist[i] in ['PZTel_10', 'PZTel_11', 'PZTel_12', 'PZTel_13']:
        buffer = 10
        aperturesize = np.random.choice((3.5,4,4.5))
    elif planetlist[i] in ['PZTel_20', 'PZTel_21', 'PZTel_22', 'PZTel_23', 'PZTel_24', 'PZTel_25', 'PZTel_26']:
        buffer = 10
        aperturesize = np.random.choice((2.5,3,3.5))


    array_include_x = np.arange(6, 50)
    array_exclude_x = np.arange((x_mask - buffer), (x_mask + buffer))
    set_to_sample_x = set(array_include_x) - set(array_exclude_x)
    tuple_to_sample_x = tuple(set_to_sample_x)

    array_include_y = np.arange(6, 50)
    array_exclude_y = np.arange((y_mask - buffer), (y_mask + buffer))
    set_to_sample_y = set(array_include_y) - set(array_exclude_y)
    tuple_to_sample_y = tuple(set_to_sample_y)

    aperture_position = [np.random.choice(tuple_to_sample_x), np.random.choice(tuple_to_sample_y)]

    # Now, set the aperture (of size 4) and plot an image to see where it falls for each cube
    aperture_circ = CircularAperture(aperture_position, r=aperturesize)
    aperture_injection = aperture_circ.to_mask(method='center')
    injection_region = aperture_injection.to_image(shape=[imsize,imsize])

    # Only for demo to highlight the inhection region
    region_inject_planet = (abs((injection_region)-1)) * test + (((injection_region)+1)*0.0003)

    plt.imshow(region_inject_planet[0,:,:])
    plt.title(str(planettitle[i]) +',\n fake planet injection region', fontsize=18)
    plt.xlabel('[px]', fontsize=17)
    plt.ylabel('[px]', fontsize=17)
    plt.savefig(data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/'+str(planetlist[i])+'_images_fakeplanet_injection_region'+str(mol)+'.pdf', bbox_inches='tight')
    if planetlist[i] in ['GQlupb0', 'PZTel_10', 'PZTel_25']:
        plt.show()
    plt.close()


    # save the masks per frame, then construct the cube and save them in dataset shape
    injection_regions_frames[planetlist[i]]=injection_region
    # save this as a pickle
    injection_regions_cubes=np.empty((noise_temp_wl_shape,imsize,imsize))
    # revert the block to stack it first by wavelength. --> stacked frames = cube
    for k in range(noise_temp_wl_shape):
        injection_regions_cubes[k,:,:]=injection_region[:,:]
    # use the copy of the previous masked regions of the true planet, to indicate where to add nans (this way the injection matrix is collapsed into a data set in the exact same way as the new data)
    injection_regions_cubes[np.where(mask_cube_inv_copy==1)]=np.nan
    injection_data, injection_regions_cubes_shape, injection_regions_cubes_position_nan = image_deconstruct(injection_regions_cubes)

    ## Test the cube reconstruction for the injection regions
    tst= image_reconstruct(injection_data, injection_regions_cubes_shape[0], injection_regions_cubes_shape[1], injection_regions_cubes_position_nan)
    plt.imshow(tst[0, :, :])
    plt.xlabel('[px]', fontsize=17)
    plt.ylabel('[px]', fontsize=17)
    plt.title(str(planettitle[i]) +',\n injection and masked region', fontsize=18)
    plt.savefig(data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/'+str(planetlist[i])+'_images_fakeplanet_injection_and_mask'+str(mol)+'.pdf', bbox_inches='tight')
    if planetlist[i] in ['GQlupb0', 'PZTel_10', 'PZTel_25']:
        plt.show()
    plt.close()



    ## 2. GAUSSIAN BLOB

    # Use the center and aperture size to construct a gaussian blob which will serve for the test set:
    # Get gaussian
    x, y = np.meshgrid(np.arange(56), np.arange(56))


    if i<= 7:
        stdx = np.random.choice(gqlupb_stdx)
        if stdx >= aperturesize:
            stdx = aperturesize - 2
        elif stdx < 0.5:
            stdx = 0.75
        stdy = stdx
    else:
        stdx = np.random.choice(pztel_stdx)
        if stdx >= aperturesize:
            stdx = aperturesize - 2
        elif stdx < 0.5:
            stdx = 0.75
        stdy = stdx


    gaussian = models.Gaussian2D(amplitude=1, x_mean=round(aperture_position[0]), y_mean=round(aperture_position[1]), x_stddev=stdx, y_stddev=stdy)
    #gaussian = models.Gaussian2D(amplitude=1, x_mean=round(aperture_position[0]), y_mean=round(aperture_position[1]), x_stddev=(aperturesize-np.random.choice((0.3,0.4,0.5,0.6))), y_stddev=(aperturesize-np.random.choice((0.3,0.4,0.5,0.6))))

    # save the gaussians per frame, then construct the cube and save them in dataset shape
    gaussian_parameters_frames[planetlist[i]]=gaussian



    # Create an injection data cube with a gaussian aperture
    injection_gaussian_cubes=np.empty((noise_temp_wl_shape,imsize,imsize))
    # revert the block to stack it first by wavelength. --> stacked frames = cube
    for k in range(noise_temp_wl_shape):
        injection_gaussian_cubes[k,:,:]=gaussian(x,y)[:,:]
    # use the copy of the previous masked regions of the true planet, to indicate where to add nans (this way the injection matrix is collapsed into a data set in the exact same way as the new data)
    injection_gaussian_cubes[np.where(mask_cube_inv_copy==1)]=np.nan
    injection_data_gaussian, injection_gaussian_cubes_shape, injection_gaussian_cubes_position_nan = image_deconstruct(injection_gaussian_cubes)


    ## GAUSSIAN decay

    ## Test the cube reconstruction
    tst= image_reconstruct(injection_data_gaussian, injection_gaussian_cubes_shape[0], injection_gaussian_cubes_shape[1], injection_gaussian_cubes_position_nan)
    plt.imshow(tst[0,:,:])
    plt.xlabel('[px]', fontsize=17)
    plt.ylabel('[px]', fontsize=17)
    plt.colorbar()
    plt.title(str(planettitle[i]) +',\n Gaussian signal injection', fontsize=18)
    plt.savefig(data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/'+str(planetlist[i])+'_images_fakeplanet_injection_gaussian_decay_cbar'+str(mol)+'+.pdf', bbox_inches='tight')
    if planetlist[i] in ['GQlupb0', 'PZTel_10', 'PZTel_25']:
        plt.show()
    plt.close()

    ## GAUSSIAN apertures

    ## Test the cube reconstruction for the gaussian aperture
    tst= image_reconstruct(injection_data_gaussian*injection_data, injection_regions_cubes_shape[0], injection_regions_cubes_shape[1], injection_regions_cubes_position_nan)
    plt.imshow(tst[0,:,:])
    plt.xlabel('[px]', fontsize=17)
    plt.ylabel('[px]', fontsize=17)
    plt.colorbar()
    plt.title(str(planettitle[i]) +',\n Gaussian signal injection', fontsize=18)
    plt.savefig(data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/'+str(planetlist[i])+'_images_fakeplanet_injection_gaussian_aperture_cbar'+str(mol)+'.pdf', bbox_inches='tight')
    if planetlist[i] in ['GQlupb0', 'PZTel_10', 'PZTel_25']:
        plt.show()
    plt.close()



    # Bind all in data frames

    # name indices for the 'mask' injection data frame
    injection_data_frame = pd.DataFrame(injection_data)
    gaussian_decay_data_frame = pd.DataFrame(injection_data_gaussian)
    gaussian_aperture_data_frame = pd.DataFrame(injection_data_gaussian * injection_data)

    ind1 = dt_noise.loc[planetlist[i]].index
    ind0 = np.repeat(planetlist[i], len(ind1))
    arr = [ind0,ind1]
    injection_data_frame.index = arr
    gaussian_decay_data_frame.index = arr
    gaussian_aperture_data_frame.index = arr


    # bind the mask cubes into a big data set
    df1 = [injection_data_set, injection_data_frame]
    injection_data_set = pd.concat(df1)

    #bind the labels to recover them later
    df2= [injection_labels_set, injection_data_frame[0]]
    injection_labels_set = pd.concat(df2)

    #bind the labels to recover them later
    df3= [injection_data_set_gauss_decay, gaussian_decay_data_frame]
    injection_data_set_gauss_decay = pd.concat(df3)

    #bind the labels to recover them later
    df4= [injection_data_set_gauss_aperture, gaussian_aperture_data_frame]
    injection_data_set_gauss_aperture = pd.concat(df4)

    plt.imshow(np.array(injection_data_frame))
    plt.title(str(planettitle[i]) + ',\n Dataframe shape', fontsize=15)
    plt.ylabel('Spaxels', fontsize=15)
    plt.xlabel('Wavelength', fontsize=15)
    plt.savefig(data_path + 'csv_inputs/CCF_realistic_fakeplanets/injection_region/' + str(planetlist[i]) + '_images_fakeplanet_injection_inDF.pdf', bbox_inches='tight')
    if planetlist[i] in ['GQlupb0', 'PZTel_10', 'PZTel_25']:
        plt.show()
    plt.close()

injection_data_set.columns = dt_noise.columns
injection_data_set_gauss_decay.columns = dt_noise.columns
injection_data_set_gauss_aperture.columns = dt_noise.columns


pd.to_pickle(injection_regions_frames, data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/data_fakeplanet_injection_region'+str(mol)+'.pkl')
pd.to_pickle(injection_data_set, data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/injection_data_set'+str(mol)+'.pkl')
pd.to_pickle(injection_labels_set, data_path+'csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/injection_labels_set'+str(mol)+'.pkl')
pd.to_pickle(injection_data_set_gauss_decay, data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/injection_data_set_gauss_decay'+str(mol)+'.pkl')
pd.to_pickle(injection_data_set_gauss_aperture, data_path+'csv_inputs/CCF_realistic_fakeplanets/injection_region/injection_data_set_gauss_aperture'+str(mol)+'.pkl')





###************************************************************************************************************###
### Advanced train_valid and test set: Select planets similar to GQ Lup B and PZ Tel B to inject in the frames ###
###************************************************************************************************************###


dt_signals = pd.read_csv(data_path + 'csv_inputs/Planet_Signals_df.csv', index_col=0)

tempvec_GQ = np.arange(2450,2760,10)
sgvec_GQ = np.arange(3.7,4.7,0.2)

tempvec_PZ = np.arange(2900,3100,10)
sgvec_PZ = np.arange(3.7,4.7,0.2)


if mol == "H2O":
    c0 = [1]
    d0 = [0,1]
elif mol == "CO":
    c0= [0,1]
    d0= [1]

template_characteristics_GQ = {'Temp': tempvec_GQ, 'Surf_grav': sgvec_GQ, 'H2O': c0, 'CO': d0, 'NH3':[0], 'CH4':[0]}
template_characteristics_PZ = {'Temp': tempvec_PZ, 'Surf_grav': sgvec_PZ, 'H2O': c0, 'CO': d0, 'NH3':[0],'CH4':[0]}


PlanetsGrid = CreatePlanetsGrid(dt_signals, dt_noise)
#test = PlanetsGrid.grid_subset(template_characteristics_GQ, 'GQlupb0', 8, 0)

# wrap a for loop around this :D
planet_grid_train_valid_tmp=pd.DataFrame()
planet_grid_test_tmp=pd.DataFrame()
# setting j allows to avoid training data leaking into the test data.
for i in range(19):

    if planetlist[i][0:5]=="GQlup":
        template = template_characteristics_GQ
        j=i
        ncubes = 8

    elif planetlist[i][0:5]=="PZTel":
        template = template_characteristics_PZ
        j=i-8
        ncubes = 11

    tmp_train_valid, tmp_test = PlanetsGrid.grid_subset(template , planetlist[i], ncubes, j)
    planet_grid_train_valid_tmp = pd.concat([planet_grid_train_valid_tmp, tmp_train_valid])
    planet_grid_test_tmp = pd.concat([planet_grid_test_tmp, tmp_test])


TempCol = planet_grid_train_valid_tmp.columns.get_loc("tempP")
saved_atmospheric_metainfo_train_valid_tmp=planet_grid_train_valid_tmp.columns[TempCol:len(planet_grid_train_valid_tmp.columns)]
saved_atmospheric_metainfo_train_valid=planet_grid_train_valid_tmp[saved_atmospheric_metainfo_train_valid_tmp]

TempCol = planet_grid_test_tmp.columns.get_loc("tempP")
saved_atmospheric_metainfo_test_tmp=planet_grid_test_tmp.columns[TempCol:len(planet_grid_test_tmp.columns)]
saved_atmospheric_metainfo_test=planet_grid_test_tmp[saved_atmospheric_metainfo_test_tmp]

# Here save the places where we actually do have a planet. The rest we can ignore.
#saved_atmospheric_metainfo * np.repeat(injection_labels_set,6)
saved_planets_wl_range_train_valid = planet_grid_train_valid_tmp[dt_noise.columns] # align the wavelength range axis of the simulated planets with the WL of the noise)
planet_grid_train_valid_tmp2 = saved_planets_wl_range_train_valid * injection_data_set_gauss_aperture
saved_atmospheric_characteristics_train_valid = saved_atmospheric_metainfo_train_valid * np.array(injection_labels_set)
planet_grid_train_validset = planet_grid_train_valid_tmp2.join(saved_atmospheric_characteristics_train_valid)

saved_planets_wl_range_test = planet_grid_test_tmp[dt_noise.columns] # align the wavelength range axis of the simulated planets with the WL of the noise)
planet_grid_test_tmp2 = saved_planets_wl_range_test * injection_data_set_gauss_decay
saved_atmospheric_characteristics_test = saved_atmospheric_metainfo_test * np.array(injection_labels_set)
planet_grid_testset = planet_grid_test_tmp2.join(saved_atmospheric_characteristics_test)

pd.to_pickle(planet_grid_train_validset, data_path+'csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/planets_grid'+str(mol)+'.pkl')
pd.to_pickle(planet_grid_testset, data_path+'csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/planets_grid_testset'+str(mol)+'.pkl')


###************************************************###
### Cross-correlate and try different alpha values ###
###************************************************###


## Read some data
data_file_templates = os.path.join(data_path, "csv_inputs/Molecular_Templates_df.csv")
templates = pd.read_csv(data_file_templates, index_col=0)

# Base template
template_characteristics = {'Temp': 2800,
                            'Surf_grav': 4.1,
                            'H2O': a0,
                            'CO': b0}



template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]



TempCol = template.columns.get_loc("tempP")
tf = template.drop(template.columns[TempCol:], axis=1)
tw = pd.to_numeric(tf.columns)
tf = np.array(tf).flatten()



data_signal_file = os.path.join(data_path,'csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/planets_grid'+str(mol)+'.pkl')
data_test_signal_file = os.path.join(data_path,'csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/planets_grid_testset'+str(mol)+'.pkl')
data_noise_file = os.path.join(data_path,'csv_inputs/CCF_realistic_fakeplanets/noise_and_planets_spectra/noise_cubes_GQlupbx_PZTel1x_PZTel2x.pkl')


noise_raw = pd.read_pickle(data_noise_file)
signal_raw = pd.read_pickle(data_signal_file)
signal_test_raw = pd.read_pickle(data_signal_file)


TempCol_dt= signal_raw.columns.get_loc("tempP")
df_signal = signal_raw.drop(signal_raw.columns[TempCol_dt:], axis=1)
df_signal_test = signal_test_raw.drop(signal_test_raw.columns[TempCol_dt:], axis=1)

# signal
dw = pd.to_numeric(df_signal.columns)
y_meta = signal_raw[signal_raw.columns[TempCol_dt:]]
df_signal = np.array(df_signal)

# test set signal
dw_test = pd.to_numeric(df_signal_test.columns)
y_meta_test = signal_test_raw[signal_test_raw.columns[TempCol_dt:]]
df_signal_test = np.array(df_signal_test)

# noise
df_noise = np.array(noise_raw)
index_noise = noise_raw.index


# Test the class function for different alpha values: test one alpha

# CCF over signal and noise separately (for the predefined template)



start_global = time.time()
ccf_tmp = CCFSignalNoise(signal=df_signal,
                         noise=df_noise,
                         dw=dw,
                         tw=tw,
                         tf=tf,
                         rvmin=-2000, rvmax=2000, drv=1,
                         mode="doppler", skipedge=0,
                         num_processes=int(maxcore))



new_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=20.0)
end_global = time.time()

time_it = (end_global-start_global)
print(time_it)
t0 = open('csv_inputs/CCF_realistic_fakeplanets/time_it.txt', 'w')
t0.write(str(time_it))
t0.close()



# Now make the SNR tests for different alpha values.

#*

plt.plot(new_cc_signal_and_noise[y_meta[mol]==1][1])
plt.title('Cross-correlation on simulated spectrum containing water', fontsize=13)
plt.xlabel("Radial Velocity span [Km/s]", fontsize=12)
plt.ylabel("Normalised cross-correlation values", fontsize=12)
plt.savefig(data_path + 'csv_inputs/CCF_realistic_fakeplanets/plots/crosscorrelation_planets_on_simulation'+str(mol)+'.pdf', bbox_inches='tight')
plt.show()
plt.close()


tmp_data_frame=pd.DataFrame(new_cc_signal_and_noise)
tmp_data_frame.columns = drvs
tmp_data_frame.index = index_noise


y_meta.index = index_noise
tmp_data_frame_tosave = tmp_data_frame.join(y_meta)
labels = np.array(y_meta[str(mol)] == 1.0)


# Run the experiment over a range of alpha values

# results of an SNR with a threshold of 3
results = test_onCCF_rv0_SNR(tmp_data_frame, 3.0)
scores_snr = np.array(results["SNR"])

metrics.roc_auc_score(labels, scores_snr)



# Compute for different alpha values

alpha_values = np.linspace(0, 500, 501)


auc_results = []
aucPR_results = []
f1_results = []
accuracy_results = []
average_snr = []


for tmp_alpha in tqdm(alpha_values):
    tmp_cc_signal_and_noise, drvs = ccf_tmp.get_data(alpha=tmp_alpha)
    tmp_data_frame = pd.DataFrame(tmp_cc_signal_and_noise)
    tmp_data_frame.columns = drvs

    tmp_results = test_onCCF_rv0_SNR(tmp_data_frame, 27.0)
    tmp_scores_snr = np.array(tmp_results["SNR"])


    sigmabar = tmp_data_frame.iloc[labels == True].drop(range(-200, 200), axis=1).std(axis=1)
    stat = tmp_data_frame.iloc[labels == True][0] / sigmabar
    average_snr_tmp = np.mean(stat)

    auc_results.append(metrics.roc_auc_score(labels, tmp_scores_snr))

    lr_precision_PCT, lr_recall_PCT, _ = metrics.precision_recall_curve(labels, tmp_scores_snr)
    aucPR_results.append(metrics.auc(lr_recall_PCT, lr_precision_PCT))

    f1_results.append(metrics.f1_score(labels, tmp_results["Y_pred"]))

    accuracy_results.append(metrics.accuracy_score(labels, tmp_results["Y_pred"]))

    average_snr.append(average_snr_tmp)

plt.style.use('seaborn')
plt.figure(figsize=(12, 8))
plt.plot(alpha_values, auc_results, label="classical SNR ROC AUC")
plt.plot(alpha_values, aucPR_results, label="classical SNR PR AUC")
plt.plot(alpha_values, f1_results, label="classical SNR F1")
plt.plot(alpha_values, accuracy_results, label="classical SNR Accuracy")
plt.legend()
plt.xlabel("Alpha value", fontsize = 15)
plt.ylabel("AUC", fontsize = 15)
plt.title("Performance of SNR as a decreasing function of noise level", fontsize = 18)
plt.savefig(plot_path+"SNR_per_alpha/realistic_fakeplanets_ccf_4ml_trim_norepetition_simple"+str(mol)+".pdf", bbox_inches='tight')
plt.show()
plt.close()

# for which alpha values do we have an AUC ROC of 0.5, 0.55, 0.6, 0.65, 0.7, max?



avg_snr = np.array(average_snr)
interp_alpha = np.interp(np.array((0.1,0.2,0.3,0.4,0.5,1,2,3,4,5)), avg_snr, alpha_values)


#interp_alpha = np.array((8,10,11,16,21))
### MULTIPROCESSING apply for other templates - train and validation data

Teff=[2300, 2500, 2700, 2800, 2900, 3100] #1600 2500
SG=[3.7, 3.9, 4.1, 4.3]
nparam = 2
ATSMP = list(map(tuple, np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T))

# for the alpha multiprocessing
nprocess = len(interp_alpha)
parameters = list(zip(interp_alpha * nprocess))
#
for atsmp in tqdm(ATSMP):
    template_characteristics = {'Temp': atsmp[0],
                                'Surf_grav': atsmp[1],
                                'H2O': a0,
                                'CO': b0}

    template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]

    TempCol = template.columns.get_loc("tempP")
    tf = template.drop(template.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()

    # CCF over signal and noise separately (for the predefined template)

    ccf_tmp = CCFSignalNoise(signal=df_signal,
                             noise=df_noise,
                             dw=dw,
                             tw=tw,
                             tf=tf,
                             rvmin=-2000, rvmax=2000, drv=1,
                             mode="doppler", skipedge=0,
                             num_processes=maxcore)


# evaluate all alpha values
    pool = Pool(nprocess)
    results = pool.starmap(ccf_tmp.get_data, parameters)
    pool.close()
    for i in range(len(interp_alpha)):
        df_temp_tosave = pd.DataFrame(results[i][0])
        df_temp_tosave.columns = results[i][1]
        df_temp_tosave.index = index_noise
        final_tosave = df_temp_tosave.join(y_meta)
        pd.to_pickle(final_tosave,
                     data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_sets/final_'+str(mol)+'_crosscorr_data_alpha_' + str(
                         round(interp_alpha[i])) + '_temp' + str(template_characteristics['Temp']) + '_sg' + str(template_characteristics['Surf_grav']) + '.pkl')



### MULTIPROCESSING apply for other templates - train and validation data

Teff=[2300, 2500, 2700, 2800, 2900, 3100] #1600 2500
SG=[3.7, 3.9, 4.1, 4.3]
nparam = 2
ATSMP = list(map(tuple, np.array(np.meshgrid(Teff, SG)).reshape(nparam, (len(Teff)*len(SG))).T))

# for the alpha multiprocessing
nprocess = 5
parameters = list(zip(interp_alpha * nprocess))
#
for atsmp in tqdm(ATSMP):
    template_characteristics = {'Temp': atsmp[0],
                                'Surf_grav': atsmp[1],
                                'H2O': a0,
                                'CO': b0}

    template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]

    TempCol = template.columns.get_loc("tempP")
    tf = template.drop(template.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()

    # CCF over signal and noise separately (for the predefined template)

    ccf_tmp = CCFSignalNoise(signal=df_signal_test,
                             noise=df_noise,
                             dw=dw_test,
                             tw=tw,
                             tf=tf,
                             rvmin=-2000, rvmax=2000, drv=1,
                             mode="doppler", skipedge=0,
                             num_processes=maxcore)


# evaluate all alpha values
    pool = Pool(nprocess)
    results = pool.starmap(ccf_tmp.get_data, parameters)
    pool.close()
    for i in range(5):
        df_temp_tosave = pd.DataFrame(results[i][0])
        df_temp_tosave.columns = results[i][1]
        df_temp_tosave.index = index_noise
        final_tosave = df_temp_tosave.join(y_meta)
        pd.to_pickle(final_tosave,
                     data_path + 'csv_inputs/CCF_realistic_fakeplanets/final_test_sets/final_testset_'+str(mol)+'_crosscorr_data_alpha_' + str(
                         round(interp_alpha[i])) + '_temp' + str(template_characteristics['Temp']) + '_sg' + str(template_characteristics['Surf_grav']) + '.pkl')


