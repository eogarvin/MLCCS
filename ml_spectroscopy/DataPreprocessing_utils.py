# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:04:30 2021

@author: emily
"""
# =============================================================================
# Utility packages 
# =============================================================================

# LIBRARIES
import csv
import numpy as np
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
import math

## ACTIVE SUBDIR
subdir=path_init()

## PATHS
data_path = subdir+"30_data/DataSets/"
plots_path = subdir+"60_plots/"
code_path = subdir + "50_code/"

# =============================================================================
# opens spectrums from pickle files into data
# =============================================================================

def open_spectrum(file_dir):
    with open(file_dir, 'r') as f:
        datareader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        data = np.array([row for row in datareader])
        if np.shape(data)[0] == 1 and len(np.shape(data)) > 1:
            data = data[0]
    return data


# =============================================================================
# Preprocess planet signals into  data frame 
# =============================================================================

def planetSignal_preprocessing(dir_pathP, nfiles, mol_cols, file_wavelength_range,  savedir, addCO=False, addFE=False):
    # For specific file number selection
    # b=[1,5000,10000,15000,20000,25000, 30000, 35000]
    # a=os.listdir(dir_pathP)
    # c = [ a[i] for i in b]

    # setup empty arrays and data frames
    nmol = len(mol_cols)
    molP0 = np.zeros((nfiles, nmol))
    molP = pd.DataFrame(molP0)
    molP = molP.set_axis(mol_cols, axis=1, inplace=False)

    tempP = np.empty((nfiles, 1))
    tempP[:] = np.nan

    logP = np.empty((nfiles, 1))
    logP[:] = np.nan
    
    CO = np.empty((nfiles, 1))
    CO[:] = np.nan
    
    FE = np.empty((nfiles, 1))
    FE[:] = np.nan

    fluxP = np.empty((nfiles, 2358))
    fluxP[:] = np.nan

    # iteration number
    i = 0

    # Data preprocessing (from raw files to Pandas)
    for file in os.listdir(dir_pathP):
        # for file in c:  #To select specific files

        # Use the file name to extract info
        if addCO==False & addFE==False:
            string, mol, temp, log = file[:-4].split('_')
        elif addCO==True & addFE ==False:
            string, mol, temp, log, co = file[:-4].split('_')
        elif addCO==False & addFE==True: 
            string, mol, temp, log, fe = file[:-4].split('_')
        elif addCO==True & addFE == True: 
            string, mol, temp, log, co, fe = file[:-4].split('_')
        # Remove the extra H2 - could be removed for future sets
        if mol[-2:] == 'H2':
            mol = mol[:-3]
        mol0 = mol.split('-')

        # stack up the molecules by mols of interest
        for m in mol_cols:
            if m in mol0:
                molP[m][i] = 1

        # stack up the temperatures
        Temp = temp[1:]
        tempP[i] = Temp

        # stack up the surface gravities
        Log = log[1:]
        logP[i] = Log
        
        # stack up the C/O
        Co = co[2:]
        CO[i] = Co
        
        # stack up the Fe
        Fe = fe[3:]
        FE[i] = Fe

        # stack up the fluxes in a 2d array (rows = planet flux, cols = wavelength)
        flux = open_spectrum(dir_pathP + '/' + file)
        fluxP[i, :] = flux

        i = i + 1
    # end

    # temperature as pandas
    tempP = pd.DataFrame(tempP)
    tempP = tempP.set_axis(["tempP"], axis=1, inplace=False)

    # surface gravity as pandas (! name changes from logP to loggP)
    loggP = pd.DataFrame(logP)
    loggP = loggP.set_axis(["loggP"], axis=1, inplace=False)
    
    # C/O as pandas
    COP = pd.DataFrame(CO)
    COP = COP.set_axis(["CO_ratio"], axis=1, inplace=False)
    
    # Metallicity as pandas 
    FeP = pd.DataFrame(FE)
    FeP = FeP.set_axis(["Fe"], axis=1, inplace=False)

    # try append wavelength as columns
    with open(file_wavelength_range, 'rb') as fp:
        wavelength_range0 = pickle.load(fp)

    wavelength_range1 = [float(num) for num in wavelength_range0]
    wavelength_range = [round(num, 5) for num in wavelength_range1]
    # wavelength_range = [("W"+str(num)) for num in wavelength_range2]

    # flux as pandas
    fluxP = pd.DataFrame(fluxP)
    fluxP = fluxP.set_axis(wavelength_range, axis=1, inplace=False)

    # Join data sets
    planet_data0 = fluxP.join(tempP)
    planet_data1 = planet_data0.join(loggP)
    planet_data2 = planet_data1.join(COP)
    planet_data3 = planet_data2.join(FeP)
    planet_data = planet_data3.join(molP)

    planet_data.to_csv(savedir + '/' + 'Planet_Signals_df2.csv')

    # end
    return planet_data


# =============================================================================
# dir_pathP="C:/Users/emily/Documents/ML_spectroscopy_thesis/30_data/spectrav1/new_planet_signals"
# mol_cols=['H2O','CO','CH4','NH3']
# nfiles=36960
# file_wavelength_range = "C:/Users/emily/Documents/ML_spectroscopy_thesis/30_data/spectrav1/wavelength_names_header"
# dir_save= "C:/Users/emily/Documents/ML_spectroscopy_thesis/30_data/spectrav1"
#
# planetSignal_df=planetSignal_preprocessing(dir_pathP, nfiles, mol_cols, file_wavelength_range, dir_save)
# =============================================================================


# =============================================================================
#  Preprocess planet signals into  data frame 
# =============================================================================

def templates_preprocessing(dir_pathT, ntemplates, molecules, file_wavelength_range, savedir, addCO=False, addFE=False):
    # For specific file number selection
    # b=[1,100,300,500,700]
    # a=os.listdir(dir_pathT)
    # c = [a[i] for i in b]

    # setup empty arrays and data frames
    nmol = len(molecules)
    molT0 = np.zeros((ntemplates, nmol))
    molT = pd.DataFrame(molT0)
    molT = molT.set_axis(molecules, axis=1, inplace=False)

    tempT = np.empty((ntemplates, 1))
    tempT[:] = np.nan

    logT = np.empty((ntemplates, 1))
    logT[:] = np.nan
      
    CO = np.empty((ntemplates, 1))
    CO[:] = np.nan
    
    FE = np.empty((ntemplates, 1))
    FE[:] = np.nan

    fluxT = np.empty((ntemplates, 2358))
    fluxT[:] = np.nan

    # iteration number
    i = 0

    # Data preprocessing (from raw files to Pandas)
    for file in os.listdir(dir_pathT):
        # for file in c:  #To select specific files

        # Use the file name to extract info
        if addCO==False & addFE==False:
            string, mol, temp, log = file[:-4].split('_')
        elif addCO==True & addFE ==False:
            string, mol, temp, log, co = file[:-4].split('_')
        elif addCO==False & addFE==True: 
            string, mol, temp, log, fe = file[:-4].split('_')
        elif addCO==True & addFE == True: 
            string, mol, temp, log, co, fe = file[:-4].split('_')
        # Remove the extra H2 - could be removed for future sets
        if mol[-2:] == 'H2':
            mol = mol[:-3]
        mol0 = mol.split('-')

        # stack up the molecules by mols of interest
        for m in molecules:
            if m in mol0:
                molT[m][i] = 1

        # stack up the temperatures
        Temp = temp[1:]
        tempT[i] = Temp

        # stack up the surface gravities
        Log = log[1:]
        logT[i] = Log
        
        # stack up the C/O
        Co = co[2:]
        CO[i] = Co
        
        # stack up the Fe
        Fe = fe[3:]
        FE[i] = Fe

        # stack up the fluxes in a 2d array (rows = planet flux, cols = wavelength)
        flux = open_spectrum(dir_pathT + '/' + file)
        fluxT[i, :] = flux

        i = i + 1
    # end    
    
    
    # temperature as pandas
    tempT = pd.DataFrame(tempT)
    tempT = tempT.set_axis(["tempP"], axis=1, inplace=False)

    # surface gravity as pandas (! name changes from logP to loggP)
    loggT = pd.DataFrame(logT)
    loggT = loggT.set_axis(["loggP"], axis=1, inplace=False)
    
    
    # C/O as pandas
    COT = pd.DataFrame(CO)
    COT = COT.set_axis(["CO_ratio"], axis=1, inplace=False)
    
    # Metallicity as pandas 
    FeT = pd.DataFrame(FE)
    FeT = FeT.set_axis(["Fe"], axis=1, inplace=False)

    # try append wavelength as columns
    with open(file_wavelength_range, 'rb') as fp:
        wavelength_range0 = pickle.load(fp)

    wavelength_range1 = [float(num) for num in wavelength_range0]
    wavelength_range = [round(num, 5) for num in wavelength_range1]
    # wavelength_range = [("W"+str(num)) for num in wavelength_range2]

    # flux as pandas
    fluxT = pd.DataFrame(fluxT)
    fluxT = fluxT.set_axis(wavelength_range, axis=1, inplace=False)

    # Join data sets
    template_data0 = fluxT.join(tempT)
    template_data1 = template_data0.join(loggT)
    template_data2 = template_data1.join(COT)
    template_data3 = template_data2.join(FeT)
    template_data = template_data3.join(molT)

    template_data.to_csv(savedir + '/' + 'Molecular_Templates_df2.csv')

    # end
    return template_data


# =============================================================================
# Performs a flattening only accross spatial pixels. We end up with vectors of spaxels
# =============================================================================

def pixelsbyspaxels0(data):
    out = np.empty(((data.shape[1] * data.shape[2]), (data.shape[0])))
    out[:] = np.NAN
    for i in range(0, data.shape[0]):
        out[:, i] = data[i, :, :].flatten()
    return (out)


# =============================================================================
# Performs a flattening only accross spatial pixels. We end up with vectors of spaxels
# appends the planet position by a binary variable, based on a mask constructed from the spatial results of the ccf 
# =============================================================================

def pixelsbyspaxels1(data, mask):
    out = np.empty(((data.shape[1] * data.shape[2]), (data.shape[0] + 1)))
    out[:] = np.NAN
    out[:, 0] = mask.flatten()
    for i in range(0, data.shape[0]):
        out[:, i + 1] = data[i, :, :].flatten()
    return (out)


# =============================================================================
# reconstructs a 2d array of spaxel vectors into a 3d array of images accross wavelengths
# =============================================================================

def spaxelsbypixels0(data):
    sbp0 = int(np.sqrt(data.shape[0]))
    sbp1 = int(data.shape[1])
    out = np.empty((sbp1, sbp0, sbp0))
    out[:] = np.NAN
    for i in range(0, sbp1):
        out[i, :, :] = data[:, i].reshape(sbp0, sbp0)
    return (out)


# =============================================================================
# To deconstruct an image into vectors of spaxels, removing NAN values.
# =============================================================================

def image_deconstruct(data_img, mask=None):
    if mask is None:
        data_vec = pixelsbyspaxels0(data_img)
        data_vec_shape = data_vec.shape  # aa=np.argwhere(np.isnan(data_vec))
        where_nan = np.argwhere(~np.isnan(data_vec))
        data_vec_nanrm = data_vec[~np.isnan(data_vec).any(axis=1)]
        data_vec_mask = None
        return data_vec_nanrm, data_vec_shape, where_nan

    else:
        data_vec = pixelsbyspaxels0(data_img)
        data_vec_shape = data_vec.shape  # aa=np.argwhere(np.isnan(data_vec))
        where_nan = np.argwhere(~np.isnan(data_vec))
        data_vec_nanrm = data_vec[~np.isnan(data_vec).any(axis=1)]

        data_vec_mask = pixelsbyspaxels1(data_img, mask)
        data_vec_nanrm_mask = data_vec_mask[~np.isnan(data_vec_mask).any(axis=1)]
        return data_vec_nanrm, data_vec_shape, where_nan, data_vec_nanrm_mask


# =============================================================================
# To reconstruct vectors of spaxels into an image, reusing out arguments from image_deconstruct. 
# The input image (arg 1) can be a function of the original image but the positional arguments (arg 2,arg 3) have to remain unchanged. 
# Use number of pixels of the original image size (before nan removal), and new number of spaxels
# =============================================================================

def image_reconstruct(data_vec_nanrm, original_npixels, new_nspaxels, where_nan):
    reconstruct_image = np.empty((original_npixels, new_nspaxels))
    reconstruct_image[:] = np.nan

    it = 0
    for i in np.unique(where_nan[:, 0]):
        reconstruct_image[i, :] = data_vec_nanrm[it]
        it = it + 1

    reconstructed_image = spaxelsbypixels0(reconstruct_image)

    return reconstructed_image


# =============================================================================
# To import wavelengths into a list format to be able to directly use them in functions
# =============================================================================

def importWavelength_asList(dir_file, extension='txt'):
    extensions = ['txt', 'pickle']
    if extension not in extensions:
        raise ValueError("Unexpected extension" % extension)

    if extension == 'txt':
        wr01 = open(dir_file + ".txt", 'r').read().split()
        wr02 = [float(num) for num in wr01]
        wr = [round(num, 5) for num in wr02]

    elif extension == 'pickle':
        with open(dir_file, 'rb') as fp:
            wr11 = pickle.load(fp)
            wr12 = [float(num) for num in wr11]
            wr = [round(num, 5) for num in wr12]

    return (wr)


# =============================================================================
# Cutout an image 
# =============================================================================

def cutout_image(image, planet_positionx, planet_positiony, cutout_rad=18, ):
    z = int(cutout_rad)
    image_cutout = image[:, (planet_positiony - z):(planet_positiony + z),
                   (planet_positionx - z):(planet_positionx + z)]
    return image_cutout


# =============================================================================
# spot (strongest ccf) planet in the data cubes
#
# !!! This function should be improved for the following cases: we have many planets, or we have a ccf peak outlier, or ccf of the planet is not super bright and an outlier rules it out
# =============================================================================


def spot_planet_incube(planetFilename, WRFilename, WR_extension, aperturesize, template_characteristics, centering_data=False, mol='H2O'):
    filename = planetFilename
    extension = WR_extension

    dir_file_planet = data_path + 'True_HCI_data'
    dir_file_WR = data_path + 'wavelength_ranges'
    # If templates have more molecules, remember to adapt the number of dropped end columns in the function
    dir_file_mol_template = data_path + 'csv_inputs/Molecular_Templates_df.csv'
    # aperture 
    # Where to save data sets
    savedirccf = data_path + 'csv_inputs/True_CCF_data/'+str(mol)+'/'
    savedirdata = data_path + 'csv_inputs/True_Spectrum_Data'
    # plot location
    dirplot = plots_path + 'Data_preprocessing_output_figures/'

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
    template_planetHCI = MT_df.loc[MT_df["tempP"] == template_characteristics['Temp']].loc[MT_df["loggP"] == template_characteristics['Surf_grav']].loc[MT_df["H2O"] == template_characteristics['H2O']].loc[MT_df["CO"] == template_characteristics['CO']]

    # if the selection is not well specified (e.g some precisions were not given on additional molecules)
    if template_planetHCI.shape[0] > 1:
        template_planetHCI = template_planetHCI.head(1)
        pd.DataFrame(template_planetHCI)
        print("Warning: Several templates available for this request, 1st template was selected to continue the task. Not all elements from the template have been defined, please check for temperature, surface gravity, or any additional molecule request which may be forgotten")

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

    ### Take the maximum points of all the spatial positions and spot the brightest point (assuming it is the planet) 
    # Spot the max ccf to find the planet. !!! This function should be improved for the following cases: we have many planets, or we have a ccf peak outlier, or ccf of the planet is not super bright and an outlier rules it out
    find_planetHCI = np.max(PlanetHCI_reconstructed_ccf, axis=0)

    # Find the maximum ccf location and draw a Circular aperture of chosen radius around it
    find_planetHCI_2 = np.argwhere(find_planetHCI == np.nanmax(find_planetHCI))
    
    if centering_data==False:
        positions = [int(find_planetHCI_2.flatten()[[1]]), int(find_planetHCI_2.flatten()[[0]])]
    elif centering_data==True:
        # corrected as it was searching for the file locally. let's see if it works.
        path_planet0=subdir+"30_data/Data_Planets/"
        if planetFilename[:2]=="GQ":
            cube_centering=importWavelength_asList(path_planet0+"GQlupb/Centering_cubes"+planetFilename[6:])
            planet_position=[len(PlanetHCI_reconstructed_ccf[0])/2,len(PlanetHCI_reconstructed_ccf[2])/2]
            positions=[planet_position[0]+(cube_centering[0]), planet_position[1]-(cube_centering[2])]
        elif planetFilename[:2]=="PZ":
            if planetFilename[:7]=="PZTel_1":
                cube_centering=importWavelength_asList(path_planet0+"PZTel_1/Centering_cubes"+planetFilename[7:])
                planet_position=[len(PlanetHCI_reconstructed_ccf[0])/2,len(PlanetHCI_reconstructed_ccf[2])/2]
                positions=[planet_position[0]+(cube_centering[0]), planet_position[1]-(cube_centering[2])]
            elif planetFilename[:7]=="PZTel_2":
                cube_centering=importWavelength_asList(path_planet0+"PZTel_2/Centering_cubes"+planetFilename[7:])
                planet_position=[len(PlanetHCI_reconstructed_ccf[0])/2,len(PlanetHCI_reconstructed_ccf[2])/2]
                positions=[planet_position[0]+(cube_centering[0]), planet_position[1]-(cube_centering[2])]
        elif planetFilename[:2]=="RO":
            cube_centering=importWavelength_asList(path_planet0+"ROXs42B/Centering_cubes"+planetFilename[7:])
            planet_position=importWavelength_asList(path_planet0+"ROXs42B/planet_position")
            positions=[planet_position[0]+(cube_centering[0]), planet_position[2]-(cube_centering[2])]       
        
    aperture00 = CircularAperture(positions, r=aperturesize)
    imshape = ((int(PlanetHCI_reconstructed_ccf.shape[1]), int(PlanetHCI_reconstructed_ccf.shape[2])))
    aperture_masks00 = aperture00.to_mask(method='center')
    aperture_image00 = aperture_masks00.to_image(shape=imshape)
    aperture_data00 = aperture_image00 * find_planetHCI

    # Output CSVs 

    # Cross correlation 
    PlanetHCIccf_nanrm, PlanetHCIccf_vec_shape, PlanetHCIccf_where_nan, PlanetHCIccf_nanrm_mask = image_deconstruct(
        PlanetHCI_reconstructed_ccf, aperture_image00)
    PlanetHCI_crosscorr_dt = pd.DataFrame(PlanetHCIccf_nanrm_mask)
    PlanetHCI_crosscorr_dt.to_csv(savedirccf + filename + '_crosscorr_dt.csv')

    # Data set: Here we add a column to the data indicating if we have the presence of the planet or not. 
    PlanetHCI_nanrm_spectr, PlanetHCI_shape_spectr, PlanetHCI_nan_spectr, PlanetHCI_nanrm_mask_spectr = image_deconstruct(
        Planet_HCI, aperture_image00)
    PlanetHCI_spectrum_dt = pd.DataFrame(PlanetHCI_nanrm_mask_spectr)
    PlanetHCI_spectrum_dt = PlanetHCI_spectrum_dt.set_axis(np.append("Planet", Planet_WR), axis=1, inplace=False)
    PlanetHCI_spectrum_dt.to_csv(savedirdata + '/' + filename + '_spectrum_dt.csv')

    # Plot it out
    figure, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].imshow(Planet_HCI[1, :, :])
    axes[0, 0].set_title("Original img.")
    axes[0, 0].set_xlabel("[px]")
    axes[0, 0].set_ylabel("[px]")
    axes[0, 0].label_outer()
    axes[0, 1].imshow(Planet_HCI_reconstructed[1, :, :])
    axes[0, 1].set_title("Reconstructed img.")
    axes[0, 1].set_xlabel("[px]")
    axes[0, 1].set_ylabel("[px]")
    axes[0, 1].label_outer()
    axes[1, 0].imshow(find_planetHCI)
    axes[1, 0].set_title("Maximum CCF")
    axes[1, 0].set_xlabel("[px]")
    axes[1, 0].set_ylabel("[px]")
    axes[1, 0].label_outer()
    axes[1, 1].imshow(aperture_data00)
    axes[1, 1].set_title("companion location")
    axes[1, 1].set_xlabel("[px]")
    axes[1, 1].set_ylabel("[px]")
    axes[1, 1].label_outer()   
    plt.suptitle("View of " + filename)
    plt.tight_layout()
    plt.savefig(dirplot + filename + 'aperture.png')
    plt.clf()

    return



# =============================================================================
# spot (strongest ccf) planet in the data cubes
#
# !!! This function can only be used for plotting, it is not useable for saving data with planet positions. for this task, see function above. 
# For code cleaning, should separate the data task and plotting task and call these functions separately. 
# =============================================================================


def spot_planet_incube2(planetFilename, WRFilename, WR_extension, aperturesize, template_characteristics, rv=0, RV=True,
                        MAX=False):
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

        # Find the maximum ccf location and draw a Circular aperture of chosen radius around it
        find_planetHCI_2 = np.argwhere(find_planetHCI == np.nanmax(find_planetHCI))
        positions = [int(find_planetHCI_2.flatten()[[1]]), int(find_planetHCI_2.flatten()[[0]])]
        aperture00 = CircularAperture(positions, r=aperturesize)
        imshape = ((int(PlanetHCI_reconstructed_ccf.shape[1]), int(PlanetHCI_reconstructed_ccf.shape[2])))
        aperture_masks00 = aperture00.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=imshape)
        aperture_data00 = aperture_image00 * find_planetHCI

        # Plot it out
        figure, axes = plt.subplots(nrows=2, ncols=2)
        axes[0, 0].imshow(Planet_HCI[1, :, :])
        axes[0, 0].set_title("Original image")
        axes[0, 0].label_outer()
        axes[0, 1].imshow(Planet_HCI_reconstructed[1, :, :])
        axes[0, 1].set_title("Reconstructed image frame")
        axes[0, 1].label_outer()
        axes[1, 0].imshow(find_planetHCI)
        axes[1, 0].set_title("Maximum CCF")
        axes[1, 0].label_outer()
        axes[1, 1].imshow(aperture_data00)
        axes[1, 1].set_title("Aperture")
        axes[1, 1].label_outer()
        plt.suptitle("View of " + filename)
        plt.savefig(dirplot + filename + 'aperture.png')

    if RV == True:
        find_planetHCI_RV = PlanetHCI_reconstructed_ccf[int(rv + 1999), :, :]

        find_planetHCI_RV2 = np.argwhere(find_planetHCI_RV == np.nanmax(find_planetHCI_RV))
        positions = [int(find_planetHCI_RV2.flatten()[[1]]), int(find_planetHCI_RV2.flatten()[[0]])]
        aperture00 = CircularAperture(positions, r=aperturesize)
        imshape = ((int(PlanetHCI_reconstructed_ccf.shape[1]), int(PlanetHCI_reconstructed_ccf.shape[2])))
        aperture_masks00 = aperture00.to_mask(method='center')
        aperture_image00 = aperture_masks00.to_image(shape=imshape)
        aperture_data00 = aperture_image00 * find_planetHCI_RV

        figure, axes = plt.subplots(nrows=2, ncols=2)
        axes[0, 0].imshow(Planet_HCI[1, :, :])
        axes[0, 0].set_title("Original image")
        axes[0, 0].label_outer()
        axes[0, 1].imshow(Planet_HCI_reconstructed[1, :, :])
        axes[0, 1].set_title("Reconstructed image frame")
        axes[0, 1].label_outer()
        axes[1, 0].imshow(find_planetHCI_RV)
        axes[1, 0].set_title("Maximum CCF at RV=" + str(int(rv)))
        axes[1, 0].label_outer()
        axes[1, 1].imshow(aperture_data00)
        axes[1, 1].set_title("Aperture")
        axes[1, 1].label_outer()
        plt.suptitle("View of " + filename)
        
        plt.savefig(dirplot2 + filename + 'aperture.png')

    return


# =============================================================================
# trim the data's edges in wavelength for easy stack up
# =============================================================================


def trim_data_WR(ls_data, dir_path_WR, savedir_spec_true, include_synthetic=False):
    ls_planetFilename = []
    for i in range(0, len(ls_data)):
        planetFilename = ls_data[i]
        if planetFilename == 'synthetic':
            ls_planetFilename.append(ls_data[i][3:])
        else:
            ls_planetFilename.append(ls_data[i][3:][:-4])

    # Open all wagelength ranges
    keys = ls_planetFilename
    ls_WR = {key: None for key in keys}
    list_sets = []

    for i in range(0, len(ls_data)):
        planetFilename = ls_planetFilename[i]
        if planetFilename == 'synthetic':
            temp = importWavelength_asList(data_path + "wavelength_ranges/WR_" + planetFilename, extension='pickle')
            ls_WR[planetFilename] = temp
        else:
            temp = importWavelength_asList(data_path + "wavelength_ranges/WR_" + planetFilename)
            ls_WR[planetFilename] = temp
        # list of sets of ranges to be intersected  
        list_sets.append(set(temp))

    ls_int0 = reduce(set.intersection, list_sets)
    ls_int = list(ls_int0)
    ls_int.sort()

    dir_path_dt = data_path + "csv_inputs/True_Spectrum_Data/"
    ls_data_dt = os.listdir(dir_path_dt)

    ls_data_dt2 = []
    for j in range(0, len(ls_planetFilename)):
        ls_data_dt3 = []
        for i in range(0, len(ls_data_dt)):
            if ls_planetFilename[j] == ls_data_dt[i][0:(len(ls_planetFilename[j]))]:
                ls_data_dt3.append(ls_data_dt[i])
        ls_data_dt2.append(ls_data_dt3)

    # =============================================================================
    # if include_synthetic==False:
    #     #ls_data_dt.remove('synthetic_spectrum_df.csv')
    #     ls_planetFilename.remove('synthetic')
    # elif include_synthetic==True:
    #     ls_data_dt.append('Planet_Signals_df.csv')
    # =============================================================================

    # import the data, trim it to the proper wavelength and remove the planet=1 rows to get only noise left
    for j in range(0, len(ls_data_dt2)):

        for i in range(0, len(ls_data_dt2[j])):

            if ls_planetFilename[j] == 'synthetic':
                data = pd.read_csv(data_path + "csv_inputs/Planet_Signals_df.csv", index_col=0)
            else:
                data = pd.read_csv(dir_path_dt + ls_data_dt2[j][i], index_col=0)

            ls_data_range = (
            ls_WR[ls_planetFilename[j]].index(min(ls_int)), ls_WR[ls_planetFilename[j]].index(max(ls_int)))
            fin_range_data = ls_WR[ls_planetFilename[j]][ls_data_range[0]:ls_data_range[1] + 1]

            # Check the data's range is the same as the intersection to avoid any data set mismatch 
            if ls_int != fin_range_data:
                print("Warning: " + ls_planetFilename[
                    j] + "does not have the same range as the intersection, please check manually")

            if ls_planetFilename[j] == 'synthetic':
                data0 = data.iloc[:, int(np.where(data.columns == str(min(ls_int)))[0]):int(
                    np.where(data.columns == str(max(ls_int)))[0] + 1)]
                TempCol = data.columns.get_loc("tempP")
                data1 = data[data.columns[TempCol:]]
                data0.join(data1)
                data0.to_csv(data_path + "csv_inputs/Synthetic_Spectrum_Data/synthetic_spectrum_trim")


            else:
                data0 = data[data['Planet'] == 0].drop('Planet', axis=1)
                data00 = data0.iloc[:, int(np.where(data0.columns == str(min(ls_int)))[0]):int(
                    np.where(data0.columns == str(max(ls_int)))[0] + 1)]
                #data00.to_csv(savedir_spec_true + ls_data_dt2[j][i][:-16] + '_Spectrum_noise_trim.csv')
                data00.to_pickle(savedir_spec_true + ls_data_dt2[j][i][:-16] + '_Spectrum_noise_trim.pkl')

    return


# =============================================================================
# Padd the data's edges in wavelength for easy stack up
# =============================================================================


def padd_data_WR(ls_data, savedir_spec_true):
    # All planet names in the folder
    ls_planetFilename = []
    for i in range(0, len(ls_data)):
        planetFilename = ls_data[i]
        if planetFilename == 'WR_synthetic':
            ls_planetFilename.append(ls_data[i][3:])
        else:
            ls_planetFilename.append(ls_data[i][3:][:-4])

    ls_planetFilename.append("synthetic")
    ls_data.append("WR_synthetic")

    # Open all wagelength ranges
    keys = ls_planetFilename
    ls_WR = {key: None for key in keys}
    list_sets = []

    for i in range(0, len(ls_data)):
        planetFilename = ls_planetFilename[i]
        if planetFilename == 'synthetic':
            temp = importWavelength_asList(data_path + "wavelength_ranges/WR_" + planetFilename, extension='pickle')
            ls_WR[planetFilename] = temp
        else:
            temp = importWavelength_asList(data_path + "wavelength_ranges/WR_" + planetFilename)
            ls_WR[planetFilename] = temp
        # list of sets of ranges to be intersected  
        list_sets.append(set(temp))

    # ls_int0=list(set(list_sets))
    # unlist: 
    ls_int0 = list(chain.from_iterable(list_sets))
    # find superset (remove duplicates)
    ls_int1 = list(set(ls_int0))
    ls_int = list(ls_int1)
    ls_int.sort()

    dir_path_dt = data_path + "csv_inputs/True_Spectrum_Data/"
    ls_data_dt = os.listdir(dir_path_dt)

    # remove the extra synthetic data range
    ls_planetFilename.remove('synthetic')

    ls_data_dt2 = []
    for j in range(0, len(ls_planetFilename)):
        ls_data_dt3 = []
        for i in range(0, len(ls_data_dt)):
            if ls_planetFilename[j] == ls_data_dt[i][0:(len(ls_planetFilename[j]))]:
                ls_data_dt3.append(ls_data_dt[i])
        ls_data_dt2.append(ls_data_dt3)

    # import the data, padd it to the proper wavelength and remove the planet=1 rows to get only noise left

    for j in range(0, len(ls_data_dt2)):
        for i in range(0, len(ls_data_dt2[j])):
            data0 = pd.read_csv(dir_path_dt + ls_data_dt2[j][i], index_col=0)
            data = data0[data0['Planet'] == 0].drop('Planet', axis=1)

            ls_int = np.array(ls_int)

            # Create the padding with proper column names on each side of the data
            left_range_minofmin = int(np.argwhere(ls_int == min(ls_int)))
            left_range_maxofmin = (int(np.argwhere(ls_int == min(ls_WR[ls_planetFilename[j]]))) - 1)
            left_range = ls_int[left_range_minofmin:left_range_maxofmin]
            left_padding = pd.DataFrame(np.zeros((data.shape[0], len(left_range))))
            left_padding = left_padding.set_axis(left_range, axis=1, inplace=False)

            right_range_minofmax = (int(np.argwhere(ls_int == max(ls_WR[ls_planetFilename[j]]))) + 1)
            right_range_maxofmax = int(np.argwhere(ls_int == max(ls_int)))
            right_range = ls_int[right_range_minofmax:right_range_maxofmax]
            right_padding = pd.DataFrame(np.zeros((data.shape[0], len(right_range))))
            right_padding = right_padding.set_axis(right_range, axis=1, inplace=False)

            # join the paddings on each side of the data
            dt1 = left_padding.join(data)
            data00 = dt1.join(right_padding)
            #data00.to_csv(savedir_spec_true + ls_data_dt2[j][i][:-16] + "_Spectrum_noise_padded.csv")
            data00.to_pickle(savedir_spec_true + ls_data_dt2[j][i][:-16] + "_Spectrum_noise_padded.pkl")

    return


# =============================================================================
# Combine data cubes into a reshaped mean of cubes
# =============================================================================


def combine_cubes(ls_planetFilename, directory_planet_folders, save_path):
    for j in range(0, len(ls_planetFilename)):

        path_planet = directory_planet_folders + '/' + ls_planetFilename[j]

        ls_planet = os.listdir(path_planet)
        ls_cubes = []
        ls_spectrum = []
        for i in range(0, len(ls_planet)):
            if ls_planet[i][:15] == 'Centering_cubes':
                ls_cubes.append(ls_planet[i][:-4])
            elif ls_planet[i][:16] == 'spectrum_PSF_sub':
                ls_spectrum.append(ls_planet[i])
            elif ls_planet[i][:16] == 'wavelength_range':
                WR = ls_planet[i][:-4]
            else:
                pass

        if len(ls_cubes) != len(ls_spectrum):
            Warning("The number of cubes does not match the number of fits images")

        hdu_list0 = fits.open(path_planet + '/' + ls_spectrum[0])
        hdu_list0.info()
        Planet_HCI = hdu_list0[0].data
        hdu_list0.close()
        shifted_img = np.empty((len(ls_cubes), Planet_HCI.shape[0], Planet_HCI.shape[1], Planet_HCI.shape[2]))
        shifted_img[:] = np.nan
        im_centered = np.empty((Planet_HCI.shape[0], Planet_HCI.shape[1], Planet_HCI.shape[2]))
        im_centered[:] = np.nan

        for i in range(0, len(ls_cubes)):
            hdu_list0 = fits.open(path_planet + "/" + ls_spectrum[i])
            hdu_list0.info()
            Planet_HCI = hdu_list0[0].data
            hdu_list0.close()
            Planet_HCI = Planet_HCI[:, ::-1, :]  # To get the north up, as python opens fits upside down
            # Planet_WR=importWavelength_asList(path_planet +"/" + WR)
            Planet_shift = importWavelength_asList(path_planet + "/" + ls_cubes[0])

            shifted_img[i, :, :, :] = sp.ndimage.shift(Planet_HCI, [0, Planet_shift[0], Planet_shift[2]])

        im_centered[0].data = np.mean(shifted_img, axis=0)

        # Write the new HDU structure to outfile
        fits.writeto(save_path + "0res_" + ls_planetFilename[j] + ".fits", im_centered, overwrite=True)

    return


# =============================================================================
# split a data frame according to the number of frames in time. 
# - used for spectrum_dataset_4ml_norep, in order to split and distribute the signals
# =============================================================================

def split_dataframe(df, length_levels):
        dfs = {}
        chunk = math.floor(df.shape[0]/(length_levels))

        for n in range(length_levels):
            df_temp = df.iloc[n*chunk:(n+1)*chunk]
            df_temp = df_temp.reset_index(drop=True)
            dfs[n] = df_temp

        return dfs

# =============================================================================
# Transform trimmed or padded noise data into a spectrum with injected planets
# =============================================================================
#
#
# def spectrum_dataset_4ML(data0, planet_signals0, alpha=10, mol='H2O'):
#
#     save_double_index=data0.index
#
#     # Join a column with the searched molecule
#     zerodt = pd.DataFrame({mol: np.zeros((data0.shape[0]))})
#     zerodt.index = save_double_index
#     data1 = data0.join(zerodt)
#     planet_signals0_trim = planet_signals0[(data1.columns)]
#
#      # Separate between a set with the molecule and the other without
#     data_mol = planet_signals0_trim[planet_signals0_trim[mol] == 1]
#     data_mol = data_mol.drop(mol, axis=1)
#     data_nomol = planet_signals0_trim[planet_signals0_trim[mol] == 0]
#     data_nomol = data_nomol.drop(mol, axis=1)
#
#     # Shuffle the data to decorrelate the spatial positions
#     savedcolumns = data1.columns
#     data2=data1.drop(mol, axis=1)
#
#     # insert data with no H2O
#     data3 = data2.sample(data1.shape[0], replace=False)
#     if data3.shape[0]>data_nomol.shape[0]:
#         data3_temp=data3.sample(data_nomol.shape[0], replace=False) # decorrelate the spatial pixels but save the positions
#         indexer=data3_temp.index
#         datanomol = np.array(data3_temp) + alpha * np.array(data_nomol)
#         datanomol = pd.DataFrame(datanomol)
#         datanomol.index=indexer
#     elif data3.shape[0]<=data_nomol.shape[0]:
#         data3_temp=data3
#         indexer=data3_temp.index
#         datanomol = np.array(data3_temp) + alpha * np.array(data_nomol.sample(data3.shape[0], replace=False))
#         datanomol = pd.DataFrame(datanomol)
#         datanomol.index=indexer
#     zerodt = pd.DataFrame({mol: np.zeros((datanomol.shape[0]))})
#     zerodt.index = indexer
#     datanomol = datanomol.join(zerodt)
#     datanomol = datanomol.set_axis(savedcolumns, axis=1, inplace=False)
#
#     # insert data with H2O
#     data4 = data2.sample(data1.shape[0], replace=False)
#     if data4.shape[0]>data_mol.shape[0]:
#         data4_temp=data4 # decorrelate the spatial pixels but save the positions
#         indexer=data4_temp.index
#         datamol = np.array(data4_temp) + alpha * np.array(data_mol.sample(data4.shape[0], replace=True))
#         datamol = pd.DataFrame(datamol)
#         datamol.index=indexer
#     elif data4.shape[0]<=data_nomol.shape[0]:
#         data4_temp=data4.sample(data_mol.shape[0], replace=True) # decorrelate the spatial pixels but save the positions
#         indexer=data4_temp.index
#         datamol = np.array(data4_temp) + alpha * np.array(data_mol)
#         datamol = pd.DataFrame(datamol)
#         datamol.index=indexer
#     onedt = pd.DataFrame({mol: np.zeros((datamol.shape[0]))})+1
#     onedt.index = indexer
#     datamol = datamol.join(onedt)
#     datamol = datamol.set_axis(savedcolumns, axis=1, inplace=False)
#
#     # concatenate all 3 data sets and shuffle them for a shuffled train/test separation among h2o and no h2o
#     data5 = pd.concat([data1, datanomol, datamol])
#
#     indexer = data5.index
#
#     zerodt_data1 = pd.DataFrame({'subclass': np.zeros((data1.shape[0]))})
#     zerodt_datanomol = pd.DataFrame({'subclass': np.zeros((datanomol.shape[0]))})
#     zerodt_datamol = pd.DataFrame({'subclass': np.zeros((datamol.shape[0]))})
#
#     zerodt_data1[:]='pureNoise'
#     zerodt_datanomol[:]='molNoise'
#     zerodt_datamol[:]=mol
#
#     zerodt_data5=pd.concat([zerodt_data1, zerodt_datanomol, zerodt_datamol])
#     zerodt_data5.index = indexer
#
#     data6 = pd.concat([data5, zerodt_data5], axis=1)
#     data6['subclass']=pd.Categorical(data6['subclass'])
#
#     data = data6.sample(data5.shape[0], replace=False)
#
#     return data


# =============================================================================
# Transform trimmed or padded noise data into a spectrum with injected planets
# Here we make use of the noise and signals only once (no sampling with replacement)
# =============================================================================

# can also take the chance to make one data set with separated temperatures? 
def spectrum_dataset_4ML_norep(data0, planet_signals0, alpha=10, mol='H2O', balancing=50, seed=100):
    
    import random
    random.seed(seed)


    save_double_index=data0.index
    save_double_index.levels[0]
    length_levels=len(save_double_index.levels[0])
    
    # Join a column with the searched molecule
    #zerodt = pd.DataFrame({mol: np.zeros((data0.shape[0]))})
    #zerodt.index = save_double_index
    #data1 = data0.join(zerodt)
        
    planet_signals0_trim = pd.concat([planet_signals0[list(data0.columns)], planet_signals0[['tempP', 'loggP','CO_ratio', 'Fe', 'H2O','CO', 'CH4', 'NH3']]], axis=1)
    
     # Separate between a set with the molecule and the other without - shuffling and random picking is to be made here
    data_mol = planet_signals0_trim[planet_signals0_trim[mol] == 1]
    data_mol = data_mol.sample(data_mol.shape[0], replace=False)       
    data_mol_split=split_dataframe(data_mol, length_levels)
    
    data_nomol = planet_signals0_trim[planet_signals0_trim[mol] == 0]
    data_nomol = data_nomol.sample(data_nomol.shape[0], replace=False)
    data_nomol_split=split_dataframe(data_nomol, length_levels)

     
    # Shuffle the data to decorrelate the spatial positions
    data=pd.DataFrame([])
    
    
    for ind in range(0, int(length_levels)):
    
        data_temp0=data0.loc[(save_double_index.levels[0][ind], slice(None)), :]
        
        #Decorrelate spatial positions
        data_temp=data_temp0.sample(data_temp0.shape[0], replace=False)
        ind_save=data_temp.index
        
        # Randomly pick molecular signals
        
        if balancing==50:
            MOL_to_input0=data_mol_split[ind].sample(round(data_temp.shape[0]/2), replace=False)
            noMOL_to_input0=data_nomol_split[ind].sample(round(data_temp.shape[0]/4), replace=False)
        elif balancing==30:
            MOL_to_input0=data_mol_split[ind].sample(round(data_temp.shape[0]/3.33333), replace=False)
            noMOL_to_input0=data_nomol_split[ind].sample(round(data_temp.shape[0]/3.33333), replace=False)
        elif balancing==20:
            MOL_to_input0=data_mol_split[ind].sample(round(data_temp.shape[0]/5), replace=False)
            noMOL_to_input0=data_nomol_split[ind].sample(round(data_temp.shape[0]/5), replace=False)
            
        
        # prepare the datasets to scale - dataset of same shape and size of the noise
        MOL_to_input=MOL_to_input0.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O','CO', 'CH4', 'NH3'], axis=1)
        noMOL_to_input=noMOL_to_input0.drop(['tempP', 'loggP','CO_ratio', 'Fe', 'H2O','CO', 'CH4', 'NH3'], axis=1)
        PureNoise_to_input=pd.DataFrame(np.zeros((data_temp.shape[0]-(MOL_to_input.shape[0]+noMOL_to_input.shape[0]), noMOL_to_input.shape[1])))
        PureNoise_to_input.columns=noMOL_to_input.columns
        dataset_to_input=pd.concat([MOL_to_input,noMOL_to_input,PureNoise_to_input], axis=0)
        dataset_to_input.index=ind_save  # we index it as the noise

        #store info to be used after the scaling
        MOL_keepinfo=MOL_to_input0[['tempP', 'loggP','CO_ratio', 'Fe', 'H2O','CO', 'CH4', 'NH3']]
        zerodt_data1 = pd.DataFrame({'subclass': np.zeros((MOL_keepinfo.shape[0]))})
        zerodt_data1[:]='molSignal'     # Allows to keep track of how the data was splitted and if molecules really correspond
        zerodt_data1.index = MOL_keepinfo.index
        MOL_keepinfo_dt=pd.concat([MOL_keepinfo,zerodt_data1], axis=1)  
        
        noMOL_keepinfo=noMOL_to_input0[['tempP', 'loggP', 'CO_ratio', 'Fe', 'H2O','CO', 'CH4', 'NH3']]
        zerodt_data2 = pd.DataFrame({'subclass': np.zeros((noMOL_keepinfo.shape[0]))})
        zerodt_data2[:]='molNoise'
        zerodt_data2.index = noMOL_keepinfo.index
        noMOL_keepinfo_dt=pd.concat([noMOL_keepinfo,zerodt_data2], axis=1)  

        
        PureNoise_keepinfo=pd.DataFrame(np.zeros((data_temp.shape[0]-(MOL_keepinfo.shape[0]+noMOL_keepinfo.shape[0]), noMOL_keepinfo.shape[1])))
        PureNoise_keepinfo.columns = noMOL_keepinfo.columns
        PureNoise_keepinfo[['tempP', 'loggP','CO_ratio','Fe']]=np.nan
        zerodt_data3 = pd.DataFrame({'subclass': np.zeros((PureNoise_keepinfo.shape[0]))})
        zerodt_data3[:]='pureNoise'
        zerodt_data3.index = PureNoise_keepinfo.index
        PureNoise_keepinfo_dt=pd.concat([PureNoise_keepinfo,zerodt_data3], axis=1)  

        
        PureNoise_keepinfo_dt.columns=noMOL_keepinfo_dt.columns 
        
        # Create the data set of signals
        dataset_keepinfo=pd.concat([MOL_keepinfo_dt, noMOL_keepinfo_dt, PureNoise_keepinfo_dt], axis=0)
        dataset_keepinfo.index=ind_save # Index it as noise data
        
        data_temp0 = data_temp + alpha * dataset_to_input
        data_temp1 = pd.concat([data_temp0, dataset_keepinfo], axis=1)       # bind the information bacl into the data
        data_temp2=data_temp1.sample(data_temp1.shape[0], replace=False)  # 
        
        data=data.append(data_temp2)
    
    return data



# =============================================================================
# Function to transform the spectrums 4ml  into cross-correlated templates.
# =============================================================================
# The Spectrums come with a given scale, molecule and planet to be inputted. 
# The option of normalizing the cross correlation is set to True as a default 
# but can be changed. 
# a = the scale at which the signal is inputted into the noise
# teff = the temperature of the template
# sg = the surface gravity of the template
# mol = the molecule of interest for the template
# planet = the planet's name as in the spectrum 4ml data
# The function breaks up atismd and distributes it within 
# =============================================================================

def signal_to_ccf_4ml(a, teff, sg, mol, planet, data_path=data_path, normalized0=True):

    data = pd.read_pickle(data_path+"data_4ml/spectrums_4ml_trim/"+str(mol)+"_"+str(planet)+"_"+str(a)+"_spectrums_4ml_trim.pkl")
    indexer = data.index

    if mol=='H2O':
        template_characteristics = {'Temp': teff, 'Surf_grav': sg, 'H2O': 1, 'CO': 0}
    elif mol=='CO':
        template_characteristics = {'Temp': teff, 'Surf_grav': sg, 'H2O': 0, 'CO': 1}

    templates = pd.read_csv(data_path+"csv_inputs/Molecular_Templates_df.csv", index_col=0)
    template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]
    del templates

    TempCol = template.columns.get_loc("tempP")
    tf = template.drop(template.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()

    df = data.drop([mol,'subclass'], axis=1)
    dw = pd.to_numeric(df.columns)
    y = data[[mol, 'subclass']]
    df = np.array(df)

    del data

    rv1, cc1 = crosscorrRV_vec(dw, df, tw, tf, -2000, 2000, 1, mode="doppler", normalized=normalized0, skipedge=10, edgeTapering=None)

    del df
    ccf_dt1 = pd.DataFrame(cc1)
    ccf_dt1.index = indexer
    ccf_dt1 = ccf_dt1.set_axis(list(rv1), axis=1, inplace=False)

    del cc1
    y = pd.DataFrame(y)
    y.index = indexer
    ccf_dt01 = pd.concat([ccf_dt1, y], axis=1)

    ccf_dt01.to_pickle(data_path+"data_4ml/ccf_4ml_trim/"+str(mol)+"_"+str(planet)+"_scale"+str(a)+"_temp"+str(teff)+"_sg"+str(sg)+"_ccf_4ml_trim.pkl")

    return str(mol)+"_"+str(planet)+"_scale"+str(a)+"_temp"+str(teff)+"_sg"+str(sg)

# =============================================================================
# Function to transform the spectrums 4ml  into cross-correlated templates.
# For Parrallelized computing
# =============================================================================
# The spectrums come with a given scale, molecule and planet to be inputted. 
# The option of normalizing the cross correlation is set to True as a default 
# but can be changed. 
# a_teff_sg_mol_planet to enter as a tuple of dim 5 with alond the dimensions:
# a = the scale at which the signal is inputted into the noise
# teff = the temperature of the template
# sg = the surface gravity of the template
# mol = the molecule of interest for the template
# planet = the planet's name as in the spectrum 4ml data
# The function breaks up atismd and distributes it within 
# =============================================================================


def parallel_signal_to_ccf_4ml(a_teff_sg_mol_planet, data_path=data_path, normalized0=True):

    a = a_teff_sg_mol_planet[0] # scale at which the signal was injected
    teff = a_teff_sg_mol_planet[1] # effective temperature chosen for template
    sg = a_teff_sg_mol_planet[2] # surface gravity for template
    mol = a_teff_sg_mol_planet[3] # molecules of interest
    planet = a_teff_sg_mol_planet[4] # planet name

    data = pd.read_pickle(data_path+"data_4ml/spectrums_4ml_trim/"+str(mol)+"_"+str(planet)+"_"+str(a)+"_spectrums_4ml_trim.pkl")
    indexer = data.index


    if mol=='H2O':
        template_characteristics = {'Temp': teff, 'Surf_grav': sg, 'H2O': 1, 'CO': 0}
    elif mol=='CO':
        template_characteristics = {'Temp': teff, 'Surf_grav': sg, 'H2O': 0, 'CO': 1}

    templates = pd.read_csv(data_path+"csv_inputs/Molecular_Templates_df.csv", index_col=0)
    template = templates.loc[(templates['tempP'] == template_characteristics['Temp']) & (templates["loggP"] == template_characteristics['Surf_grav']) & (templates["H2O"] == template_characteristics['H2O']) & (templates["CO"] == template_characteristics['CO'])]
    del templates

    TempCol = template.columns.get_loc("tempP")
    tf = template.drop(template.columns[TempCol:], axis=1)
    tw = pd.to_numeric(tf.columns)
    tf = np.array(tf).flatten()

    df = data.drop([mol, 'subclass'], axis=1)
    dw = pd.to_numeric(df.columns)
    y = data[[mol, 'subclass']]
    df = np.array(df)

    del data

    rv1, cc1 = crosscorrRV_vec(dw, df, tw, tf, -2000, 2000, 1, mode="doppler", normalized=normalized0, skipedge=10, edgeTapering=None)

    del df
    ccf_dt1 = pd.DataFrame(cc1)
    ccf_dt1.index = indexer
    ccf_dt1 = ccf_dt1.set_axis(list(rv1), axis=1, inplace=False)

    del cc1
    y = pd.DataFrame(y)
    y.index = indexer
    ccf_dt01 = pd.concat([ccf_dt1, y], axis=1)

    ccf_dt01.to_pickle(data_path+"data_4ml/ccf_4ml_trim/"+str(mol)+"_"+str(planet)+"_scale"+str(a)+"_temp"+str(teff)+"_sg"+str(sg)+"_ccf_4ml_trim.pkl")

    return str(mol)+"_"+str(planet)+"_scale"+str(a)+"_temp"+str(teff)+"_sg"+str(sg)





