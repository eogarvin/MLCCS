
#sys.path.append(code_path + "ml_spectroscopy/ml_spectroscopy")
#sys.path.append("C:/Users/emily/Documents/ML_spectroscopy_thesis/50_code/ml_spectroscopy")
import pylab as pl
from ml_spectroscopy.config import path_init
import numpy as np
import random


## This class takes the planet signals data set and reshapes them into the noise dataset shape (in terms of row numbers)
## The goal of this grid is to easily inject planets into the images at later stage, using the mask location properties to mask out all planets we don't need
## The class function also cares for the number of cubes; as each cube receives a different planet (with slight variations within the spectra) the class and makes sure that each cube gets a different planet subset, to avoid any leaking of the training dataset into the test data set.




class CreatePlanetsGrid:

    def __init__(self,  signals, noise, num_processes=32):

        self.signals = signals
        self.noise = noise
        self.m_num_processes = num_processes



    def grid_subset(self, template: dict, planet_name: str, ncubes: int, cube: int):
        seed = cube # use the cube number to set the seed
        random.seed(seed)

        # desired data size
        nrows = len(self.noise.loc[planet_name])  # ncols = len(self.dt_signals.columns)

        # template values
        temperatures = template['Temp']
        surface_gravities = template['Surf_grav']
        water = template['H2O']
        carbmon = template['CO']
        ammonia = template['NH3']
        methane = template['CH4']

        # get indices of spectra which match the individual parameter ranges
        indices_t = np.where(np.in1d(self.signals['tempP'], temperatures))[0]
        indices_sg = np.where(np.in1d(self.signals['loggP'], surface_gravities))[0]
        indices_w = np.where(np.in1d(self.signals['H2O'], water))[0]
        indices_c = np.where(np.in1d(self.signals['CO'], carbmon))[0]
        indices_a = np.where(np.in1d(self.signals['NH3'], ammonia))[0]
        indices_m = np.where(np.in1d(self.signals['CH4'], methane))[0]


        # intersection of all allowed indiv. parameter values to get the set of allowed spectra
        intersection = list(
            set(indices_t) & set(indices_sg) & set(indices_w) & set(indices_c) & set(indices_a) & set(indices_m))
        # by splitting the intersection list by the number of cubes, we ensure that the data is well separated in terms of training and validation set, and that the same data is not seen twice.
        chunks = np.array_split(intersection, ncubes)

        local_cube_sample = chunks[cube]
        # sample the index to get the proper dataset size
        sample_indx = np.random.choice(local_cube_sample, size=nrows, replace=True)
        sample = self.signals.iloc[sample_indx]

        choice = np.random.choice(sample_indx, size=1, replace=False)
        unique_indx = np.repeat(choice, nrows)
        unique = self.signals.iloc[unique_indx]

        ind1 = self.noise.loc[planet_name].index
        ind0 = np.repeat(planet_name, len(ind1))
        arr = [ind0, ind1]
        sample.index = arr
        sample.columns = self.signals.columns

        unique.index = arr
        unique.columns = self.signals.columns

        return sample, unique

