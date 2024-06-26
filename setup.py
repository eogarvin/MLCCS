from setuptools import setup

setup(name='MLCCS',
      version='0.1',
      description='MLCCS is a Python codebase to apply Machine Learning on Cross-Correlation for Spectroscopy , to detect exoplanets by using molecular templates, or to search for molecules on exoplanets with extra sensitivity.',
      url='https://github.com/eogarvin/MLCCS',
      author='Emily O. Garvin',
      author_email='egarvin@phys.ethz.ch',
      license='BSD 3',
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'astropy',
            'PyAstronomy',
            'pandas',
            'sklearn',
            'seaborn',
            'tqdm',
            'photutils',
            'xgboost',
            'tensorflow',
            #'tensorflow-gpu',
            'keras',
            'scipy'],
      packages=['MLCCS'],
      zip_safe=False)
