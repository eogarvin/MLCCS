
<h1 align="center">🪐 MLCCS 🪐</h1>

This public repository shares the MLCCS codes from the following research paper: 

**E. O. Garvin et al. (in prep):** "Machine Learning for Exoplanet Detection in High-Contrast Spectroscopy: Revealing Exoplanets by Leveraging Hidden Molecular Signatures in
Cross-Correlated Spectra with Convolutional Neural Networks." (Submitted at A&A).

**MLCCS** is a Python codebase to apply _**M**achine **L**earning on **C**ross-**C**orrelation for **S**pectroscopy_ , to detect exoplanets by using molecular templates, or to search for molecules on exoplanets with extra sensitivity. 


## 💻 Installation

The code is organized as a Python package, and can be installed using `pip`.

```bash
git clone git@github.com:eogarvin/MLCCS.git
cd MLCCS
pip install .
```

## 📖 Documentation

Documentation is not yet available for this code, but the code has many comments the users can rely on. 

Necessary ingredients to start the code: 
- A grid of companion spectra with indications of structural parameters (Temperature, Surface Gravity, Molecules)
- SINFONI residual data cubes with centering information
- A grid of templates.
The code is able to work from there, to insert planets in noise and compile the train/test datasets. We are currently preparing a demo dataset to start the codes.

## 🤖 Authors and implementation of the codes

All codes have been written by Emily Garvin, with additional contributions from Markus Bonse. The codes have been successfully installed, investigated and tested by Jonas Spiller.

## 📚 Citing this code

If you use the codes or part of them for your work, we kindly request you to cite: 

E. O. Garvin et al. (in prep). "Machine Learning for Exoplanet Detection in High-Contrast Spectroscopy: Revealing Exoplanets by Leveraging Hidden Molecular Signatures in
Cross-Correlated Spectra with Convolutional Neural Networks." (Submitted at A&A).  

## 🤓 Contact and Support

Feel free to contact Emily for support with data preparation or processing, codes installation, or to get access to prepared training and test data sets. You can use the following means to get in touch:

egarvin[at]phys.ethz.ch

https://eogarvin.github.io/






## 📒 The codes we used

- We made use of petitRADTRANS to prepare the data and templates:

  https://gitlab.com/mauricemolli/petitRADTRANS
  
  https://petitradtrans.readthedocs.io

- We borrowed and adapted the cross-correlation function (crosscorrRV) codes from PyAstronomy

  https://pyastronomy.readthedocs.io

- This ReadMe template was inspired by the awesome (art!)pieces of work by T. D. Gebhard (github.com/timothygebhard)



We have a companion paper (Nath+2024), who also aims to seach for planets by leveraging cross-correlation spectroscopy, but they focus on detection in the spatial dimension. Our codes live in different spaces and are independent from eachother, but our papers are tied as companions. You can check out their work: https://github.com/digirak/NathRanga2024
