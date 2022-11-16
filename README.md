# Princess
Predicting tool for CBC GW observations

This tool aims to link astrophysical models with gravitational wave observations for compact binary coalescences.
It's separated in different package, and will be upgrade in the next months.

For an easy start you can simply follow the file Getting_started.py. It will present step by step all the set up needed, and contains the parameters you would like to set for your study. The first section of this document will be dedicated to an Introduction and a first use of the code through the _Getting_Started_ file. The next sections are detailing the different function for each internal main packages of the code.

## Introduction and basic use: Getting_Started.py

### Requierment
This program has been made and tested with Python 3.6.
This code is used with the following packages, please make sure all of them are installed with the right version.

### First use of the code: _Getting_Started_

For a first use a pre-made code(notebook) Getting_started.py(.ipy) contains a full  guide line for the calculation of the background from the preparation of the catalogues to the analysis of the obtained background. Getting_Started.py also gather all the parameter needed for the future analysis.
### Structure of the code.

**Entry Files**  
AuxiliaryFiles: Contains all the files related to the detection files 
* Overlap reduction functions(ORFs)
* Power spectral densities (PSDs)
* Power integrated curves (PICs)

**Catalogs**  
Catalogs: contains all the catalogs made by the program. The _Ana_xxx.txt_ files contains the basic statistics about the _xxx.txt_ catalog.

**Packages**  
* Princess.Starter: This package contains functions to load your astrophysical catalogue and make it usable with Princess.stochastic  
* Princess.Stochastic: Make prediction on the astrophysical gravitational wave background.  
* Princess.AuxiliaryPacks: All useful package as constants or basic functions. 

**Results**
Results files will have the same names as the catalogs it refers to. 
* /Ana: contains the files made by the analysis set up by the user from Getting_Started


## Princess.Starter

**Content**
* AstroModel.py: AstroModel class and method
* Detection.py: Contain all classes and methods related to the detection.
* Htild.py: Calculation of the frequency domain waveform

## Princess.stochastic

**Content**
* Princess.py: Tools to calculate the background of a catalogue
* SNR.py: Calculation of the SNR of a given background

## Princess.AuxiliaryPacks

**Content**
Basic_function.py: All basic function useful
Kst.py: contain the constants used in the program
Pix.py: contains the drawings appearing in the code


