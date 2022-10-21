'''This programs aims to calculate the background from a CBC population.
    Following all this steps is required to launch the program in a safe way :)

    Current status :

    -
    -
'''

import os
import numpy as np


'''
    1) Choose and check your basic catalogues : 
    -------
     In the Following code the parameters are referenced as and must have these names in your initial catalogue : 
     
     Mc : Chirp mass in the source frame [Msun]
     q : mass ratio in the source frame
     m1 : mass of the first component (m1>m2) in the source frame [Msun]
     m2 : mass of the secondary component (m2<m1) in the source frame [Msun]
     Xsi : Chieff
     Chi1 : individual spin factor of the first component
     Chi2 : individual spin factor of the second component
     
     a0 : semi-major axis at the formaiton of the second compact object [Rsun]
     e0 : eccentricity at the formation of the second compact object
     
     inc : inclinaison angle [rad]
     
     zm : redshift of merger
     zf : redshift of formation
     
     flag : this colums cam contain a flag to differenciate sources
     
     and set up all the links :
    -------
    Models : np.array of str.
            Will be used to create appropriate file names for new catalogues and results.
    Paths : np.array of str.
            Same size as Models, link to the directory where are initial catalogs.
    co_object : list of str.
            List of compact objects catalogues ['BBH', 'BNS', 'BHNS', 'mix']
            If they are mixed in the catalogue, use 'mix' and steup adequate flags to differenciate them.
    OPTIONAL
    Flags : list of int or str.
            Are some flags present in your catalogue, like 1 for Orig 2 for Exch.
    Flags_str : list of str.
            Will be used to distinguish the flagged populations in file names.
    IDD : list of str.
            Must have the same lenght as Models.
            Can be used to randomly take the spins contributions chi1 and chi2.
'''

Models = ['Field', 'YSCs']
#Flags = [1, 2, 3]
#Flags_str = ['Orig', 'Exch','Iso']
co_object = ['BBH']#, 'BNS'] # 'BHNS'
path_2_cat_folder =  '/home/perigois/Documents/GreatWizardOz/'
IDD = ['iso','dyn']
Flags = {'1': 'Orig', '2':'Exch', '3':'Iso'}
Cat_list = dict({})
for co in range(len(co_object)) :
    for m in range(len(Models)) :
        name = co_object[co]+'_'+Models[m]
        #Cat_list[name] = [path_2_cat_folder+'cat_'+co_object[co]+'_'+IDD[m]+'.dat', co_object[co] , Models[m], IDD[m]]
        Cat_list[name] = [path_2_cat_folder + 'Cat_stochastic_' + IDD[m] +'_'+ co_object[co] +   's.dat', co_object[co], Models[m],
                          IDD[m]]


'''
    2) Context of the study, choose the Network and the frequency range you want :
    -------list(newdict.keys())
        SNR_thrs : 
            Describe all Networks currently available in the code and their detectability threshold.
            HLV : Hanford(H), Livingson(L), Virgo(V) at Advanced design sensitivity.
            HLVIK : HLV + Indigo(I) + Kagra(K) at design sensitivity.
            ET : Einstein Telescope(ET) at design sensitivity #Check/Add configuration.
            ET2CE : ET + two Cosmic Explorer(CE) located and oriented as Hanford and Livingston respectively.
            LISA : At design sensitivity.
        Networks : list of str. 
            The programm will evaluate the residual background assuming a perfect substraction of sources, 
            and its detectability for each Network of this list.
            Currently available Network are ['HLV','HLVIK','ET','ET+2CE','LISA'].
        Freq : np.array of floats.
            Frenquency range for the calculation, for cost reason len(Freq)<3000.
         
        Net_Compo : dict
            Composition of networks in Networks
        Detectors : dict
            Information about individual detectors [Name, Source, Lenght, Deltaf, fcut_min]
            Name : (str) Name used in pycbc or path to file
            Source : 'PyCBC' or 'file' : where to extract the PSD.
            Lenght : lenght of the PSD
            Deltaf : f bin size for the PSD
            fcut_min : starting frequency for the PSD
            For more information on the PSD from PyCBC : https://pycbc.org/pycbc/latest/html/pycbc.psd.html
        ORF_path : path where to find the Gamma (Overlap redustion functions) for the study. 
            The different available files are detailed in AuxiliaryFiles/ORFs/ORFs.info.
            AuxiliaryFiles/ORFs/ORFs.py : generate overlap reduction functions for a specific set of detectors.
            AuxiliaryFiles/ORFs/Generate_Gamma_Files.py : Code to generate the appropriate file used by the code.
            
'''

Networks = ['ET', 'ET2CE']
SNR_thrs = {'HLV' : 12., 'HLVIK' : 12., 'ET' : 12., '3ET2CE' : 12.,'2ET2CE' : 12., 'LISA' :5.}
Net_compo = {'HLV' : ['H','L','V'],
             'HLVIK' : ['H','L','V','I','K'],
             #'ET' : ['ET','ET'], # '' is needed for combinations, arrays must have a size>1
             #'ET2CE' : ['ET','CE','CE'],
             'ET' : ['E1','E2','E3'],
             '3ET2CE' : ['E1','E2','E3','CE1','CE2'],
             '2ET2CE' : ['CE1','CE2','EL1','EL2'],
             'LISA' :['LISA','']}

#Composition for 3G Configuration ET : triangle with arm lenght 10km
Detectors = {'H' : ['PyCBC','aLIGODesignSensitivityP1200087',2490, 1, 10],
             'L' : ['PyCBC','aLIGODesignSensitivityP1200087',2490, 1, 10],
             'V' : ['PyCBC','AdVDesignSensitivityP1200087',2490, 1, 10],
             'I' : ['PyCBC','aLIGODesignSensitivityP1200087',2490, 1, 10],
             'K': ['PyCBC','KAGRADesignSensitivityT1600593', 2490, 1, 10],
             'ET': ['PyCBC','EinsteinTelescopeP1600143', 2490, 1, 1],
             'E1': ['Coba','EinsteinTelescopeP1600143', 2490, 1, 1],
             'E2': ['Coba','EinsteinTelescopeP1600143', 2490, 1, 1],
             'E3': ['Coba','EinsteinTelescopeP1600143', 2490, 1, 1],
             'CE': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
             'CE1': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
             'CE2': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
             'EL1': ['Coba','EinsteinTelescopeP1600143', 2490, 1, 1],
             'EL2': ['Coba','EinsteinTelescopeP1600143', 2490, 1, 1]
             #'LISA' : ['file','AuxiliaryFiles/PSDs/LISA_Des/', 2490, 1, 1]
             }
#Composition for Desined sensitivities
# Detectors = {'H' : ['PyCBC','aLIGODesignSensitivityP1200087',2490, 1, 10],
#              'L' : ['PyCBC','aLIGODesignSensitivityP1200087',2490, 1, 10],
#              'V' : ['PyCBC','AdVDesignSensitivityP1200087',2490, 1, 10],
#              'I' : ['PyCBC','aLIGODesignSensitivityP1200087',2490, 1, 10],
#              'K': ['PyCBC','KAGRADesignSensitivityT1600593', 2490, 1, 10],
#              'ET': ['PyCBC','EinsteinTelescopeP1600143', 2490, 1, 1],
#              'E1': ['PyCBC','EinsteinTelescopeP1600143', 2490, 1, 1],
#              'E2': ['PyCBC','EinsteinTelescopeP1600143', 2490, 1, 1],
#              'E3': ['PyCBC','EinsteinTelescopeP1600143', 2490, 1, 1],
#              'CE': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
#              'CE1': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
#              'CE2': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
#              'EL1': ['Coba','EinsteinTelescopeP1600143', 2490, 1, 1],
#              'EL2': ['Coba','EinsteinTelescopeP1600143', 2490, 1, 1]
#              #'LISA' : ['file','AuxiliaryFiles/PSDs/LISA_Des/', 2490, 1, 1]
#              }

#ORF_path =



#Freq = np.concatenate([np.logspace(-6,0,50),np.linspace(2,2500,2499)])
Freq = np.linspace(1,2500,2499)


'''
    3) Setup the ana output you want :
    in addition to the files containing the energydensity spectrum this code will write a file with useful stats : 
    specific values of omega, number of non-resolved sources...
    -------
    ANA_Values : list of str
        Names of the values you need
    Ref_values : list of floats
        Reference frequencies for the energy density. 
        Ususal values are: 0.001 for LISA, 1Hz and 10 Hz for 3G and 25Hz for 2G. 
        These values have to belong to Freq!
        
'''

Omega_ana_freq = [1.,10.,25.]