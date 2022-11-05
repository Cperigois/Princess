'''This programs aims to calculate the background from a CBC population.
    Following all this steps is required to launch the program in a safe way :)

    Current status :

    -
    -
'''

import os
import numpy as np
import Starter as start
import Stochastic.Princess as SP



'''
    1) Prepare you model(s) : 
    -------
     In the Following code the parameters are referenced as and must have these names in your initial catalogue : 
     
     Mc : Chirp mass in the source frame [Msun]
     q : mass ratio in the source frame
     m1 : mass of the first component (m1>m2) in the source frame [Msun]
     m2 : mass of the secondary component (m2<m1) in the source frame [Msun]
     Xsi : Chieff effective spin
     Chi1 : individual spin factor of the first component
     Chi2 : individual spin factor of the second component
     
     a0 : semi-major axis at the formaiton of the second compact object [Rsun]
     e0 : eccentricity at the formation of the second compact object
     
     inc : inclinaison angle [rad]
     
     zm : redshift of merger
     zf : redshift of formation
     
     flag : this columns cam contain a flag to differenciate sources
     
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

path_2_cat_folder =  '/home/perigois/Documents/GreatWizardOz/'
Flags = {'1': 'Orig', '2':'Exch', '3':'Iso'}
Cat = 'Cat_stochastic_dyn_BBHs.dat'
Model = 'YSCs'
astromodel1 = start.AstroModel(cat_name= 'BBH_YSCs', path = path_2_cat_folder+Cat, sep = "\t", index_col = None)
astromodel1.makeHeader(['Mc','q','zm','zf','t_del','Z','a0','e','flag'])


# Set spin option:
#   True if you want to include spin in your calculations of the waveforms and background and you have the spin in your catalogue
#   False if you don't want to use the spin
#   Model if you want Princess to generate spin values - Option available later -
spin_option = False
IncAndPos = False
orbit_evo = False

# Prepare the spin
astromodel1.makeCat(flags = Flags)

print('Astromodel loaded and ready :)')


'''
    2) Set up the detectors:
    Define the detectors and Networks you want to use for your analysis
'''

H = Detector.Detector(det_name = 'H', Pycbc = True, psd_file = 'aLIGODesignSensitivityP1200087', freq = Freq )
L = Detector.Detector(det_name = 'L', Pycbc = True, psd_file = 'aLIGODesignSensitivityP1200087', freq = Freq )
V = Detector.Detector(det_name = 'V', Pycbc = True, psd_file = 'AdVDesignSensitivityP1200087', freq = Freq )


HLV = Detector.Network(net_name = 'HLV',compo=[H,L,V], pic_file = , efficiency = 0.5 )

Networks = ['HLV']

astromodel1.compute_SNR(Networks)


'''
    3) Calculate the corresponding background:
'''

# Frequency range for the calculation
Freq = np.linspace(5, 250, 244)

# Waveform you would like to use from Pycbc, write Ajith for analytical ones (longer computation)
WF_approx = 'IMRPhenomD'

# Threshold you choose for a source to be detectable, default is 12 \cite.
SNR_thrs = 12

for sub_cat in Flags.keys :
    name_cat = astromodel1.name+'_'+Flags[sub_cat]+'.dat'
    SP.Omega(cat = name_cat, frange = Freqs)


'''
    4) Analysis :
    in addition to the files containing the energy density spectrum this code will write a file with useful stats : 
    specific values of omega, number of non-resolved sources...
    -------
    ANA_Values : list of str
        Names of the values you need
    Ref_values : list of floats
        Reference frequencies for the energy density. 
        Ususal values are: 0.001 for LISA, 1Hz and 10 Hz for 3G and 25Hz for 2G. 
        These values have to belong to your Freq range!

'''

Omega_ana_freq = [1., 10., 25.]



'''
    4) Dectecability of the background:
    Run the SNR for the correspondig background.
'''

 SNR_Omega()



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