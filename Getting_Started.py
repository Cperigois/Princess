'''
Princess: Guide for a first calculation

This file goes step-by-step through the calculation of an astrophysical background starting from a CBC catalogue. For mode details about the code structure and basic calculations please visit the README.md file. Princess as two companioin paper which give more details on the physica behind these calculations.

This toolkit aims to be user friendly and useful to the collaboration. If you have any comments, issues or requests please contact the administrator (caroleperigois@outlook.com).

The calculation of the background goes through four main step, defining the four sections of this file.

    Prepare your model
    Prepare your detectors and Networks
    Calculate the background
    Analyse our background

'''

import os
import numpy as np
#from Starter.Detection import Detector, Network
from Stochastic import Princess as SP
from Starter.AstroModel import Astromodel
from Starter import Detection
import Useful_functions as UF

'''
1. Prepare your model:

All population synthesis codes may have different outputs. In the next steps the code will re-build catalogues insuring that it contains all the required parameters for the next steps of calculation. Your astrophysical model will be set in a class Princess.Astromodel and takes in entry several parameter.

    cat_name: (str) is the name you want to use for your model
    original_cat_path: (str) path to your original astrophysical catalogue
    cat_sep: (str) separator used in your original catalogue (default is tab)
    index_column: (bool) does your original file contain a columns with indexes (default is None)
    flags: (dict) this option allow to differenciate different types of CBCs is a Model. If you add this option you need to set up a dictionnary of the different categories. For example in the checks the original catalogue contain a column called 'flag' wher can be found identifiers 1 for isolated BBH and 2 for cluster BBH. Therefore the dictionnary looks like Flags = {'1': 'Iso', '2':'Cluster'}. In the next steps the code will build two catalogues out from the initial model (default is None).
'''

path_2_cat_folder =  '/home/perigois/Documents/GreatWizardOz/'
Flags = {'1': 'Orig', '2':'Exch', '3':'Iso'}
Cat = 'Cat_stochastic_dyn_BBHs.dat'
astromodel1 = Astromodel(cat_name= 'BBH_YSCs', original_cat_path = path_2_cat_folder+Cat, cat_sep = "\t", index_column = None, flags = Flags)


'''
    If your original catalogue do not have header you can set one using the method makeHeader on your model. In the next liste are the labels allowed by the code, please note that for the masse you will need to have or the chirp mass and the mass ratio, or the two masses m1 and m2.
     
     Mc : Chirp mass in the source frame [Msun]
     q : mass ratio in the source frame
     m1 : mass of the first component (m1>m2) in the source frame [Msun]
     m2 : mass of the secondary component (m2<m1) in the source frame [Msun]
     Dl : luminosity distance [Mpc]
     
     Xeff : Chieff effective spin
     s1 : individual spin factor of the first component
     s2 : individual spin factor of the second component
     theta1 : angle between the first spin component and the angular momentum of the binary [rad]
     theta2 : angle between the second spin component and the angular momentum of the binary [rad] 
     
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


#astromodel1.makeHeader(['Mc','q','zm','zf','t_del','Z','a0','e','flag'])

'''
Set spin option: True if you want to include spin in your calculations of the waveforms and background and you have the spin in your catalogue 'Zero' if you don't want to use the spin, the code set all spins to 0. Model if you want Princess to generate spin values - Option available later -
'''

spin_option = 'Zero'
IncAndPos = False
orbit_evo = False

'''
Finally the MakeCat method generate the catalogue with all requiered parameters
'''

#astromodel1.makeCat(flags = Flags)
print('Astromodel loaded and ready :)')


'''
    2. Detectors and Networks

In this part is detailed the context of the study starting by defining the range of frequency Freq and the waveforms WF_approx to use. In this version the range has to be linear in the future specific function will be added to allow log scale, in particular for LISA band. The available waveforms are the ones define in PyCBC and the analytic one from Ajith2011. The calculation with Ajith waveforms is computationnalymore expensive and therefore not recommended.

'''

# Frequency range for the calculation
#Freq = np.linspace(5, 250, 244)
Freq = np.arange(240)+10

# Waveform you would like to use from Pycbc, write Ajith for analytical ones (longer computation)
WF_approx = "IMRPhenomD"

'''
In Princess two classes has been build for this purpose in the file Starter/Detection in order to define Detectors and combine them to build a Network. The Detector class takes in entry several parameters:

    det_name: (str) is the name you give to the detector
    Pycbc: (bool) True if the sensitivity is available in PyCBC, else False
    psd_file: (str) name of the sensitivity in PyCBC, or file where your sensitivity is stored
    freq: (np.array) frequency range of the study

'''

H = Detection.Detector(det_name = 'H', Pycbc = True, psd_file = 'aLIGODesignSensitivityP1200087', freq = Freq )
L = Detection.Detector(det_name = 'L', Pycbc = True, psd_file = 'aLIGODesignSensitivityP1200087', freq = Freq )
V = Detection.Detector(det_name = 'V', Pycbc = True, psd_file = 'AdVDesignSensitivityP1200087', freq = Freq )


'''
The second class Network allow to combine different detectors to build a Network.A Network takes in entry:

    net_name: (str) Name of the network
    compo: (list of detectors) List of the detectors in the network.
    pic_file: (str) link to the file of the power integrated curve.
    efficiency: (float) between 0 and 1 define the effective time of observation of the Network. For example during O3a, in the Hanford-Livinstone-Virgo network only 50% of the data can be used with the three pipelines. The rest of the time at least on detector pipeline was unusuable.
    SNR_thrs: (int or float) Define the SNR threshold for which we assume a source is detectable
    SNR_sub: (int or float) Define the SNR threshold to substract the sources. For example its commonly assumed that in HLV all source with an SNR above 8 are resolved. However for a reason of parameter uncertainty the calculation of the residual background is done by subtracting only source with a SNR above 12.

If only one detector is used in the study it still has to be set as a detector.

The variable Networks gather all the networks used in the study.
'''

HLV = Detection.Network(net_name = 'HLV',compo=[H,L,V], pic_file = 'AuxiliaryFiles/PICs/Design_HLV_flow_10.txt' , efficiency = 0.5,SNR_thrs = 12 )

Networks = [HLV]

'''Finally the method compute_SNR, compute the SNR of each sources of the model, and update the catalogue(s) with a new parameter column named by net_name containing the SNR in the corresponding network.'''

astromodel1.compute_SNR(Networks, Freq, approx = WF_approx)


'''
    3. Calculate the corresponding background:

Prepare the calculation of the background with the class Princess :

    freq: (np.array) frequency range, preferentially the one used since the beginning. In linear scale before the LISA update
    approx: (str) waveform approximation
    freq_ref: (list of float) list of frequency of interest for the study, is usually 10Hz for 3G detectors and 25Hz for 2G.

Then the calculation is done by using the method Omega_pycbc

'''

# Threshold you choose for a source to be detectable, default is 12 \cite.



Zelda = SP.Princess(Freq, approx = WF_approx, freq_refs = [10., 25.])
Zelda.Omega_pycbc(astromodel1, Networks)


'''
    4. Analyse the background

This part of the code aims to extract reference values for the predicted background. Usual values are the amplitude at 10 and 25 Hz, the SNR, the number of resolved soures, the ratio of detected sources, and the ratio between residuals and total backgrouns at a reference value.

'''


for sub_cat in astromodel1.catalogs :
    Omega_name = 'Results/Omega_e0/'+sub_cat
    SP.Analysis(astromodel1, Networks)

Omega_ana_freq = [1., 10., 25.]



'''
    4) Dectecability of the background:
    Run the SNR for the correspondig background.
'''


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

'''
    4) Gather all the files from this computation in a single folder.
'''
UF.move_computation(name ='Test', astromodels_list=[AM_BBH, AM_BNS, AM_BHNS])