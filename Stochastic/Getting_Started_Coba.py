'''This programs aims to calculate the background from a CBC population.
    Following all this steps is required to launch the program in a safe way :)

    Current status :

    -
    -
'''

import os
import numpy as np
import pandas as pd

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
Models = ['logmet_0.2','logmet_0.3','logmet_0.4']
#Flags = [1, 2, 3]
#Flags_str = ['Orig', 'Exch','Iso']
co_object = ['BBH', 'BNS', 'BHNS']
path_2_cat_folder =  '/home/perigois/Documents/GreatWizardOz/Coba/'
IDD = ['iso','dyn']
Flags = {'1': 'Orig', '2':'Exch', '3':'Iso'}


# for cat in Cat_list.keys() :
#     if co_object[co] == 'BBH' :
#         for m in range(len(Models)) :
#             name = co_object[co]+'_'+Models[m]
#             Cat_list[name] = [path_2_cat_folder+'Catalog_co_'+co_object[co]+'_fc_MIX_kick150_'+Models[m]+'_magsp_0.1_a_1.0_sn_rapid.dat', co_object[co] , Models[m], 'iso']
#             #Cat_list[name] = [path_2_cat_folder + 'Cat_stochastic_' + IDD[m] +'_'+ co_object[co] +   's.dat', co_object[co], Models[m],IDD[m]]
#     else :
#         name = co_object[co]
#         Cat_list[name] = [path_2_cat_folder+'Catalog_co_'+co_object[co]+'_fc_Iso_logmet_0.3_magsp_0.1_a_3.0_sn_delayed.dat', co_object[co],co_object[co],'iso']
#         #Cat_list[co_object[co]] = [path_2_cat_folder + 'Cat_stochastic_' + IDD[m] + '_' + co_object[co] + 's.dat', co_object[co], Models[m], IDD[m]]

#set up catalogs :
columns=['path', 'Model', 'co_type', 'Duration', 'Flags']
Cat_list = { 'BBH_logmet_0.2' :[path_2_cat_folder+'Catalog_co_BBH_fc_MIX_kick150_logmet_0.2_magsp_0.1_a_1.0_sn_rapid.dat','logmet_0.2','BBH',10,False],
             'BBH_logmet_0.3' :[path_2_cat_folder+'Catalog_co_BBH_fc_MIX_kick150_logmet_0.3_magsp_0.1_a_1.0_sn_rapid.dat','logmet_0.3','BBH',10,False],
             'BBH_logmet_0.4' :[path_2_cat_folder+'Catalog_co_BBH_fc_MIX_kick150_logmet_0.4_magsp_0.1_a_1.0_sn_rapid.dat','logmet_0.4','BBH',10,False],
             'BNS' :[path_2_cat_folder+'Catalog_co_BNS_fc_Iso_logmet_0.3_magsp_0.1_a_3.0_sn_delayed.dat','logmet_0.3','BNS',10,False],
             'BHNS' :[path_2_cat_folder+'Catalog_co_BHNS_fc_Iso_logmet_0.3_magsp_0.1_a_3.0_sn_delayed.dat','logmet_0.3','BHNS',100,False]
             #'YSC_logmet_0.4' :[path_2_cat_folder+'ET_COBA_bychannel-20220509T220133Z-001/ET_COBA_bychannel/sigma04_rapid_A1.0_chi01_YSC/Catalogs/Catalog_co_BBH_fc_YSC_kick150_logmet_0.4_magsp_0.1_a_1.0_sn_rapid.dat','logmet_0.3','BBH',100,False],
             #'field_logmet_0.4' :[path_2_cat_folder+'ET_COBA_bychannel-20220509T220133Z-001/ET_COBA_bychannel/sigma04_rapid_A1.0_chi01_field/Catalogs/Catalog_co_BBH_fc_field_kick150_logmet_0.4_magsp_0.1_a_1.0_sn_rapid.dat','logmet_0.3','BBH',100,False],
             #'GC_logmet_0.4' :[path_2_cat_folder+'ET_COBA_bychannel-20220509T220133Z-001/ET_COBA_bychannel/sigma04_rapid_A1.0_chi01_GC/Catalogs/Catalog_co_BBH_fc_GC_kick150_logmet_0.4_magsp_0.1_a_1.0_sn_rapid.dat','logmet_0.3','BBH',100,False],
             #'NSC_logmet_0.4' :[path_2_cat_folder+'ET_COBA_bychannel-20220509T220133Z-001/ET_COBA_bychannel/sigma04_rapid_A1.0_chi01_NSC/Catalogs/Catalog_co_BBH_fc_NSC_kick150_logmet_0.4_magsp_0.1_a_1.0_sn_rapid.dat','logmet_0.3','BBH',100,False],
             }
Cat_list_df = pd.DataFrame.from_dict(Cat_list, columns= columns, orient='index')

orbit_evo = False
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

Networks = ['3ETcryo10' ,'3ETcryo15','3ETamb10' ,'3ETamb15' ,
             '3ET2CEcryo10','3ET2CEamb10','3ET2CEcryo15' ,'3ET2CEamb15' ,'2ETcryo15al' ,'2ETcryo15misal' ,'2ETamb15al',
             '2ETamb15misal','2ETcryo20al','2ETcryo20misal','2ETamb20al','2ETamb20misal','2ET2CEcryo15al' ,
             '2ET2CEcryo15misal','2ET2CEamb15al','2ET2CEamb15misal','2ET2CEcryo20al','2ET2CEcryo20misal',
             '2ET2CEamb20al' ,'2ET2CEamb20misal']
SNR_thrs = { '3ETcryo10' : 12.,
             '3ETcryo15' : 12.,
             '3ETamb10' : 12.,
             '3ETamb15' : 12.,

             '3ET2CEcryo10' : 12.,
             '3ET2CEamb10' : 12.,
             '3ET2CEcryo15' : 12.,
             '3ET2CEamb15' : 12.,

             '2ETcryo15al' : 12.,
             '2ETcryo15misal' : 12.,
             '2ETamb15al' : 12.,
             '2ETamb15misal' : 12.,
             '2ETcryo20al' : 12.,
             '2ETcryo20misal' : 12.,
             '2ETamb20al' : 12.,
             '2ETamb20misal' : 12.,

             '2ET2CEcryo15al' : 12.,
             '2ET2CEcryo15misal' : 12.,
             '2ET2CEamb15al' : 12.,
             '2ET2CEamb15misal' : 12.,
             '2ET2CEcryo20al' : 12.,
             '2ET2CEcryo20misal' :12.,
             '2ET2CEamb20al' : 12.,
             '2ET2CEamb20misal' : 12.}
Net_compo = {'3ETcryo10' : ['E1','E2','E3'],
             '3ETcryo15' : ['E1','E2','E3'],
             '3ETamb10' : ['E1','E2','E3'],
             '3ETamb15' : ['E1','E2','E3'],
             
             '3ET2CEcryo10' : ['E1','E2','E3','CE1','CE2'],
             '3ET2CEamb10' : ['E1','E2','E3','CE1','CE2'],
             '3ET2CEcryo15' : ['E1','E2','E3','CE1','CE2'],
             '3ET2CEamb15' : ['E1','E2','E3','CE1','CE2'],
             
             '2ETcryo15al' : ['EL1', 'EL2'],
             '2ETcryo15misal' : ['EL1', 'EL2'],
             '2ETamb15al' : ['EL1', 'EL2'],
             '2ETamb15misal' : ['EL1', 'EL2'],
             '2ETcryo20al' : ['EL1', 'EL2'],
             '2ETcryo20misal' : ['EL1', 'EL2'],
             '2ETamb20al' : ['EL1', 'EL2'],
             '2ETamb20misal' : ['EL1', 'EL2'],
             
             '2ET2CEcryo15al' : ['CE1', 'CE2','EL1', 'EL2'],
             '2ET2CEcryo15misal' : ['CE1', 'CE2','EL1', 'EL2'],
             '2ET2CEamb15al' : ['CE1', 'CE2','EL1', 'EL2'],
             '2ET2CEamb15misal' : ['CE1', 'CE2','EL1', 'EL2'],
             '2ET2CEcryo20al' : ['CE1', 'CE2','EL1', 'EL2'],
             '2ET2CEcryo20misal' : ['CE1', 'CE2','EL1', 'EL2'],
             '2ET2CEamb20al' : ['CE1', 'CE2','EL1', 'EL2'],
             '2ET2CEamb20misal' : ['CE1', 'CE2','EL1', 'EL2']

             }

Net_psd = {'3ETcryo10': ['ETcryo10', 'ETcryo10', 'ETcryo10'],
             '3ETcryo15': ['ETcryo15', 'ETcryo15', 'ETcryo15'],
             '3ETamb10': ['ETamb10', 'ETamb10', 'ETamb10'],
             '3ETamb15': ['ETamb15', 'ETamb15', 'ETamb15'],

             '3ET2CEcryo10': ['ETcryo10', 'ETcryo10', 'ETcryo10', 'CE1', 'CE2'],
             '3ET2CEamb10': ['ETamb10', 'ETamb10', 'ETamb10', 'CE1', 'CE2'],
             '3ET2CEcryo15': ['ETcryo15', 'ETcryo15', 'ETcryo15', 'CE1', 'CE2'],
             '3ET2CEamb15': ['ETamb15', 'ETamb15', 'ETamb15', 'CE1', 'CE2'],

             '2ETcryo15al': ['ETcryo15', 'ETcryo15'],
             '2ETcryo15misal': ['ETcryo15', 'ETcryo15'],
             '2ETamb15al': ['ETamb15', 'ETamb15'],
             '2ETamb15misal': ['ETamb15', 'ETamb15'],
             '2ETcryo20al': ['ETcryo20', 'ETcryo20'],
             '2ETcryo20misal': ['ETcryo20', 'ETcryo20'],
             '2ETamb20al': ['ETamb20', 'ETamb20'],
             '2ETamb20misal': ['ETamb20', 'ETamb20'],

             '2ET2CEcryo15al': ['CE1', 'CE2', 'ETcryo15', 'ETcryo15'],
             '2ET2CEcryo15misal': ['CE1', 'CE2', 'ETcryo15', 'ETcryo15'],
             '2ET2CEamb15al': ['CE1', 'CE2', 'ETamb15', 'ETamb15'],
             '2ET2CEamb15misal': ['CE1', 'CE2', 'ETamb15', 'ETamb15'],
             '2ET2CEcryo20al': ['CE1', 'CE2', 'ETcryo20', 'ETcryo20'],
             '2ET2CEcryo20misal': ['CE1', 'CE2', 'ETcryo20', 'ETcryo20'],
             '2ET2CEamb20al': ['CE1', 'CE2','ETamb20', 'ETamb20'],
             '2ET2CEamb20misal': ['CE1', 'CE2', 'ETamb20', 'ETamb20']

             }

Net_files = {'3ETcryo10': ['ET_10km_cryo.txt', 'ORF.dat', '3G'],
             '3ETcryo15': ['ET_15km_cryo.txt', 'ORF.dat', '3G'],
             '3ETamb10': ['ET_10km_290K.txt', 'ORF.dat', '3G'],
             '3ETamb15': ['ET_15km_290K.txt', 'ORF.dat', '3G'],

             '3ET2CEcryo10': ['ET_10km_cryo.txt', 'ORF.dat', '3G'],
             '3ET2CEamb10': ['ET_10km_290K.txt', 'ORF.dat', '3G'],
             '3ET2CEcryo15': ['ET_15km_cryo.txt', 'ORF.dat', '3G'],
             '3ET2CEamb15': ['ET_15km_290K.txt', 'ORF.dat', '3G'],

             '2ETcryo15al': ['ET_15km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ETcryo15misal': ['ET_15km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],
             '2ETamb15al': ['ET_15km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ETamb15misal': ['ET_15km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G'],
             '2ETcryo20al': ['ET_20km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ETcryo20misal': ['ET_20km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],
             '2ETamb20al': ['ET_20km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ETamb20misal': ['ET_20km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G'],

             '2ET2CEcryo15al': ['ET_15km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ET2CEcryo15misal': ['ET_15km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],
             '2ET2CEamb15al': ['ET_15km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ET2CEamb15misal': ['ET_15km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G'],
             '2ET2CEcryo20al': ['ET_20km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ET2CEcryo20misal': ['ET_20km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],
             '2ET2CEamb20al': ['ET_20km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],
             '2ET2CEamb20misal': ['ET_20km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G']

             }

#Composition for 3G Configuration ET : triangle with arm lenght 10km
Detectors = {'ETcryo10': ['Coba','ET_10km_cryo.txt', 2490, 1, 1],
             'ETcryo15': ['Coba','ET_15km_cryo.txt', 2490, 1, 1],
             'ETcryo20': ['Coba','ET_20km_cryo.txt', 2490, 1, 1],
             'ETamb10': ['Coba','ET_10km_290K.txt', 2490, 1, 1],
             'CE1': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
             'CE2': ['PyCBC','CosmicExplorerP1600143', 2490, 1, 1],
             'ETamb15': ['Coba','ET_15km_290K.txt', 2490, 1, 1],
             'ETamb20': ['Coba','ET_20km_290K.txt', 2490, 1, 1]
             }


#Freq = np.concatenate([np.logspace(-6,0,50),np.linspace(2,2500,2499)])
Freq = np.linspace(1,2500,2499)


'''
    3) Setup the ana output you want :
    in addition to the files containing the energydensity spectrum this code will write a file zwith useful stats : 
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