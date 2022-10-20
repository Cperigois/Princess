import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.cosmology import Planck15
import numpy as np
import math
import pycbc.psd
import pycbc.waveform
import pycbc.filter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    Networks = {'3ETcryo10' : {'Files':['ET_10km_cryo.txt', 'ORF.dat'], 'PSDs' : ['ETcryo10', 'ETcryo10', 'ETcryo10'], 'Gamma_Compo' : ['E1', 'E2', 'E3'] },
                '3ETcryo15' : {'Files':['ET_15km_cryo.txt', 'ORF.dat', '3G'],'PSDs' :  ['ETcryo15', 'ETcryo15', 'ETcryo15'], 'Gamma_Compo' : ['E1', 'E2', 'E3'] },
                '3ETamb10' : {'Files':  ['ET_10km_290K.txt', 'ORF.dat', '3G'],'PSDs' : ['ETamb10', 'ETamb10', 'ETamb10'], 'Gamma_Compo' : ['E1', 'E2', 'E3'] },
                '3ETamb15' : {'Files':  ['ET_15km_290K.txt', 'ORF.dat', '3G'],'PSDs' : ['ETamb15', 'ETamb15', 'ETamb15'], 'Gamma_Compo' : ['E1', 'E2', 'E3'] },

                '3ET2CEcryo10' : {'Files': ['ET_10km_cryo.txt', 'ORF.dat', '3G'],'PSDs' : ['ETcryo10', 'ETcryo10', 'ETcryo10', 'CE1', 'CE2'], 'Gamma_Compo' : ['E1', 'E2', 'E3', 'CE1', 'CE2'] },
                '3ET2CEamb10' : {'Files': ['ET_10km_290K.txt', 'ORF.dat', '3G'],'PSDs' : ['ETamb10', 'ETamb10', 'ETamb10', 'CE1', 'CE2'], 'Gamma_Compo' : ['E1', 'E2', 'E3', 'CE1', 'CE2'] },
                '3ET2CEcryo15' : {'Files':  ['ET_15km_cryo.txt', 'ORF.dat', '3G'],'PSDs' : ['ETcryo15', 'ETcryo15', 'ETcryo15', 'CE1', 'CE2'], 'Gamma_Compo' : ['E1', 'E2', 'E3', 'CE1', 'CE2'] },
                '3ET2CEamb15' : {'Files': ['ET_15km_290K.txt', 'ORF.dat', '3G'],'PSDs' : ['ETamb15', 'ETamb15', 'ETamb15', 'CE1', 'CE2'], 'Gamma_Compo' : ['E1', 'E2', 'E3', 'CE1', 'CE2'] },

                '2ETcryo15al' : {'Files': ['ET_15km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' : ['ETcryo15', 'ETcryo15'], 'Gamma_Compo' :  ['EL1', 'EL2'] },
                '2ETcryo15misal' : {'Files': ['ET_15km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' :  ['ETcryo15', 'ETcryo15'], 'Gamma_Compo' : ['EL1', 'EL2'] },
                '2ETamb15al' : {'Files':  ['ET_15km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' : ['ETamb15', 'ETamb15'], 'Gamma_Compo' :   ['EL1', 'EL2']},
                '2ETamb15misal' : {'Files': ['ET_15km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' : ['ETamb15', 'ETamb15'], 'Gamma_Compo' :  ['EL1', 'EL2'] },
                '2ETcryo20al' : {'Files': ['ET_20km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' :  ['ETcryo20', 'ETcryo20'], 'Gamma_Compo' :  ['EL1', 'EL2'] },
                '2ETcryo20misal' : {'Files':['ET_20km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' : ['ETcryo20', 'ETcryo20'], 'Gamma_Compo' :  ['EL1', 'EL2'] },
                '2ETamb20al' : {'Files':['ET_20km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' :  ['ETamb20', 'ETamb20'], 'Gamma_Compo' :  ['EL1', 'EL2'] },
                '2ETamb20misal' : {'Files':['ET_20km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' :['ETamb20', 'ETamb20'], 'Gamma_Compo' :  ['EL1', 'EL2'] },

                '2ET2CEcryo15al' : {'Files': ['ET_15km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' :  ['CE1', 'CE2', 'ETcryo15', 'ETcryo15'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] },
                '2ET2CEcryo15misal' : {'Files':['ET_15km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' : ['CE1', 'CE2', 'ETcryo15', 'ETcryo15'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] },
                '2ET2CEamb15al' : {'Files':['ET_15km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' :  ['CE1', 'CE2', 'ETamb15', 'ETamb15'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] },
                '2ET2CEamb15misal' : {'Files':  ['ET_15km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' : ['CE1', 'CE2', 'ETamb15', 'ETamb15'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] },
                '2ET2CEcryo20al' : {'Files': ['ET_20km_cryo.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' : ['CE1', 'CE2', 'ETcryo20', 'ETcryo20'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] },
                '2ET2CEcryo20misal' : {'Files':['ET_20km_cryo.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' : ['CE1', 'CE2', 'ETcryo20', 'ETcryo20'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] },
                '2ET2CEamb20al' : {'Files':  ['ET_20km_290K.txt', 'ORF_3G_2ETal.dat', '3G'],'PSDs' :  ['CE1', 'CE2', 'ETamb20', 'ETamb20'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] },
                '2ET2CEamb20misal' : {'Files':['ET_20km_290K.txt', 'ORF_3G_2ETmisal.dat', '3G'],'PSDs' : ['CE1', 'CE2', 'ETamb20', 'ETamb20'], 'Gamma_Compo' : ['CE1', 'CE2', 'EL1', 'EL2'] }
    }


    Detectors = {'ETcryo10': ['Coba', 'ET_10km_cryo.txt', 3001, 1, 1],
                 'ETcryo15': ['Coba', 'ET_15km_cryo.txt', 3001, 1, 1],
                 'ETcryo20': ['Coba', 'ET_20km_cryo.txt', 3001, 1, 1],
                 'ETamb10': ['Coba', 'ET_10km_290K.txt', 3001, 1, 1],
                 'CE1': ['PyCBC', 'CosmicExplorerP1600143', 3001, 1, 1],
                 'CE2': ['PyCBC', 'CosmicExplorerP1600143', 3001, 1, 1],
                 'ETamb15': ['Coba', 'ET_15km_290K.txt', 3001, 1, 1],
                 'ETamb20': ['Coba', 'ET_20km_290K.txt', 3001, 1, 1]
                 }

    yr = 365 * 24 * 3600  # s
    T_obs = yr #yr
    B= 50 # power laws list from -b to +b
    beta = np.arange(2*B+1)-B
    H0 = 2.183e-18  # s-1
    Omega_PI = pd.DataFrame({'f' : np.arange(3000)+1})
    for idNet in list(Networks.keys()) :
        #Calculation of Seff
        Network = Networks[idNet]
        PSDs = Network['PSDs']
        Files = Network['Files']
        Gamma_compo = Network['Gamma_Compo']
        Seff = np.zeros(3000)
        for det1id in range(len(PSDs)):
            for det2t in range(len(PSDs)-1-det1id) :
                det2id = det2t+1+det1id
                det1_info = Detectors[PSDs[det1id]]
                det2_info = Detectors[PSDs[det2id]]
                if det1_info[0] == 'PyCBC':
                    PSD1 = pycbc.psd.from_string(psd_name=det1_info[1], length=det1_info[2], delta_f=det1_info[3],
                                                low_freq_cutoff=det1_info[4])
                if det1_info[0] == 'Coba':
                    df_bis = pd.read_csv('../PSDs/Coba/' + det1_info[1], sep=' ', index_col=None)
                    PSD1 = pycbc.psd.read.from_numpy_arrays(df_bis['f'], df_bis['HF_LF'], length=det1_info[2],
                                                               delta_f=det1_info[3], low_freq_cutoff=det1_info[4])
                if det2_info[0] == 'PyCBC':
                    PSD2 = pycbc.psd.from_string(psd_name=det2_info[1], length=det2_info[2], delta_f=det2_info[3],
                                                low_freq_cutoff=det2_info[4])
                if det2_info[0] == 'Coba':
                    df_bis = pd.read_csv('../PSDs/Coba/' + det2_info[1], sep=' ', index_col=None)
                    PSD2 = pycbc.psd.read.from_numpy_arrays(df_bis['f'], df_bis['HF_LF'], length=det2_info[2],
                                                               delta_f=det2_info[3], low_freq_cutoff=det2_info[4])
                gamma_name = Gamma_compo[det1id]+Gamma_compo[det2id]
                gamma = pd.read_csv('../ORFs/'+Files[1], sep = '\t', index_col= None)
                Seff += np.power(gamma[gamma_name],2.) /(np.array(PSD1[1:])*np.array(PSD2[1:]))
                #Seff += 1./ (np.array(PSD1[1:]) * np.array(PSD2[1:]))
                #print(Seff)
        Omega_eff =np.power(Seff, -0.5)*np.power(gamma['freq'],3.)*2.*math.pi*math.pi/(3.*H0*H0)
        omega_beta = np.array([])
        fmax = 3000
        fmin = 1
        rho = 1.
        fref  =10.
        for bid in range(len(beta)) :
            b = beta[bid]
            omgbet = rho/np.sqrt(2*T_obs) *np.power(np.sum(np.power(gamma['freq']/fref, 2.*b)/np.power(Omega_eff,2.)), -0.5)
            omega_beta =np.append(omega_beta,omgbet)
        Omgpi = np.array([])
        for f in gamma['freq']:
            Omgpi= np.append(Omgpi, np.max([omega_beta[b+B]*(f/10.)**b  for b in beta ]))
        Omega_PI[idNet] = Omgpi
    Omega_PI.to_csv('ET_PICs_rho1_T1_nogamma.txt', sep = '\t', index = None)
