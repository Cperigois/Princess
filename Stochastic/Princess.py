import pandas as pd
import os
import Stochastic.SNR as SNR
from astropy.cosmology import Planck15
import Stochastic.Kst as K
import numpy as np
import Stochastic.Basic_Functions as BF
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import jv
from scipy.integrate import quad
from scipy.optimize import fsolve
import pycbc.waveform as wf

from Getting_Started import Networks


class Princess:

    def __init__(self, Freq, Omega_ana_freq = [10,25], Networks):
        """Create an instance of your calculations parameters.
        Parameters
        ----------
        Freq : np.array
            Frequency band of the caclulation
        Omega_ana_freq : str
            Frequency of reference for the built of analysis file
        Network: np.array of Networks class type
            For which Network is done the calculation
        """
        self.Freq = Freq
        self.Networks = Networks
        self.Omega_ana_freq = Omega_ana_freq



    def Omega_ecc(self, cat):

        df = pd.read_csv('Catalogs/'+cat, delimiter = '\t', index_col = None )
        check_file = os.path.exists('Results/Omega_e0/' + cat)
        if check_file == False:
            Omega = pd.DataFrame({'f':Freq, 'Total': np.zeros(len(self.Freq))})
            Omega_e0 = pd.DataFrame({'f':Freq, 'Total': np.zeros(len(self.Freq))})
            for n in self.Networks :
                Omega[n]= np.zeros(len(self.Freq))
                Omega_e0[n] = np.zeros(len(self.Freq))
            for i in range(len(df['Mc'])) :
                evt = df.iloc[i]
                args = [ evt['Mc'], evt['q'], evt['Xsi'], evt['zm'], evt['zf'], evt['a0'], evt['e0'] ]
                Omg, Omg_e0 = GWk(args, co_type, 0 )
                Omega['Total'] += Omg
                Omega_e0['Total'] += Omg_e0
                for N in self.Networks :
                    if evt[N]< self.SNR_thrs[N] :
                        Omega[N] += Omg
                        Omega_e0[N] += Omg_e0
            Omega.to_csv('Results/Omega/'+path, index = False, sep = '\t')
            Omega_e0.to_csv('Results/Omega_e0/'+path, index = False, sep = '\t')
        else :
            Omega_e0 = pd.read_csv('Results/Omega_e0/'+path, index_col = False, sep = '\t')
        check_file = os.path.exists('Results/Ana/' + path)
        if check_file == False:
            output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq] + ['SNR']
            Ana = pd.DataFrame(index=output_index, columns=['Total'] + self.Networks)
            Ana['Total']['N_source'] = 0
            Ana['Total'][['Omg_'+str(i)+'_Hz' for i in self.Omega_ana_freq]] = Search_Omg(Omega_e0['Total'], self.Omega_ana_freq)
            Ana['Total']['SNR'] = SNR.SNR_Omega(Omega_e0['Total'])
            print(Ana['Total'])
            for N in range(len(self.Networks)):
                Ana[Networks[N].net_name][['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq]] = Search_Omg(Omega_e0[Networks[N].net_name], self.Omega_ana_freq)
                Ana[N]['SNR'] = SNR.SNR_Omega(Omega_e0[Networks[N].net_name],N)
                residual = df[df[N]<self.SNR_thrs[N]]
                Ana[N]['Nsource'] = len(residual[N])
                print(Ana[N])
            Ana.to_csv('Results/Ana/'+path, sep = '\t')

    def Ana(self, path) :
        Omega_e0 = pd.read_csv('Results/Omega_e0/' + path, index_col=False, sep='\t')
        output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq] + ['SNR_Total']+ ['SNR_Residual']
        Ana = pd.DataFrame(index=output_index, columns=['Total'] + self.Networks)
        Ana['Total']['N_source'] = 0
        Ana['Total'][['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq]] = Search_Omg(Omega_e0['Total'],
                                                                                              self.Omega_ana_freq)
        Ana['Total']['SNR_Total'] = SNR.SNR_Omega(Omega_e0['Total'])
        Ana['Total']['SNR_Residual'] = 0
        print(Ana['Total'])
        for N in range(len(self.Networks)):
            Ana[self.Networks[N].net_name][['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq]] = Search_Omg(
                Omega_e0[Networks[N].net_name], self.Omega_ana_freq)
            Ana[self.Networks[N].net_name]['SNR_residual'] = SNR.SNR_Omega(Omega_e0[self.Networks[N].net_name], self.Networks[N].pic_file)
            Ana[self.Networks[N].net_name]['SNR_total'] = SNR.SNR_Omega(Omega_e0['Total'],
                                                                           self.Networks[N].pic_file)
            residual = df[df[self.Networks[N].net_name] < self.Networks[N].SNR_thrs]
            Ana[self.Networks[N].net_name]['Nsource'] = len(residual.zm)
            print(Ana[N])
        Ana.to_csv('Results/Ana/' + path, sep='\t')


    def Omega(cat, Freq, Networks, SNR_thrs):

        df = pd.read_csv('Catalogs/'+cat, delimiter = '\t', index_col = None )
        check_file = os.path.exists('Results/Omega_e0/' + cat)
        if check_file == False:
            Omega_e0 = pd.DataFrame({'f':Freq, 'Total': np.zeros(len(Freq))})
            for n in Networks :
                Omega_e0[n] = np.zeros(len(Freq))
            if orbit_evo == False :
                for i in range(len(df['Mc'])):
                    evt = df.iloc[i]
                    args = [evt['Mc'], evt['q'], (evt['m1']*evt['s1']+evt['m2']*evt['s2'])/(evt['m1']+evt['m2']), evt['zm']]
                    Omg_e0 = GWk_noEcc(args, co_type, 0)
                    Omega_e0['Total'] += Omg_e0
                    for N in Networks:
                        if evt[N] < SNR_thrs[N]:
                            Omega_e0[N] += Omg_e0
            Omega_e0.to_csv('Results/Omega_e0/' + path, index=False, sep='\t')

        else :
            Omega_e0 = pd.read_csv('Results/Omega_e0/'+path, index_col = False, sep = '\t')

        check_file = os.path.exists('Results/Ana/' + path)
        if check_file == False:
            output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in Omega_ana_freq] + ['SNR']
            Ana = pd.DataFrame(index=output_index, columns=['Total'] + Networks)
            Ana['Total']['N_source'] = 0
            Ana['Total'][['Omg_'+str(i)+'_Hz' for i in Omega_ana_freq]] = Search_Omg(Omega_e0['Total'], Omega_ana_freq)
            Ana['Total']['SNR'] = SNR.SNR_Omega(Omega_e0['Total'])
            print(Ana['Total'])
            for N in GS.Networks:
                Ana[N][['Omg_' + str(i) + '_Hz' for i in Omega_ana_freq]] = Search_Omg(Omega_e0[N], Omega_ana_freq)
                Ana[N]['SNR'] = SNR.SNR_Omega(Omega_e0[N],N)
                residual = df[df[N]<self.SNR_thrs[N]]
                Ana[N]['Nsource'] = len(residual[N])
                print(Ana[N])
            Ana.to_csv('Results/Ana/'+path, sep = '\t')


    def Omega_pycbc(path, model):

        cat = GS.Cat_list_df.loc[model]

        check_file = os.path.exists('Results/Omega_e0/' + path)
        if check_file == False:
            df = pd.read_csv('Catalogs/' + path, delimiter='\t', index_col=None)
            Omega_e0 = pd.DataFrame({'f':GS.Freq, 'Total': np.zeros(len(GS.Freq))})
            for n in GS.Networks :
                Omega_e0[n] = np.zeros(len(GS.Freq))
            if GS.orbit_evo ==False :
                for i in range(len(df['m1'])):
                    evt = df.iloc[i]
                    args = [evt['m1'], evt['m2'], evt['zm'], evt['Dl'], evt['s1'], evt['s2']]
                    Omg_e0 = GWk_noEcc_Pycbcwf(args, cat.co_type, 0)* np.power(GS.Freq,3.) * K.C / cat.Duration
                    Omega_e0['Total'] += Omg_e0
                    for N in GS.Networks:
                        if evt[N] < GS.SNR_thrs[N]:
                            Omega_e0[N] += Omg_e0
                Omega_e0.to_csv('Results/Omega_e0/' + path, index=False, sep='\t')


        check_file = os.path.exists('Results/Ana/' + path)
        if check_file == False:
            output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in GS.Omega_ana_freq] + ['SNR']
            Ana = pd.DataFrame(index=output_index, columns=['Total'] + GS.Networks)
            Ana['Total']['N_source'] = 0
            Ana['Total'][['Omg_'+str(i)+'_Hz' for i in GS.Omega_ana_freq]] = Search_Omg(Omega_e0['Total'], GS.Omega_ana_freq)
            Ana['Total']['SNR'] = SNR.SNR_Omega(Omega_e0['Total'])
            print(Ana['Total'])
            for N in GS.Networks:
                Ana[N][['Omg_' + str(i) + '_Hz' for i in GS.Omega_ana_freq]] = Search_Omg(Omega_e0[N], GS.Omega_ana_freq)
                Ana[N]['SNR'] = SNR.SNR_Omega(Omega_e0[N],N)
                residual = df[df[N]<GS.SNR_thrs[N]]
                Ana[N]['Nsource'] = len(residual[N])
                print(Ana[N])
            Ana.to_csv('Results/Ana/'+path, sep = '\t')

def Search_Omg(Omega, freq_ref):
    interp = InterpolatedUnivariateSpline(GS.Freq, Omega)
    out = np.zeros(len(freq_ref))
    for i in range(len(freq_ref)) :
        out[i] = interp(freq_ref[i])
    return out
