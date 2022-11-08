import pandas as pd
import os
import Stochastic.SNR as SNR
from astropy.cosmology import Planck15
import Stochastic.Kst as K
import numpy as np
import Stochastic.Basic_Functions as BF
from Stochastic.Htild import GWk_noEcc_Pycbcwf
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import jv
from scipy.integrate import quad
from scipy.optimize import fsolve
import pycbc.waveform as wf


class Princess:

    def __init__(self, Freq, approx, Omega_ana_freq = [10,25]):
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
        #self.dict_Networks = {  }
        #for key in Neworks.keys() :
        #    self.dict_Networks[key] = Detection.Network(net_name = key, compo = Networks[key][0] , pic_file = Networks[key][1], freq = Networks[key][2], efficiency = Networks[key][3], SNR_thrs =Networks[key][4] )

        #self.Networks = Networks
        self.Omega_ana_freq = Omega_ana_freq
        self.approx = approx



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

    def Ana(self, path, Networks) :
        Omega_e0 = pd.read_csv('Results/Omega_e0/' + path, index_col=False, sep='\t')
        output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq] + ['SNR_Total']+ ['SNR_Residual']
        Ana = pd.DataFrame(index=output_index, columns=['Total'] + self.Networks)
        Ana['Total']['N_source'] = 0
        Ana['Total'][['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq]] = BF.Search_Omg(Omega_e0['Total'],
                                                                                              self.Omega_ana_freq)
        Ana['Total']['SNR_Total'] = SNR.SNR_Omega(Omega_e0['Total'])
        Ana['Total']['SNR_Residual'] = 0
        print(Ana['Total'])
        for N in range(len(self.Networks)):
            Ana[self.Networks[N].net_name][['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq]] = 0
            for i in self.Omega_ana_freq:
                Ana[self.Networks[N].net_name]['Omg_' + str(i) + '_Hz'] = BF.Search_Omg(Omega_e0[Networks[N].net_name], self.Omega_ana_freq)
            Ana[self.Networks[N].net_name]['SNR_residual'] = SNR.SNR_Omega(Omega_e0[self.Networks[N].net_name], self.Networks[N].pic_file)
            Ana[self.Networks[N].net_name]['SNR_total'] = SNR.SNR_Omega(Omega_e0['Total'],
                                                                           self.Networks[N].pic_file)
            residual = df[df[self.Networks[N].net_name] < self.Networks[N].SNR_thrs]
            Ana[self.Networks[N].net_name]['Nsource'] = len(residual.zm)
            print(Ana[N])
        Ana.to_csv('Results/Ana/' + path, sep='\t')


    def Omega(self, cat, Freq, Networks):

        df = pd.read_csv('Catalogs/'+cat, delimiter = '\t', index_col = None )
        check_file = os.path.exists('Results/Omega_e0/' + cat)
        if check_file == False:
            Omega_e0 = pd.DataFrame({'f':Freq, 'Total': np.zeros(len(Freq))})
            for n in range(len(Networks)) :
                Omega_e0[Networks[n].net_name] = np.zeros(len(Freq))
            if orbit_evo == False :
                for i in range(len(df['Mc'])):
                    evt = df.iloc[i]
                    args = [evt['Mc'], evt['q'], (evt['m1']*evt['s1']+evt['m2']*evt['s2'])/(evt['m1']+evt['m2']), evt['zm']]
                    Omg_e0 = GWk_noEcc(args, co_type, 0)
                    Omega_e0['Total'] += Omg_e0
                    for N in Networks:
                        if evt[N] < SNR_thrs[N]:
                            Omega_e0[N] += Omg_e0
            Omega_e0.to_csv('Results/Omega_e0/' + cat, index=False, sep='\t')


    def Omega_pycbc(self, astromodel, Networks):
        for cat in range(len(astromodel.catalogs)) :
            Cat = pd.read_csv(astromodel.catalogs[cat], delimiter='\t', index_col=None)
            Omega_e0 = pd.DataFrame({'f':self.Freq, 'Total': np.zeros(len(self.Freq))})
            for N in range(len(Networks)) :
                Omega_e0[Networks[N].net_name] = np.zeros(len(self.Freq))
            for evt in range(len(Cat['m1'])):
                event = Cat.iloc[[evt]]
                Omg_e0 = GWk_noEcc_Pycbcwf(event, self.Freq, self.approx, evt, len(Cat.zm))* np.power(self.Freq,3.) * K.C / astromodel.duration
                Omega_e0['Total'] += Omg_e0
                for N in range(len(Networks)):
                    if evt[N] < Networks[N].SNR_thrs:
                        Omega_e0[Networks[N].net_name] += Omg_e0
            Omega_e0.to_csv('Results/Omega_e0/' + astromodel.catalogs[cat], index=False, sep='\t')


