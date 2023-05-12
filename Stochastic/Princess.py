import pandas as pd
import os
import Stochastic.SNR as SNR
import Stochastic.Kst as K
import numpy as np
import Stochastic.Basic_Functions as BF
from Starter.Htild import GWk_noEcc_Pycbcwf


class Princess:

    def __init__(self, Freq, astromodel, approx, Networks, inclination, Omega_ana_freq = [10,25]):
        """Create an instance of your calculations parameters.
        Parameters
        ----------
        Freq : np.array
            Frequency band of the caclulation
        Omega_ana_freq : list of float
            Frequency of reference for the built of analysis file
        Network: np.array of Networks class type
            For which Network is done the calculation

        """
        self.Freq = Freq
        #self.dict_Networks = {  }
        #for key in Neworks.keys() :
        #    self.dict_Networks[key] = Detection.Network(net_name = key, compo = Networks[key][0] , pic_file = Networks[key][1], freq = Networks[key][2], efficiency = Networks[key][3], SNR_thrs =Networks[key][4] )

        self.Networks = Networks
        self.Omega_ana_freq = Omega_ana_freq
        self.approx = approx
        self.inclination = inclination
        self.astromodel = astromodel
        output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq] + ['SNR_Total'] + ['SNR_Residual']
        self.anadict = {}
        for cat in astromodel.catalogs :
            self.anadict[cat] = pd.DataFrame(index=output_index, columns=['Total'] + [Networks[i].net_name for i in range(len(Networks))])

    def Write_results(self) :
        check_file = os.path.exists('Results/Analysis/')
        if check_file == False:
            os.mkdir('Results/Analysis/')
        for cat in self.astromodel.catalogs:
            df = self.anadict[cat]
            df.to_csv('Results/Analysis/' + cat, sep='\t')

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
                args = [ evt['Mc'], evt['q'], evt['Xsi'], evt['z'], evt['zf'], evt['a0'], evt['e0'] ]
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
            Omega_e0 = pd.read_read_csvcsv('Results/Omega_e0/'+path, index_col = False, sep = '\t')
        check_file = os.path.exists('Results/Analysis/' + path)
        if check_file == False:
            output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq] + ['SNR']
            Ana = pd.DataFrame(index=output_index, columns=['Total'] + self.Networks)
            Ana['Total']['N_source'] = 0
            Ana['Total'][['Omg_'+str(i)+'_Hz' for i in self.Omega_ana_freq]] = Search_Omg(Omega_e0['Total'], self.Omega_ana_freq)
            Ana['Total']['SNR'] = SNR.SNR_Omega(Omega_e0['Total'])
            for N in range(len(self.Networks)):
                Ana[Networks[N].net_name][['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq]] = Search_Omg(Omega_e0[Networks[N].net_name], self.Omega_ana_freq)
                Ana[N]['SNR'] = SNR.SNR_Omega(Omega_e0[Networks[N].net_name],N)
                residual = df[df[N]<self.SNR_thrs[N]]
                Ana[N]['Nsource'] = len(residual[N])
                print(Ana[N])
            Ana.to_csv('Results/Analysis/'+path, sep = '\t')

    def Analysis(self, Networks) :
        for c in range(len(self.astromodel.catalogs)):
            cat = self.astromodel.catalogs[c]
            Omega_e0 = pd.read_csv('Results/Omega_e0/' + cat, index_col=False, sep='\t')
            Ana = self.anadict[cat]
            for i in self.Omega_ana_freq :
                Ana['Total']['Omg_' + str(i) + '_Hz' ] = BF.Search_Omg(Freq = Omega_e0['f'], Omega = Omega_e0['Total'], freq_ref = [i])
            for N in range(len(Networks)):
                for i in self.Omega_ana_freq:
                    Ana[Networks[N].net_name]['Omg_' + str(i) + '_Hz'] = BF.Search_Omg(Freq = Omega_e0['f'], Omega = Omega_e0[Networks[N].net_name], freq_ref = self.Omega_ana_freq)
                Ana[Networks[N].net_name]['SNR_residual'] = SNR.SNR_bkg(Omega_e0['f'], Omega_e0[Networks[N].net_name], Networks[N])
                Ana[Networks[N].net_name]['SNR_total'] = SNR.SNR_bkg(Omega_e0['f'],Omega_e0['Total'], Networks[N])
                print(SNR.SNR_bkg(Omega_e0['f'], Omega_e0['Total'], Networks[N]))

            self.anadict[cat] = Ana


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
                    args = [evt['Mc'], evt['q'], (evt['m1']*evt['s1']+evt['m2']*evt['s2'])/(evt['m1']+evt['m2']), evt['z']]
                    Omg_e0 = GWk_noEcc(args, co_type, 0)
                    Omega_e0['Total'] += Omg_e0
                    for N in Networks:
                        if evt[N] < SNR_thrs[N]:
                            Omega_e0[N] += Omg_e0
            Omega_e0.to_csv('Results/Omega_e0/' + cat, index=False, sep='\t')


    def Omega_pycbc(self, Networks):
        fd_table = pd.read_csv('./AuxiliaryFiles/factor_table.dat', index_col=None, sep='\t')
        for cat in range(len(self.astromodel.catalogs)) :
            Cat = pd.read_csv('Catalogs/'+self.astromodel.catalogs[cat], delimiter='\t', index_col=None)
            Omega_e0 = pd.DataFrame({'f':self.Freq-1., 'Total': np.zeros(len(self.Freq))})
            Ana = self.anadict[self.astromodel.catalogs[cat]]
            Ana['Total']['N_source'] = len((Cat.z))
            for N in range(len(Networks)) :
                Omega_e0[Networks[N].net_name] = np.zeros(len(self.Freq))
                Ana[Networks[N].net_name]['N_source'] =0
            for evt in range(len(Cat['m1'])):
                event = Cat.iloc[[evt]]
                if  self.inclination == 'Rand' :
                    i = np.random.randint(0, len(fd_table.inc),1, int)
                    r = fd_table.iloc[i]
                    event['inc'] = np.arccos(r.inc.values)
                elif self.inclination == 'Optimal':
                    event['inc'] = 0.
                else :
                    print("inclination error")
                Omg_e0 = GWk_noEcc_Pycbcwf(evt = event,
                                           freq = self.Freq,
                                           approx = self.approx,
                                           n = evt,
                                           size_catalogue = len(Cat.z)) * np.power(self.Freq-1.,3.) * K.C / self.astromodel.duration
                Omega_e0['Total'] += Omg_e0
                for N in range(len(Networks)):
                    SNR = 0
                    for d in Networks[N].compo :
                        conf = 'f'+str(d.configuration)
                        SNR+= event[str(d.det_name)]* fd_table[conf][i[0]]
                    if float(SNR) < Networks[N].SNR_thrs:
                        Ana[Networks[N].net_name]['N_source'] += 1
                        Omega_e0[Networks[N].net_name] += Omg_e0
            Omega_e0.to_csv('Results/Omega_e0/' + self.astromodel.catalogs[cat], index=False, sep='\t')
            print('Written :  Results/Omega_e0/' , self.astromodel.catalogs[cat])


