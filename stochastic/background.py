import pandas as pd
import os
import json
import stochastic.snr as SNR
import stochastic.constants as K
import numpy as np
import stochastic.basic_functions as BF
from astrotools.htild import GWk_no_ecc_pycbcwf
from astrotools.astromodel import AstroModel as AM
import astrotools.detection as DET

params = json.load(open('Run/Params.json', 'r'))

def process_background_computation():
    # Compute background and analysis
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Results"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Results")
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Results/Omega/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Results/Omega")
    for astomodel in params['astro_model_list'].keys():
        Zelda = Princess(astromodel=AM(name = astomodel))
        Zelda.Make_Ana_Output()
        Zelda.compute_Omega()
        Zelda.Analysis()
        Zelda.Write_results()


class Princess:

    def __init__(self, astromodel, inclination = 'Optimal', Omega_ana_freq = [10,25]):
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
        #self.dict_Networks = {  }
        #for key in Neworks.keys() :
        #    self.dict_Networks[key] = Detection.Network(name = key, compo = Networks[key][0] , pic_file = Networks[key][1], freq = Networks[key][2], efficiency = Networks[key][3], SNR_thrs =Networks[key][4] )

        self.Omega_ana_freq = Omega_ana_freq
        self.inclination = inclination
        self.astromodel = astromodel

    def Make_Ana_Output(self):
        output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq] + ['SNR_Total'] + [
            'SNR_Residual']
        self.anadict = {}
        for cat in self.astromodel.catalogs:
            self.anadict[cat] = pd.DataFrame(index=output_index,
                                             columns=['Total'] + [i for i in list(params['network_list'].keys())])
        print(self.anadict)

    def Write_results(self) :
        check_file = os.path.exists('Run/' + params['name_of_project_folder'] + "/Results/Analysis/")
        if check_file == False:
            os.mkdir('Run/' + params['name_of_project_folder'] + "/Results/Analysis/")
        for cat in self.astromodel.catalogs:
            df = self.anadict[cat]
            print('Results/Analysis/' + cat)
            df.to_csv('Run/' + params['name_of_project_folder'] + "/Results/Analysis/" + cat, sep='\t')

    def compute_Omega(self):
        net_list_GB = list([])
        for net in params['network_list'].keys():
            network = DET.Network(name = net)
            if network.type == 'LISA':
                print("Princess not ready for this computation")
            elif network.type == 'PTA':
                print("Princess not ready for this computation")
            else :
                net_list_GB.append(network)
            self.net_list_GB = net_list_GB
            if len(net_list_GB) > 0:
                self.Omega_pycbc()

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
            Omega_e0 = pd.read_csv('Results/Omega_e0/'+path, index_col = False, sep = '\t')
        check_file = os.path.exists('Results/Analysis/' + path)
        if check_file == False:
            output_index = ['N_source'] + ['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq] + ['SNR']
            Ana = pd.DataFrame(index=output_index, columns=['Total'] + self.Networks)
            Ana['Total']['N_source'] = 0
            Ana['Total'][['Omg_'+str(i)+'_Hz' for i in self.Omega_ana_freq]] = Search_Omg(Omega_e0['Total'], self.Omega_ana_freq)
            Ana['Total']['SNR'] = SNR.SNR_Omega(Omega_e0['Total'])
            for N in range(len(self.Networks)):
                Ana[Networks[N].name][['Omg_' + str(i) + '_Hz' for i in self.Omega_ana_freq]] = Search_Omg(Omega_e0[Networks[N].name], self.Omega_ana_freq)
                Ana[N]['SNR'] = SNR.SNR_Omega(Omega_e0[Networks[N].name],N)
                residual = df[df[N]<self.SNR_thrs[N]]
                Ana[N]['Nsource'] = len(residual[N])
                print(Ana[N])
            Ana.to_csv('Results/Analysis/'+path, sep = '\t')

    def Analysis(self) :
        for c in range(len(self.astromodel.catalogs)):
            cat = self.astromodel.catalogs[c]
            Omega_e0 = pd.read_csv('Run/' + params['name_of_project_folder'] + "/Results/Omega/" + cat, index_col=False, sep='\t')
            Ana = self.anadict[cat]
            for i in self.Omega_ana_freq :
                Ana['Total']['Omg_' + str(i) + '_Hz' ] = BF.Search_Omg(Freq = Omega_e0['f'], Omega = Omega_e0['Total'], freq_ref = i)
            for network in self.net_list_GB:
                for i in self.Omega_ana_freq:
                    Ana[network.name]['Omg_' + str(i) + '_Hz'] = BF.Search_Omg(Freq = Omega_e0['f'], Omega = Omega_e0[network.name], freq_ref = i)
#                SNRres = SNR.SNR_bkg(Omega_e0['f'], Omega_e0[Networks[N].name], Networks[N])
#                SNRtot = SNR.SNR_bkg(Omega_e0['f'],Omega_e0['Total'], Networks[N])
#                print(SNRres,' ', Networks[N].name)
                Ana[network.name]['SNR_Residual'] = SNR.SNR_bkg(Omega_e0['f'], Omega_e0[network.name], network)
                Ana[network.name]['SNR_Total'] = SNR.SNR_bkg(Omega_e0['f'],Omega_e0['Total'], network)
            self.anadict[cat] = Ana

    def Analysis_noResidual(self, Networks):
        for c in range(len(self.astromodel.catalogs)):
            cat = self.astromodel.catalogs[c]
            Omega_e0 = pd.read_csv('Run/' + params['name_of_project_folder'] + "/Results/Omega/" + cat, index_col=False, sep='\t')
            Ana = self.anadict[cat]
            for i in self.Omega_ana_freq:
                Ana['Total']['Omg_' + str(i) + '_Hz'] = BF.Search_Omg(Freq=Omega_e0['f'], Omega=Omega_e0['Total'],
                                                                      freq_ref=i)
            for N in range(len(Networks)):
                for i in self.Omega_ana_freq:
                    Ana[Networks[N].name]['Omg_' + str(i) + '_Hz'] = BF.Search_Omg(Freq=Omega_e0['f'],
                                                                                   Omega=Omega_e0['Total'],
                                                                                   freq_ref=i)
                Ana[Networks[N].name]['SNR_Total'] = SNR.SNR_bkg(Omega_e0['f'], Omega_e0['Total'], Networks[N])
            self.anadict[cat] = Ana

    def Omega(self, cat, Freq, Networks):

        df = pd.read_csv('Catalogs/'+cat, delimiter = '\t', index_col = None )
        check_file = os.path.exists('Results/Omega_e0/' + cat)
        if check_file == False:
            Omega_e0 = pd.DataFrame({'f':Freq, 'Total': np.zeros(len(Freq))})
            for n in range(len(Networks)) :
                Omega_e0[Networks[n].name] = np.zeros(len(Freq))
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

    def Omega_pycbc_old(self):
        fd_table = pd.read_csv('./AuxiliaryFiles/factor_table.dat', index_col=None, sep='\t')
        Freq_GB = np.linspace(1,2500,2500)
        waveform = params['detector_params']['types']['3G']['waveform']
        for cat in range(len(self.astromodel.catalogs)) :
            Cat = pd.read_csv('./Run/' + params['name_of_project_folder'] + '/Astro_Models/Catalogs/'+self.astromodel.catalogs[cat], delimiter='\t', index_col=None)
            Omega_e0 = pd.DataFrame({'f':Freq_GB-1., 'Total': np.zeros(len(Freq_GB))})
            Ana = self.anadict[self.astromodel.catalogs[cat]]
            Ana['Total']['N_source'] = len((Cat.z))
            for N in self.net_list_GB :
                network = N
                Omega_e0[network.name] = np.zeros(len(Freq_GB))
                Ana[network.name]['N_source'] =0
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
                Omg_e0 = GWk_no_ecc_pycbcwf(evt = event,
                                           freq = Freq_GB,
                                           approx = waveform,
                                           n = evt,
                                           size_catalogue = len(Cat.z)) * np.power(Freq_GB-1.,3.) * K.C / self.astromodel.duration
                Omega_e0['Total'] += Omg_e0
                for N in self.net_list_GB :
                    SNR = 0
                    network = N
                    for d in network.compo.keys() :
                        detector = DET.Detector(name = d)
                        conf = 'f'+str(detector.configuration)
                        if self.inclination == 'Rand':
                            fd = fd_table[conf][i[0]]
                        elif self.inclination == 'Optimal':
                            fd = 1
                        SNR+= event[d]* fd
                    # Loop through each threshold in the list of SNR thresholds
                    for snr_thr in network.SNR_thrs:
                        if event[network.name] < snr_thr:
                            col_name = network.name+'_thrs_'+str(snr_thr)
                            # Initialize the dictionary for this threshold if not already present
                            if snr_thr not in Ana[network.name]:
                                Ana[col_name] = {'N_source': 0}

                            # Increment the source count for this threshold
                            Ana[col_name]['N_source'] += 1

                            # Ensure the Omega_e0 dictionary has an entry for this threshold
                            if col_name not in Omega_e0.keys:
                                Omega_e0[col_name] = np.zeros(len(Freq_GB))

                            # Add to the Omega_e0 value for this threshold
                            Omega_e0[col_name] += Omg_e0
            Omega_e0.to_csv('Run/' + params['name_of_project_folder'] + "/Results/Omega/" + self.astromodel.catalogs[cat], index=False, sep='\t')
            print('Written : Run/' + params['name_of_project_folder'] + "/Results/Omega/" , self.astromodel.catalogs[cat])

    def Omega_pycbc(self):
        # Load factor table
        fd_table = pd.read_csv('./AuxiliaryFiles/factor_table.dat', sep='\t')
        Freq_GB = np.linspace(1, 2500, 2500)
        waveform = params['detector_params']['types']['3G']['waveform']

        # Loop through all catalogs
        for cat_path in self.astromodel.catalogs:
            # Load catalog data
            catalog_path = f'./Run/{params["name_of_project_folder"]}/Astro_Models/Catalogs/{cat_path}'
            Cat = pd.read_csv(catalog_path, delimiter='\t')

            # Initialize Omega and analysis dictionaries
            Omega_e0 = pd.DataFrame({'f': Freq_GB - 1., 'Total': np.zeros(len(Freq_GB))})
            Ana = self.anadict[cat_path]
            Ana['Total']['N_source'] = len(Cat.z)

            # Initialize network-specific data
            for network in self.net_list_GB:
                Omega_e0[network.name] = np.zeros(len(Freq_GB))
                Ana[network.name]['N_source'] = 0

            # Process each event in the catalog
            for _, event in Cat.iterrows():
                # Handle inclination
                if self.inclination == 'Rand':
                    i = np.random.randint(0, len(fd_table))
                    event['inc'] = np.arccos(fd_table.iloc[i].inc)
                elif self.inclination == 'Optimal':
                    event['inc'] = 0.0
                else:
                    raise ValueError("Invalid inclination value. Must be 'Rand' or 'Optimal'.")

                # Calculate Omega for the event
                Omg_e0 = (
                        GWk_no_ecc_pycbcwf(
                            evt=event,
                            freq=Freq_GB,
                            approx=waveform,
                            n=event.name,  # Row index
                            size_catalogue=len(Cat.z)
                        )
                        * np.power(Freq_GB - 1., 3.)
                        * K.C
                        / self.astromodel.duration
                )
                Omega_e0['Total'] += Omg_e0

                # Evaluate for each network
                for network in self.net_list_GB:
                    if network.name not in event.keys():
                        for d in network.compo.keys():
                            detector = DET.Detector(name=d)
                            conf = f'f{detector.configuration}'
                            fd = fd_table[conf][i] if self.inclination == 'Rand' else 1
                            SNR += event[d] * fd
                    else : SNR = event[network.name]

                    # Handle thresholds for the network
                    for snr_thr in network.SNR_thrs:
                        col_name = f"{network.name}_thrs_{snr_thr}"

                        if SNR < snr_thr:
                            # Initialize analysis dictionary for threshold if not present
                            Ana.setdefault(col_name, {'N_source': 0})
                            Ana[col_name]['N_source'] += 1

                            # Ensure Omega_e0 column exists for this threshold
                            if col_name not in Omega_e0:
                                Omega_e0[col_name] = np.zeros(len(Freq_GB))

                            Omega_e0[col_name] += Omg_e0

            # Save Omega results to file
            result_path = f'Run/{params["name_of_project_folder"]}/Results/Omega/{cat_path}'
            Omega_e0.to_csv(result_path, index=False, sep='\t')
            print(f'Written: {result_path}')



