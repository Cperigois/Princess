import pandas as pd
import os
import json
import stochastic.snr as SNR
import stochastic.constants as K
import numpy as np
import stochastic.basic_functions as BF
from astrotools.htild import GWk_no_ecc_pycbcwf
from astrotools.astromodel import AstroModel as AM
from astrotools.detection import Detector as DET
import astrotools.detection as detection

params = json.load(open('Run/Params.json', 'r'))

def process_background_computation():
    # Compute background and analysis
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Results"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Results")
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Results/Omega/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Results/Omega")
    for astomodel in params['astro_model_list'].keys():
        Zelda = Princess(astromodel=AM(name = astomodel))
        Zelda.Network_list()
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
        """
        Initializes the analysis output structure.
        Creates a DataFrame for each catalog, containing results for 'Total'
        and each network with its respective SNR thresholds.
        """
        output_index = ['N_source'] + [f'Omg_{i}_Hz' for i in self.Omega_ana_freq] + ['SNR_Total', 'SNR_Residual']
        self.anadict = {}

        for cat in self.astromodel.catalogs:
            columns = ['Total']

            # Add network names and their respective thresholds
            for network in self.net_list_GB:
                columns.append(network.name)
                columns.extend([f'{network.name}_thrs_{thr}' for thr in network.SNR_thrs])

            # Create DataFrame with specified index and columns
            self.anadict[cat] = pd.DataFrame(index=output_index, columns=columns)

    def Write_results(self) :
        check_file = os.path.exists('Run/' + params['name_of_project_folder'] + "/Results/Analysis/")
        if check_file == False:
            os.mkdir('Run/' + params['name_of_project_folder'] + "/Results/Analysis/")
        for cat in self.astromodel.catalogs:
            df = pd.DataFrame(self.anadict[cat])
            print('Results/Analysis/' + cat)
            df.to_csv('Run/' + params['name_of_project_folder'] + "/Results/Analysis/" + cat, sep='\t')

    def Network_list(self):
        """
                Initializes the network list.
        """
        self.net_list_GB = []

        for net in params['network_list'].keys():
            network = detection.Network(name=net)

            if network.type in ['LISA', 'PTA']:
                print("Princess not ready for this computation")
            else:
                self.net_list_GB.append(network)

    def compute_Omega(self):
        """
        Initializes the network list and computes Omega using pycbc only once if applicable.
        """

        # Ensure Omega_pycbc is only executed if there is at least one valid network
        if self.net_list_GB:
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

    def Analysis(self):
        """
        Performs the analysis for each astrophysical model catalog.
        Computes Omega at specified frequencies and SNR for each network and its thresholds.
        Updates the analysis dictionary with the results.
        """
        project_folder = params['name_of_project_folder']

        for cat in self.astromodel.catalogs:
            # Load Omega data
            omega_path = f'Run/{project_folder}/Results/Omega/{cat}'
            Omega_e0 = pd.read_csv(omega_path, index_col=False, sep='\t')

            # Retrieve the analysis dictionary for the catalog
            Ana = self.anadict[cat]

            # Compute Omega values at specified reference frequencies for 'Total'
            for freq in self.Omega_ana_freq:
                Ana['Total'][f'Omg_{freq}_Hz'] = BF.Search_Omg(
                    Freq=Omega_e0['f'], Omega=Omega_e0['Total'], freq_ref=freq
                )

            # Process each network in the analysis
            for network in self.net_list_GB:
                # Compute the total SNR for the network
                SNR_network_total = SNR.SNR_bkg(Omega_e0['f'], Omega_e0['Total'], network)

                # Process each SNR threshold for the network
                for thrs in network.SNR_thrs:
                    column_name = f'{network.name}_thrs_{thrs}'

                    # Compute Omega values at specified frequencies for each threshold
                    for freq in self.Omega_ana_freq:
                        Ana[column_name][f'Omg_{freq}_Hz'] = BF.Search_Omg(
                            Freq=Omega_e0['f'], Omega=Omega_e0[column_name], freq_ref=freq
                        )

                    # Compute residual and total SNR for the network at the threshold
                    Ana[column_name]['SNR_Residual'] = SNR.SNR_bkg(Omega_e0['f'], Omega_e0[column_name], network)
                    Ana[column_name]['SNR_Total'] = SNR_network_total

            # Save the updated analysis dictionary
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


    def Omega_pycbc(self):
        """
        Computes the Omega function using the pycbc waveform model for all events in the AstroModel catalogs.
        Saves the results in the appropriate results folder.
        """
        # Load factor table
        fd_table = pd.read_csv('./AuxiliaryFiles/factor_table.dat', sep='\t')

        # Define frequency range and waveform type
        Freq_GB = np.linspace(1, 2500, 2500)
        waveform = params['detector_params']['types']['3G']['waveform']
        project_folder = params["name_of_project_folder"]

        # Load all networks once
        networks = {network.name: network for network in self.net_list_GB}

        # Process each catalog
        for cat_path in self.astromodel.catalogs:
            catalog_path = f'./Run/{project_folder}/Astro_Models/Catalogs/{cat_path}'
            Cat = pd.read_csv(catalog_path, delimiter='\t')

            # Ensure that self.anadict[cat_path] is a dictionary
            if cat_path not in self.anadict or not isinstance(self.anadict[cat_path], dict):
                self.anadict[cat_path] = {'Total': {'N_source': len(Cat.z)}}
            Ana = self.anadict[cat_path]

            # Initialize Omega dataframe
            Omega_e0 = pd.DataFrame({'f': Freq_GB - 1.})
            Omega_e0['Total'] = np.zeros_like(Freq_GB)

            # Initialize Omega and source counts for each network and its thresholds
            for network_name, network in networks.items():
#                network = NET.load(name = network_name)
                for snr_thr in network.SNR_thrs:
                    threshold_col = f"{network.name}_thrs_{snr_thr}"
                    Omega_e0[threshold_col] = np.zeros_like(Freq_GB)
                    Ana.setdefault(threshold_col, {}).update({'N_source': 0})

            # Process each event in the catalog
            for _, event in Cat.iterrows():
                # Set inclination
                if self.inclination == 'Rand':
                    i = np.random.randint(0, len(fd_table))
                    event['inc'] = np.arccos(fd_table.iloc[i]['inc'])
                elif self.inclination == 'Optimal':
                    event['inc'] = 0.0
                else:
                    raise ValueError("Invalid inclination value. Must be 'Rand' or 'Optimal'.")

                # Compute Omega for the event
                Omg_e0 = (
                        GWk_no_ecc_pycbcwf(
                            evt=event,
                            freq=Freq_GB,
                            approx=waveform,
                            n=event.name,
                            size_catalogue=len(Cat.z)
                        )
                        * np.power(Freq_GB - 1., 3.)
                        * K.C
                        / self.astromodel.duration
                )
                Omega_e0['Total'] += Omg_e0

                # Compute SNR for each network
                for network_name, network in networks.items():
                    SNR = 0  # Initialize SNR

                    if network_name not in event:
                        for d in network.compo.keys():
                            detector = DET.load(d)
                            conf = detector.configuration
                            fd = fd_table.iloc[i][conf] if self.inclination == 'Rand' else 1
                            SNR += event[d] * fd
                    else:
                        SNR = event[network_name]

                    # Check threshold levels
                    for snr_thr in network.SNR_thrs:
                        col_name = f"{network_name}_thrs_{snr_thr}"

                        if SNR < snr_thr:
                            Ana.setdefault(col_name, {}).update(
                                {'N_source': Ana.get(col_name, {}).get('N_source', 0) + 1})

                            if col_name not in Omega_e0:
                                Omega_e0[col_name] = np.zeros_like(Freq_GB)

                            Omega_e0[col_name] += Omg_e0

            # Save results
            result_path = f'Run/{project_folder}/Results/Omega/{cat_path}'
            Omega_e0.to_csv(result_path, index=False, sep='\t')
            print(f'Written: {result_path}')





