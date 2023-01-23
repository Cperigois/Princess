import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Starter.AstroModel import Astromodel
from Starter.Detection import Network, Detector
import Useful_Functions as UF


class IndividualAnalysis:

    def __init__(self, iteration = 100, network = 'HLV', pastro_thrs = 0, SNR_thrs = 0, FAR_thrs = 2 , binary_type = ['BBH', 'BNS', 'NSBH'], params = {'m1' : [0, 100, 100],'q' : [0, 1, 20],'zm': [0, 5, 30]}):
        """Gather all parameters of your study
         Parameters
         ----------
        params : dictionnary
            Parameters the user want to compare and the range [minimum, maximum, bins]. Default are : {'m1': [0, 100, 100], 'q': [0, 1, 20], 'zm': [0, 5, 30]}
        iteration : int
            Number of iteration to extract the errors on observations
        Network : Network from class Network
            Network to use in the study
        binary_type : list of str
            Type of binaries to be compared. Default is ['BBH', 'BNS', 'NSBH'].
        pastro_thrs : float
            Threshold to select sources from the data for the comparison. Default is 0.
        SNR_thrs : float
            Threshold to select sources from the data for the comparison. Default is 0.
        FAR_thrs :float
            Threshold to select sources from the data for the comparison. Default is 2.
         """

        # Set class variables
        self.params = params
        self.Network = network
        self.iteration = iteration
        self.pastro_thrs = pastro_thrs
        self.SNR_thrs = SNR_thrs
        self.FAR_thrs = FAR_thrs
        self.binary_type = binary_type
        self.param_list = params.keys()

        #Read Data from LVK
        data_list = pd.read_csv('AuxiliaryFiles/LVK_data/GW_list.csv', dtype = {'id' : str, 'commonName' : str, 'binary' : str, 'run' : str, 'version' : str,'catalog.shortName' : str, 'GPS' : float,'reference' : str, 'jsonurl' : str, 'mass_1_source' : float, 'mass_1_source_lower' : float, 'mass_1_source_upper' : float, 'mass_2_source' : float, 'mass_2_source_lower' : float,'mass_2_source_upper' : float, 'network_matched_filter_snr' : float, 'network_matched_filter_snr_lower' : float, 'network_matched_filter_snr_upper' : float, 'luminosity_distance' : float, 'luminosity_distance_lower' : float, 'luminosity_distance_upper' : float, 'chi_eff,chi_eff_lower' : float, 'chi_eff_upper' : float, 'total_mass_source' : float, 'total_mass_source_lower' : float,'total_mass_source_upper' : float, 'chirp_mass_source' : float, 'chirp_mass_source_lower' : float, 'chirp_mass_source_upper' : float, 'chirp_mass' : float, 'chirp_mass_lower' : float, 'chirp_mass_upper' : float,'redshift' : float, 'redshift_lower' : float, 'redshift_upper' : float, 'far' : float, 'far_lower' : float, 'far_upper' : float, 'p_astro' : float, 'p_astro_lower' : float, 'p_astro_upper' : float, 'final_mass_source' : float,'final_mass_source_lower' : float, 'final_mass_source_upper' : float})

        for b in self.binary_type :
            list_events = data_list[(data_list['binary'] == b) & (data_list['network_matched_filter_snr'] > SNR_thrs) & (
                            data_list['far'] < FAR_thrs) & (data_list['pastro'] > pastro_thrs)]
            posteriors = dict({})
            for p in self.param_list :
                opts = self.params[p]
                posteriors[p] = dict({'bins': UF.newbin_lin(np.linspace(opts[0],opts[1],opts[2])), 'values': np.zeros(opts[2])})
            for evt in list_events.commonName :
                posterior = pd.read_csv('AuxiliaryFiles/LVC_data/Posterior/' + evt + '_post.dat', sep='\t', index_col=None)
                for p in self.param_list:
                    hist, bini, patch = plt.hist(posterior[p], bins=posteriors[p]['bins'], density=True)
                    posteriors[p]['values']+= hist
            for p in self.param_list:
                df = pd.DataFrame(posteriors[p])
                df.to_csv('Results/'+self.Network.net_name+'_'+b+'_'+p+'.txt')

    def SNR_real(self, Catalogue_path, update_file = False) :
        """This function build the real SNR of each source of the calogue. Can also update the catalogue file
                ----------
                Model : AstroModel
                    Built from the Princess.Starter.AstroModel class.
                update_files : bool
                    If True update the catalogue file with and additionnal column containing the SNR. The new column will be named as the Network.
                Returns
                -------
                SNR_ real : np.array
                    array containing the real SNR for each source of the input catalogue                """

        factors = pd.read_csv('AuxiliaryFiles/factor_table.dat', sep = '\t', index_col = None)
        size_factor = len(factors[factors.columns[0]])
        cat = pd.read_csv('../Catalogs/' + Catalogue_path, sep='\t', index_col=None)
        size_cat = len(cat[cat.keys[0]])
        fdet = factors.iloc[np.random.randint(size_factor, size_cat)]
        SNR = np.zeros(size_cat)
        for d in self.Network.compo:
            SNR += np.power(cat['SNR_' + d] * fdet['f' + d],2.)
        if update_file == True :
            cat['SNR_' + self.Network.net_name] = np.sqrt(SNR)
            cat.to_csv('Catalogs/' + Catalogue_path, sep='\t', index=False)

        return np.sqrt(SNR)


    def Full_Analysis(self, update_file = False) :
        """This function does the full analysis for all models
                ----------
                Model : AstroModel
                    Built from the Princess.Starter.AstroModel class.
                update_files : bool
                    If True update the catalogue file with and additionnal column containing the SNR. The new column will be named as the Network.
        """

        output = dict({})
        for p in self.param_list:
            output[p] = pd.Dataframe({'bins': UF.newbin_lin(np.linspace(self.params[p][0],self.params[p][1],self.params[p][2]))})
        for C in self.Model.catalogs:
            cat = pd.read_csv('../Catalogs/' + C, sep='\t', index_col=None)
            name = C[: len(C) - 4]
            for i in self.iteration :
                SNR = self.SNR_real(C, update_file)
                det_bool = pd.Series(SNR>self.Network.SNR_thrs)
                det = cat[det_bool.values]
                ndet = len(det_bool)
                for p in self.param_list:
                    output[p][str(i)], bini, patch = plt.hist(det[p], bins=output[p]['bins'], density=True)
            for p in self.params :
                data = output[p]
                df = pd.DataFrame({'bins': output[p]['bins'],
                                   'mean': np.mean(data, axis =1),
                                   'std': np.std(data, axis=1),
                                   '5%': data.quantile(0.05, axis =1),
                                   '95%': data.quantile(0.95, axis =1) })
                df.to_csv('Results/Individual/Ana_'+ name + '_'+ p +'.dat')
            f = open('Results/Individual/Ana_'+ name + '_ndet.dat', "w")
            f.write("Mean number of detections : {0} \nStandard deviation : {1}\nMinimum of detections: {2} \nMaximum detections : {3} ".format(np.mean(ndet), np.std(ndet), np.min(ndet), np.max(ndet)))
            f.close()




