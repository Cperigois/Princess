import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import Useful_Functions as UF


class IndividualAnalysis:

    def __init__(self, name = 'AnaTest', iteration = 100, Network = 'HLV', pastro_thrs = 0.85, SNR_thrs = 0, FAR_thrs = 2 , binary_type = ['BBH', 'BNS', 'NSBH'], params = {'m1' : [0, 100, 100],'q' : [0, 1, 20],'zm': [0, 5, 30]}):
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
        self.Network = Network
        self.iteration = iteration
        self.pastro_thrs = pastro_thrs
        self.SNR_thrs = SNR_thrs
        self.FAR_thrs = FAR_thrs
        self.binary_type = binary_type
        self.param_list = list(params.keys())
        self.name = name


        if os.path.exists('Results/'+name) ==False :
            os.mkdir('Results/'+name)
        if os.path.exists('Results/'+self.name+'/'+self.Network.net_name+'_'+self.binary_type[0]+'_'+self.param_list[0]+'.txt') == False :
            #Read Data from LVK
            data_list = pd.read_csv('AuxiliaryFiles/LVK_data/GW_list.csv', dtype = {'id' : str, 'commonName' : str, 'binary' : str, 'run' : str, 'version' : str,'catalog.shortName' : str, 'GPS' : float,'reference' : str, 'jsonurl' : str, 'mass_1_source' : float, 'mass_1_source_lower' : float, 'mass_1_source_upper' : float, 'mass_2_source' : float, 'mass_2_source_lower' : float,'mass_2_source_upper' : float, 'network_matched_filter_snr' : float, 'network_matched_filter_snr_lower' : float, 'network_matched_filter_snr_upper' : float, 'luminosity_distance' : float, 'luminosity_distance_lower' : float, 'luminosity_distance_upper' : float, 'chi_eff,chi_eff_lower' : float, 'chi_eff_upper' : float, 'total_mass_source' : float, 'total_mass_source_lower' : float,'total_mass_source_upper' : float, 'chirp_mass_source' : float, 'chirp_mass_source_lower' : float, 'chirp_mass_source_upper' : float, 'chirp_mass' : float, 'chirp_mass_lower' : float, 'chirp_mass_upper' : float,'redshift' : float, 'redshift_lower' : float, 'redshift_upper' : float, 'far' : float, 'far_lower' : float, 'far_upper' : float, 'p_astro' : float, 'p_astro_lower' : float, 'p_astro_upper' : float, 'final_mass_source' : float,'final_mass_source_lower' : float, 'final_mass_source_upper' : float})

            for b in self.binary_type :
                list_events = data_list[(data_list['binary'] == b) & (data_list['network_matched_filter_snr'] > SNR_thrs) & (
                                data_list['far'] < FAR_thrs) & (data_list['p_astro'] > pastro_thrs)]
                posteriors = dict({})
                for p in self.param_list :
                    opts = self.params[p]
                    posteriors[p] = dict({'bins': UF.newbin_lin(np.linspace(opts[0],opts[1],opts[2])), 'values': np.zeros(opts[2]-1)})
                    #print(np.linspace(opts[0],opts[1],opts[2]))
                    #print(UF.newbin_lin(np.linspace(opts[0],opts[1],opts[2])))
                for evt in list_events.commonName :
                    posterior = pd.read_csv('AuxiliaryFiles/LVK_data/Posterior/' + evt + '_post.dat', sep='\t', index_col=None)
                    for p in self.param_list:
                        hist, bini, patch = plt.hist(posterior[p], bins=np.linspace(self.params[p][0],self.params[p][1],self.params[p][2]), density=True)
                        #print(len(hist), ' ', len(bini), ' ', len(posteriors[p]['bins']), len(posteriors[p]['values']))
                        posteriors[p]['values']+= hist
                for p in self.param_list:
                    df = pd.DataFrame(posteriors[p])
                    df.to_csv('Results/'+name+'/'+self.Network.net_name+'_'+b+'_'+p+'.txt', sep = '\t', index = None)

    def SNR_real(self, Catalogue_path, update_file = False) :
        """This function build the real SNR of each source of the catalogue. Can also update the catalogue file
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
        cat = pd.read_csv('Catalogs/' + Catalogue_path, sep='\t', index_col=None)
        size_cat = len(cat[list(cat.columns)[0]])
        fdet = factors.iloc[np.random.randint(0, high=size_factor, size = size_cat)]
        SNR = np.zeros(size_cat)
        for d in self.Network.compo:
            SNR += np.power(np.multiply(cat[d.det_name] , fdet['f' + d.configuration].values),2.)
        if update_file == True :
            cat['SNR_real_' + self.Network.net_name] = np.sqrt(SNR)
            cat.to_csv('Catalogs/' + Catalogue_path, sep='\t', index=False)

        return np.sqrt(SNR)


    def Full_Analysis(self, Model, update_file = False) :
        """This function does the full analysis for all models
                ----------
                Model : AstroModel
                    Built from the Princess.Starter.AstroModel class.
                update_files : bool
                    If True update the catalogue file with and additionnal column containing the SNR. The new column will be named as the Network.
        """

        output = dict({})
        ndet = np.zeros(self.iteration)
        for p in self.param_list:
            output[p] = pd.DataFrame({'bins': UF.newbin_lin(np.linspace(self.params[p][0],self.params[p][1],self.params[p][2]))})
        for C in Model.catalogs:
            cat = pd.read_csv('Catalogs/' + C, sep='\t', index_col=None)
            name_cat = C[: len(C) - 4]
            for i in range(self.iteration) :
                SNR = self.SNR_real(C, update_file)
                det_bool = pd.Series(SNR>self.Network.SNR_thrs)
                det = cat[det_bool.values]
                ndet[i] = len(det[self.param_list[0]])
                for p in self.param_list:
                    output[p][str(i)], bini, patch = plt.hist(det[p], bins=np.linspace(self.params[p][0],self.params[p][1],self.params[p][2]), density=True)
                    plt.close()
                    test = pd.DataFrame(output[p])
                    #print(test.describe())
                    test.to_csv('Results/'+self.name+'/test.dat', sep='\t', index=False)
            for p in self.params :
                data = output[p].drop(columns=['bins'])
                df = pd.DataFrame({'bins': output[p]['bins'],
                                   'mean': np.mean(data, axis =1),
                                   'std': np.std(data, axis=1),
                                   '5%': data.quantile(0.05, axis =1),
                                   '95%': data.quantile(0.95, axis =1) })
                df.to_csv('Results/'+self.name+'/Ana_'+ name_cat + '_'+ p +'.dat', sep = '\t', index = None)
            f = open('Results/'+self.name+'/Ana_'+ name_cat + '_ndet.dat', "w")
            f.write("Mean number of detections : {0} \nStandard deviation : {1} \nMinimum of detections : {2} \nMaximum detections : {3} ".format(np.mean(ndet), np.std(ndet), np.min(ndet), np.max(ndet)))
            f.close()

    def show_off(self):
        for b in self.binary_type:
            for p in self.param_list :
                det = pd.read_csv('Results/' + name + '/' + self.Network.net_name + '_' + b + '_' + p + '.txt', sep = '\t', index = None)
                ana = pd.read_csv('Results/'+self.name+'/Ana_'+ name_cat + '_'+p+'.dat', sep = '\t', index = None)

                plt.fill_between(det.bins, 0, det['values'], color = 'grey', label = 'LVK events', alpha = 0.4)
                plt.plot(ana.bins, ana.mean, color = 'crimson', linewidth = 3, label = 'Prediction')
                plt.legend(fontsize = 12)
                plt.fill_between(ana.bins, ana.mean-ana.std, ana.mean+ana.std, color = 'crimson', label = 'LVK events', alpha = 0.4)








