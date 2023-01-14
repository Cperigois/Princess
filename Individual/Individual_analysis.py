from Starter.AstroModel import *
from Starter.Detection import *
import Useful_Functions as UF


class Individual_analysis:

    def __init__(self, params = {'m1' : [0, 100, 100],'q' : [0, 1, 20],'zm': [0, 5, 30]}, iteration, Network = ['HLV'], binary_type = 'BBH',  pastro_thrs = 0, SNR_thrs = 0, FAR_thrs = 2  ):
        """Gather all pqrqmeters of your study
         Parameters
         ----------
        params : dictionnary
            Parameters the user want to compare and the range [minimum, maximum, bins]. Default are : {'m1': [0, 100, 100], 'q': [0, 1, 20], 'zm': [0, 5, 30]}
        iteration : int
            Number of iteration to extract the errors on observations
        Network : Network from class Network
            Network to use in the study
        binary_type : str
            Type of binaries to be compared. Default is 'BBH'.
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
        self.param_list = params.keys()

        #Read Data from LVK
        data_list = pd.read_csv('AuxiliaryFiles/LVK_data/GW_list.txt')
        list_events = data_list[(data_list['binary']==binary_type) & (data_list['network_matched_filter_snr']>SNR_thrs) & (data_list['far']<FAR_thrs) & (data_list['pastro']>pastro_thrs) ]
        posteriors = dict({})
        for p in param_list :
            opts = params[p]
            posteriors[p] = dict({'bins': UF.newbin_lin(np.linspace(opts[0],opts[1],opts[2])), 'values': np.zeros(opts[2])})
        for evt in list_events.commonName :
            posterior = pd.read_csv('AuxiliaryFiles/LVC_data/Posterior/' + evt + '_post.dat',
                sep='\t', index_col=None)
            for p in param_list:
                hist, bini, patch = plt.hist(posterior[p], bins=output[bins], density=True)
                posteriors[p]['values']+= hist
        for p in param_list:
            df = pd.DataFrame(posteriors[p])
            df.to_csv('Results/'+Network.net_name+'_'+binary_type+'_'+p+'.txt')


    def SNR_real(self, Model, Network, stat) :
        """This function build the real SNR of each source of the calogue. Can also update the catalogue file
                ----------
                Model : AstroModel
                    Built from the Princess.Starter.AstroModel class.
                parameters : list
                    List of parameters to be analyzed, default ['m1', 'q', 'z']
                Network : Network
                    Built from the Princess.Starter.Network class
                stats : int
                    Number of iteration for the uncertainties
                Returns
                -------
                SNR_ real : np.array
                    array containing the real SNR for esach source of the input catalogue                """

        factors = pd.read_csv('AuxiliaryFiles/factor_table.dat', sep = '\t', index_col = None)
        size_factor = len(factors[factors.columns[0]])
        print(size_factor)

        for C in Model.catalogs:
            ndet = np.array([])
            m1_distrib  = pd.DataFrame([])
            q_distrib = pd.DataFrame([])
            z_distrib = pd.DataFrame([])
            cat = pd.read_csv('../Catalogs/' + C, sep='\t', index_col=None)
            size_cat = len(cat[cat.columns[0]])

            for n in range(stat) :
                fdet = factors.iloc[np.random.randint(size_factor, size_cat)]
                cat['SNR_'+Network.net_name] = np.zeros(size_cat)
                for d in Network.compo :
                    cat['SNR_'+Network.net_name] += cat['SNR_'+d]*fdet['f'+d]
                cat['SNR_'+Network.net_name] = np.sqrt(cat['SNR_'+Network.net_name])
                det = cat[cat['SNR_'+Network.net_name]> Network.SNR_thrs]
                m1_distrib[str(n)], bins, patches = plt.hist(det['m1'], bins = np.linspace(params.m1[0],params.m1[1],params.m1[2]))
                q_distrib[str(n)], bins, patches = plt.hist(det['m2']/det['m1'], bins = np.linspace(params.q[0],params.q[1],params.q[2]))
                z_distrib[str(n)], bins, patches = plt.hist(det['zm'], bins = np.linspace(params.zm[0],params.zm[1],params.zm[2]))

    cat.to_csv('Catalogs/' + Model.cat_name + '.dat', sep='\t', index=False)

    def histos_obs(self, Model, Network, stat) :
        """This function extract expected observed parameters distribution from CBCs catalogues, and write it in files.
                ----------
                Model : AstroModel
                    Built from the Princess.Starter.AstroModel class.
                parameters : list
                    List of parameters to be analyzed, default ['m1', 'q', 'z']
                Network : Network
                    Built from the Princess.Starter.Network class
                stats : int
                    Number of iteration for the uncertainties
                Returns
                -------
                SNR_ real : np.array
                    array containing the real SNR for esach source of the input catalogue                """

    def Read_posteriors(self):
        """This function built the sum of posteriors fron LVK data, with the user constrains on the detectability, and write them in proper files.
                ----------

                Returns
                -------
        """

    def Full_Analysis(self):
