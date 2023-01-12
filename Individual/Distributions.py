from Starter.AstroModel import *
from Starter.Detection import *


class Individual_analysis:

    def __init__(self, params = {'m1' : [0, 100, 100],'q' : [0, 1, 20],'zm': [0,5,30]}):
        """Create an instance of your model.
         Parameters
         ----------
         cat_name : str
             Name of the reshaped catalogue in the folder catalogue
         duration : int of float
             Duration of your supposed catalogue, ie your initial catalogue is showing all sources arriving on the detector in X years.
             (default duration = 1)
         original_cat_path : str
             Path to your original catalogue
         sep_cat: str
            Used to read your original catalogue with pandas " ", "," or "\t"
         index_column: bool
            Used to read your original catalogue with pandas. True if you have a column with indexes in your original file.
            (default = None)
         flags: dict
            Dictionary of a possible flag column in your catalogue (Can be used to distinguish the type of binary, the formation channel...)
         """

        # Set class variables
        self.params = params

    def SNR_real(self, Model, Network, stat) :
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
                """

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

