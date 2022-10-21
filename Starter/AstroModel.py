import math
import numpy as np
from scipy.integrate import quad

class Astromodel:

    def __init__(self, cat_name = None, duration = 1,  original_cat_path = None, sep_cat = None, index_column = None):
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
         """

        # Set class variables
        self.original_cat_path = original_cat_path
        self.cat_name = cat_name
        self.duration = duration
        self.sep_cat = sep_cat
        self.index_column = index_column

    def make_header(self, header):
        """Create or modify the header of your original catalogue. Careful this function WILL MODIFY your original file.
        Parameters
        ----------
        header : list of str
             Names of your columns in the original file. Some names are important and should be labeled with specific names.

             zm: Redshift of merger
             m1,m2: Mass of the compact objects in solar masses
             Mc: Chirp mass in solar masses
             q: Mass ratio (q<=1)
             e: Eccentricity at the formation of the first compact object
             Dl: Luminosity distance in Mpc
             inc: Inclination of the source in rad
             Xeff: Effective spin of the binary
             chi1, chi2: Individual spins of the two CO.
             theta1,theta2: Angle between individual spins and the angular momentum of the binary
        """
        Cat = pd.read_csv(self.original_cat_path, sep = self.sep_cat, index_col = self.index_column, columns = header)
        Cat.to_csv(self.original_cat_path, sep = self.sep_cat, index_col = self.index_column, header = True)

    def makekCat(self):
        Cat= pd.read_csv(self.original_cat_path, sep = self.sep_cat, index_col = self.index_column)
        Col = list(Cat.columns)
        if 'zm' not in Col:
            Cat['zm'] = Cat['z']
        if 'Mc' not in Col:
            Cat['Mc'], Cat['q'] = BF.m1_m2_to_mc_q(Cata['m1'], Cata['m2'])
        if 'm1' not in Col:
            Cat['m1'], Cat['m2'] = BF.mc_q_to_m1_m2(Cata['Mc'], Cata['q'])
        if 'Xeff' not in Col:
            if model[1] == 'BBH':
                Cata['Xsi'] = BF.Xsi_compil(Cata['m1'], Cata['m2'], Cata['theta1'], Cata['theta2'], Cata['chi1'],
                                            Cata['chi2'])  # Add differend spin model
            else:
                Cata['Xsi'] = np.zeros(len(Cata['m1']))
        if 'Dl' not in Col:
            zm = np.array(Cata['zm'], float)
            dl = np.array([])
            for z in zm:
                dl = np.append(dl, cosmo.Planck15.luminosity_distance(z).value)
            Cata['Dl'] = dl
        if GS.orbit_evo == False:
            OutCat = pd.DataFrame(
                {'Mc': Cata['Mc'], 'q': Cata['q'], 'Xsi': Cata['Xsi'], 'zm': Cata['zm']})
        else:
            OutCat = pd.DataFrame(
                {'m1': Cata['m1'], 'm2': Cata['m2'], 'spinz1': Cata['spinz1'], 'zm': Cata['zm'],
                 'spinz2': Cata['spinz2'], 'a0': Cata['a0'],
                 'e0': Cata['e0']})
        if 'inc' in Col:
            OutCat['inc'] = Cata['inc']
        print(Cata.describe())
        fileexist = os.path.exists(outname + '3G_SNRs.dat')
        if fileexist == False:
            SNRfile = SNR.SNRwf(Cata, outname)
        else:
            SNRfile = outname + '3G_SNRs.dat'
        print('ici')
        for N in GS.Networks:
            # OutCat[N] = SNR.Cat(Cata, Net=N)
            OutCat[N] = SNR.SNR_Net_wf(SNRfile, Net=N)
        OutCat.to_csv('Catalogs/' + outname, sep='\t', index=False)
        truc = OutCat.describe()
        truc.to_csv('Catalogs/Ana_' + outname, sep='\t')

    def makecat(self):
