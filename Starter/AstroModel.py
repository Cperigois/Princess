import math
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from Stochastic import Basic_Functions as BF
from Starter.Htild import GWk_noEcc_Pycbcwf

class Astromodel:

    def __init__(self, cat_name = None, duration = 1,  original_cat_path = None, cat_sep = None, index_column = None, flags ={'':''}):
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
        self.original_cat_path = original_cat_path
        self.cat_name = cat_name
        self.duration = duration
        self.sep_cat = cat_sep
        self.index_column = index_column
        self.flags = flags
        self.catalogs = []
        for x in flags.keys() :
            if x == '':
                self.catalogs.append(self.cat_name + '.dat')
            else :
                self.catalogs.append(self.cat_name + '_' + flags[x] + '.dat')


    def makeHeader(self, header):
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
             flag: ID for the binary type/formation channel or more.
        """
        Cat = pd.read_csv(self.original_cat_path, sep = self.sep_cat, index_col = self.index_column, names = header)
        Cat.to_csv(self.original_cat_path, sep = self.sep_cat, index = None, header = True)

    def makeCat(self, flags = {}):
        """Create the catalogue(s).
        Parameters
        ----------
        flag: dict
            Dictionary of the flag ids in the original catalogue
        spin_opt

        """

        Cat = pd.read_csv(self.original_cat_path, sep = self.sep_cat, index_col = self.index_column, dtype = float)
        print(Cat.describe())
        OutCat = pd.DataFrame()
        Col = list(Cat.columns)
        OutCat['zm'] = Cat['zm']

        # Check the masses calculations
        if 'Mc' not in Col:
            OutCat['Mc'], OutCat['q'] = BF.m1_m2_to_mc_q(Cat['m1'], Cat['m2'])
        if 'm1' not in Col:
            OutCat['m1'], OutCat['m2'] = BF.mc_q_to_m1_m2(Cat['Mc'], Cat['q'])
        # Compute luminosity distance
        if 'Dl' not in Col:
            zm = np.array(Cat['zm'], float)
            dl = np.array([])
            for z in zm:
                dl = np.append(dl, Planck15.luminosity_distance(z).value)
            OutCat['Dl'] = dl
        # Generate the spin
        OutCat['s1'], OutCat['s2'] = self.makeSpin(GS.spin_option, len(Cat['zm']))
        if GS.orbit_evo == True:
            OutCat['a0'] = Cat['a0']
            OutCat['e0'] = Cat['e0']

        if GS.IncAndPos == True:
            if 'inc' not in Col:
                OutCat['inc'] = np.random.uniform(0,2*math.pi, len(OutCat['m1']))
            else :
                OutCat['inc'] = Cat['inc']
            if 'ra' not in Col:
                OutCat['ra'] = np.random.uniform(0,2*math.pi, len(OutCat['m1']))
            else :
                OutCat['ra'] = Cat['ra']
            if 'dec' not in Col:
                OutCat['dec'] = np.random.uniform(0,2*math.pi, len(OutCat['m1']))
            else :
                OutCat['dec'] = Cat['dec']
        if flags != {} :
            for key in flags.keys():
                print(key)
                flagCat = OutCat[Cat['flag'] == int(key)]
                flagCat.to_csv('Catalogs/' + self.cat_name + '_' + flags[key] + '.dat', sep='\t', index=False)
                truc = flagCat.describe()
                truc.to_csv('Catalogs/Ana_' + self.cat_name + '_' + flags[key] + '.dat', sep='\t')
        else :
            OutCat.to_csv('Catalogs/' + self.cat_name + '.dat', sep='\t', index=False)


    def makeSpin(self, option, size):

        # Available models : zeros, Maxwellian (in prep.), Maxwellian_dynamics (in prep.)
        if option == 'Zero':
            s1 = np.zeros(size)
            s2 = np.zeros(size)
        return s1, s2


    def compute_SNR(self, Networks, freq, approx, real = True, ):
        flow = int(np.min(freq))
        fsize = flow+len(freq)
        for cat in self.catalogs :
            Cat = pd.read_csv('Catalogs/'+cat, sep='\t', index_col=False)
            print('SNR calculation for ', cat)
            ntot = len(Cat.zm)
            for N in Networks:
                SNR_N = np.zeros(len(Cat.zm))
                psd_compo = np.empty((len(N.compo), len(freq)+1+flow))
                for d in range(len(N.compo)):
                    psd_compo[d] = N.compo[d].Make_psd()
                for evt in range(len(Cat.zm)):
                    event = Cat.iloc[[evt]]
                    htildsq = GWk_noEcc_Pycbcwf(event, freq = freq, approx=approx,n = evt, ntot = ntot)
                    for d in range(len(N.compo)) :
                        Sn = psd_compo[d]
                        SNR = np.sum(4.*htildsq/Sn[flow:fsize])
                        SNR_N[evt]+= SNR
                Cat['snr_'+N.net_name+'_opt'] = np.sqrt(SNR_N)
                if real == True :
                    SNR_N = np.zeros(len(Cat.zm))
                    
                    for evt in range(len(Cat.zm)):

            Cat.to_csv(cat, sep='\t', index=False)

        def compute_SNR(self, Networks, freq, approx):
            flow = int(np.min(freq))
            fsize = flow + len(freq)
            for cat in self.catalogs:
                Cat = pd.read_csv('Catalogs/' + cat, sep='\t', index_col=False)
                print('SNR calculation for ', cat)
                ntot = len(Cat.zm)
                for N in Networks:
                    SNR_N = np.zeros(len(Cat.zm))
                    psd_compo = np.empty((len(N.compo), len(freq) + 1 + flow))
                    for d in range(len(N.compo)):
                        psd_compo[d] = N.compo[d].Make_psd()
                    for evt in range(len(Cat.zm)):
                        event = Cat.iloc[[evt]]
                        htildsq = GWk_noEcc_Pycbcwf(event, freq=freq, approx=approx, n=evt, ntot=ntot)
                        for d in range(len(N.compo)):
                            Sn = psd_compo[d]
                            SNR = np.sum(4. * htildsq / Sn[flow:fsize])
                            SNR_N[evt] += SNR
                    Cat[N.net_name] = np.sqrt(SNR_N)
                Cat.to_csv(cat, sep='\t', index=False)









