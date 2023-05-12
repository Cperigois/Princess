import math
import os
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from Stochastic import Basic_Functions as BF
from Starter.Htild import GWk_noEcc_Pycbcwf
from scipy.interpolate import InterpolatedUnivariateSpline

class AstroModel:

    def __init__(self, cat_name = None, duration = 1,  original_cat_path = None, cat_sep = None, index_column = None, flags ={}, spin_option = "Zeros", orbit_evolution = False, inclination_position = False):
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
         spin_option : str
            Choose an option to eventually generate the spin among {"Zeros", "Isotropic", "Dynamic"}. Default is 'Zeros'
         flags: dict
            Dictionary of a possible flag column in your catalogue (Can be used to distinguish the type of binary, the formation channel...)
         """

        # Set class variables
        self.original_cat_path = original_cat_path
        self.cat_name = cat_name
        self.duration = duration
        self.sep_cat = cat_sep
        self.index_column = index_column
        self.spin_option = spin_option
        self.flags = flags
        self.orbit_evolution = orbit_evolution
        self.inclination_position =inclination_position
        self.catalogs = []
        print(len(flags.keys()))
        if len(flags.keys())==0 :
            self.catalogs = [self.cat_name + '.dat']
        else :
            for x in flags.keys() :
                self.catalogs.append(self.cat_name + '_' + flags[x] + '.dat')


    def makeHeader(self, header):
        """Create or modify the header of your original catalogue. Careful this function WILL MODIFY your original file.
        Parameters
        ----------
        header : list of str
             Names of your columns in the original file. Some names are important and should be labeled with specific names.

             z: Redshift of merger
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

    def makeCat(self):
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
        OutCat['z'] = Cat['z']

        # Check the masses calculations
        if 'Mc' not in Col:
            OutCat['m1'] = Cat['m1']
            OutCat['m2'] = Cat['m2']
            OutCat['Mc'], OutCat['q'] = BF.m1_m2_to_mc_q(Cat['m1'], Cat['m2'])
        elif 'm1' not in Col:
            OutCat['Mc'] = Cat['Mc']
            OutCat['q'] = Cat['q']
            OutCat['m1'], OutCat['m2'] = BF.mc_q_to_m1_m2(Cat['Mc'], Cat['q'])
        else:
            OutCat['Mc'] = Cat['Mc']
            OutCat['q'] = Cat['q']
            OutCat['m1'] = Cat['m1']
            OutCat['m2'] = Cat['m2']
        # Compute luminosity distance
        if 'Dl' not in Col:
            zm = np.array(Cat['z'], float)
            dl = np.array([])
            for z in zm:
                dl = np.append(dl, Planck15.luminosity_distance(z).value)
            OutCat['Dl'] = dl
        else :
            OutCat['Dl'] = Cat['Dl']
        # Generate the spin

        if self.spin_option== 'In_Catalogue' :
            OutCat['s1'] = Cat['chi1']
            OutCat['s2'] = Cat['chi2']
        elif self.spin_option == 'Chi&Theta' :
            OutCat['s1'] = Cat['chi1']*np.cos(Cat['theta1'])
            OutCat['s2'] = Cat['chi2']*np.cos(Cat['theta2'])
        else:
            OutCat['s1'], OutCat['s2'] = self.makeSpin(self.spin_option, len(Cat['z']))

        if self.orbit_evolution == True:
            OutCat['a0'] = Cat['a0']
            OutCat['e0'] = Cat['e0']

        if self.inclination_position == True :
            if 'inc' not in Col:
                OutCat['inc'] = np.random.uniform(0, math.pi, len(OutCat['m1']))
            else :
                OutCat['inc'] = Cat['inc']
            if 'ra' not in Col:
                OutCat['ra'] = np.random.uniform(0, 2*math.pi, len(OutCat['m1']))
            else :
                OutCat['ra'] = Cat['ra']
            if 'dec' not in Col:
                OutCat['dec'] = np.random.uniform(0,2*math.pi, len(OutCat['m1']))
            else :
                OutCat['dec'] = Cat['dec']
        if self.flags != {} :
            for key in self.flags.keys():
                print(key)
                flagCat = OutCat[Cat['flag'] == int(key)]
                flagCat.to_csv('Catalogs/' + self.cat_name + '_' + self.flags[key] + '.dat', sep='\t', index=False)
                truc = flagCat.describe()
                truc.to_csv('Catalogs/Ana_' + self.cat_name + '_' + self.flags[key] + '.dat', sep='\t')
        else :
            OutCat.to_csv('Catalogs/' + self.cat_name + '.dat', sep='\t', index=False)


    def makeSpin(self, option, size):

        # Available models : zeros, Maxwellian (in prep.), Maxwellian_dynamics (in prep.)

        if option == 'Rand_Dynamics' :
            sigmaSpin = 0.1
            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            V_1 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            V_2 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            costheta1 = 2. * np.random.uniform(0.0, 1.0, size = size) - 1.0
            costheta2 = 2. * np.random.uniform(0.0, 1.0, size = size) - 1.0

            s1 = V_1 * costheta1
            s2 = V_2 * costheta2

        elif option == 'Rand' :
            sigmaSpin = 0.1
            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            V_1 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            V_2 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            s1 = V_1
            s2 = V_2
        else :
            s1 = np.zeros(size)
            s2 = np.zeros(size)

        return s1, s2


    def compute_SNR_opt(self, Networks, freq, approx ):
        """Calculate the optimal SNR for each event of the catalogue and update the catalogue with it.
                Parameters
                ----------
                Networks: list of str
                    List of Networks considered in the study
                freq: np.array
                    Frequencies
                approx: str
                    Approximants used for the waveforms
                """
        deltaf = freq[1]-freq[0]
        for cat in self.catalogs :
            Cat = pd.read_csv('Catalogs/'+cat, sep='\t', index_col=False)
            print('SNR calculation for ', cat)
            ntot = len(Cat.z)
            for N in Networks:
                SNR_N = np.zeros(len(Cat.z))
                psd_compo = np.empty((len(N.compo), len(freq)))

                SNR_det = pd.DataFrame()
                for d in range(len(N.compo)):
                    if N.compo[d] not in list(cat):
                        det = N.compo[d]
                        print(det)
                        psd_compo[d] = det.Make_psd()
                        SNR_det[N.compo[d].det_name] = np.zeros(len(Cat.z))
                for evt in range(len(Cat.z)):
                    event = Cat.iloc[[evt]]
                    htildsq = GWk_noEcc_Pycbcwf(evt=event, freq = freq, approx=approx, n = evt, size_catalogue = ntot, inc_option= 'Optimal')
                    if isinstance(htildsq,int) :
                        f = open('Catalogs/' + cat + '_errors.txt', "a")
                        if os.path.getsize('Catalogs/'+cat+'_errors.txt') == 0:
                            f.write('m1 m2 z\n')
                        f.write('{0} {1} {2}\n'.format(event['m1'].values, event['m2'].values, event['z'].values))
                        f.close()
                        #print('Merger out of range, event notify in the file: Catalogs/{0}_errors.txt'.format(cat))
                    for d in range(len(N.compo)) :
                        if str(N.compo[d]) not in list(cat) :
                            Sn = psd_compo[d]
                            comp = deltaf*4.*htildsq/Sn
                            comp = np.nan_to_num(comp, nan = 0, posinf = 0)
                            SNR = comp.sum()
                            SNR_det[str(N.compo[d].det_name)][evt] = np.sqrt(SNR)
                        else :
                            SNR = event[str(N.compo[d])]**2
                        SNR_N[evt]+= SNR
                Cat = pd.concat([Cat, SNR_det], axis =1)
                Cat['snr_'+N.net_name+'_opt'] = np.sqrt(SNR_N)
            Cat = Cat.T.groupby(level=0).first().T
            Cat.to_csv('Catalogs/'+cat, sep='\t', index=False)

    #def Build_Catalogue(self, mass1_distrib = np.array([np.linspace(0,1,100), np.linspace(0,50,100)]), mass2_distrib = np.array([np.linspace(0,1,100), np.linspace(0,50,100)]), z_distrib = np.array([np.linspace(0,1,100), np.linspace(0,15,100)])):
        #interpm1 = InterpolatedUnivariateSpline(mass1_distrib[0], mass1_distrib[1])













