import math
import os
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from Stochastic import Basic_Functions as BF
from Starter.Htild import GWk_noEcc_Pycbcwf
from scipy.interpolate import InterpolatedUnivariateSpline

class AstroModel:

    def __init__(self, name:str = 'model', duration:float = 1,  original_path:str = None, sep:str = None,
                 index_column:bool = None, flags:dict ={}, spin_option:str = "Zeros", orbit_evolution:bool = False,
                 inclination_position:bool = False):
        """
        Create an instance of the model.
        Parameters
        ----------
        :param name (str): Name for all outputs from this model. Default is model.
        :param duration (float): Duration of the catalogs in yr. Default is 1.
        :param original_path (str): Path to the user orginal catalog.
        :param sep (str): Delimiter or the user input catalog. Default is '\t'
        :param index_column (bool): True if the input catalog of the user contain an index columns. Default is False.
        :param flags (dict): Dictionnary of specific flags in the catalogs if there is a dedicated column. Default is {}.
        :param spin_option (str): Set which spin components are in the initial catalogs and eventually how it should be
         generated. Default is 'Zeros.
        :param orbit_evolution (bool): True if the user want to account for the evolution of the binary. Default is False.
        :param inclination_position (bool): If True generate an inclination, right acsension and declinaison for each
        source of the catalogue. Default is False.
        """

        # Set class variables
        self.original_path = original_path
        self.name = name
        self.duration = duration
        self.sep_cat = sep
        self.index_column = index_column
        self.spin_option = spin_option
        self.flags = flags
        self.orbit_evolution = orbit_evolution
        self.inclination_position =inclination_position
        self.catalogs = []
        print(len(flags.keys()))
        if len(flags.keys())==0 :
            self.catalogs = [self.name + '.dat']
        else :
            for x in flags.keys() :
                self.catalogs.append(self.name + '_' + flags[x] + '.dat')

    def make_catalog(self):
        """
        Create a catalog with the adequate parameters for further calculations.
        The catalog will be named after the parameter self.name, and saved in the folder Catalogs/
        """

        Cat = pd.read_csv(self.original_path, sep = self.sep_cat, index_col = self.index_column, dtype = float)
        print(Cat.describe())
        OutCat = pd.DataFrame()
        Col = list(Cat.columns)

        available_spin_option = ['Chi&Theta', 'Chi&cosTheta', 'Rand_dynamics', 'Rand_aligned', 'Zeros']
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

        if self.spin_option not in available_spin_option :
            raise ValueError((f"{self.spin_option} is not an option for the spin. Please choose among "
                              f"['Chi&Theta', 'Rand_dynamics', 'Rand', 'Zeros'] \n 'Chi&Theta'(Chi&cosTheta'): You have "
                              f"spin magnitudes chi and (cos)theta angle for both components of the binary. The program "
                              f"will compute the spins s1 and s2 from the data.\n 'Rand_dynamics': generate randon "
                              f"magnitude between 0 and 1 and random theta angles. \n 'Rand_aligned': generate random "
                              f"magnitude and assumes aligned spins (i.e. costheta1=costheta2=1). \n 'Zeros': Set both "
                              f"spin to 0. "))

        elif self.spin_option == 'Chi&Theta' :
            OutCat['chi1'] = Cat['chi1']
            OutCat['chi2'] = Cat['chi2']
            OutCat['costheta1'] = np.cos(Cat['theta1'])
            OutCat['costheta2'] = np.cos(Cat['theta2'])

        elif self.spin_option == 'Chi&cosTheta':
            OutCat['chi1'] = Cat['chi1']
            OutCat['chi2'] = Cat['chi2']
            OutCat['costheta1'] = Cat['costheta1']
            OutCat['costheta2'] = Cat['costheta2']

        else :
            OutCat['chi1'], OutCat['chi2'], OutCat['costheta1'], OutCat['costheta2'] = self.generate_spin(len(Cat['z']))

        OutCat['s1'], OutCat['s2'] = self.compute_spins(OutCat['chi1'], OutCat['chi2'], OutCat['costheta1'],
                                                        OutCat['costheta2'])


        #Compute spin components and add it to the output.
        if 'chip' in Col :
            Outcat['chip'] = Cat['chip']
        else :
            OutCat['chip'] = self.compute_chip(OutCat['m1'], OutCat['m2'], OutCat['chi1'], OutCat['chi2'],
                                               OutCat['costheta1'], OutCat['costheta2'])
        if 'chieff' in Col :
            OutCat['chieff'] = Cat['chieff']
        else :
            OutCat['chieff'] = self.compute_chieff(OutCat['m1'], OutCat['m2'], OutCat['chi1'], OutCat['chi2'],
                                                   OutCat['costheta1'], OutCat['costheta2'])

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
                flagCat.to_csv('Catalogs/' + self.name + '_' + self.flags[key] + '.dat', sep='\t', index=False)
                truc = flagCat.describe()
                truc.to_csv('Catalogs/Ana_' + self.name + '_' + self.flags[key] + '.dat', sep='\t')
        else :
            OutCat.to_csv('Catalogs/' + self.name + '.dat', sep='\t', index=False)


    def generate_spin(self, size:int)->tuple:
        """
        Generate spin magnitude and theta angle from the model set by the user : self.spin_option.
        Parameters
        ----------
        :param size (int): size of the catalog.
        :return: tuple of np.array chi1, chi2, costheta1, costheta2
        """

        if self.spin_option == 'Rand_dynamics' :
            sigmaSpin = 0.1
            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            chi1 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            chi2 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            costheta1 = 2. * np.random.uniform(0.0, 1.0, size = size) - 1.0
            costheta2 = 2. * np.random.uniform(0.0, 1.0, size = size) - 1.0


        elif self.spin_optionn == 'Rand_aligned' :
            sigmaSpin = 0.1
            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            chi1 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            v1_L = np.random.normal(0.0, sigmaSpin, size = size)
            v2_L = np.random.normal(0.0, sigmaSpin, size = size)
            v3_L = np.random.normal(0.0, sigmaSpin, size = size)
            chi2 = np.sqrt(v1_L * v1_L + v2_L * v2_L + v3_L * v3_L)

            costheta1 = np.ones(size)
            costheta2 = np.ones(size)

        elif self.spin_option == 'Zeros' :
            chi1 = np.zeros(size)
            chi2 = np.zeros(size)
            costheta1 = np.zeros(size)
            costheta2 = np.zeros(size)


        return chi1, chi2, costheta1, costheta2

    def compute_spins(self, chi1:np.ndarray, chi2:np.ndarray, costheta1:np.ndarray, costheta2:np.ndarray)->tuple:
        """
        Compute spins of the two components s_i = chi_i*costheta_i
        Parameters
        ----------
        :param chi1 (np.1darray): Array containing the magnitudes of the first components.
        :param chi2 (np.1darray): Array containing the magnitudes of the second components.
        :param costheta1 (np.1darray): Array containing the theta cosine of the first components.
        :param costheta2 (np.1darray): Array containing the theta cosine of the second components.
        :return: tuple containing two np.1darray s1(s2) is the projection on the z axis of the spin for firts(seconds)
                components.
        """

        s1 = chi1*costheta1
        s2 = chi2*costheta2
        return s1, s2

    def compute_chieff(self, m1:np.ndarray, m2:np.ndarray, chi1:np.ndarray, chi2:np.ndarray, cos_theta_1:np.ndarray,
                       cos_theta_2:np.ndarray)->np.ndarray:
        """
        Compute the effective spin of each binaries.
        Parameters
        ----------
        :param m1 (np.1darray): Array containing the masses of the first component.
        :param m2 (np.1darray): Array containing the masses of the second component.
        :param chi1 (np.1darray): Array containing the magnitudes of the first components.
        :param chi2 (np.1darray): Array containing the magnitudes of the second components.
        :param cos_theta_1 (np.1darray): Array containing the theta cosine of the first components.
        :param cos_theta_2 (np.1darray): Array containing the theta cosine of the second components.
        :return (np.1darray): Array with the effective spin of each binaries.
        """

        chieff = (chi1 * cos_theta_1 * m1 + chi2 * cos_theta_2 * m2) / (m1 + m2)

        return chieff


    def compute_chip(self, m1, m2, chi1, chi2, cos_theta_1, cos_theta_2):
        """
        Compute the precessing spin of each binaries.
        Parameters
        ----------
        :param m1 (np.1darray): Array containing the masses of the first component.
        :param m2 (np.1darray): Array containing the masses of the second component.
        :param chi1 (np.1darray): Array containing the magnitudes of the first components.
        :param chi2 (np.1darray): Array containing the magnitudes of the second components.
        :param cos_theta_1 (np.1darray): Array containing the theta cosine of the first components.
        :param cos_theta_2 (np.1darray): Array containing the theta cosine of the second components.
        :return (np.1darray): Array with the precessing spin of each binaries.
        """

        chip1 = (2. + (3. * m2) / (2. * m1)) * chi1 * m1 * m1 * (1. - cos_theta_1 * cos_theta_1) ** 0.5
        chip2 = (2. + (3. * m1) / (2. * m2)) * chi2 * m2 * m2 * (1. - cos_theta_2 * cos_theta_2) ** 0.5
        chipmax = np.maximum(chip1, chip2)
        chip = chipmax / ((2. + (3. * m2) / (2. * m1)) * m1 * m1)
        return chip




    def compute_SNR_opt(self, Networks:list, freq:np.ndarray, approx:str):
        """
        Calculate the optimal SNR for each event of the catalogue and save it iwith additionnal columns in the catalog
        Catalogs/<self.name>.dat.
        Parameters
        ----------
        :param Networks (list): List of class Network instances.
        :param freq (np.1darray): Array of frequencies.
        :param approx (str): Approximants used for the waveforms
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
                missing_detectors =0
                for d in range(len(N.compo)):
                    if N.compo[d] not in list(cat):
                        det = N.compo[d]
                        psd_compo[d] = det.Make_psd()
                        SNR_det[N.compo[d].name] = np.zeros(len(Cat.z))
                        print(N.compo[d].name)
                        missing_detectors +=1
                if missing_detectors>0:
                    for evt in range(len(Cat.z)):
                        event = Cat.iloc[[evt]]
                        htildsq = GWk_noEcc_Pycbcwf(evt=event, freq = freq, approx=approx, n = evt, size_catalogue = ntot,
                                                    inc_option= 'Optimal')
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
                                SNR_det[str(N.compo[d].name)][evt] = np.sqrt(SNR)
                            else :
                                SNR = event[str(N.compo[d])]**2
                            SNR_N[evt]+= SNR
                Cat = pd.concat([Cat, SNR_det], axis =1)
                Cat['snr_'+N.name+'_opt'] = np.sqrt(SNR_N)
                Cat = Cat.T.groupby(level=0).first().T
                Cat.to_csv('Catalogs/'+cat, sep='\t', index=False)

    #def Build_Catalogue(self, mass1_distrib = np.array([np.linspace(0,1,100), np.linspace(0,50,100)]), mass2_distrib = np.array([np.linspace(0,1,100), np.linspace(0,50,100)]), z_distrib = np.array([np.linspace(0,1,100), np.linspace(0,15,100)])):
        #interpm1 = InterpolatedUnivariateSpline(mass1_distrib[0], mass1_distrib[1])













