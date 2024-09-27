import math
import os
import numpy as np
import pandas as pd
import json
import pickle
from astropy.cosmology import Planck15
from stochastic import basic_functions as BF
from astrotools.htild import GWk_no_ecc_pycbcwf
import astrotools.detection as DET
from scipy.interpolate import InterpolatedUnivariateSpline

params = json.load(open('Run/Params.json', 'r'))


def initialization():
    for model in params['astro_model_list'].keys():
        astromodel = AstroModel(name=params['astro_model_list'][model]['name'],
                                original_path=params['astro_model_list'][model]['original_path'],
                                spin_model=params['astro_model_list'][model]['spin_model'],
                                duration=params['astro_model_list'][model]['duration'])
        print(astromodel.name,' ',astromodel.original_path)
        astromodel.save()

def process_astromodel():
    # -------------------------------------      Main code       ---------------------------------------------------
    # Make sure directories are created
    if not os.path.exists('Run/' + params['name_of_project_folder']):
        os.mkdir('Run/' + params['name_of_project_folder'])
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/")
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs")

    for astromodel in params['astro_model_list'].keys():
        am = AstroModel(name = astromodel)
        am.make_catalog()
        am.compute_SNR()

class AstroModel:

    def __init__(self, name:str = 'model', duration:float = 1,  original_path:str = None, sep:str = '\t',
                 index_column:bool = None, flags:dict ={}, spin_model:str = "Zeros", orbit_evolution:bool = False,
                 inclination_position:bool = True):
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
        self.name = name
        #print(name)
        if (not os.path.exists('Run/' + params['name_of_project_folder'] + '/' + self.name + '_AM.pickle')) or (
                params['overwrite']['astromodel'] == True):
            self.original_path = original_path
            self.duration = duration
            self.sep_cat = sep
            self.index_column = index_column
            self.spin_model = spin_model
            self.flags = flags
            self.orbit_evolution = orbit_evolution
            self.inclination_position =inclination_position
            self.catalogs = []
            if len(flags.keys())==0 :
                self.catalogs = [self.name + '.dat']
            else :
                for x in flags.keys() :
                    self.catalogs.append(self.name + '_' + flags[x] + '.dat')
            self.save()
        else :
            self.load()


    def make_catalog(self):
        """
        Create a catalog with the adequate parameters for further calculations.
        The catalog will be named after the parameter self.name, and saved in the folder Catalogs/
        """
        print(self.original_path)
        Cat = pd.read_csv(self.original_path, sep = self.sep_cat, index_col = self.index_column, dtype = float)
        print(Cat.describe())
        OutCat = pd.DataFrame()
        Col = list(Cat.columns)

        available_spin_option = ['Spin&Theta', 'Spin&cosTheta', 'Rand_dynamics', 'Rand_aligned', 'Zeros']
        Cat = Cat.rename(columns = params['AM_params']['input_parameters'])
        print(Cat.describe())

        if 'z' not in Col :
            BF.build_interp()
            OutCat['z'] = BF.dl_to_z_Planck15(Cat['dl'])
        else :
            OutCat['z'] = Cat['z']

        # Check the masses calculations
        if 'Mc' not in Col:
            OutCat['m1'] = Cat['m1']
            OutCat['m2'] = Cat['m2']
            OutCat['Mc'], OutCat['q'] = BF.m1_m2_to_mc_q(OutCat['m1'], OutCat['m2'])
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
            zm = np.array(OutCat['z'], float)
            dl = np.array([])
            for z in zm:
                dl = np.append(dl, Planck15.luminosity_distance(z).value)
            OutCat['Dl'] = dl
        else :
            OutCat['Dl'] = Cat['Dl']
        # Generate the spin

        if self.spin_model not in available_spin_option :
            raise ValueError((f"{self.spin_model} is not an option for the spin. Please choose among "
                              f"['Spin&Theta', 'Rand_dynamics', 'Rand', 'Zeros'] \n 'Spin&Theta'(Spin&cosTheta'): You have "
                              f"spin magnitudes chi and (cos)theta angle for both components of the binary. The program "
                              f"will compute the spins s1 and s2 from the data.\n 'Rand_dynamics': generate randon "
                              f"magnitude between 0 and 1 and random theta angles. \n 'Rand_aligned': generate random "
                              f"magnitude and assumes aligned spins (i.e. costheta1=costheta2=1). \n 'Zeros': Set both "
                              f"spin to 0. "))

        elif self.spin_model == 'Spin&Theta' :
            OutCat['s1'] = Cat['s1']
            OutCat['s2'] = Cat['s2']
            OutCat['costheta1'] = np.cos(Cat['theta1'])
            OutCat['costheta2'] = np.cos(Cat['theta2'])

        elif self.spin_model == 'Spin&cosTheta':
            OutCat['s1'] = Cat['s1']
            OutCat['s2'] = Cat['s2']
            OutCat['costheta1'] = Cat['costheta1']
            OutCat['costheta2'] = Cat['costheta2']

        else :
            OutCat['chi1'], OutCat['chi2'], OutCat['costheta1'], OutCat['costheta2'] = self.generate_spin(len(OutCat['z']))

        OutCat['chi1'], OutCat['chi2'] = self.compute_spins(OutCat['s1'], OutCat['s2'], OutCat['costheta1'],
                                                        OutCat['costheta2'])


        #Compute spin components and add it to the output.
        if 'chip' in Col :
            OutCat['chip'] = Cat['chip']
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
            OutCat.to_csv('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs/" + self.name + '.dat', sep='\t', index=False)


    def generate_spin(self, size:int)->tuple:
        """
        Generate spin magnitude and theta angle from the model set by the user : self.spin_option.
        Parameters
        ----------
        :param size (int): size of the catalog.
        :return: tuple of np.array chi1, chi2, costheta1, costheta2
        """

        if self.spin_model == 'Rand_dynamics' :
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


        elif self.spin_model == 'Rand_aligned' :
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

        elif self.spin_model == 'Zeros' :
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

    def compute_SNR(self):
        det_list_2G = list([])
        det_list_3G = list([])
        for det in params['detector_list'].keys():
            detector = DET.Detector(name = det)
            if detector.type == 'LISA' :
                print("Princess not ready for this computation")
            elif detector.type == 'PTA' :
                print("Princess not ready for this computation")
            elif detector.type == '2G' :
                det_list_2G.append(detector)
            elif detector.type == '3G':
                det_list_3G.append(detector)
        if len(det_list_2G)>0:
            print(det_list_2G[0].freq)
            self.SNR(det_list = det_list_2G, waveform = params['detector_params']['types']['2G']['waveform'], freq = det_list_2G[0].freq )
        if len(det_list_3G)>0:
            self.SNR(det_list= det_list_3G, waveform = params['detector_params']['types']['3G']['waveform'], freq = det_list_3G[0].freq )
        self.compute_SNR_Networks()


    def SNR(self, det_list:list, waveform:str, freq:np.array):
        """
        Calculate the optimal SNR for each event of the catalogue and save it iwith additionnal columns in the catalog
        Catalogs/<self.name>.dat.
        Parameters
        ----------
        :param Networks (list): List of class Network instances.
        :param freq (np.1darray): Array of frequencies.
        :param approx (str): Approximants used for the waveforms
        """
        for cat in self.catalogs :
            Cat = pd.read_csv('./Run/' + params['name_of_project_folder'] + '/Astro_Models/Catalogs/'+cat, sep='\t', index_col=False)
            print('SNR calculation for ', cat)
            ntot = len(Cat.z)
            df = pd.DataFrame({})
            for i in det_list:
                Cat[i.name] = np.zeros(ntot)
            for evt in range(len(Cat.z)):
                event = Cat.iloc[[evt]]
                htildsq = GWk_no_ecc_pycbcwf(evt=event, freq = freq, approx=waveform, n = evt, size_catalogue = ntot,
                                            inc_option= params['Inclination'])
                if isinstance(htildsq,int) :
                    f = open('Catalogs/' + cat + '_errors.txt', "a")
                    if os.path.getsize('Catalogs/'+cat+'_errors.txt') == 0:
                        f.write('m1 m2 z\n')
                        f.write('{0} {1} {2}\n'.format(event['m1'].values, event['m2'].values, event['z'].values))
                        f.close()
                            #print('Merger out of range, event notify in the file: Catalogs/{0}_errors.txt'.format(cat))
                else:
                    for d in det_list:
                        Sn = d.psd
                        comp = d.deltaf*4.*htildsq/Sn
                        comp = np.nan_to_num(comp, nan = 0, posinf = 0)
                        SNR = comp.sum()
                        Cat[d.name][evt] = np.sqrt(SNR)
                Cat.to_csv('./Run/' + params['name_of_project_folder'] + '/Astro_Models/Catalogs/'+cat, sep='\t', index=False)

    #def Build_Catalogue(self, mass1_distrib = np.array([np.linspace(0,1,100), np.linspace(0,50,100)]), mass2_distrib = np.array([np.linspace(0,1,100), np.linspace(0,50,100)]), z_distrib = np.array([np.linspace(0,1,100), np.linspace(0,15,100)])):
        #interpm1 = InterpolatedUnivariateSpline(mass1_distrib[0], mass1_distrib[1])

    def compute_SNR_Networks(self):
        for net in params['network_list'].keys():
            network = DET.Network(name=net)
            for cat in self.catalogs:
                Cat = pd.read_csv('./Run/' + params['name_of_project_folder'] + '/Astro_Models/Catalogs/' + cat, sep='\t', index_col=False)
                Cat[net] = np.zeros(len(Cat.m1))
                for det in network.compo:
                    Cat[net]+= Cat[det]**2
                Cat[net] = np.sqrt(Cat[net])
        Cat.to_csv('./Run/' + params['name_of_project_folder'] + '/Astro_Models/Catalogs/' + cat, sep='\t', index=False)

    def load(self):
        """try load self.name.txt"""
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_AM.pickle', 'rb')
        data_pickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(data_pickle)

    def save(self):
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_AM.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()










