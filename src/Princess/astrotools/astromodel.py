print(f"Loading {__name__}")
import math
import os
import numpy as np
import pandas as pd
import json
import pickle
from Princess.astrotools.utils import m1_m2_to_mc_q, mc_q_to_m1_m2
from Princess.gwtools.Network import Network
from Princess.gwtools.Detector import Detector
from Princess.cosmology.cosmology import Cosmology
from Princess.gwtools.snr import SNR_catalog, compute_SNR_Networks_catalog
from Princess.Run.settings import PARAMS_FILE

# Check PARAMS_FILE value
if not PARAMS_FILE or not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"The file parameter {PARAMS_FILE} is missing,. Execute Run.settings.Make_params_file() first.")

# Charge le fichier de paramètres
with open(PARAMS_FILE, "r") as f:
    params = json.load(f)


class AstroModel:

    def __init__(self, name:str = 'model', duration:float = 1,  original_path:str = None, sep:str = None,
                 index_column:bool = None, flags:dict ={}, spin_model:str = "Zeros", orbit_evolution:bool = False,
                 inclination_position:bool = True):
        """
        Initializes an AstroModel instance and loads or creates necessary data.

        :param name: (str) Name for all outputs from this model. Default is 'model'.
        :param duration: (float) Duration of the catalogs in years. Default is 1
        :param original_path: (str) Path to the user's original catalog file.
        :param sep: (str) Delimiter for the user's input catalog. Default is '\t'.
        :param index_column: (bool) True if the input catalog has an index column. Default is False.
        :param flags: (dict) Dictionary of specific flags in the catalog, if applicable. Default is {}.
        :param spin_model: (str) Determines how spin components are handled/generated. Default is 'Zeros'.
        :param orbit_evolution: (bool) True to account for binary evolution. Default is False.
        :param inclination_position: (bool) If True, generates inclination, right ascension, and declination
        for each catalog source. Default is True.
        """



        self.name = name
        self.original_path = original_path
        self.duration = duration
        self.sep_cat = params['astro_model_list'][self.name].get('sep', None)
        self.index_column = index_column
        self.spin_model = spin_model
        self.flags = flags
        self.orbit_evolution = orbit_evolution
        self.inclination_position = inclination_position
        self.catalogs = []

        # Build the list of catalogs based on flags
        if not flags:  # No specific flags, single catalog name
            self.catalogs = [f"{self.name}.dat"]
        else:  # Append flag-specific catalog names
            for key, flag_name in flags.items():
                self.catalogs.append(f"{self.name}_{flag_name}.dat")

        # Determine if model needs to be loaded or created
        model_path = f"Run/{params['name_of_project_folder']}/{self.name}_AM.pickle"
        if not os.path.exists(model_path) or params['overwrite']['astromodel']:
            self.save()
        else:
            self.load(model_path)

    @classmethod
    def load(cls, model_path):
        """
        Loads a previously saved AstroModel instance from a pickle file.
        If the loaded object is a dictionary, it converts it back into an AstroModel instance.
        """
        try:
            with open(model_path, 'rb') as f:
                obj = pickle.load(f)

            if isinstance(obj, dict):  # Si c'est un dictionnaire, reconstruire un objet AstroModel
                return cls(**obj)
            return obj  # Si c'est déjà un objet AstroModel, on le retourne directement

        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file for AstroModel not found: {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading AstroModel from {model_path}: {e}")

    def make_catalog(self):
        """
        Create a catalog with the appropriate parameters for further calculations.
        The catalog will be named after the parameter `self.name` and saved in the folder `Catalogs/`.
        """
        # Print the path to the input file
        print(f"Input file path: {self.original_path}")

        # Attempt to load the source catalog
        try:
            Cat = pd.read_csv(self.original_path, sep=self.sep_cat, index_col=self.index_column, engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.original_path} not found.")
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

        print("Overview of loaded data:")
        print(Cat.describe())

        # Initialize the output catalog
        OutCat = pd.DataFrame()
        Col = list(Cat.columns)

        # Handle identifiers (if required by parameters)
        if params['AM_params'].get('keepID', False):
            id_col = params['AM_params'].get('ID_col', None)
            if id_col and id_col in Col:
                OutCat[id_col] = Cat[id_col]
            else:
                raise KeyError(f"Specified ID column '{id_col}' not found in input data.")

        # Rename input columns if specified in parameters
        Cat.rename(columns=params['AM_params'].get('input_parameters', {}), inplace=True)
        # Load cosmology models (parameters from Advanced_params.py
        cosmology = Cosmology.load(params['Cosmo_model'])
        cosmology.info()
        # Handle redshift (z)
        if 'z' not in Col:  # If 'z' is missing, compute it from luminosity distance
            OutCat['z'] = cosmology.compute_z(Cat['dl'])
        else:
            OutCat['z'] = Cat['z']

        # Validate and calculate masses
        if 'Mc' not in Col:  # If chirp mass is missing, calculate it from m1 and m2
            OutCat['m1'] = Cat['m1']
            OutCat['m2'] = Cat['m2']
            OutCat['Mc'], OutCat['q'] = m1_m2_to_mc_q(OutCat['m1'], OutCat['m2'])
        elif 'm1' not in Col:  # If m1 and m2 are missing, calculate them from Mc and q
            OutCat['Mc'] = Cat['Mc']
            OutCat['q'] = Cat['q']
            OutCat['m1'], OutCat['m2'] = mc_q_to_m1_m2(Cat['Mc'], Cat['q'])
        else:  # If all are present, copy directly
            OutCat['Mc'] = Cat['Mc']
            OutCat['q'] = Cat['q']
            OutCat['m1'] = Cat['m1']
            OutCat['m2'] = Cat['m2']

        # Compute luminosity distance (Dl) if missing
        if 'Dl' not in Col:
            OutCat['Dl'] = cosmology.compute_dl(OutCat['z'])
        else:
            OutCat['Dl'] = Cat['Dl']

        # Handle spin generation based on the selected model
        available_spin_option = ['Spin&Theta', 'Spin&cosTheta', 'Rand_dynamics', 'Rand_aligned', 'Zeros']
        if self.spin_model not in available_spin_option:
            raise ValueError(f"Invalid spin model: {self.spin_model}. Choose from: {available_spin_option}")

        if self.spin_model == 'Spin&Theta':  # Input contains spin and theta
            OutCat['s1'] = Cat['s1']
            OutCat['s2'] = Cat['s2']
            OutCat['costheta1'] = np.cos(Cat['theta1'])
            OutCat['costheta2'] = np.cos(Cat['theta2'])
        elif self.spin_model == 'Spin&cosTheta':  # Input contains spin and cosTheta
            OutCat['s1'] = Cat['s1']
            OutCat['s2'] = Cat['s2']
            OutCat['costheta1'] = Cat['costheta1']
            OutCat['costheta2'] = Cat['costheta2']
        else:  # Generate spins dynamically based on random values
            OutCat['s1'], OutCat['s2'], OutCat['costheta1'], OutCat['costheta2'] = self.generate_spin(len(OutCat['z']))

        # Compute chi1 and chi2 from spins
        OutCat['chi1'], OutCat['chi2'] = self.compute_spins(OutCat['s1'], OutCat['s2'], OutCat['costheta1'],
                                                            OutCat['costheta2'])

        # Calculate chip and chieff
        OutCat['chip'] = Cat['chip'] if 'chip' in Col else self.compute_chip(OutCat['m1'], OutCat['m2'], OutCat['chi1'],
                                                                             OutCat['chi2'], OutCat['costheta1'],
                                                                             OutCat['costheta2'])
        OutCat['chieff'] = Cat['chieff'] if 'chieff' in Col else self.compute_chieff(OutCat['m1'], OutCat['m2'],
                                                                                     OutCat['chi1'], OutCat['chi2'],
                                                                                     OutCat['costheta1'],
                                                                                     OutCat['costheta2'])

        # Add orbital parameters if evolution is enabled
        if self.orbit_evolution:
            OutCat['a0'] = Cat.get('a0', np.nan)
            OutCat['e0'] = Cat.get('e0', np.nan)

        # Handle inclination and sky positions
        if self.inclination_position:
            OutCat['inc'] = Cat.get('inc', np.random.uniform(0, math.pi, len(OutCat['m1'])))
            OutCat['ra'] = Cat.get('ra', np.random.uniform(0, 2 * math.pi, len(OutCat['m1'])))
            OutCat['dec'] = Cat.get('dec', np.random.uniform(0, 2 * math.pi, len(OutCat['m1'])))

        # Handle additional flags and save results
        if self.flags:
            for key, flag_name in self.flags.items():
                flagCat = OutCat[Cat['flag'] == int(key)]
                flagCat.to_csv(f'Catalogs/{self.name}_{flag_name}.dat', sep='\t', index=False)
                summary = flagCat.describe()
                summary.to_csv(f'Catalogs/Ana_{self.name}_{flag_name}.dat', sep='\t')
        else:
            output_path = f"Run/{params['name_of_project_folder']}/Astro_Models/Catalogs/{self.name}.dat"
            OutCat.to_csv(output_path, sep='\t', index=False)
            self.columns = OutCat.columns.to_numpy
            print(f"Catalog saved to: {output_path}")
        self.check_SNR_reboot()

    def check_SNR_reboot(self):

        self.SNR_2G = False
        self.SNR_3G = False
        self.SNR_LISA = False
        self.SNR_PTA = False


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
            detector = Detector.load(det)
            if detector.type == 'LISA' :
                print("Princess not ready for this computation")
            elif detector.type == 'PTA' :
                print("Princess not ready for this computation")
            elif detector.type == '2G' :
                det_list_2G.append(detector)
            elif detector.type == '3G':
                det_list_3G.append(detector)

        #changes in the SNR function import gwtools.tools as gwt, takes in entry catalog path the list of catalogs from self.catalogs
        # catalog_path = f'./Run/{params["name_of_project_folder"]}/Astro_Models/Catalogs':
        # takes name of the model and RETURN the dataframe with the SNRs, and save the new catalogs
        if ((not self.SNR_2G) or params['overwrite']['individual_snr']) and len(det_list_2G)>0:
            SNR_catalog(catalog_name= self.name, det_list = det_list_2G, waveform = params['detector_params']['types']['2G']['waveform'], freq = det_list_2G[0].freq )
            self.SNR_2G = True
        elif self.SNR_2G :
            print('SNR for 2G detectors already computed')
        else :
            print("Possible error...Check values for SNR check point am.SNR_2G and the presence of 2G detectors in catalog columns")
        if ((not self.SNR_3G) or params['overwrite']['individual_snr']) and len(det_list_3G)>0:
            SNR_catalog(catalog_name= self.name, det_list= det_list_3G, waveform = params['detector_params']['types']['3G']['waveform'], freq = det_list_3G[0].freq )
            self.SNR_3G = True
        elif self.SNR_3G :
            print('SNR for 3G detectors already computed')
        else :
            print("Possible error...Check values for SNR check point am.SNR_3G and the presence of 2G detectors in catalog columns")
        compute_SNR_Networks_catalog(catalog_name = self.name)
        self.save()


    def save(self):
        """
        Saves the current AstroModel instance to a pickle file for later use.
        """
        model_path = f"Run/{params['name_of_project_folder']}/{self.name}_AM.pickle"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"AstroModel '{self.name}' saved successfully.")
        except Exception as e:
            print(f"Error saving AstroModel '{self.name}': {e}")






