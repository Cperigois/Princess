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

params = json.load(open('Run/Params.json', 'r'))


def process_astromodel():
    """
    Main function to process all astromodels defined in the configuration.
    Loads existing models if available, otherwise initializes, saves,
    generates catalogs, and computes SNR.
    """
    # Ensure necessary directories exist
    base_path = f"Run/{params['name_of_project_folder']}"
    astro_models_path = f"{base_path}/Astro_Models"
    catalogs_path = f"{astro_models_path}/Catalogs"

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(astro_models_path, exist_ok=True)
    os.makedirs(catalogs_path, exist_ok=True)

    # Process each astromodel defined in the parameter configuration
    for model_key, model_params in params['astro_model_list'].items():
        try:
            model_name = model_params['name']
            model_save_path = f"{base_path}/{model_name}_AM.pickle"

            # Vérifier si le modèle existe déjà
            if os.path.exists(model_save_path) and (not params['overwrite']['astromodel']) :
                am = AstroModel.load(model_save_path)
                print(f"Loaded existing AstroModel: {am.name}")
            else:
                # Initialisation d'un nouveau modèle
                am = AstroModel(
                    name=model_params['name'],
                    original_path=model_params['original_path'],
                    spin_model=model_params['spin_model'],
                    duration=model_params['duration']
                )
                print(f"Initialized new AstroModel: {am.name}")

                # Générer le catalogue pour le modèle
                am.make_catalog()
                am.compute_SNR()
                # Sauvegarde du modèle pour un usage futur
                am.save()
                print('ICI')
            # if the user ask for the rurun of SNR without rerunning the astromodel
            if params['overwrite']['individual_snr'] and (not params['overwrite']['astromodel']):
                am.check_SNR_reboot() # Set to False the computation of SNRs to ensure its recomputation
                am.compute_SNR()

        except Exception as e:
            print(f"Error processing AstroModel '{model_key}': {e}")


class AstroModel:

    def __init__(self, name:str = 'model', duration:float = 1,  original_path:str = None, sep:str = None,
                 index_column:bool = None, flags:dict ={}, spin_model:str = "Zeros", orbit_evolution:bool = False,
                 inclination_position:bool = True):
        """Initializes an AstroModel instance and loads or creates necessary data.

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

        # Handle redshift (z)
        if 'z' not in Col:  # If 'z' is missing, compute it from luminosity distance
            BF.build_interp()
            OutCat['z'] = BF.dl_to_z_Planck15(Cat['dl'])
        else:
            OutCat['z'] = Cat['z']

        # Validate and calculate masses
        if 'Mc' not in Col:  # If chirp mass is missing, calculate it from m1 and m2
            OutCat['m1'] = Cat['m1']
            OutCat['m2'] = Cat['m2']
            OutCat['Mc'], OutCat['q'] = BF.m1_m2_to_mc_q(OutCat['m1'], OutCat['m2'])
        elif 'm1' not in Col:  # If m1 and m2 are missing, calculate them from Mc and q
            OutCat['Mc'] = Cat['Mc']
            OutCat['q'] = Cat['q']
            OutCat['m1'], OutCat['m2'] = BF.mc_q_to_m1_m2(Cat['Mc'], Cat['q'])
        else:  # If all are present, copy directly
            OutCat['Mc'] = Cat['Mc']
            OutCat['q'] = Cat['q']
            OutCat['m1'] = Cat['m1']
            OutCat['m2'] = Cat['m2']

        # Compute luminosity distance (Dl) if missing
        if 'Dl' not in Col:
            OutCat['Dl'] = [Planck15.luminosity_distance(z).value for z in OutCat['z']]
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
            detector = BF.load_detector(name_detector= det, project_folder='Run/'+params['name_of_project_folder']+'/')
            if detector['type'] == 'LISA' :
                print("Princess not ready for this computation")
            elif detector['type'] == 'PTA' :
                print("Princess not ready for this computation")
            elif detector['type'] == '2G' :
                det_list_2G.append(detector)
            elif detector['type'] == '3G':
                det_list_3G.append(detector)
        if ((not self.SNR_2G) or params['overwrite']['individual_snr']) and len(det_list_2G)>0:
            self.SNR(det_list = det_list_2G, waveform = params['detector_params']['types']['2G']['waveform'], freq = det_list_2G[0]['freq'] )
            self.SNR_2G = True
        elif self.SNR_2G :
            print('SNR for 2G detectors already computed')
        else :
            print("Possible error...Check values for SNR check point am.SNR_2G and the presence of 2G detectors in catalog columns")
        if ((not self.SNR_3G) or params['overwrite']['individual_snr']) and len(det_list_3G)>0:
            self.SNR(det_list= det_list_3G, waveform = params['detector_params']['types']['3G']['waveform'], freq = det_list_3G[0]['freq'] )
            self.SNR_3G = True
        elif self.SNR_3G :
            print('SNR for 3G detectors already computed')
        else :
            print("Possible error...Check values for SNR check point am.SNR_3G and the presence of 2G detectors in catalog columns")
        self.compute_SNR_Networks()
        self.save()



    def SNR(self, det_list: list, waveform: str, freq: np.array):
        """
        Calculate the optimal SNR for each event of the catalogue and save it with additional columns.
        """

        for cat in self.catalogs:
            Cat = pd.read_csv('./Run/' + params['name_of_project_folder'] + '/Astro_Models/Catalogs/' + cat, sep='\t',
                              index_col=False)
            print('SNR calculation for', cat)
            ntot = len(Cat.z)

            # Initialisation des colonnes SNR à zéro pour chaque détecteur
            for i in det_list:
                Cat[i['name']] = 0.0

            for evt in range(len(Cat)):
                event = Cat.iloc[[evt]]  # Sélectionne une seule ligne sous forme de DataFrame
                htildsq = GWk_no_ecc_pycbcwf(evt=event, freq=freq, approx=waveform, n=evt, size_catalogue=ntot,
                                             inc_option=params['Inclination'])
                if isinstance(htildsq, int):  # Si htildsq est un entier, erreur
                    error_file = 'Catalogs/' + cat + '_errors.txt'
                    with open(error_file, "a") as f:
                        if os.path.getsize(error_file) == 0:
                            f.write('m1 m2 z\n')
                        f.write(f"{event['m1'].iloc[0]} {event['m2'].iloc[0]} {event['z'].iloc[0]}\n")
                else:
                    for d in det_list:
                        Sn = d['psd']
                        comp = d['deltaf'] * 4. * htildsq / Sn
                        comp = np.nan_to_num(comp, nan=0, posinf=0)
                        SNR = np.sqrt(comp.sum())  # Calcul final de la SNR
                        Cat.at[evt, d['name']] = SNR  # Correction de l'affectation

            # Sauvegarde du fichier avec les nouvelles colonnes
            output_file = './Run/' + params['name_of_project_folder'] + '/Astro_Models/Catalogs/' + cat
            Cat.to_csv(output_file, sep='\t', index=False)


    def compute_SNR_Networks(self):
        # Charger la table de facteurs
        try:
            fd_table = pd.read_csv('AuxiliaryFiles/factor_table.dat', sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError("Le fichier 'factor_table.dat' est introuvable dans 'AuxiliaryFiles/'.")

        # Parcourir les réseaux dans la liste des réseaux
        for net in params['network_list'].keys():
            network = DET.Network(name=net)

            # Parcourir chaque catalogue
            for cat in self.catalogs:
                catalog_path = f"./Run/{params['name_of_project_folder']}/Astro_Models/Catalogs/{cat}"

                try:
                    Cat = pd.read_csv(catalog_path, sep='\t')
                except FileNotFoundError:
                    print(f"Le fichier catalogue '{cat}' est introuvable.")
                    continue

                # Vérifier si les colonnes nécessaires existent
                if 'm1' not in Cat.columns:
                    raise KeyError(f"'m1' column is missing in {cat}.")

                # Initialisation des colonnes pour les résultats
                Cat[f'{net}_optimal'] = 0.0
                Cat[net] = 0.0

                # Sélectionner un facteur aléatoire dans la table des facteurs
                fd = fd_table.iloc[np.random.randint(0, len(fd_table))]

                # Calculer les contributions des détecteurs
                for det in network.compo.keys():
                    if det in Cat.columns:
                        Cat[f'{net}_optimal'] += Cat[det] ** 2
                        config = DET.Detector(name = det).configuration
                        Cat[net] += (Cat[det] * fd[config]) ** 2
                    else:
                        print(f"Detector '{det}' pre computations are missing in '{cat}'.")

                # Finaliser les calculs des SNR pour le réseau
                Cat[net] = np.sqrt(Cat[net])
                Cat[f'{net}_optimal'] = np.sqrt(Cat[f'{net}_optimal'])

                # Sauvegarder les résultats dans le fichier
                output_path = f"./Run/{params['name_of_project_folder']}/Astro_Models/Catalogs/{cat}"
                try:
                    Cat.to_csv(output_path, sep='\t', index=False)
                except Exception as e:
                    print(f"Problem during the saving of file {cat}: {e}")


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





def load_detector(name_detector: str, project_folder: str = "Run") -> object:
    """
    Load a Detector instance from a pickle file.

    Parameters
    ----------
    name_detector : str
        The name of the detector.
    project_folder : str, optional
        The base directory where the detector pickle files are stored, by default 'Run'.

    Returns
    -------
    object
        The loaded Detector instance, or None if loading fails.

    Raises
    ------
    FileNotFoundError
        If the specified detector file does not exist.
    Exception
        For any other issues during the loading process.
    """
    file_path = os.path.join(project_folder, f"{name_detector}_DET.pickle")
    try:
        with open(file_path, "rb") as file:
            detector_instance = pickle.load(file)
            print(f"Detector '{name_detector}' successfully loaded from {file_path}.")
            return detector_instance
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the detector: {e}")
        return None






