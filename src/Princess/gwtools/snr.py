import os
import numpy as np
import pandas as pd
import json

from pycbc.waveform import get_fd_waveform
from pycbc.filter.matchedfilter import sigma

from Princess.gwtools.Network import Network
from Princess.gwtools.Detector import Detector
from Princess.gwtools.waveform import GWk_no_ecc_pycbcwf
from Princess.cosmology.cosmology import Cosmology
from Princess.gwtools.utils import fmerg_f
from Princess.Run.settings import PARAMS_FILE

# Check PARAMS_FILE value
if not PARAMS_FILE or not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"The file parameter {PARAMS_FILE} is missing,. Execute Run.settings.Make_params_file() first.")

# Charge le fichier de paramètres
with open(PARAMS_FILE, "r") as f:
    params = json.load(f)
    
    def SNR_source(detector, mtot:float, z:float, q:float, waveform_approx:str)->float:
        """
        TEST FUNCTION USE ONLY FOR TESTS
        Compute the snr of one specific source, assuming spins are 0 and the best sky location.
        Parameters
        ----------
        :param mtot (float): Total mass of the system in Msun.
        :param z (float): Redshift of the merger.
        :param q (float): Mass ratio of the system. By convention q<1.
        :param waveform_approx (str): Waveform to use for the computation of the SNR.
        :return (snr): Signal-to-noise ration of the soure in the detector.
        """
        cosmology = Cosmology.load(params['Cosmo_model'])
        luminosity_distance = cosmology.luminosity_distance(z)
        m1 = mtot * q * (1. + z) / (1 + q)
        m2 = m1 / q
        flim = fcut_f(m1=m1, m2=m2, xsi=0, zm=z)
        if flim > detector.freq[0]+0.15:
            psd = detector.Make_psd()
            hp, hc = pycbc.waveform.get_fd_waveform(approximant=waveform_approx, mass1=m1 * (1. + z), mass2 = m2 * (1. + z),
                                                spin1x=0., spin1y=0., spin1z=0, spin2x=0., spin2y=0., spin2z=0,
                                                delta_f=float(detector.freq[1]-detector.freq[0]), f_lower=float(detector.freq[0]), distance=luminosity_distance, f_ref=20.,
                                                inclinaison=0)
            snr_l = pycbc.filter.matchedfilter.sigma(np.sqrt(2) * hp, psd=psd,
                                                 low_frequency_cutoff=float(detector.freq[0]),
                                                 high_frequency_cutoff=float(np.max(detector.freq)))
        else :
            snr_l = 0
        return snr_l


def SNR_catalog( catalog_name:str, det_list: list, waveform: str, freq: np.array):
    """
    Computes the optimal Signal-to-Noise Ratio (SNR) for each event in the catalog
    and saves the updated catalog with additional SNR columns.

    Parameters
    ----------
    det_list : list
        List of `Detector` objects used for SNR computation.
    waveform : str
        The waveform model used for gravitational wave signal generation.
    freq : np.array
        Frequency array used for the SNR computation.

    Notes
    -----
    - If an event produces an invalid `htildsq` value, it is logged in an error file.
    - The function modifies and saves the catalog with SNR values for each detector.
    """

    catalog_path = f'./Run/{params["name_of_project_folder"]}/Astro_Models/Catalogs'
    Cat = pd.read_csv(f'{catalog_path}/{catalog_name}.dat', sep='\t', index_col=False)
    print(f'SNR calculation for {catalog_name}')
    ntot = len(Cat.z)

    # Initialize SNR columns to zero for each detector
    for det in det_list:
        Cat[det.name] = 0.0

    error_file = f'{catalog_path}/{catalog_name}_errors.txt'

    for evt in range(len(Cat)):
        event = Cat.iloc[[evt]]  # Select a single event as a DataFrame
        htildsq, freqency = GWk_no_ecc_pycbcwf(evt=event, freq=freq, approx=waveform, n=evt, size_catalogue=ntot,
                                     inc_option=params['Inclination'])
        if isinstance(htildsq, int):  # If htildsq is an integer, log an error
            if not os.path.exists(error_file):  # Check if file exists before writing
                with open(error_file, "w") as f:
                    f.write('m1 m2 z\n')

            with open(error_file, "a") as f:
                f.write(f"{event['m1'].iloc[0]} {event['m2'].iloc[0]} {event['z'].iloc[0]}\n")
        else:
            for det in det_list:
                Sn = det.psd
                comp = det.deltaf * 4. * htildsq / Sn
                comp = np.nan_to_num(comp, nan=0, posinf=0)
                SNR = np.sqrt(comp.sum())  # Compute final SNR
                Cat.at[evt, det.name] = SNR  # Assign the computed SNR to the event

    # Save the updated catalog with new SNR columns
    Cat.to_csv(f'{catalog_path}/{catalog_name}.dat', sep='\t', index=False)


def SNR_pycbc_catalog(catalog_name:str, det_list: list, waveform: str, catalogue_path = None):
    if catalogue_path == None :
        catalogue_path = f'./Run/{params["name_of_project_folder"]}/Astro_Models/Catalogs'
    Cat = pd.read_csv(f'{catalogue_path}/{catalog_name}.dat', sep='\t', index_col=False)
    print(f'SNR calculation for {catalog_name}')
    ntot = len(Cat.z)

    if 'Dl' not in Cat.columns :
        # Initialize SNR columns to zero for each detector
        cosmology = Cosmology.load(params['Cosmo_model'])
        cosmology.info()
        Cat['Dl']= cosmology.compute_dl(Cat['z'])
        Cat.to_csv(f'{catalogue_path}/{catalog_name}.dat', sep='\t', index=False)
        print('Luminosity distance computation complete.')
    else :
        print('Luminosity distance already in original catalogue.')

    if 'fmerg' not in Cat.columns :
        Cat['fmerg']= fmerg_f(Cat['m1'],Cat['m2'], np.zeros(len(Cat['m1'])), Cat['z'])
        Cat.to_csv(f'{catalogue_path}/{catalog_name}.dat', sep='\t', index=False)
        print('Merging frequency computation complete.')
    else :
        print('Merging frequency already in original catalogue.')

    Cat= Cat[Cat['fmerg']>10]
    for det in det_list:
        low_freq = 10
        high_freq = 1000
        delta_f = 1
        print(f"Starting computations for {det}")
        detector = Detector.load(det)
        psd_data = detector.get_psd_pycbc_compatible()
        opt_snr = [sigma(get_fd_waveform(approximant=waveform,
                                                                            mass1=m1 * (1. + z),
                                                                            mass2=m2 * (1. + z),
                                                                            spin1x=0., spin1y=0., spin1z=0.,
                                                                            spin2x=0., spin2y=0., spin2z=0.,
                                                                            delta_f=delta_f, f_lower=low_freq,
                                                                            distance=ld,
                                                                            inclination=0., f_ref=20.)[0],
                                             psd=psd_data,
                                             low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
                   for m1, m2, ld, z in zip(Cat["m1"], Cat["m2"], Cat["Dl"], Cat["z"])]

        Cat[detector.name+'_pycbc'] = opt_snr  # Assign the computed SNR to the event

    # Save the updated catalog with new SNR columns
    Cat.to_csv(f'{catalogue_path}/{catalog_name}.dat', sep='\t', index=False)



def SNR_single(event, det_list: list, network: Network, waveform: str, freq: np.array, savefile: bool = False,
               name='event'):
    """
    Computes the optimal Signal-to-Noise Ratio (SNR) for a single event using a list of detectors.

    The function calculates the SNR for each detector in the list and combines them for the given network.
    If requested, the results are saved in a `.dat` file.

    Parameters
    ----------
    event : dict
        Dictionary containing the event parameters, including 'm1', 'm2', 'z', and 'Dl'.
    det_list : list
        List of `Detector` objects used for SNR computation.
    network : Network
        The network of detectors for combining SNRs.
    waveform : str
        The waveform model used for gravitational wave signal generation.
    freq : np.array
        Frequency array used for the SNR computation.
    savefile : bool, optional
        If True, saves the SNR results to a file (default is False).
    name : str, optional
        Name of the event, used for the output filename (default is 'event').

    Returns
    -------
    pd.Series
        Series containing the combined SNR for the specified network.

    Notes
    -----
    - If an event produces an invalid `htildsq` value, it is logged in an error file.
    - The function modifies and saves the catalog with SNR values for each detector.
    - The function checks if required keys ('m1', 'm2', 'z', 'Dl') are present in the event.
    """

    # Check if all necessary keys are present in the event
    required_keys = ['m1', 'm2', 'z', 'Dl']
    if not all(key in event for key in required_keys):
        print(f"Error: Missing required keys for SNR computation. Required: {required_keys}. "
              f"Currently available: {event.keys()}")
        return None

    # Initialize an empty DataFrame to store SNRs for each detector
    SNRs = pd.DataFrame(index=[0])  # Single row for this event

    # Compute the squared waveform for the event
    htildsq, frequency = GWk_no_ecc_pycbcwf(evt=event, freq=freq, approx=waveform, n=event, size_catalogue=1,
                                 inc_option='Optimal', disable_progress_bar=True)

    # Check for invalid htildsq value
    if isinstance(htildsq, float):
        error_file = "error_log.txt"  # Path to the error log file
        if not os.path.exists(error_file):
            with open(error_file, "w") as f:
                f.write('m1 m2 z\n')

        with open(error_file, "a") as f:
            f.write(f"{event['m1']} {event['m2']} {event['z']}\n")
        print('Event out of frequency band')
        return htildsq # Return a Series with zero SNR

    else:
        # Compute SNR for each detector
        for det in det_list:
            Sn = det.psd  # Power spectral density
            comp = det.deltaf * 4. * htildsq / Sn
            comp = np.nan_to_num(comp, nan=0, posinf=0, neginf=0)  # Handle inf and NaN values
            SNR = np.sqrt(np.sum(comp))
            SNRs[det.name] = SNR

        #print("Individual SNRs per detector:")
        #print(SNRs)

        # Combine SNRs from all detectors in the network
        SNRs = combine_snr_detectors(det_list=det_list, net=network, SNRs=SNRs)

        #print("Combined SNR for the network:")
        #print(SNRs)

        # Save the updated SNR catalog to a file if requested
        if savefile:
            SNRs.to_csv(f'{name}.dat', sep='\t', index=False)

        return SNRs  # Return the dataframe

def combine_snr_detectors(det_list: list, net, SNRs: pd.DataFrame, savefile: bool = False, name: str = 'event'):
    """
    Combines the SNRs from multiple detectors for a given network.

    The function calculates the optimal and network SNRs by summing the squares of the individual
    detector contributions. A random factor from the factor table is used to simulate different
    configurations.

    Parameters
    ----------
    det_list : list
        List of `Detector` objects.
    net : str
        Name of the network (e.g., 'HLV', 'ET').
    SNRs : pd.DataFrame
        DataFrame containing the individual SNRs for each detector.
    savefile : bool, optional
        If True, saves the combined SNR results to a file (default is False).
    name : str, optional
        Name of the event, used for the output filename (default is 'event').

    Returns
    -------
    pd.DataFrame
        Updated DataFrame containing the combined SNRs for the network.

    Raises
    ------
    FileNotFoundError
        If 'factor_table.dat' is not found in 'AuxiliaryFiles/'.

    Notes
    -----
    - This function uses a factor table to apply configuration-specific corrections.
    - It computes both optimal and realistic SNRs for the network.
    """



    # Load factor table for configuration-specific corrections
    try:
        fd_table = pd.read_csv('AuxiliaryFiles/factor_table.dat', sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError("'factor_table.dat' file is not found in 'AuxiliaryFiles/'.")

    # Initialize columns for combined SNRs
    SNRs[f'{net.name}_optimal'] = 0.0
    SNRs[net.name] = 0.0

    # Randomly select a factor row (position and inclination)
    fd = fd_table.iloc[np.random.randint(0, len(fd_table))]

    # Compute combined SNR contributions for each detector
    for det in det_list:
        if det.name in SNRs.columns:
            SNRs[f'{net.name}_optimal'] += SNRs[det.name] ** 2
            config = det.configuration
            SNRs[net.name] += (SNRs[det.name] * fd[config]) ** 2
        else:
            print(f"Detector '{det.name}' pre-computations are missing in '{net.name}'.")

    # Finalize combined SNR computation
    SNRs[net.name] = np.sqrt(SNRs[net.name])
    SNRs[f'{net.name}_optimal'] = np.sqrt(SNRs[f'{net.name}_optimal'])

    # Save the combined SNRs to a file if requested
    if savefile:
        SNRs.to_csv(f'{net.name}_combined_snr.dat', sep='\t', index=False)

    return SNRs

def compute_SNR_Networks_catalog(catalog_name: str):
    # Charger la table de facteurs
    try:
        fd_table = pd.read_csv('AuxiliaryFiles/factor_table.dat', sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError("Le fichier 'factor_table.dat' est introuvable dans 'AuxiliaryFiles/'.")

    # Parcourir les réseaux dans la liste des réseaux
    for net in params['network_list'].keys():
        network = Network(name=net)

        # Parcourir chaque catalogue
        catalog_path = f"./Run/{params['name_of_project_folder']}/Astro_Models/Catalogs/{catalog_name}.dat"

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
                detector_object = Detector.load(name = det)
                config = detector_object.configuration
                Cat[net] += (Cat[det] * fd[config]) ** 2
            else:
                print(f"Detector '{det}' pre computations are missing in '{cat}'.")

        # Finaliser les calculs des SNR pour le réseau
        Cat[net] = np.sqrt(Cat[net])
        Cat[f'{net}_optimal'] = np.sqrt(Cat[f'{net}_optimal'])

        # Sauvegarder les résultats dans le fichier
        output_path = f"./Run/{params['name_of_project_folder']}/Astro_Models/Catalogs/{catalog_name}.dat"
        try:
            Cat.to_csv(output_path, sep='\t', index=False)
        except Exception as e:
            print(f"Problem during the saving of file {cat}: {e}")




def SNR_LISA( catalog_name:str, det_list: list, freq: np.array):
    """
    Computes the optimal Signal-to-Noise Ratio (SNR) for each event in the catalog
    and saves the updated catalog with additional SNR columns.

    Parameters
    ----------
    det_list : list
        List of `Detector` objects used for SNR computation.
    waveform : str
        The waveform model used for gravitational wave signal generation.
    freq : np.array
        Frequency array used for the SNR computation.

    Notes
    -----
    - If an event produces an invalid `htildsq` value, it is logged in an error file.
    - The function modifies and saves the catalog with SNR values for each detector.
    """

    catalog_path = f'./Run/{params["name_of_project_folder"]}/Astro_Models/Catalogs'
    Cat = pd.read_csv(f'{catalog_path}/{catalog_name}.dat', sep='\t', index_col=False)
    print(f'SNR calculation for {catalog_name}')
    ntot = len(Cat.z)

    # Initialize SNR columns to zero for each detector
    for det in det_list:
        Cat[det.name] = 0.0

    error_file = f'{catalog_path}/{catalog_name}_errors.txt'

    for evt in range(len(Cat)):
        event = Cat.iloc[[evt]]  # Select a single event as a DataFrame
        htildsq, freqency = GWk_no_ecc_pycbcwf(evt=event, freq=freq, approx=waveform, n=evt, size_catalogue=ntot,
                                     inc_option=params['Inclination'])
        if isinstance(htildsq, int):  # If htildsq is an integer, log an error
            if not os.path.exists(error_file):  # Check if file exists before writing
                with open(error_file, "w") as f:
                    f.write('m1 m2 z\n')

            with open(error_file, "a") as f:
                f.write(f"{event['m1'].iloc[0]} {event['m2'].iloc[0]} {event['z'].iloc[0]}\n")
        else:
            for det in det_list:
                Sn = det.psd
                comp = det.deltaf * 4. * htildsq / Sn
                comp = np.nan_to_num(comp, nan=0, posinf=0)
                SNR = np.sqrt(comp.sum())  # Compute final SNR
                Cat.at[evt, det.name] = SNR  # Assign the computed SNR to the event

    # Save the updated catalog with new SNR columns
    Cat.to_csv(f'{catalog_path}/{catalog_name}.dat', sep='\t', index=False)

    #check 1803.01944
