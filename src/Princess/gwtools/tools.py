
import math
import os
import numpy as np
import pandas as pd
import json
import pickle
from astropy.cosmology import Planck15
from Princess.astrotools.detection import Detector
from Princess.gwtools.Network import Network
from Princess.stochastic import basic_functions as BF
import Princess.gwtools.Detector as DET
import gwtools.Network as NET
from Princess.gwtools.htild import GWk_no_ecc_pycbcwf


def SNR_single_old(event, det_list: list, network : Network, waveform: str, freq: np.array, savefile: bool = True, name = 'event'):
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

    if not all(key in event.keys() for key in ['m1', 'm2', 'z', 'Dl']):
        print(f'Error, some info are missing for the snr computation. \n '
              f'Check that you event contain at least m1 m2, z, Dl items. \n'
              f'Currently available {event.keys()}')
    SNRs = pd.DataFrame({})
    for det in det_list:
        SNRs[det.name] = 0.0
    htildsq = GWk_no_ecc_pycbcwf(evt=event, freq=freq, approx=waveform, n=evt, size_catalogue=ntot,
                                    inc_option='Optimal', disable_progress_bar= True)
    if isinstance(htildsq, int):  # If htildsq is an integer, log an error
        if not os.path.exists(error_file):  # Check if file exists before writing
            with open(error_file, "w") as f:
                f.write('m1 m2 z\n')

        with open(error_file, "a") as f:
            f.write(f"{event['m1'].iloc[0]} {event['m2'].iloc[0]} {event['z'].iloc[0]}\n")
        print(f'event out of frequency band')
    else:
        for det in det_list:
            Sn = det.psd
            comp = det.deltaf * 4. * htildsq / Sn
            comp = np.nan_to_num(comp, nan=0, posinf=0)
            SNR = np.sqrt(comp.sum())  # Compute final SNR
            SNRs.at[det.name] = SNR  # Assign the computed SNR to the event

        SNRs = combine_snr_detectors(det_list, network, SNRs)

        # Save the updated catalog with new SNR columns
        if savefile == True :
            SNRs.to_csv(f'{cat}.dat', sep='\t', index=False)
        return SNRs[network.name]


def combine_snr_detecors_old(det_list, net, SNRs, savefile: bool = False, name : str = 'event'):
    # Loading factor table
    try:
        fd_table = pd.read_csv('AuxiliaryFiles/factor_table.dat', sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError("'factor_table.dat' file is not found in 'AuxiliaryFiles/'.")

    # Column initialisation
    SNRs[f'{net}_optimal'] = 0.0
    SNRs[net] = 0.0

    # Select a random position, inclination
    fd = fd_table.iloc[np.random.randint(0, len(fd_table))]

    # Compute detectors contributions
    if det in SNRs.columns:
        SNRs[f'{net}_optimal'] += SNRs[det] ** 2
        detector_object = det_list[det]
        config = detector_object.configuration
        SNRs[net] += (SNRs[det] * fd[config]) ** 2
    else:
        print(f"Detector '{det}' pre computations are missing in '{cat}'.")

    # Finalise computation
    SNRs[net] = np.sqrt(Cat[net])
    SNRs[f'{net}_optimal'] = np.sqrt(Cat[f'{net}_optimal'])

    return SNRs


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
    htildsq = GWk_no_ecc_pycbcwf(evt=event, freq=freq, approx=waveform, n=event, size_catalogue=1,
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



def SNR(name, catalog_path, det_list: list, waveform: str, freq: np.array):
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

    for cat in catalog_path:
        Cat = pd.read_csv(f'{cat}.dat', sep='\t', index_col=False)
        print(f'SNR calculation for {cat}')
        ntot = len(Cat.z)

        # Initialize SNR columns to zero for each detector
        for det in det_list:
            Cat[det.name] = 0.0

        error_file = f'{cat}_errors.txt'

        for evt in range(len(Cat)):
            event = Cat.iloc[[evt]]  # Select a single event as a DataFrame
            htildsq = GWk_no_ecc_pycbcwf(evt=event, freq=freq, approx=waveform, n=evt, size_catalogue=ntot,
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
        Cat.to_csv(f'{cat}.dat', sep='\t', index=False)


def compute_SNR_Networks(self):
    # Charger la table de facteurs
    try:
        fd_table = pd.read_csv('AuxiliaryFiles/factor_table.dat', sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError("'factor_table.dat' file is not found in 'AuxiliaryFiles/'.")

    # Parcourir les réseaux dans la liste des réseaux
    for net in params['network_list'].keys():
        network = DET.Network(name=net)

        # Parcourir chaque catalogue
        for cat in self.catalogs:
            catalog_path = f"./Run/{params['name_of_project_folder']}/Astro_Models/Catalogs/{cat}"

            try:
                Cat = pd.read_csv(catalog_path, sep='\t')
            except FileNotFoundError:
                print(f" '{cat}.dat' file in not found. Check that th name ends with .dat")
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
                    detector_object = DET.Detector.load(name = det)
                    config = detector_object.configuration
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


