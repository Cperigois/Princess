import pandas
import os
import json


def set_old(_projectFolder, _paramDictionnary, _advParamDictionnary):

    output = {**_paramDictionnary, **_advParamDictionnary}
    json_object = json.dumps(output, indent=len(output.keys()))
    with open('Run/Params.json', "w") as file:
        file.write(json_object)  # encode dict into JSON
    if not os.path.exists('Run/' + _projectFolder):
        os.mkdir('Run/' + _projectFolder)
    with open('Run/' + _projectFolder + '/Params.json', "w") as file:
        file.write(json_object)  # encode dict into JSON
    print("Done writing dict into Run/Params.json file and in Run/", str(_projectFolder), "/Params.json")


def set(_projectFolder, _paramDictionnary, _advParamDictionnary):
    try:
        # Validate inputs
        if not isinstance(_paramDictionnary, dict) or not isinstance(_advParamDictionnary, dict):
            raise ValueError("Both _paramDictionnary and _advParamDictionnary must be dictionaries.")

        # Merge dictionaries
        output = {**_paramDictionnary, **_advParamDictionnary}
        json_object = json.dumps(output, indent=4)  # Pretty print JSON with a standard indent level

        # Define file paths
        base_params_path = os.path.join('Run', 'Params.json')
        project_folder_path = os.path.join('Run', _projectFolder)
        project_params_path = os.path.join(project_folder_path, 'Params.json')

        # Ensure 'Run' and project folder exist
        os.makedirs('Run', exist_ok=True)
        os.makedirs(project_folder_path, exist_ok=True)

        # Write to base Params.json
        with open(base_params_path, "w") as file:
            file.write(json_object)

        # Write to project-specific Params.json
        with open(project_params_path, "w") as file:
            file.write(json_object)

        print(f"Successfully wrote Params.json to {base_params_path} and {project_params_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")

def clean():
    params = json.load(open('Run/Params.json', 'r'))
    for am in params['astro_model_list'].keys():
        os.remove('Run/' + params['name_of_project_folder']+'/'+ am +'_AM.pickle')
    for det in params['detector_list'].keys():
        os.remove('Run/' + params['name_of_project_folder']+'/'+ det +'_DET.pickle')
    for net in params['network_list'].keys():
        os.remove('Run/' + params['name_of_project_folder'] + '/' + net + '_NET.pickle')
    os.remove('Run/Params.json')

""" Types of detectors notes : 
    - LISA and PTA types are not available
    - min_fsize stands to ensure that the delta_f stads below or equal to 1. Otherwise Pycbc interpolation crashes.
    - 'refs' stands for the reference frequency for Omega compotation.
"""

types = {"2G": {'freq': {'min': 10., 'max' : 2000., 'scale': 'lin', 'ref': [25.], 'min_fsize' : 1990}, 'waveform': "IMRPhenomD"},
         "3G": {'freq':{'min': 1., 'max' : 2500., 'scale': 'lin', 'ref': [10.,25.], 'min_fsize' : 2500}, 'waveform': "IMRPhenomD"},
         'LISA': { 'freq':{'min': 1.e-4, 'max' : 0.1, 'scale': 'log', 'ref': 0.001}, 'waveform': "Ajith"},
         'PTA': {  'freq':{'min': 1.e-10, 'max' : 1.e-7, 'scale': 'log', 'ref': 1.e-9},'waveform': "Inspiral"}
        }

"""             *** ASTROMODEL ***           """

"""
Parameters of the user input catalogue. Please do not change the left column.
{<Input catalogue names> : <Names for output catalogues>}
"""

input_parameters = {
    "m1": "m1",  # mass of compact object 1
    "mrem1": "m1",
    "Mass1": "m1",

    "m2": "m2",  # mass of compact object 2
    "mrem2": "m2",
    "Mass2": "m2",

    "chi1": "chi1",  # spin magnitude of compact object 1
    "th1": "theta1",  # angle between angular momentum and spin for compact object 1
    "cos_nu1": "costheta1",  # cosine of tilt 1st supernova
    "cmu1": "costheta1",  # cosine of tilt 1st supernova

    "chi2": "chi2",  # spin magnitude of compact object 2
    "th2": "theta2",  # angle between angular momentum and spin for compact object 2
    "cos_nu2": "costheta2",  # cosine of tilt 2nd supernova
    "cmu2": "costheta2",  # cosine of tilt 2nd supernova

    "z_merg": "z",  # redshift at merger
    "z_form": "zForm",  # redshift at merger

    "ID": "id",  # Id of the binary
    "time_delay [yr]": "timeDelay",  # time delay
    "time_delay": "timeDelay",  # time delay
    "mZAMS1": "mzams1",  # Zero Age Main Sequence mass of the 1st component
    "mZAMS2": "mzams2",  # Zero Age Main Sequence mass of the 2nd component
    "M_SC": "m_sc",  # mass of star cluster
    "Z progenitor": "z_progenitor",
    "Channel": "Channel",
    "Ng": "Ng",
    "tSN1": "tsn1",
    "tSN2": "tsn2"
}

Inclination = 'Rand' # To be chosen among {'InCat', 'Rand', 'Optimal'}
Position = 'Optimal'
orbit_evo = False
keepID = False
ID_col = 'Name'

"""             *** Sampling parameters ***           """

"""
Parameters for the sampling of the catalogue.
"""

sampling_size = 100  # suggested 100 for primary checks and 500000 for full analysis
sampling_number_of_walkers = 16  # number of MCMC walkers
sampling_chain_length = 500  # 50 #500  # length of MCMC chain
sampling_bandwidth_KDE = 0.075  # KDE bandwidth to use

"""             *** Detector settings ***           """

"""
Parameters for the sampling of the catalogue.
"""

# List of accessible detectors from text file
detectors_avail = ["Livingston_O1", "Livingston_O2", "Livingston_O3a", "Livingston_O3b", "Hanford_O1", "Hanford_O2",
                   "Hanford_O3a", "Virgo_O2", "Virgo_O3a", "Virgo_O3b", "Virgo_O4", "Virgo_O5", "LIGO_O4", "LIGO_O5", "LIGO_Design",
                   "ET_Design", "ET_10km", "ET_15km", "ET_20km"]

# For psd read from files, the values were set when constructing the files
# For pycbc psd, the min and max frequency can be modified, as long as it is understood where the model breaks
# For delta_freq_min for pycbc psd, tests showed that a value of 0.01 was generating malloc error, which is the
# reason why the minimum value was set to 0.015.
psd_attributes = {
    "Livingston_O1": {"psd_name": "Livingston_O1_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                      "delta_freq_min": 0.025},
    "Livingston_O2": {"psd_name": "Livingston_O2_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                      "delta_freq_min": 0.025},
    "Livingston_O3a": {"psd_name": "Livingston_O3a_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                       "delta_freq_min": 0.025},
    "Livingston_O3b": {"psd_name": "Livingston_O3b_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                       "delta_freq_min": 0.025},
    "Hanford_O1": {"psd_name": "Hanford_O1_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                   "delta_freq_min": 0.025},
    "Hanford_O2": {"psd_name": "Hanford_O2_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                   "delta_freq_min": 0.025},
    "Hanford_O3a": {"psd_name": "Hanford_O3a_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                    "delta_freq_min": 0.025},
    "Hanford_O3b": {"psd_name": "Hanford_O3b_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                    "delta_freq_min": 0.025},
    "Virgo_O2": {"psd_name": "Virgo_O2_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                 "delta_freq_min": 0.025},
    "Virgo_O3a": {"psd_name": "Virgo_O3a_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                  "delta_freq_min": 0.025},
    "Virgo_O3b": {"psd_name": "Virgo_O3b_psd", "in_pycbc": False, "min_freq": 16.0, "max_freq": 1023.0,
                  "delta_freq_min": 0.025},
    "LIGO_Design": {"psd_name": "aLIGODesignSensitivityP1200087", "in_pycbc": True, "min_freq": 0.01,
                    "max_freq": 2048.0, "delta_freq_min": 0.015},
    "ET_Design": {"psd_name": "EinsteinTelescopeP1600143", "in_pycbc": True, "min_freq": 0.01,
                  "max_freq": 2048.0, "delta_freq_min": 0.015},
    "ET_10km": {"psd_name": "ET10_CoBa_psd", "in_pycbc": False, "min_freq": 0.01,
                  "max_freq": 10000.0, "delta_freq_min": 0.25},
    "ET_15km": {"psd_name": "ET15_Coba_psd", "in_pycbc": False, "min_freq": 0.01,
                  "max_freq": 10000.0, "delta_freq_min": 0.015},
    "ET_20km": {"psd_name": "ET20_Coba_psd", "in_pycbc": False, "min_freq": 0.01,
                  "max_freq": 10000.0, "delta_freq_min": 0.015},
    "LIGO_O4": {"psd_name": "LIGO_O4_psd", "in_pycbc": False, "min_freq": 10.21659,
                  "max_freq": 4.995378e+03, "delta_freq_min": 0.015},#https://emfollow.docs.ligo.org/userguide/capabilities.html
    "Virgo_O4": {"psd_name": "Virgo_O4_psd", "in_pycbc": False, "min_freq": 10.0,
                  "max_freq": 10000.0, "delta_freq_min": 0.1},#https://emfollow.docs.ligo.org/userguide/capabilities.html
    "KAGRA_O4": {"psd_name": "Kagra_O4_psd", "in_pycbc": False, "min_freq": 1.0,
                  "max_freq": 10000.0, "delta_freq_min": 0.1},#https://emfollow.docs.ligo.org/userguide/capabilities.html
    "LIGO_O5": {"psd_name": "LIGO_O5_psd", "in_pycbc": False, "min_freq": 5.0,
                  "max_freq": 5000, "delta_freq_min": 0.015},#https://emfollow.docs.ligo.org/userguide/capabilities.html
    "Virgo_O5": {"psd_name": "Virgo_O5_psd", "in_pycbc": False, "min_freq": 10.0,
                  "max_freq": 10000.0, "delta_freq_min": 0.1},#https://emfollow.docs.ligo.org/userguide/capabilities.html
    "KAGRA_O5": {"psd_name": "Kagra_O5_psd", "in_pycbc": False, "min_freq": 1.0,
                  "max_freq": 10000.0, "delta_freq_min": 0.4999}, #https://emfollow.docs.ligo.org/userguide/capabilities.html
    "CE_20km": {"psd_name": "CE_20km_psd", "in_pycbc": False, "min_freq": 1.0,
                  "max_freq": 5000.0, "delta_freq_min": 0.4999}, #https://emfollow.docs.ligo.org/userguide/capabilities.html
    "CE_40km": {"psd_name": "CE_40km_psd", "in_pycbc": False, "min_freq": 1.0,
                  "max_freq": 5000.0, "delta_freq_min": 0.4999} #https://emfollow.docs.ligo.org/userguide/capabilities.html
}

"""             *** Event Selection ***           """

"""
These parameters are used to select the confident events from LVK.
The standard population paper uses pastro>0.9, FAR<0.25 and no constrain on the SNR (i.e. SNR>0)
"""

pAstroLimit = 0  # use only GW events with a pastro > pAstroLimit
farLimit = 100  # use only GW events with a far < farLimit
snrLimit = 0  # use only GW events with a snr > snrLimit
available_obs_runs = {"O1": {'detector': 'Livingston_O1', 'delta_freq': 1.0, 'duration': 0.1331},
                      # 48.6 days (arxiv 1606.04856, section 2, page 8)
                      'O2': {'detector': 'Livingston_O2', 'delta_freq': 1.0, 'duration': 0.3231},
                      # 118  days (arxiv 1811.12907, section 2B, page 4)
                      'O3a': {'detector': 'Livingston_O3a', 'delta_freq': 1.0, 'duration': 0.2230},
                      # 81.4  days (arxiv 2010.14527, section 2, page 10)
                      'O3b': {'detector': 'Livingston_O3b', 'delta_freq': 1.0,
                              'duration': 0.2053}}  # 75.0 days (GWTC 3 paper)

"""             *** Bayes Model Processing ***           """

"""
Parameters for the computation of the match and the efficiency.
"""

bayes_model_processing_waveform_approximant = "IMRPhenomD"  # waveform approximant the fastest beeing "IMRPhenomD"
# but to account for precession we recommand "IMRPhenomPv2"
option_SNR_computation = 0
bayes_model_processing_bandwidth_KDE = 0.075  # KDE bandwidth to use
bayes_option_compute_likelihood = "All"
bayes_option_multichannel = "NoRate"

"""             *** Parameter output ***           """

"""
All parameters in this file has to end in the dictionnary, for the creation of the json file.
"""

advParams = {"AM_params": {'input_parameters': input_parameters, 'keepID' : keepID, 'ID_col' : ID_col},
             "detector_params": {'detectors_avail': detectors_avail,
                                 'psd_attributes': psd_attributes,
                                 'types': types},
             "Inclination" : Inclination
             }
