import os
import json
import shutil
import importlib.util

PARAMS_FILE = "../Princess/Run/Params.json"  # Global parameter containing the path to Params.json

def import_module_from_path(module_name, file_path):
    """Dynamically import a package from its path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def Make_param_file(getting_started_path="Princess/Run/getting_started.py",
                    advanced_params_path="Princess/Run/advanced_params.py",
                    params_file_path=None):
    """Generate Params.json, to gather all the parameters set in getting_started.py and advanced_params.py"""

    # Importation dynamique des modules
    GS = import_module_from_path("GS", getting_started_path)
    AP = import_module_from_path("AP", advanced_params_path)

    # If necessary create project folder
    project_folder = os.path.join('Run', GS.name_of_project_folder)
    os.makedirs(project_folder, exist_ok=True)

    # Creates dictionnary with getting_starded parameters
    param_dictionary = {
        'name_of_project_folder': GS.name_of_project_folder,
        'astro_model_list': GS.astro_model_list,
        'detector_list': GS.detector_list,
        'network_list': GS.network_list,
        'frequency_size': GS.frequency_size,
        'n_cpu_max': GS.n_cpu_max,
        'overwrite': {
            'astromodel': GS.rerun_astromodels,
            'individual_snr': GS.rerun_snr_computation,
            'detectors': GS.rerun_detectors
        },
        'results': {
            'cleaning': GS.run_data_cleaning,
            'plots': GS.run_plots
        }
    }

    # Creates dictionnary with advanced parameters
    advParams = {
        "AM_params": {
            'input_parameters': AP.input_parameters,
            'keepID': AP.keepID,
            'ID_col': AP.ID_col
        },
        "detector_params": {
            'detectors_avail': AP.detectors_avail,
            'psd_attributes': AP.psd_attributes,
            'types': AP.types
        },
        "Inclination": AP.Inclination,
        "Cosmo_model": AP.cosmo_model
    }

    # Définition du chemin du fichier Params.json
    params_file_path = params_file_path or os.path.join('Run', 'Params.json')

    # Sauvegarde des paramètres et mise à jour de la variable globale
    global PARAMS_FILE
    PARAMS_FILE = set(GS.name_of_project_folder, param_dictionary, advParams, params_file_path)


def set(_projectFolder, _paramDictionnary, _advParamDictionnary, params_file_path):
    """Save parameters in Params.json, and update the value of PARAMS_FILE"""

    try:
        # Validation des entrées
        if not isinstance(_paramDictionnary, dict) or not isinstance(_advParamDictionnary, dict):
            raise ValueError("Les paramètres doivent être des dictionnaires.")

        # Merging dictionnaries
        output = {**_paramDictionnary, **_advParamDictionnary}
        json_object = json.dumps(output, indent=4)

        # Eventually creates folders
        os.makedirs(os.path.dirname(params_file_path), exist_ok=True)

        # Write .json
        with open(params_file_path, "w") as file:
            file.write(json_object)

        print(f"Paramètres enregistrés dans {params_file_path}.")

        return params_file_path  # Retourne le chemin du fichier Params.json

    except Exception as e:
        print(f"Erreur lors de l'écriture des fichiers : {e}")
        return None

def clean():
    params = json.load(open('Run/Params.json', 'r'))
    for am in params['astro_model_list'].keys():
        os.remove('Run/' + params['name_of_project_folder']+'/'+ am +'_AM.pickle')
    for det in params['detector_list'].keys():
        os.remove('Run/' + params['name_of_project_folder']+'/'+ det +'_DET.pickle')
    for net in params['network_list'].keys():
        os.remove('Run/' + params['name_of_project_folder'] + '/' + net + '_NET.pickle')
    os.remove('Run/Params.json')


def make_params_template(path):
    """Copy getting_started.py and advanced_params.py in path"""

    try:
        os.makedirs(destination, exist_ok=True)  # Eventually creates destination folder
        shutil.copy("Princess/Run/getting_started.py", destination)
        shutil.copy("Princess/Run/advanced_params.py", destination)
        print(f"Files copy to {destination}.")
    except Exception as e:
        print(f"Error while loading the files : {e}")
