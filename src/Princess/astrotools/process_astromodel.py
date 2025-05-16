print(f"Loading {__name__}")
import os
import json
import importlib.resources
from Princess.astrotools.astromodel import AstroModel
from Princess.Run.settings import PARAMS_FILE

# Check PARAMS_FILE value
if not PARAMS_FILE or not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"The file parameter {PARAMS_FILE} is missing,. Execute Run.settings.Make_params_file() first.")

# Charge le fichier de paramètres
with open(PARAMS_FILE, "r") as f:
    params = json.load(f)

def process_astromodel():
    """
    Main function to process all astromodels defined in the configuration.
    Loads existing models if available, otherwise initializes, saves,
    generates catalogs, and computes SNR.
    """

    # Import parameter file
    with importlib.resources.open_text("Princess.Run", "Params.json") as f:
        params = json.load(f)

    # Ensure necessary directories exist
    base_path = f"Run/{params['name_of_project_folder']}"
    astro_models_path = f"{base_path}/Astro_Models"
    catalogs_path = f"{astro_models_path}/Catalogs"

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(astro_models_path, exist_ok=True)
    os.makedirs(catalogs_path, exist_ok=True)

    # Process each astromodel defined in the parameter configuration
    print('Begin astromodel processing')
    for model_key, model_params in params['astro_model_list'].items():
        try:
            model_name = model_params['name']
            model_save_path = f"{base_path}/{model_name}_AM.pickle"

            # Vérifier si le modèle existe déjà
            print(not params['overwrite']['astromodel'])
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
                # Sauvegarde du modèle pour un usage future
                am.save()
                print('ICI')
            # if the user ask for the rurun of SNR without rerunning the astromodel
            if params['overwrite']['individual_snr'] and (not params['overwrite']['astromodel']):
                am.check_SNR_reboot() # Set to False the computation of SNRs to ensure its recomputation
                am.compute_SNR()

        except Exception as e:
            print(f"Error processing AstroModel '{model_key}': {e}")