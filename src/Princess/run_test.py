import os
import json
from Princess.astrotools.astromodel import process_astromodel
from Princess.gwtools.initialization import initialization
from Princess.stochastic.background import process_background_computation
import test.advanced_params as AP
import shutil



if __name__ == '__main__':
    params = json.load(open('test/Params.json', 'r'))

    # Make sure directories are created
    if not os.path.exists('Run/' + params['name_of_project_folder']):
        os.mkdir('Run/' + params['name_of_project_folder'])
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/")
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs")

    #Read and reshape astrophysical models, save instance of each astrophysical models for later
#    AM.initialization()

    #Read and reshape detectors and networks, save instances for each of them
    initialization()

    #Compute SNR and individual analysis for the individual detection
    process_astromodel()

    #Compute backgrounds, residuals and the corresponding analysis
    process_background_computation()

    #Copy the getting_starting file to the project repository
    source_file = "Run/getting_started"
    destination_folder = f"Run/{params['name_of_project_folder']}/"
    shutil.copy(source_file, destination_folder)

    if params['results']['cleaning'] == True :
        AP.clean()

