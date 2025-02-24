import os
import json
import Run.getting_started as GS # Initiate the file Params.json
import astrotools.astromodel as AM
import astrotools.detection as DET
import stochastic.background as BKG
import Run.advanced_params as AP
import shutil



if __name__ == '__main__':
    params = json.load(open('Run/Params.json', 'r'))

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
    DET.initialization()

    #Compute SNR and individual analysis for the individual detection
    AM.process_astromodel()

    #Compute backgrounds, residuals and the corresponding analysis
    BKG.process_background_computation()

    #Copy the getting starter file to the project repository
    source_file = "Run/getting_started"
    destination_folder = f"Run/{params['name_of_project_folder']}/"
    shutil.copy(source_file, destination_folder)

    if params['results']['cleaning'] == True :
        AP.clean()

