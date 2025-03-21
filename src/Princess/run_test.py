import os
import json
from Princess.astrotools.astromodel import process_astromodel
from Princess.gwtools.initialization import initialization
from Princess.stochastic.background import process_background_computation
from Princess.astrotools.catalogue_generation import generate_population, save_population
from Princess.test.settings import Make_param_file, clean
import shutil



if __name__ == '__main__':

    Make_param_file()
    params = json.load(open('test/Params.json', 'r'))

    # Make sure directories are created
    if not os.path.exists('Run/' + params['name_of_project_folder']):
        os.mkdir('Run/' + params['name_of_project_folder'])
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/")
    if not os.path.exists('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs/"):
        os.mkdir('Run/' + params['name_of_project_folder'] + "/Astro_Models/Catalogs")

    #Create a small fake population
    mass_params = {"low": 2, "high": 200}  # Masse in Msun
    z_params = {"low": 0.01, "high": 25}  # Redshift

    #Generate 300 merging binaries
    population = generate_population(
        num_sources=300,
        mass_method="m1_m2",
        mass_distribution="uniform",
        z_distribution="uniform",
        mass_params=mass_params,
        z_params=z_params
    )

    # Save the catalogue
    save_population(population, "test/Catalogue.dat")



    #Read and reshape astrophysical models, save instance of each astrophysical models for later
#    AM.initialization()

    #Read and reshape detectors and networks, save instances for each of them
    initialization()

    #Compute SNR and individual analysis for the individual detection
    process_astromodel()

    #Compute backgrounds, residuals and the corresponding analysis
    process_background_computation()

    #Copy the getting_starting file to the project repository
    source_file = "test/getting_started.py"
    destination_folder = f"Run/{params['name_of_project_folder']}/"
    shutil.copy(source_file, destination_folder)

    if params['results']['cleaning'] == True :
        clean()

