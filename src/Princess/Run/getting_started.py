print(f"Loading {__name__}")

import os
import sys
import Run.advanced_params as AP

"""
Parameters of the user input catalogue. Please do not change the right column.
{<Input catalogue names> : <Names for output catalogues>}
"""

sys.path.append('../')
"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

name_of_project_folder = 'snr_test'
n_cpu_max = 4  # Number maximal of cpu used by the code
param_dictionary = {'name_of_project_folder': name_of_project_folder}

"""               *** ASTROMODELS ***                 """
"""
        class AstroModel:
        Parameters: 
        Optional: spinModel: among:
                  'InCat'(by default option) means that the spins are already in your catalogues
                  'Rand_Isotropic': Build random aligned spins with magnitude from a maxwellian law sigma = 0.1
                  'Rand_Dynamics': Build random misaligned spins with magnitude from a maxwellian law sigma = 0.1
                  'Zeros' (default is 'InCat', assuming that your spins are in your initial catalog
"""

path =  '/home/perigois/PycharmProjects/Princess/src/Princess/Run/OtherChannels/Astro_Models/Catalogs/'
rerun_astromodels = False

astro_model_1 = {
    'name': 'snrtestfield',
    'original_path': path + 'Field.dat',
    'spin_model': 'Zeros',
    'duration': 1
}


astro_model_list = {astro_model_1['name']: astro_model_1}

rerun_snr_computation = True

frequency_size = 1000 # need to be an int

"""               *** Detectors and Network ***                 """
"""
        Set the runs you want to use
        List of available detectors : 
"""

# Define detectors

detector_L1 = {'name' : 'LO1', 'origin': 'Princess', 'configuration' : 'L', 'reference' : 'Livingston_O1', 'type' : '2G'}
detector_H1 = {'name' : 'HO1', 'origin': 'Princess', 'configuration' : 'H', 'reference' : 'Hanford_O1', 'type' : '2G'}



detector_list = {detector_L1['name']: detector_L1,
                 detector_H1['name']: detector_H1}

"               ***                 "
network_LVK_O1 = {'name' : 'LVK_O1',
             'compo' : {detector_L1['name'] : detector_L1, detector_H1['name'] : detector_H1},#https://emfollow.docs.ligo.org/userguide/capabilities.html
             'pic_file' : 'AuxiliaryFiles/PICs/Design_HLVIK_flow_10.txt',
             'efficiency' : 1.,
             'SNR_thrs' : [8, 20, 50]
             }



network_list = {network_LVK_O1['name']: network_LVK_O1}

rerun_detectors = True  # Force to reload the detectors and their instances.
                        # To be kept True except if the detectors are heavy to be loaded.

"""               *** Background computation ***                 """
"""
        Choose option for background computation
"""
rerun_background = True


"""               *** Post processing ***                 """
"""
        Choose if you want to compute multichannel analysis
"""
run_data_cleaning = False
run_plots = False
