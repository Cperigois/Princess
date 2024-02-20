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

name_of_project_folder = 'Test_Arnaud'
#n_cpu_max = 4  # Number maximal of cpu used by the code
param_dictionary = {'name_of_project_folder': name_of_project_folder}
AP.set(name_of_project_folder, param_dictionary, AP.advParams)

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

path_popIII =  '/home/perigois/Documents/Catalogs_channel/'
astromodel_1 = {'name': 'source_test',
                'original_path': path_popIII+'test_Arnaud.dat',
                'spin_model' : 'Zeros',
                'duration': 15}

#astromodel_2 = {'name': 'TOP5_H22',
#                'original_path': path_popIII+'TOP5_H22.dat',
#                'spin_model' : 'Rand_aligned',
#                'duration': 10}

astro_model_list = {astromodel_1['name']: astromodel_1}#,
                    #astromodel_2['name']: astromodel_2}
rerun_snr_computation = False

frequency_size = 2500 # need to be an int

"""               *** Detectors and Network ***                 """
"""
        Set the runs you want to use
        List of available detectors : 
"""

# Define detectors
detector_1 = {'name' : 'ET', 'origin': 'Pycbc', 'configuration' : 'ET', 'psd_file' : 'EinsteinTelescopeP1600143', 'type' : '3G'}
detector_2 = {'name' : 'Livnigston', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'Livingston_O3a_psd', 'type' : '2G'}

detector_list = {detector_1['name']: detector_1}

"               ***                 "
network_1 = {'name' : 'ET_net',
             'compo' : {detector_1['name'] : detector_1},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : 12
             }
network_2 = {'name' : 'HLV',
             'compo' : {detector_2['name'] : detector_2, detector_3 : detector_3['name'], detector_4}

network_list = {network_1['name']: network_1}

rerun_detectors = False

"""               *** Background computation ***                 """
"""
        Choose option for background computation
"""
rerun_background = False


"""               *** Post processing ***                 """
"""
        Choose if you want to compute multichannel analysis
"""
run_data_cleaning = True
run_plots = False

"""---------------------------------------------------"""

"""        *** Main Code, should not change ***       """

"""  1- Set the directory for all intermediate and definitive results  """

if not os.path.exists('Run/' + name_of_project_folder):
    os.mkdir('Run/' + name_of_project_folder)

"""  2- Gather and save the parameter used in the study  """

param_dictionary = {'name_of_project_folder': name_of_project_folder,
                    'astro_model_list': astro_model_list,
                    'detector_list': detector_list,
                    'network_list' : network_list,
                    'frequency_size' : frequency_size,
                    'n_cpu_max': n_cpu_max,
                    'overwrite': {'astromodel': rerun_snr_computation,
                                  'detectors': rerun_detectors},
                    'results': {'cleaning': run_data_cleaning,
                                'plots': run_plots}
                    }
AP.set(name_of_project_folder, param_dictionary, AP.advParams)