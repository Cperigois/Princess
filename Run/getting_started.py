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

name_of_project_folder = 'GW170817_LVK'
n_cpu_max = 4  # Number maximal of cpu used by the code
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

path =  '/home/perigois/Documents/SNR_GW170817/'
astromodel_1 = {'name': 'GW170817',
                'original_path': path+'GW170817_post.dat',
                'spin_model' : 'Spin&cosTheta',
                'duration': 1}


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
detector_L1 = {'name' : 'LO1', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'Livingston_O1', 'type' : '2G'}
detector_L2 = {'name' : 'LO2', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'Livingston_O2', 'type' : '2G'}
detector_L3a = {'name' : 'LO3a', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'Livingston_O3a', 'type' : '2G'}
detector_L3b = {'name' : 'LO3b', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'Livingston_O3b', 'type' : '2G'}
detector_L4 = {'name' : 'LO4', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'LIGO_O4', 'type' : '2G'}

detector_H1 = {'name' : 'HO1', 'origin': 'Princess', 'configuration' : 'H', 'psd_file' : 'Hanford_O1', 'type' : '2G'}
detector_H2 = {'name' : 'HO2', 'origin': 'Princess', 'configuration' : 'H', 'psd_file' : 'Hanford_O2', 'type' : '2G'}
detector_H3a = {'name' : 'HO3a', 'origin': 'Princess', 'configuration' : 'H', 'psd_file' : 'Hanford_O3a', 'type' : '2G'}
detector_H3b = {'name' : 'HO3b', 'origin': 'Princess', 'configuration' : 'H', 'psd_file' : 'Hanford_O3b', 'type' : '2G'}
detector_H4 = {'name' : 'HO4', 'origin': 'Princess', 'configuration' : 'H', 'psd_file' : 'LIGO_O4', 'type' : '2G'}

#detector = {'name' : 'ET10km', 'origin': 'Princess', 'configuration' : 'ET', 'psd_file' : 'ET10_CoBa', 'type' : '3G'}

detector_list = {detector_L1['name']: detector_L1,
                 detector_L2['name']: detector_L2,
                 detector_L3a['name']: detector_L3a,
                 detector_L3b['name']: detector_L3b,
                 detector_L4['name']: detector_L4,
                 detector_H1['name']: detector_H1,
                 detector_H2['name']: detector_H2,
                 detector_H3a['name']: detector_H3a,
                 detector_H3b['name']: detector_H3b,
                 detector_H4['name']: detector_H4}

"               ***                 "
network_O1 = {'name' : 'HL_O1',
             'compo' : {detector_L1['name'] : detector_L1, detector_H1['name'] : detector_H1},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 1.,
             'SNR_thrs' : 8
             }

network_O2 = {'name' : 'HL_O2',
             'compo' : {detector_L2['name'] : detector_L2, detector_H2['name'] : detector_H2},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : 8
             }

network_O3a = {'name' : 'HL_O3a',
             'compo' : {detector_L3a['name'] : detector_L3a, detector_H3a['name'] : detector_H3a},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : 8
             }

network_O3b = {'name' : 'HL_O3b',
             'compo' : {detector_L3b['name'] : detector_L3b, detector_H3b['name'] : detector_H3b},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : 8
             }

network_O4 = {'name' : 'HL_O4',
             'compo' : {detector_L4['name'] : detector_L4, detector_H4['name'] : detector_H4},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : 8
             }


network_list = {network_O1['name']: network_O1,
                network_O2['name']: network_O2,
                network_O3a['name']: network_O3a,
                network_O3b['name']: network_O3b,
                network_O4['name']: network_O4,}

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