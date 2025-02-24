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

name_of_project_folder = 'Test'
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
detector_V1 = {'name' : 'VO1', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'Virgo_O1', 'type' : '2G'}
detector_H1 = {'name' : 'HO1', 'origin': 'Princess', 'configuration' : 'H', 'psd_file' : 'Hanford_O1', 'type' : '2G'}

detector_ET = {'name' : 'ET10km', 'origin': 'Princess', 'configuration' : 'ET', 'psd_file' : 'ET10_CoBa', 'type' : '3G'}
detector_CE_h = {'name' : 'CE40km', 'origin': 'Princess', 'configuration' : 'H', 'psd_file' : 'CE_40km', 'type' : '3G'}
detector_CE_h = {'name' : 'CE40km', 'origin': 'Princess', 'configuration' : 'L', 'psd_file' : 'CE_40km', 'type' : '3G'}



detector_list = {detector_L1['name']: detector_L1,
                 detector_H1['name']: detector_H1,
                 detector_V1['name']: detector_V1,
                 detector_ET['name']: detector_ET,
                 detector_CE['name']: detector_CE}

"               ***                 "
network_HLV = {'name' : 'HLV_O1',
             'compo' : {detector_L1['name'] : detector_L1, detector_H1['name'] : detector_H1, detector_V1['name'] : detector_V1},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 1.,
             'SNR_thrs' : 8
             }

network_ET = {'name' : 'ET10',
             'compo' : {detector_ET['name'] : detector_ET},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : 8
             }

network_ET2CE = {'name' : 'ET2CE',
             'compo' : {detector_ET['name'] : detector_ET, detector_CE_h['name'] : detector_CE_h, detector_CE_l['name'] : detector_CE_l},
             'pic_file' : 'AuxiliaryFiles/PICs/ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : 8
             }

network_list = {network_HVL['name']: network_HLV,
                network_ET['name']: network_ET,
                network_ET2CE['name']: network_ET2CE}

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