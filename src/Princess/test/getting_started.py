import os
import sys
import importlib.resources
import Princess.test.advanced_params as AP

"""
Parameters of the user input catalogue. Please do not change the right column.
{<Input catalogue names> : <Names for output catalogues>}
"""


sys.path.append('../')
"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

name_of_project_folder = 'test'
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


path =  'test/'
rerun_astromodels = True

astro_model_1 = {
    'name': 'Catalogue_test',
    'original_path': path + 'Catalogue.dat',
    'spin_model': 'Zeros',
    'duration': 1
}


astro_model_list = {astro_model_1['name']: astro_model_1
                    }


rerun_snr_computation = True

frequency_size = 2000 # need to be an int

"""               *** Detectors and Network ***                 """
"""
        Set the runs you want to use
        List of available detectors : 
"""

# Define detectors

detector_L4 = {'name' : 'LO4', 'origin': 'Princess', 'configuration' : 'L', 'reference' : 'LIGO_O4', 'type' : '2G'}
detector_H4 = {'name' : 'HO4', 'origin': 'Princess', 'configuration' : 'H', 'reference' : 'LIGO_O4', 'type' : '2G'}
detector_V4 = {'name' : 'VO4', 'origin': 'Princess', 'configuration' : 'V', 'reference' : 'Virgo_O4', 'type' : '2G'}
detector_K4 = {'name' : 'KO4', 'origin': 'Princess', 'configuration' : 'K', 'reference' : 'KAGRA_O4', 'type' : '2G'}

detector_L5 = {'name' : 'LO5', 'origin': 'Princess', 'configuration' : 'L', 'reference' : 'LIGO_O5', 'type' : '2G'}
detector_H5 = {'name' : 'HO5', 'origin': 'Princess', 'configuration' : 'H', 'reference' : 'LIGO_O5', 'type' : '2G'}
detector_V5 = {'name' : 'VO5', 'origin': 'Princess', 'configuration' : 'V', 'reference' : 'Virgo_O5', 'type' : '2G'}
detector_K5 = {'name' : 'KO5', 'origin': 'Princess', 'configuration' : 'K', 'reference' : 'KAGRA_O5', 'type' : '2G'}
detector_I5 = {'name' : 'IO5', 'origin': 'Princess', 'configuration' : 'I', 'reference' : 'LIGO_O5', 'type' : '2G'}

detector_ET1 = {'name' : 'ET10km', 'origin': 'Princess', 'configuration' : 'E1', 'reference' : 'ET_10km', 'type' : '3G'}
detector_ET2 = {'name' : 'ET10km', 'origin': 'Princess', 'configuration' : 'E2', 'reference' : 'ET_10km', 'type' : '3G'}
detector_ET3 = {'name' : 'ET10km', 'origin': 'Princess', 'configuration' : 'E3', 'reference' : 'ET_10km', 'type' : '3G'}

detector_CE20 = {'name' : 'CE_H_20km', 'origin': 'Princess', 'configuration' : 'H', 'reference' : 'CE_20km', 'type' : '3G'}
detector_CE40H = {'name' : 'CE_H_40km', 'origin': 'Princess', 'configuration' : 'H', 'reference' : 'CE_40km', 'type' : '3G'}
detector_CE40L = {'name' : 'CE_L_40km', 'origin': 'Princess', 'configuration' : 'L', 'reference' : 'CE_40km', 'type' : '3G'}



detector_list = {detector_L4['name']: detector_L4,
                 detector_H4['name']: detector_H4,
                 detector_V4['name']: detector_V4,
                 detector_K4['name']: detector_K4,
                 detector_L5['name']: detector_L5,
                 detector_H5['name']: detector_H5,
                 detector_V5['name']: detector_V5,
                 detector_K5['name']: detector_K5,
                 detector_I5['name']: detector_I5,
                 detector_ET1['name']: detector_ET1,
                 detector_ET2['name']: detector_ET2,
                 detector_ET3['name']: detector_ET3,
                 detector_CE20['name']: detector_CE20,
                 detector_CE40H['name']: detector_CE40H,
                 detector_CE40L['name']: detector_CE40L
                 }

"               ***                 "
network_LVK_O4 = {'name' : 'LVK_O4',
             'compo' : {detector_L4['name'] : detector_L4, detector_H4['name'] : detector_H4,
                        detector_V4['name'] : detector_V4, detector_K4['name'] : detector_K4},#https://emfollow.docs.ligo.org/userguide/capabilities.html
             'pic_file' : 'AuxiliaryFiles/PICs/Design_HLVIK_flow_10.txt',
             'efficiency' : 1.,
             'SNR_thrs' : [8, 20, 50]
             }
network_LVK_O5 = {'name' : 'LVK_O5',
             'compo' : {detector_L5['name'] : detector_L5, detector_H5['name'] : detector_H5,
                        detector_V5['name'] : detector_V5, detector_K5['name'] : detector_K5,
                        detector_I5['name'] : detector_I5},#https://emfollow.docs.ligo.org/userguide/capabilities.html
             'pic_file' : 'AuxiliaryFiles/PICs/Design_HLVIK_flow_10.txt',
             'efficiency' : 1.,
             'SNR_thrs' : [8, 20, 50]
             }

network_ET = {'name' : 'ET',
             'compo' : {detector_ET1['name'] : detector_ET1,
                        detector_ET2['name'] : detector_ET2,
                        detector_ET3['name'] : detector_ET3},
             'pic_file' : 'AuxiliaryFiles/PICs/PIC_ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : [8, 20, 50]
             }

network_ET2CE4020 = {'name' : 'ET2CE4020',
             'compo' : {detector_ET1['name'] : detector_ET1,
                        detector_ET2['name'] : detector_ET2,
                        detector_ET3['name'] : detector_ET3,
                        detector_CE20['name'] : detector_CE20,
                        detector_CE40L['name'] : detector_CE40L},
             'pic_file' : 'AuxiliaryFiles/PICs/PIC_CE4020ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : [8, 20, 50]
             }


network_list = {network_LVK_O4['name']: network_LVK_O4,
                network_LVK_O5['name']: network_LVK_O5,
                network_ET['name']: network_ET,
                network_ET2CE4020['name']: network_ET2CE4020}

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

