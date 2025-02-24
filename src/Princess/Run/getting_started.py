import os
import sys

import pandas as pd

import Run.advanced_params as AP

"""
Parameters of the user input catalogue. Please do not change the right column.
{<Input catalogue names> : <Names for output catalogues>}
"""




sys.path.append('../')
"""----------------------TO FILL----------------------"""

"""             *** GENERIC PARAMETERS ***            """

name_of_project_folder = 'PopIII'
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

path =  '/home/perigois/Documents/Princess_savings_02_2024/Princess/Catalogs/'
rerun_astromodels = True

astro_model_1 = {
    'name': 'KRO1_H22',
    'original_path': path + 'KRO1_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_2 = {
    'name': 'KRO1_J19',
    'original_path': path + 'KRO1_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_3 = {
    'name': 'KRO1_LB20',
    'original_path': path + 'KRO1_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_4 = {
    'name': 'KRO1_SW20',
    'original_path': path + 'KRO1_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_5 = {
    'name': 'KRO5_H22',
    'original_path': path + 'KRO5_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_6 = {
    'name': 'KRO5_J19',
    'original_path': path + 'KRO5_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_7 = {
    'name': 'KRO5_LB20',
    'original_path': path + 'KRO5_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_8 = {
    'name': 'KRO5_SW20',
    'original_path': path + 'KRO5_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_9 = {
    'name': 'LAR1_H22',
    'original_path': path + 'LAR1_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_10 = {
    'name': 'LAR1_J19',
    'original_path': path + 'LAR1_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_11 = {
    'name': 'LAR1_LB20',
    'original_path': path + 'LAR1_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_12 = {
    'name': 'LAR1_SW20',
    'original_path': path + 'LAR1_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_13 = {
    'name': 'LAR5_H22',
    'original_path': path + 'LAR5_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_14 = {
    'name': 'LAR5_J19',
    'original_path': path + 'LAR5_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_15 = {
    'name': 'LAR5_LB20',
    'original_path': path + 'LAR5_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_16 = {
    'name': 'LAR5_SW20',
    'original_path': path + 'LAR5_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_17 = {
    'name': 'LOG1_H22',
    'original_path': path + 'LOG1_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_18 = {
    'name': 'LOG1_J19',
    'original_path': path + 'LOG1_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_19 = {
    'name': 'LOG1_LB20',
    'original_path': path + 'LOG1_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_20 = {
    'name': 'LOG1_SW20',
    'original_path': path + 'LOG1_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_21 = {
    'name': 'LOG2_H22',
    'original_path': path + 'LOG2_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_22 = {
    'name': 'LOG2_J19',
    'original_path': path + 'LOG2_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_23 = {
    'name': 'LOG2_LB20',
    'original_path': path + 'LOG2_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_24 = {
    'name': 'LOG2_SW20',
    'original_path': path + 'LOG2_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_25 = {
    'name': 'LOG3_H22',
    'original_path': path + 'LOG3_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_26 = {
    'name': 'LOG3_J19',
    'original_path': path + 'LOG3_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_27 = {
    'name': 'LOG3_LB20',
    'original_path': path + 'LOG3_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_28 = {
    'name': 'LOG3_SW20',
    'original_path': path + 'LOG3_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_29 = {
    'name': 'LOG4_H22',
    'original_path': path + 'LOG4_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_30 = {
    'name': 'LOG4_J19',
    'original_path': path + 'LOG4_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_31 = {
    'name': 'LOG4_LB20',
    'original_path': path + 'LOG4_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_32 = {
    'name': 'LOG4_SW20',
    'original_path': path + 'LOG4_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_33 = {
    'name': 'LOG5_H22',
    'original_path': path + 'LOG5_H22.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_34 = {
    'name': 'LOG5_J19',
    'original_path': path + 'LOG5_J19.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_35 = {
    'name': 'LOG5_LB20',
    'original_path': path + 'LOG5_LB20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}

astro_model_36 = {
    'name': 'LOG5_SW20',
    'original_path': path + 'LOG5_SW20.dat',
    'spin_model': 'Zeros',
    'duration': 1
}


astro_model_list = {astro_model_1['name']: astro_model_1,
                    astro_model_2['name']: astro_model_2,
                    astro_model_3['name']: astro_model_3,
                    astro_model_4['name']: astro_model_4,
                    astro_model_5['name']: astro_model_5,
                    astro_model_6['name']: astro_model_6,
                    astro_model_7['name']: astro_model_7,
                    astro_model_8['name']: astro_model_8,
                    astro_model_9['name']: astro_model_9,
                    astro_model_10['name']: astro_model_10,
                    astro_model_11['name']: astro_model_11,
                    astro_model_12['name']: astro_model_12,
                    astro_model_13['name']: astro_model_13,
                    astro_model_14['name']: astro_model_14,
                    astro_model_15['name']: astro_model_15,
                    astro_model_16['name']: astro_model_16,
                    astro_model_17['name']: astro_model_17,
                    astro_model_18['name']: astro_model_18,
                    astro_model_19['name']: astro_model_19,
                    astro_model_20['name']: astro_model_20,
                    astro_model_21['name']: astro_model_21,
                    astro_model_22['name']: astro_model_22,
                    astro_model_23['name']: astro_model_23,
                    astro_model_24['name']: astro_model_24,
                    astro_model_25['name']: astro_model_25,
                    astro_model_26['name']: astro_model_26,
                    astro_model_27['name']: astro_model_27,
                    astro_model_28['name']: astro_model_28,
                    astro_model_29['name']: astro_model_29,
                    astro_model_30['name']: astro_model_30,
                    astro_model_31['name']: astro_model_31,
                    astro_model_32['name']: astro_model_32,
                    astro_model_33['name']: astro_model_33,
                    astro_model_34['name']: astro_model_34,
                    astro_model_35['name']: astro_model_35,
                    astro_model_36['name']: astro_model_36
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

network_4020CE = {'name' : '2CE4020',
             'compo' : {detector_CE20['name'] : detector_CE20, detector_CE40L['name'] : detector_CE40L},
             'pic_file' : 'AuxiliaryFiles/PICs/PIC_CE4020.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : [8, 20, 50]
             }

network_4040CE = {'name' : '2CE4040',
             'compo' : {detector_CE40H['name'] : detector_CE40H, detector_CE40L['name'] : detector_CE40L},
             'pic_file' : 'AuxiliaryFiles/PICs/PIC_CE4040.txt',
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

network_ET2CE4040 = {'name' : 'ET2CE4040',
             'compo' : {detector_ET1['name'] : detector_ET1,
                        detector_ET2['name'] : detector_ET2,
                        detector_ET3['name'] : detector_ET3,
                        detector_CE40H['name'] : detector_CE40H,
                        detector_CE40L['name'] : detector_CE40L},
             'pic_file' : 'AuxiliaryFiles/PICs/PIC_CE4040ET.txt',
             'efficiency' : 0.5,
             'SNR_thrs' : [8, 20, 50]
             }


network_list = {network_LVK_O4['name']: network_LVK_O4,
                network_LVK_O5['name']: network_LVK_O5,
                network_ET['name']: network_ET,
                network_4020CE['name']: network_4020CE,
                network_4040CE['name']: network_4040CE,
                network_ET2CE4020['name']: network_ET2CE4020,
                network_ET2CE4040['name']: network_ET2CE4040}

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
                    'overwrite': {'astromodel': rerun_astromodels,
                                  'individual_snr': rerun_snr_computation,
                                  'detectors': rerun_detectors},
                    'results': {'cleaning': run_data_cleaning,
                                'plots': run_plots}
                    }
advParams = {"AM_params": {'input_parameters': AP.input_parameters, 'keepID' : AP.keepID, 'ID_col' : AP.ID_col},
             "detector_params": {'detectors_avail': AP.detectors_avail,
                                 'psd_attributes': AP.psd_attributes,
                                 'types': AP.types},
             "Inclination" : AP.Inclination
             }
AP.set(name_of_project_folder, param_dictionary, advParams)