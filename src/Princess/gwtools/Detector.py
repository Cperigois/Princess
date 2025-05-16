print(f"Loading {__name__}")
import os
import numpy as np
import pycbc.psd
import pandas as pd
import json
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
from Princess.Run.settings import PARAMS_FILE
from Princess.gwtools.waveform import Ajith_waveform


# Check PARAMS_FILE value
if not PARAMS_FILE or not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"The file parameter {PARAMS_FILE} is missing,. Execute Run.settings.Make_params_file() first.")

# Charge le fichier de paramètres
with open(PARAMS_FILE, "r") as f:
    params = json.load(f)

class Detector:

    def __init__(self,
                 name: str,
                 configuration: str = None,
                 origin: str = 'Princess',
                 reference: str = None,
                 type: str = None,
                 psd_file: str =None):
        """
        Instance of a detector.

        Parameters
        ----------
        name (str): Name of the detector, used for labeling further data such as the signal-to-noise ratio.
        configuration (str): Location, orientation, and arm opening of the detector. Options: 'H', 'L', 'V', 'ET'.
        origin (str): Source of the PSD ('Pycbc', 'Princess', or 'User').
        psd_file (str): File or identifier for a customized PSD. Varies based on origin.
        reference (str): Reference to existing detectors.
        type (str): Optional, additional type information.
        """


        self.name = name
        # Check if the detector needs to be reloaded or created
        project_folder = os.path.join('Run', params['name_of_project_folder'])
        detector_file = os.path.join(project_folder, f'{self.name}_DET.pickle')

        if not os.path.exists(detector_file) or params['overwrite']['detectors']:
            self.initialize_detector(_configuration= configuration,
                                     _reference= reference,
                                     _origin= origin,
                                     _type= type,
                                     _psd_file= psd_file)
            self.save()
            print(f'Detector {self.name} successfully saved.')
        else:
            self.load(self.name)
            print(f'Detector {self.name} successfully loaded.')
        print(f'Detector {self.name} instance recreated.')


    def get_psd_file(self):
        "Get the correct file where to find the Psd"
        print(self.reference)
        if self.origin == 'Princess' :
            self.psd_file = params['detector_params']['psd_attributes'][self.reference]['psd_name']
        else :
            print("No customized psd available yet")

    def initialize_detector(self, _configuration, _reference, _origin, _psd_file, _type):
        """Initialize the detector, setting up necessary parameters."""
        self.configuration = _configuration
        self.origin = _origin
        if _reference == None :
            self.reference = params['detector_list'][self.name]['reference']
        else :
            self.reference = _reference

        if _psd_file == None:
            self.get_psd_file()
        else:
            self.psd_file = _psd_file

        if _type == None :
            self.type = params['detector_list'][self.name]['type']
        else :
            self.type = _type

        self.make_frequency()
        if self.freq is None:
            self.handle_missing_frequency()

    def handle_missing_frequency(self):
        """Handle the case where frequency information is missing."""
        print(
            f"Unable to find the frequency range for detector '{self.name}'.\n"
            "Please define 'freq = [np.array]' in your detector definition and recompile."
        )

    def make_frequency(self):
        frequency_min = max(params['detector_params']['types'][self.type]['freq']['min'], params['detector_params']['psd_attributes'][self.reference]['min_freq'])
        frequency_max = min(params['detector_params']['types'][self.type]['freq']['max'], params['detector_params']['psd_attributes'][self.reference]['max_freq'])

        n = max(params['frequency_size'], params['detector_params']['types'][self.type]['freq']['min_fsize'])

        scale = params['detector_params']['types'][self.type]['freq']['scale']

        if scale =='log':
            self.freq = np.logspace(np.log10(frequency_min), np.log10(frequency_max), num = n)

        elif scale =='lin':
            self.freq = np.linspace(frequency_min, frequency_max, n)
            self.deltaf = self.freq[1]-self.freq[0]
            self.make_psd()

        else :
            print ("Error in frequency array creation, please check the type, and eventually avanced param file.")

    def make_psd(self):
        """
        load the PSD of the detector
        :return (np.1Darray): PSD of the detectors for frequencies corresponding to self.freq.
        """

        if self.origin == 'Pycbc' :
            self.psd = pycbc.psd.from_string(psd_name=self.psd_file, length=len(self.freq)+2, delta_f=float(self.freq[1]-self.freq[0]),
                                    low_freq_cutoff=float(self.freq[0]))
            self.psd = self.psd[1:len(self.freq)+1]
        elif self.origin == 'Princess' :
            path = 'AuxiliaryFiles/PSDs/'+self.psd_file+'.dat'
            df = pd.read_csv(path, index_col = None, sep = '\t')
            newpath = 'Run/temp/'+self.psd_file+'.dat'
            df.to_csv(newpath, index = False, sep = '\t', header = False)
            print(self.freq)
            """ Notes for the use of Pycbc.psd.read.from_numpy_array.
            - carefully keep the lenght +1
            - math.ceil(self.freq[0]) rounds freq[0] to the integer above, insuring that the interpolation stays in the right range
            - delta_f needs to be 1 otherwise pycbc.psd.read.from_numpy_arrays crashes
            This way to do psd definitely needs to be changed : it refer to pycbc classes, and are not well adapted to Princess. 
            This issue needs to be address in the V2 of Princess.  
            """
            self.psd = pycbc.psd.read.from_numpy_arrays(np.array(df['f']), np.array(df['psd[1/Hz]']),
                                                        length=len(self.freq)+1,
                                                        delta_f=1.,
                                                        low_freq_cutoff=np.ceil(self.freq[0]))

            self.psd = self.psd[1:]
        elif self.origin == 'User' :
            self.psd = pycbc.psd.read.from_txt(self.psd_file, length=len(self.freq)+1,
                                               delta_f=max(int(self.freq[1] - self.freq[0]),1),
                                               low_freq_cutoff=int(self.freq[0]), is_asd_file=False)
            self.psd = self.psd[1:]

        return self.psd

    def reshape_psd(self, delimiter:str = '\t', have_header:bool = False, have_index:bool  = False):
        """
        ***UNDER DEVELOPPEMENT***
        Reshape your psd to fit the Make psd function and write it in a new file in AuxiliaryFiles/PSDs.
        It also uptate the variable psd_file to the new directory.
        Parameters
        ----------
        :param delimiter (str): Delimiter used in the original file.
        :param header (bool): True, if the original file contain an header.
        :param index (bool): True, if the original file contain an index column.
        """

        sens = pd.read_csv(self.psd_name, names = ['f','sens'], sep = delimiter, header = have_header , index_col = have_index)
        interp = InterpolatedUnivariateSpline(sens['f'], sens['sens'])
        df_out = pd.DataFrame({'f' :self.freq, 'asd' : interp(self.freq)})
        df_out.to_csv('../AuxiliaryFiles/PSDs/'+self.name+'.dat', header = None, index = None, sep = '\t')
        self.psd_file = '../AuxiliaryFiles/PSDs/'+self.name+'.dat'



    @classmethod
    def load(cls, name):
        """
        Load a Detector object from a pickle file.

        :param name: Name of the detector.
        :type name: str
        :return: The loaded Detector object.
        :rtype: Detector
        """
        path = f'./Run/{params["name_of_project_folder"]}/'
        with open(path + name + '_DET.pickle', 'rb') as file:
            data = pickle.load(file)

        if isinstance(data, cls):
            return data
        elif isinstance(data, dict):
            # Create a new instance without infinite recursion
            detector = object.__new__(cls)
            detector.__dict__.update(data)
            return detector
        else:
            raise TypeError("Loaded data is not a valid Detector instance or dictionary.")

    def get_psd_pycbc_compatible(self, path_dir_psd = None):
        """This function sets the values for the PSE on the interval of frequency chosen.

        Parameters
        ----------
        path_dir_psd : str
            Path towards the PSD file if the PSD is imported from a file
        """

        # Set the path towards psd files
        if path_dir_psd is None:
            path_dir_psd = './AuxiliaryFiles/PSDs/'
        else:
            path_dir_psd = clean_path(path_dir_psd)

        # Check that file exists
        namefile = path_dir_psd + self.psd_file + "_pycbc_compatible.dat"
        if not os.path.isfile(namefile):
            raise FileNotFoundError(f"Psd file was not found at {namefile}")

        # Read psd file
        psd_data = pycbc.psd.read.from_txt(filename=namefile, length=1000, delta_f=1,
                                           low_freq_cutoff=10, is_asd_file=False)

        return psd_data


    def save(self):
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_DET.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()


    def reshape_analytical_waveforms(self, htildSQ, wf_freq):

        interp = InterpolatedUnivariateSpline(wf_freq,htildSQ)
        htildSQ = interp(self.freq)

        return htildSQ

    def compute_LISA_SNR(self, catalog_name):

        catalog_path = f'./Run/{params["name_of_project_folder"]}/Astro_Models/Catalogs'
        Cat = pd.read_csv(f'{catalog_path}/{catalog_name}.dat', sep='\t', index_col=False)
        print(f'LISA SNR calculation for {catalog_name}')
        ntot = len(Cat.z)

        # Initialize SNR columns to zero for each detector
        Cat['LISA'] = 0.0

        for evt in range(len(Cat)):
            event = Cat.iloc[[evt]] # Select a single event as a DataFrame
            htildSQ = self.reshape_analytical_waveforms(Ajith_waveform(event))
            Sn = self.psd
            comp =  4. * np.trapz(htildsq / Sn, self.freq)
            comp = np.nan_to_num(comp, nan=0, posinf=0)
            SNR = np.sqrt(comp.sum())  # Compute final SNR
            Cat.at[evt, det.name] = SNR  # Assign the computed SNR to the event

            # Save the updated catalog with new SNR columns
        Cat.to_csv(f'{catalog_path}/{catalog_name}.dat', sep='\t', index=False)









