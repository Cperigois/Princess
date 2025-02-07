import os
import numpy as np
import pycbc.psd
import pandas as pd
import json
import math
import pickle
from astropy.cosmology import Planck15
from stochastic import basic_functions as BF

params = json.load(open('Run/Params.json', 'r'))

def initialization():
    for det in params['detector_list'].keys():
        detector = Detector(name=params['detector_list'][det]['name'],
                            configuration=params['detector_list'][det]['configuration'],
                            origin=params['detector_list'][det]['origin'],
                            reference=params['detector_list'][det]['reference'],
                            type= params['detector_list'][det]['type'])
        detector.save()
    for net in params['network_list'].keys():
        network = Network(name=params['network_list'][net]['name'],
                          compo=params['network_list'][net]['compo'],
                          pic_file=params['network_list'][net]['pic_file'],
                          efficiency=params['network_list'][net]['efficiency'],
                          SNR_thrs= params['network_list'][net]['SNR_thrs'])
        network.save()


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
            self.freq = np.logspace(numpy.log10(frequency_min), numpy.log10(frequency_max), num = n)

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
            self.psd = pycbc.psd.read.from_txt(psd_file, length=len(self.freq)+1,
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
        df_out = pd.DataFrame({'f' :freq, 'asd' : interp(freq)})
        df_out.to_csv('../AuxiliaryFiles/PSDs/'+self.name+'.dat', header = None, index = None, sep = '\t')
        self.psd_file = '../AuxiliaryFiles/PSDs/'+self.name+'.dat'

    def SNR_source(self, mtot:float, z:float, q:float, waveform_approx:str)->float:
        """
        TEST FUNCTION USE ONLY FOR TESTS
        Compute the snr of one specific source, assuming spins are 0 and the best sky location.
        Parameters
        ----------
        :param mtot (float): Total mass of the system in Msun.
        :param z (float): Redshift of the merger.
        :param q (float): Mass ratio of the system. By convention q<1.
        :param waveform_approx (str): Waveform to use for the computation of the SNR.
        :return (snr): Signal-to-noise ration of the soure in the detector.
        """
        luminosity_distance = Planck15.luminosity_distance(z).value
        m1 = mtot * q * (1. + z) / (1 + q)
        m2 = m1 / q
        flim = BF.fcut_f(m1=m1, m2=m2, xsi=0, zm=z)
        if flim > self.freq[0]+0.15:
            psd = self.Make_psd()
            hp, hc = pycbc.waveform.get_fd_waveform(approximant=waveform_approx, mass1=m1 * (1. + z), mass2 = m2 * (1. + z),
                                                spin1x=0., spin1y=0., spin1z=0, spin2x=0., spin2y=0., spin2z=0,
                                                delta_f=float(self.freq[1]-self.freq[0]), f_lower=float(self.freq[0]), distance=luminosity_distance, f_ref=20.,
                                                inclinaison=0)
            snr_l = pycbc.filter.matchedfilter.sigma(np.sqrt(2) * hp, psd=psd,
                                                 low_frequency_cutoff=float(self.freq[0]),
                                                 high_frequency_cutoff=float(np.max(self.freq)))
        else :
            snr_l = 0
        return snr_l

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

    def save(self):
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_DET.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

class Network:

    def __init__(self, efficiency:float = 1., SNR_thrs:float = 12, duration:float = 1,
                 name:str = None, compo:list = None ,pic_file:str = None ):
        """
        Create an instance of your network.
        Parameters
        ----------
        :param name (str): Name the network will be reffered to in further savings.
        :param compo (list): List of Detectors composing the network. The programm currently does not allow to mix
            generations of detectors.
        :param pic_file (str): Path and file referring where the PIC is stored.
        :param freq (np.1Darray): Array used to compute the sensitivity. Needs to be coherent with the detectors
            frequency ranges.
        :param efficiency (float): Duty cycle of the network, refers to the proportion of time the network has all
            detectors operational. Default is 1.
        :param SNR_thrs (float): Set the detection threshold of the detector. Default is 12.
        :param duration (float): Observation duration in yr. Default is 1.
        """

        # Set class variables
        self.name = name
        if (not os.path.exists('Run/' + params['name_of_project_folder'] + '/' + self.name + '_NET.pickle')):
            self.compo = compo
            self.pic_file = pic_file
            keys = list(compo.keys())
            self.efficiency = efficiency
            self.SNR_thrs = SNR_thrs
            self.duration = duration
            self.get_detectors_attributes(keys[0])
        else :
            self.load()

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
        with open(path + name + '_NET.pickle', 'rb') as file:
            data = pickle.load(file)

        if isinstance(data, dict):
            # Ensure data is used to create an instance of Detector
            network = cls(name)
            network.__dict__.update(data)
            return network
        elif isinstance(data, cls):
            return data
        else:
            raise TypeError("Loaded data is not a valid Network instance or dictionary.")

    def get_detectors_attributes(self, name_detector):
        with open(f"Run/{params['name_of_project_folder']}/{name_detector}_DET.pickle", 'rb') as file:
            detector_instance = pickle.load(file)
        self.freq = detector_instance['freq']
        self.type = detector_instance['type']

    def reshape_pic(self, delimiter:str='\t', Header:bool=False, index:bool=False):
        """
        Reshape your psd to fit the Make psd function and write it in a new file in AuxiliaryFiles/PSDs.
        It also uptate the variable psd_file to the new directory.
        Parameters
        ----------
        :param delimiter (str): delimiter used in the original file. Default is '\t'.
        :param Header (bool): True if the original file contain a header. Default is False.
        :param index (bool): True if the original file contain a column with indexes. Default is False.
        """

        sens = pd.read_csv(self.pic_name, names=['f', 'sens'], sep=delimiter, header=Header, index_col=index)
        interp = InterpolatedUnivariateSpline(sens['f'], sens['sens'])
        df_out = pd.DataFrame({'f': freq, 'pic': interp(freq)})
        df_out.to_csv('../AuxiliaryFiles/PICs/' + self.name + '.dat', header=None, index=None, sep='\t')
        self.pic_file = '../AuxiliaryFiles/PICs/' + self.name + '.dat'

    def SNR_individual(self, astromodel_catalogue):
        """
        Compute the signal-to-noise ratios (SNRs) of each source of a catalogues for the given network.
        Parameters
        ----------
        :param astromodel_catalogue (str): Path to a catalogues of binaries
        :return: SNR (1Darray): Array of the catalogue size containing the SNRs.
        """
        cat = pd.read_csv(astromodel_catalogue, sep = '\t', index_col = None)
        SNR = np.zeros(len(cat['m1']))
        for evt in range(len(cat['m1'])) :
            event = cat.iloc[evt]
            wf = pycbc.waveform.get_fd_waveform(approximant=approximant,
                                                      mass1=event['m1'] * (1. + event['z']),
                                                      mass2=event['m2'] * (1. + event['z']),
                                                      spin1x=0., spin1y=0., spin1z=event['s1'],
                                                      spin2x=0., spin2y=0., spin2z=event['s2'],
                                                      delta_f=self.freq[1]-self.freq[0],
                                                      f_lower=self.freq[0],
                                                      distance=event.Dl, f_ref=20.)[0]
            for d in self.compo:
                SNR[evt]+= pycbc.filter.matchedfilter.sigma(wf, psd=d.psd, low_frequency_cutoff=d.freq[0],
                                                         high_frequency_cutoff=np.max(d.freq))
            SNR[evt] = np.sqrt(SNR[evt])
        cat[self.name] = SNR
        return SNR



    def horizon(self, SNR_threshold:float = 9., mmin:float = 1., mmax:float = 10000., waveform:str = "IMRPhenomD", zmax:float = 150., mratio:float = 1.):
        """
        CONTAIN ISSUES IN SOME PART OF THE PARAMETER SPACE. Compute the horizon of a detector and write it in
        'Horizon/Horizon_'+self.name +'_'+ str(mmax)+ str(zmax)+'_'+waveform+.dat.
        Parameters
        ----------
        :param SNR_threshold (float): Signal-to-noise ratio (SNR) threshold to denine sources individually resolved.
        :param mmin (float): Minimal mass in Msun. Default is 1.
        :param mmax (float): Maximal mass in Msun. Default is 10 000.
        :param waveform (str): Waveform used to compute the SNR. Default is "IMRPhenomD".
        :param zmax (float): Maximal redshift. Default is 150.
        :param mratio (float): mass ratio considered for the computations. Default is 1.
        :return:
        """

        deltaz = [10,1,0.1,0.01, 0.001]
        Mtot = np.logspace(np.log10(mmin),np.log10(mmax),100)
        Hori  = np.zeros(len(Mtot))
        print(Mtot[0])

        for m in range(len(Mtot)):
            print(m)
            z = 0.001
            m1 = Mtot[m] * mratio / (1 + mratio)
            m2 = m1 / mratio
            zmax_1Hz = np.maximum(BF.zmax(m1,m2,0,1.2),0.001)
            for dz in deltaz :
                snr = SNR_threshold+0.001
                print(snr,' ',SNR_threshold,' ',zmax_1Hz,' ', zmax)
                while ((snr > SNR_threshold)&((z+dz)<np.minimum(zmax_1Hz,zmax))) :
                    z = z + dz
                    snr_net = 0
                    for d in self.compo:
                        snr_net += np.power(d.SNR_source(Mtot[m],z, mratio, waveform),2.)
                    snr = np.sqrt(snr_net)
                z = np.maximum(z - dz, 0.001)
            Hori[m] = z+dz
        output =  pd.DataFrame({'Mtot':Mtot, 'Horizon':Hori})
        filename = 'Horizon_'+self.name +'_'+ str(mmax)+ str(zmax)+'_'+waveform
        if os.path.exists('Horizon') ==False :
            os.mkdir('Horizon')
        output.to_csv('Horizon/'+filename+'.dat', sep = '\t', index = None)

    def load(self):
        """try load self.name.txt"""
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_NET.pickle', 'rb')
        data_pickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(data_pickle)

    def save(self):
        path = './Run/' + params['name_of_project_folder'] + '/'
        file = open(path + self.name + '_NET.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()





