print(f"Loading {__name__}")
import os
import numpy as np
import pycbc.psd
import pandas as pd
import json
import pickle
from Princess.gwtools.utils import zmaximal
import importlib.resources

# Import parameter file
with importlib.resources.open_text("Princess.Run", "Params.json") as f:
    params = json.load(f)


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
        deltaz = [10,1,0.1,0.01, 0.001]
        Mtot = np.logspace(np.log10(mmin),np.log10(mmax),100)
        Hori  = np.zeros(len(Mtot))
        print(Mtot[0])

        for m in range(len(Mtot)):
            print(m)
            z = 0.001
            m1 = Mtot[m] * mratio / (1 + mratio)
            m2 = m1 / mratio
            zmax_1Hz = np.maximum(zmaximal(m1,m2,0,1.2),0.001)
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