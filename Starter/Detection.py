import numpy as np
import pycbc.psd


class Detector:

    def __init__(self, det_name, freq, Pycbc = True, psd_file = None):
        """Define a single detector.
         Parameters
         ----------
         det_name : str
             Name of the detector
         psd_file : str
             the file where to find the psd, or the psd name from PyCBC
         PyCBC : bool
             True if the PSD ca be found on PyCBC, False if you have your own.
         freq: np.array
            Contain the frequency range for the use of the detecor
         """

        # Set class variables
        self.det_name = det_name
        self.psd_file = psd_file
        self.Pycbc = Pycbc
        self.freq = freq

    def Make_psd(self):
        """Load or calculate the psd of a detector.
        Parameters
        ----------

        Return
        ----------
        self.psd
        """
        if self.Pycbc == True :
            self.psd = pycbc.psd.from_string(psd_name=self.psd_file, length=len(self.freq)+1+ np.min(self.freq), delta_f=int(self.freq[1]-self.freq[0]),
                                    low_freq_cutoff=int(self.freq[0]))
        else :
            self.psd = pycbc.psd.read.from_txt(psd_file, length=len(self.freq)+1,  delta_f=int(self.freq[1]-self.freq[0]), low_freq_cutoff=int(self.freq[0]), is_asd_file=self.asd)
        return self.psd

    def reshape_psd(self, delimiter = '\t', Header = None, index  = None):
        """Reshape your psd to fit the Make psd function and write it in a new file in AuxiliaryFiles/PSDs.
        It also uptate the variable psd_file to the new directory.
        Parameters
        ----------
        delimiter: str
            Delimiter used in the original file
        Header: bool
            True if the file contain a header, else None
        index: bool
            True if the file contain a column with indexes, else None
        """
        sens = pd.read_csv(self.psd_name, names = ['f','sens'], sep = delimiter, header = Header , index_col = index)
        interp = InterpolatedUnivariateSpline(sens['f'], sens['sens'])
        df_out = pd.DataFrame({'f' :freq, 'asd' : interp(freq)})
        df_out.to_csv('../AuxiliaryFiles/PSDs/'+self.name+'.dat', header = None, index = None, sep = '\t')
        self.psd_file = '../AuxiliaryFiles/PSDs/'+self.name+'.dat'

class Network:

    def __init__(self, net_name = None, compo = None ,pic_file = None, freq = np.arange(500)+1, efficiency = 1., SNR_thrs = 12 ):
        """Create an instance of your model.
         Parameters
         ----------
         net_name : str
             Name of the detector
         pic_file : int of float
             name of the file where to find the PIC
         compo : list of detectors
             list of detectors in the network
         SNR_thrs : np.array int or float
             Gives the snr threshold of detection for each network in 'Networks', must have the same size than Networks
         """
        # Set class variables
        self.net_name = net_name
        self.compo = compo
        self.pic_file = pic_file
        self.freq = freq
        self.efficiency = efficiency
        self.SNR_thrs = SNR_thrs

    def reshape_pic(self, delimiter='\t', Header=None, index=None):
        """Reshape your psd to fit the Make psd function and write it in a new file in AuxiliaryFiles/PSDs.
        It also uptate the variable psd_file to the new directory.
        Parameters
        ----------
        delimiter: str
            Delimiter used in the original file
        Header: bool
            True if the file contain a header, else None
        index: bool
            True if the file contain a column with indexes, else None
        """
        sens = pd.read_csv(self.pic_name, names=['f', 'sens'], sep=delimiter, header=Header, index_col=index)
        interp = InterpolatedUnivariateSpline(sens['f'], sens['sens'])
        df_out = pd.DataFrame({'f': freq, 'pic': interp(freq)})
        df_out.to_csv('../AuxiliaryFiles/PICs/' + self.name + '.dat', header=None, index=None, sep='\t')
        self.pic_file = '../AuxiliaryFiles/PICs/' + self.name + '.dat'

    def SNR_individual(self, astromodel_catalogue):
        cat = pd.read_csv(astromodel_catalogue, sep = '\t', index_col = None)
        SNR = np.zeros(len(cat['m1']))
        for evt in range(len(cat['m1'])) :
            event = cat.iloc[evt]
            wf = pycbc.waveform.get_fd_waveform(approximant=approximant,
                                                      mass1=event['m1'] * (1. + event['zm']),
                                                      mass2=event['m2'] * (1. + event['zm']),
                                                      spin1x=0., spin1y=0., spin1z=event['spinz1'],
                                                      spin2x=0., spin2y=0., spin2z=event['spinz2'],
                                                      delta_f=self.freq[1]-self.freq[0],
                                                      f_lower=self.freq[0],
                                                      distance=event.Dl, f_ref=20.)[0]
            for d in self.compo:
                SNR[evt]+= pycbc.filter.matchedfilter.sigma(wf, psd=d.psd, low_frequency_cutoff=self.freq[0],
                                                         high_frequency_cutoff=np.max(freq))
            SNR[evt] = np.sqrt(SNR[evt])
        cat[self.net_name] = SNR

