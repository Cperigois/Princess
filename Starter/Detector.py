import math
import numpy as np
import Getting_Started.py as GS
from astropy.cosmology import Planck15


class Detector:

    def __init__(self, det_name = None, Pycbc = True, psd_file, freq = np.arange(500)+1 ):
        """Create an instance of your model.
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
         index_column: bool
            Used to read your original catalogue with pandas. True if you have a column with indexes in your original file.
            (default = None)
         flags: dict
            Dictionary of a possible flag column in your catalogue (Can be used to distinguish the type of binary, the formation channel...)
         """

        # Set class variables
        self.det_name = det_name
        self.psd_file = psd_file
        self.Pycbc = Pycbc
        self.freq = freq

    def Make_psd(self):
        if Pycbc == True :
            self.psd = pycbc.psd.from_string(psd_name=psd_file, length=len(freq), delta_f=freq[1]-freq[0],
                                    low_freq_cutoff=freq[0])
        else :
            self.psd = pycbc.psd.read.from_txt(psd_file, length=len(freq),  delta_f=freq[1]-freq[0], low_freq_cutoff=freq[0], is_asd_file=self.asd)
        return self.psd

    def reshape_psd(self, delimiter = '\t', Header = None):
        sens = pd.read_csv(self.psd_name, names = ['f','sens'], sep = delimiter, header = Header )
        interp = InterpolatedUnivariateSpline(sens['f'], sens['sens'])
        df_out = pd.DataFrame({'f' :freq, 'asd' : interp(freq)})
        df_out.to_csv('../AuxiliaryFiles/PSDs/'+self.name+'.dat', header = None, index = None, sep = '\t')
        self.psd_file = '../AuxiliaryFiles/PSDs/'+self.name+'.dat'

class Network:

    def __init__(self, net_name = None, compo = None ,pic_file = None):
        """Create an instance of your model.
         Parameters
         ----------
         net_name : str
             Name of the detector
         pic_file : int of float
             name of the file where to find the PIC
         compo : list of str
             list of detectors in the network -- see det_name detectors class
         """
        # Set class variables
        self.original_cat_path = original_cat_path
        self.cat_name = cat_name
        self.duration = duration
        self.sep_cat = sep_cat
        self.index_column = index_column
        self.flags = flags
#Here define your detectors and network, link the psd, set up all the names...