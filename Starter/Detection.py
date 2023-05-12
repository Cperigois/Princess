import os
import numpy as np
import pycbc.psd
import pandas as pd
from astropy.cosmology import Planck15
from Stochastic import Basic_Functions as BF


class Detector:

    def __init__(self, det_name, configuration, origin = 'Pycbc', psd_file = None, freq = None):
        """Define a single detector.
         Parameters
         ----------
         det_name : str
             Name of the detector
         psd_file : str
             the file where to find the psd, or the psd name from PyCBC
         origin : str
             {'Pycbc, 'Princess', 'User'}.
         freq: np.array
            Contain the frequency range for the use of the detecor
         """

        # Set class variables
        self.det_name = det_name
        self.psd_file = psd_file
        self.origin = origin
        self.configuration = configuration
        self.freq = freq
        if origin == 'Princess' :
            info = pd.read_csv('AuxiliaryFiles/PSDs/PSD_Princess.dat', index_col= 'names', sep = '\t')
            fmin = info.fmin[psd_file]
            fmax = info.fmax[psd_file]
            if (np.min(freq)< fmin) :
                self.freq = freq[(fmin > freq)]
            if np.max(freq)> fmax:
                self.freq = freq[(fmax < freq)]
        if self.freq is None:
            print('Unable to find th frequency range of the detector... \n Please add in your detector definition freq = [np.array] and recompile your detector')
    def Make_psd(self):
        """Load or calculate the psd of a detector.
        Parameters
        ----------

        Return
        ----------
        self.psd
        """
        if self.origin == 'Pycbc' :
            self.psd = pycbc.psd.from_string(psd_name=self.psd_file, length=len(self.freq), delta_f=float(self.freq[1]-self.freq[0]),
                                    low_freq_cutoff=float(self.freq[0]))
        elif self.origin == 'Princess' :
            path = 'AuxiliaryFiles/PSDs/'+self.psd_file+'_psd.dat'
            df_psd = pd.read_csv(path, index_col = None, sep = '\t', dtype = float)
            self.psd = pycbc.psd.read.from_numpy_arrays(df_psd.f, df_psd['psd[1/Hz]'],length=len(self.freq),  delta_f=self.freq[1]-self.freq[0], low_freq_cutoff=self.freq[0])
        elif self.origin == 'User' :
            self.psd = pycbc.psd.read.from_txt(psd_file, length=len(self.freq),
                                               delta_f=int(self.freq[1] - self.freq[0]),
                                               low_freq_cutoff=int(self.freq[0]), is_asd_file=self.asd)
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

    def SNR_source(self, mtot, z, q, waveform_approx):
        luminosity_distance = Planck15.luminosity_distance(z).value
        m1 = mtot * q * (1. + z) / (1 + q)
        m2 = m1 / q
        flim = BF.fcut_f(m1=m1, m2=m2, xsi=0, zm=z)
        print(flim)
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

class Network:

    def __init__(self, net_name = None, compo = None ,pic_file = None, freq = np.arange(500)+1, efficiency = 1., SNR_thrs = 12, duration = 1 ):
        """Create an instance of your model.
         Parameters
         ----------
         net_name : str
             Name of the network
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
        self.duration = duration

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
                                                      mass1=event['m1'] * (1. + event['z']),
                                                      mass2=event['m2'] * (1. + event['z']),
                                                      spin1x=0., spin1y=0., spin1z=event['spinz1'],
                                                      spin2x=0., spin2y=0., spin2z=event['spinz2'],
                                                      delta_f=self.freq[1]-self.freq[0],
                                                      f_lower=self.freq[0],
                                                      distance=event.Dl, f_ref=20.)[0]
            for d in self.compo:
                SNR[evt]+= pycbc.filter.matchedfilter.sigma(wf, psd=d.psd, low_frequency_cutoff=d.freq[0],
                                                         high_frequency_cutoff=np.max(d.freq))
            SNR[evt] = np.sqrt(SNR[evt])
        cat[self.net_name] = SNR



    def Horizon(self, SNR_threshold = 9, mmin = 1, mmax = 10000, waveform = "IMRPhenomD", zmax = 150, mratio = 1):

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
        filename = 'Horizon_'+self.net_name +'_'+ str(mmax)+'_'+waveform
        if os.path.exists('Horizon') ==False :
            os.mkdir('Horizon')
        output.to_csv('Horizon/'+filename+'.dat', sep = '\t', index = None)





