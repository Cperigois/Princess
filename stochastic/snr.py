import numpy as np
import pandas as pd
import math
import itertools as iterT
import pycbc.psd
import pycbc.waveform
import pycbc.filter
import astrotools.detection as DET
import stochastic.basic_functions as BF
import joblib
import stochastic.constants as K
from scipy.interpolate import InterpolatedUnivariateSpline


def SNR_bkg(freq_omg, Omega, Network):
    """
    Computes the signal-to-noise ratio (SNR) for a stochastic gravitational wave background.

    :param freq_omg: Array of input frequencies for Omega.
    :type freq_omg: numpy.ndarray
    :param Omega: Normalized spectral energy density of the stochastic background.
    :type Omega: numpy.ndarray
    :param Network: Detector network containing frequency and composition information.
    :type Network: object
    :return: Computed SNR value.
    :rtype: float
    """
    # Load overlap reduction functions (ORFs)
    gammafile = './AuxiliaryFiles/ORFs/ORF.dat'
    gamma = pd.read_csv('./AuxiliaryFiles/ORFs/ORF.dat', sep='\t', index_col=None)
    freq = Network.freq
    deltaF = freq[1] - freq[0]

    # Interpolate Omega over the network frequencies
    interp = InterpolatedUnivariateSpline(freq_omg, Omega)
    Omega_interp = interp(freq)

    #Interpolate all psds

    SNR = 0

    # Retrieve detectors
    compo = list(Network.compo.keys())

    for i, name_i in enumerate(compo):
        di = DET.Detector.load(name=name_i)

        for j in range(i + 1, len(compo)):
            dj = DET.Detector.load(name = Network.compo[compo[j]]['name'])

            # Construct the ORF column name
            name1 = di.configuration + dj.configuration
            name2 = dj.configuration + di.configuration
            if name1 in gamma.columns:
                gamma_ref = di.configuration + dj.configuration
            elif name2 in gamma.columns:
                gamma_ref = dj.configuration + di.configuration
            else :
                print(f'Error : No configuration available for the couple ({di.configuration}, {dj.configuration} \n'
                      f'Please check the ORF file : {gammafile}')

            # Interpolate the overlap reduction function
            interp_ORF = InterpolatedUnivariateSpline(gamma.freq, gamma[gamma_ref])
            Gammaij = interp_ORF(freq)

            # Interpolate the PSDs
            df_Pi = pd.read_csv('AuxiliaryFiles/PSDs/'+di.psd_file+'.dat', index_col=None, sep='\t')
            interpPi = InterpolatedUnivariateSpline(df_Pi['f'], df_Pi['psd[1/Hz]'])

            df_Pj = pd.read_csv('AuxiliaryFiles/PSDs/'+dj.psd_file+'.dat', index_col=None, sep='\t')
            interpPj = InterpolatedUnivariateSpline(df_Pj['f'], df_Pj['psd[1/Hz]'])

            Pi = interpPi(freq)
            Pj = interpPj(freq)

            # Compute SNR
            SNR += np.sum((Gammaij ** 2 * Omega_interp ** 2) / (freq ** 6 * Pi * Pj))

    SNR = K.Cst_snr_bkg * np.sqrt(2 * Network.duration * Network.efficiency) * np.sqrt(SNR) / deltaF

    return SNR

def SNR_bkg_1det(freq_omg, Omega, Network):
    """
    Old function for SNR background computation
    :param freq_omg:
    :param Omega:
    :param Network:
    :return:
    """
    freq = Network.freq
    deltaF = freq[1]-freq[0]
    print(deltaF)
    interp = InterpolatedUnivariateSpline(freq_omg, Omega)
    Omega_interp = interp(freq)
    SNR =0
    for d in Network.compo:
        d.Make_psd()
    interpPi = InterpolatedUnivariateSpline(np.array(freq, dtype = float), np.array(di.psd, dtype = float))
    interpPj = InterpolatedUnivariateSpline(dj.freq, dj.psd)
    Pi = interpPi(freq)**2
    Pj = interpPj(freq)**2
    print(Pi)
    SNR += np.sum(Omega_interp**2. / (freq**6. * Pi * Pj))
    print('SNR = ',SNR,' ', Network.name)
    SNR = K.Cst_snr_bkg* np.sqrt(2.* Network.duration * Network.efficiency)*np.sqrt(SNR) / deltaF
    return K.Cst_snr_bkg* np.sqrt(2* Network.duration * Network.efficiency* SNR) / deltaF

def SNR_bkgtrash(freq_omg, Omega, Network):
    """ Old function to compute the background
    :param freq_omg:
    :param Omega:
    :param Network:
    :return:
    """
    gamma =pd.read_csv('./AuxiliaryFiles/ORFs/ORF.dat', sep = '\t', index_col = None)
    freq = Network.freq
    deltaF = freq[1]-freq[0]
    print(deltaF)
    interp = InterpolatedUnivariateSpline(freq_omg, Omega)
    Omega_interp = interp(freq)
    SNR = 0
    SNR2 = 0

    for d in Network.compo:
        d.Make_psd()
    for i in range(len(Network.compo)):
        di = Network.compo[i]
        for j in range(i-1):
            dj = Network.compo[j]
            gamma_ref = str(di.configuration+dj.configuration)
            if gamma_ref not in list(gamma.columns):
                gamma_ref = str(dj.configuration+di.configuration)
            interp_ORF = InterpolatedUnivariateSpline(gamma.freq, gamma[gamma_ref])
            Gammaij = interp_ORF(freq)
            interpPi = InterpolatedUnivariateSpline(di.freq, di.psd)
            interpPj = InterpolatedUnivariateSpline(dj.freq, dj.psd)
            Pi = np.sqrt(interpPi(freq))
            Pj = np.sqrt(interpPj(freq))
            Pi[np.isnan(Pi)] = 1
            Pj[np.isnan(Pj)] = 1
            print(Pi)
            for f in range(len(freq)) :
                SNR2 +=np.power(Gammaij[f],2.) * np.power(Omega_interp[f],2.) / (np.power(f+1,6.) * Pi[f] * Pj[f])
            SNR += np.sum(np.square(Gammaij) * np.square(Omega_interp) / (np.power(freq,6.) * Pi * Pj))
    SNR = K.Cst_snr_bkg* np.sqrt(2.* Network.duration * Network.efficiency)*np.sqrt(SNR) / deltaF
    SNR2 = K.Cst_snr_bkg* np.sqrt(2.* Network.duration * Network.efficiency)*np.sqrt(SNR2) / deltaF
    print('SNR1 = ',K.Cst_snr_bkg,' ','SNR2 = ',SNR2,' ', Network.name)

    return K.Cst_snr_bkg* np.sqrt(2* Network.duration * Network.efficiency* SNR) / deltaF


def SNR_Omega(freq, Omega, Networks):
    ''' Calculate the SNR of a given spectrum Omega for each Network in Networks.
    Assuming one year of detection with a duty-cycle (ie. efficiency) of 0.5
    '''

    #Detectors = [GS.Net_compo[n] for n in Networks]
    fmin_list = np.array([])
    fmax_list = np.array([])
    SNR_final = dict({})
    # for n in range(len(Networks)) :
    #     for d in range(len(Compo)) :
    #         det = GS.Detectors[Compo[d]] #names for orfs
    #         fmin_list= np.append(fmin_list, det[4])
    #         fmax_list = np.append(fmax_list, det[4]+det[2]*det[3])
    # freq_min = np.min(fmin_list)
    # freq_max = np.max(fmax_list)



    for N in Networks :
        PSD = dict({})
        Freq_Lim = dict({})
        Compo = GS.Net_psd[N]
        combi = 0
        net_info = GS.Net_files[N]
        Gamma = pd.read_csv('AuxiliaryFiles/ORFs/'+net_info[1], delimiter = '\t')
        #Omega = GW.Search_Omg(Omega, Gamma['freq'])
        print(len(Omega))
        SNR_pair =  dict({})
        combi_orf = [''.join(j) for j in iterT.combinations(GS.Net_compo[N], 2)]
        combi_orf =  np.unique(combi_orf)
        for idx_det1 in range(len(Compo)-1) :
            for idx_det2 in range(len(Compo)-1-idx_det1) :
                idx_det2+=1+idx_det1
                if (Compo[idx_det1]+Compo[idx_det2]) not in PSD.keys() :
                    det1 = GS.Detectors[Compo[idx_det1]]
                    det2 = GS.Detectors[Compo[idx_det2]]
                    if det1[0]=='Coba' :
                        df = pd.read_csv('AuxiliaryFiles/PSDs/Coba/'+det1[1], sep = ' ', index_col = None)
                        PSD1 = pycbc.psd.read.from_numpy_arrays(df['f'], df['HF_LF'], length = det1[2], delta_f = det1[3], low_freq_cutoff=det1[4])
                    else :
                        PSD1 = pycbc.psd.from_string(psd_name=det1[1], length=det1[2], delta_f=det1[3],
                                  low_freq_cutoff=det1[4])
                    if det2[0]=='Coba' :
                        df = pd.read_csv('AuxiliaryFiles/PSDs/Coba/'+det2[1], sep = ' ', index_col = None)
                        PSD2 = pycbc.psd.read.from_numpy_arrays(df['f'], df['HF_LF'], length = det2[2], delta_f = det2[3], low_freq_cutoff=det2[4])
                    else :
                        PSD2 = pycbc.psd.from_string(psd_name=det2[1], length=det2[2], delta_f=det2[3],
                                  low_freq_cutoff=det2[4])
                    key = combi_orf[combi]
                    PSD[key] = PSD1*PSD2
                    Freq_Lim[key] = [det1[4], det1[4]+det1[3]*det1[2]]
            combi +=1
        if net_info[2] == '3G' :
            freq_min = 1
            freq_max = 2300
        for pair in list(PSD.keys()) :
            SNR_pair[pair] = 0
            print(N, ' ', pair)
        for fr in range(int(freq_max)-int(freq_min)) :
            f = fr + freq_min
            for pair in list(PSD.keys()) :
                freqs = Freq_Lim[pair]
                psd = PSD[pair]
                gamma = Gamma[pair]
                freqs[1] -= 3
                if ((f <= freqs[1] - 1) & (f >= freqs[0] + 1)):
                    if (psd[int(f) - freqs[0]] != 0):
                        SNR_pair[pair] += math.pow(Omega[fr] * gamma[fr], 2.) / math.pow(psd[int(f) - freqs[0]],
                                                                                             0.5) * math.pow(f, -6.)
        SNR_fin = 0
        for pair in list(PSD.keys()) :
            SNR_fin += SNR_pair[pair]
        SNR_final[N] = 8.13e-34*math.pow(SNR_fin,0.5)

    return SNR_final

def SNR_run(SNR_0, SNR_year, duration, tstart):
    ''' 
    :param SNR_0: Starting SNR (SNR at the end of a previous run for example) 
    :param SNR_year: SNR for one year of observation (careful to know what the duty-cycle is at this point) 
    :param duration: duration of the run in yr
    :param tstart: starting time
    :return: dataframe with two columns 'Time' and 'SNR'
    '''
    t=0
    SNR_cum = []
    Time = []
    while (t<=(duration+0.02)) :
        SNR_cum.append(pow(SNR_0*SNR_0+t*SNR_year*SNR_year,0.5))
        Time.append(tstart+t)
        t+=0.02
    print(Time)
    print(SNR_cum)
    return 0

def SNRCat(Cat, Net) :
    """Compute the SNR of each sources in each network in Net
                Parameters
                ----------
                Net : str array
                    {'HLV' , 'HLVIK', 'A+', 'ET', 'ET+2CE', 'LISA'}. More Network can be set up in "Oz/Networks.py".
                model : str
                    Name of the model, used to setup the folder
                type : str
                    {'BBH' , 'BNS', 'BHNS', 'mix'} type of binary compact objects in the initial catalogue.
                    If mix is chosen fill the condition to distinguish the different type of binaries

                Returns
                -------
                SNR : np.array, floats
                    Array of SNR for each binary of the catalogue for the given Network.
        """
    path = 'AuxiliaryFiles/Joblib/'
    Net_det = joblib.load(path+"rf_"+Net+"_SNR.joblib")
    if 'inc' not in Cat.columns :
        Cat['inc'] = np.random.uniform(0, 2*math.pi, len(Cat['Mc']))
    features = ['Mc', 'zm', 'q', 'Dl', 'inc']
    val_X = Cat[features]

    return Net_det.predict(val_X)

def SNRwf(df, cat_name) :
    approximant = "IMRPhenomPv2"
    Det_list = GS.Detectors
    grid = pd.DataFrame()
    #df = pd.read_csv(cat, delimiter='\t', index_col=False)
    df['inc'] = np.random.uniform(low=0.0, high=2 * math.pi, size=len(df['Mc']))
    #df['m1'], df['m2'] = BF.mc_q_to_m1_m2(df['Mc'], df['q'])
    #df['Dl'] = cosmo.Planck15.luminosity_distance(df["zm"]).value
    grid = pd.DataFrame(
        {'Mc': df['Mc'], 'q': df['q'], 'zm': df['zm'], 'inc': df['inc'], 'Dl': df['Dl']})
    for d in Det_list.keys():
        name = 'SNR_' + d
        det_info = Det_list[d]
        if det_info[0] == 'PyCBC' :
            PSD = pycbc.psd.from_string(psd_name=det_info[1], length=det_info[2], delta_f=det_info[3],
                                    low_freq_cutoff=det_info[4])
        if det_info[0]=='Coba' :
            df_bis = pd.read_csv('AuxiliaryFiles/PSDs/Coba/'+det_info[1], sep = ' ', index_col = None)
            PSD = pycbc.psd.read.from_numpy_arrays(df_bis['f'], df_bis['HF_LF'], length = det_info[2], delta_f = det_info[3], low_freq_cutoff=det_info[4])
            #interp = InterpolatedUnivariateSpline(dfbis['f'], dfbis['HF_LF'])
            #freq = np.linspace(det_info[4],det_info[4] + det_info[3] * det_info[2],det_info[4] + det_info[3] * det_info[2])
            #PSD = interp(freq)
        grid[name] = [pycbc.filter.matchedfilter.sigma(pycbc.waveform.get_fd_waveform(approximant=approximant,
                                                                                      mass1=m1 * (1. + z),
                                                                                      mass2=m2 * (1. + z),
                                                                                      spin1x=0., spin1y=0., spin1z=0.,
                                                                                      spin2x=0., spin2y=0., spin2z=0.,
                                                                                      delta_f=det_info[3],
                                                                                      f_lower=det_info[4],
                                                                                      distance=ld,
                                                                                      inclination=i, f_ref=20.)[0],
                                                       psd=PSD,
                                                       low_frequency_cutoff=det_info[4],
                                                       high_frequency_cutoff=det_info[4] + det_info[3] * det_info[
                                                           2]-10)
                      for m1, m2, ld, z, i in zip(df["m1"], df["m2"], df["Dl"], df["zm"], df['inc'])]
        print(d)
        grid.to_csv(cat_name + '3G_SNRs'+name+'.dat', index=None, sep='\t')
    grid.to_csv(cat_name+'3G_SNRs.dat', index=None, sep='\t')
    truc = grid.describe()
    truc.to_csv(cat_name+'_describe.dat', sep='\t')
    return(cat_name +'3G_SNRs.dat')

def SNR_Net_wf(cat,Net ) :
    combi = GS.Net_psd[Net]
    df = pd.read_csv(cat, index_col = None, sep = '\t')
    print(df.describe())
    truc = df.describe()
    truc.to_csv('Ana_indSNR_'+cat, index = False, sep = '\t')
    df[Net] = np.zeros(len(df['Mc']))
    for det in combi :
        df[Net] = np.add(df[Net],df['SNR_'+det]**2 )
    return(np.sqrt(df[Net]))


