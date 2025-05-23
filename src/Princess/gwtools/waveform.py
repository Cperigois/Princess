import pandas as pd
print(f"Loading {__name__}")
import pycbc.waveform
import numpy as np
from Princess.gwtools.progress_bar import bar_peach
from Princess.gwtools.utils import fcut_f, fmerg_f, Mc, fring_f, sigma_f, eccentricity_evolution, \
    interpolate_with_zeros_outside, orbital_frequency
import warnings
from scipy.special import jv

from scipy.interpolate import InterpolatedUnivariateSpline



def GWk_no_ecc_pycbcwf(evt, freq, approx, n, size_catalogue, inc_option='InCat', disable_progress_bar = False):
    """
    This function calculates the contribution of a binary.

    Parameters
    ----------
    evt : pandas.Series
        Given event from the re-built catalogue.
    freq : numpy.array
        Observed frequency range for the calculation.
    approx : str
        Waveform approximant.
    n : int
        Index of the event in the catalogue.
    size_catalogue : int
        Total number of events in the catalogue.
    inc_option : str, optional
        Inclination handling mode ('InCat', 'Rand', 'Optimal').

    Returns
    -------
    htildSQ : numpy.array or int
        Contribution of the source to each observed frequency, or 0 if outside range.
    """
    warnings.filterwarnings("ignore")

    flow = int(np.min(freq))
    deltaf = freq[1] - freq[0]

    # Load values from the dictionnary/Dataframe
    m1 = evt['m1']
    m2 = evt['m2']
    z = evt['z']
    chi1 = evt['chi1']
    chi2 = evt['chi2']
    Dl = evt['Dl']

    if inc_option == 'InCat':
        inc = evt.inc
    elif inc_option == 'Rand':
        inc = np.arccos(np.random.uniform(-1, 1))
    elif inc_option == 'Optimal':
        inc = 0.0

    # Cumpute cutting frequency(the last emitting frequency of a source)
    flim = fcut_f(m1=m1, m2=m2, xsi=0, zm=z)

    if flim > flow:
        # Compute GW signal with Pycbc
        hptild, hctild = pycbc.waveform.get_fd_waveform(
            approximant=approx,
            mass1=m1 * (1. + z),
            mass2=m2 * (1. + z),
            spin1x=0., spin1y=0., spin1z=chi1,
            spin2x=0., spin2y=0., spin2z=chi2,
            delta_f=deltaf,
            f_lower=flow,
            inclination=inc,
            distance=Dl,
            f_ref=20.
        )

        # Fix table size
        if len(hptild) < len(freq):
            hptild = np.concatenate((hptild, np.zeros(len(freq) - len(hptild))))
            hctild = np.concatenate((hctild, np.zeros(len(freq) - len(hctild))))
        elif len(hptild) > len(freq):
            hptild = hptild[:len(freq)]
            hctild = hctild[:len(freq)]

        htildSQ = np.array(hptild * np.conjugate(hptild) + hctild * np.conjugate(hctild), dtype=float)
        htildSQ = np.nan_to_num(htildSQ, nan=0.0)
    else:
        htildSQ = flim  # If frequency out of the detector range

    # Update the progress bar
    if disable_progress_bar == False:
        bar_peach(n, size_catalogue)

    return htildSQ, freq


def Ajith_waveform(evt, freq_gw, n, size_catalogue, eccentricity_computation = False, disable_progress_bar = False):

    hinsp ,finsp  = inspiral_waveform(evt =evt)
    hmerger ,fmerger  = merger_waveform(evt =evt)
    hringdown ,fringdown  = inspiral_waveform(evt =evt)

    htildSQ_raw = np.concatenate((hinsp, hmerger, hringdown))
    freq = np.concatenate((finsp, fmerger, fringdown))

    if eccentricity_computation == True:
        e0 = evt['e0']
        a0 = evt['a0']
        ecc_spectrum = eccentricity_spectrum(evt['m1'], evt['m2'], e0, a0, freq_gw=freq)

    else:
        ecc_spectrum = np.ones(len(freq))

    htildSQ = ecc_spectrum * interpolate_with_zeros_outside(freq, htildSQ_raw, freq_gw)


    if disable_progress_bar == False:
        bar_peach(n, size_catalogue)

    return htildSQ, freq


def inspiral_waveform(evt):
    # Load values from the dictionnary/Dataframe
    m1 = evt['m1']
    m2 = evt['m2']
    z = evt['z']
    chi1 = evt['chi1']
    chi2 = evt['chi2']
    Dl = evt['Dl']

    inc = 0

    if evt['a0'] :
        fstart = 2*orbital_frequency(m1, m2,evt['a0'])
    else :
        fstart = 1e-6
    fend = fmerg_f(m1 = m1, m2 = m2, xsi = 0, zm = z)

    freq = np.logspace(fstart, fend, 200)

    spin = (m1*chi1 + m2*chi2) /(m1+m2)
    eta = (m1*m2)/(m1+m2)**2
    a2 = -323. / 224. + 451. * eta / 168.
    a3 = (27. / 8. - 11. * eta / 6.) * spin
    nu =  np.power(m1+m2 * (1 + z) * (freq) * 1.547388618e-5, 1. / 3.)
    gamma= (1+ a2 * nu ** 2 +a3 * nu ** 3 ) * freq ** (-7/6)

    hptild = hz(m1, m2, z, Dl)* gamma * (1+ np.cos(inc) ** 2)/2.

    hctild = hz(m1, m2, z, Dl)* gamma * np.cos(inc)

    htildSQ = np.array(hptild * np.conjugate(hptild) + hctild * np.conjugate(hctild), dtype=float)

    return htildSQ, freq

def merger_waveform(evt):

    # Load values from the dictionnary/Dataframe
    m1 = evt['m1']
    m2 = evt['m2']
    z = evt['z']
    chi1 = evt['chi1']
    chi2 = evt['chi2']
    Dl = evt['Dl']
    inc = 0

    spin = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    eta = (m1 * m2) / (m1 + m2) ** 2

    fstart = fmerg_f(m1=m1, m2=m2, xsi = spin, zm=z)
    fend = fring_f(m1 = m1, m2 = m2, xsi = spin, zm = z )

    freq = np.logspace(fstart, fend, 200)

    a2 = -323. / 224. + 451. * eta / 168.
    a3 = (27. / 8. - 11. * eta / 6.) * spin
    e1 = 1.4547 * spin - 1.8897
    e2 = -1.8153 * spin + 1.6557
    nu_merg = np.power((m1+m2) * (1 + z) * (fstart) * 1.547388618e-5, 1. / 3.)
    nu_ring = np.power((m1+m2) * (1 + z) * (fend) * 1.547388618e-5, 1. / 3.)
    g_merg = np.power(1. + a2 * nu_merg * nu_merg + a3 * nu_merg * nu_merg * nu_merg, 2.)
    g_ring1 = np.power(1. + e1 * nu_merg + e2 * nu_merg * nu_merg, 2.)
    wm = g_merg / (g_ring1 * fstart)
    nu =  np.power(m1+m2 * (1 + z) * (freq) * 1.547388618e-5, 1. / 3.)

    gamma = (1 + e1 * nu + e2 * nu ** 2) * freq ** (-2/3)  # Parameters calculations

    hptild = wm * hz(m1, m2, z, Dl) * gamma * (1 + np.cos(inc) ** 2) / 2.
    hctild = wm * hz(m1, m2, z, Dl) * gamma * np.cos(inc)

    htildSQ = np.array(hptild * np.conjugate(hptild) + hctild * np.conjugate(hctild), dtype=float)

    return htildSQ, freq

def ringdown_waveform(evt):

    # Load values from the dictionnary event
    m1 = evt['m1']
    m2 = evt['m2']
    z = evt['z']
    chi1 = evt['chi1']
    chi2 = evt['chi2']
    Dl = evt['Dl']

    inc = 0

    spin = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    eta = (m1 * m2) / (m1 + m2) ** 2

    fstart = fring_f(m1=m1, m2=m2, xsi = spin , zm=z)
    fend = fcut_f(m1=m1, m2=m2, xsi = spin , zm=z)
    freq = np.logspace(fstart, fend, 200)

    #Constants from Ajith waveforms
    a2 = -323. / 224. + 451. * eta / 168.
    a3 = (27. / 8. - 11. * eta / 6.) * spin
    e1 = 1.4547 * spin - 1.8897
    e2 = -1.8153 * spin + 1.6557
    nu_merg = np.power((m1 + m2) * (1 + z) * (fstart) * 1.547388618e-5, 1. / 3.)
    nu_ring = np.power((m1 + m2) * (1 + z) * (fend) * 1.547388618e-5, 1. / 3.)
    g_merg = np.power(1. + a2 * nu_merg * nu_merg + a3 * nu_merg * nu_merg * nu_merg, 2.)
    g_ring1 = np.power(1. + e1 * nu_merg + e2 * nu_merg * nu_merg, 2.)
    g_ring2 = np.power(1. + e1 * nu_ring + e2 * nu_ring * nu_ring, 2.)
    wm = g_merg / (g_ring1 * fstart)
    wr = wm * g_ring2 * np.power(fmerg_f(m1, m2, spin, z), -4. / 3.)

    #Gamma factor
    gamma = freq / (1 + (freq - fstart) / (sigma_f(m1, m2, spin, z) / 2.) * (freq - fstart) / (sigma_f(m1,m2,spin,z) / 2.))  # Parameters calculations

    hptild = wr * hz(m1, m2, z, Dl) * gamma * (1 + np.cos(inc) ** 2) / 2.
    hctild = wr * hz(m1, m2, z, Dl) * gamma * np.cos(inc)

    htildSQ = np.array(hptild * np.conjugate(hptild) + hctild * np.conjugate(hctild), dtype=float)

    return htildSQ, freq

def hz(m1, m2, z, dl):
    G = 6.674e-8  # cm3 g-1 s-2
    c = 2.99e10  # cm s-1

    GMcz5_6 = (G * Mc(m1, m2)(1 + z)) ** (5 / 6)

    return np.sqrt(5 / 24) *  GMcz5_6/ (np.pi ** (2 / 3) * c ** (3/2) * dl)


def eccentricity_spectrum(m1, m2, e0, a0, freq_gw) :

    f0 = orbital_frequency(m1, m2, a0)
    forb, e = eccentricity_evolution(e0, f0)
    ecc_spectrum = np.zeros(len(freq_gw))
    # for eccntricity paper
    df = pd.DataFrame({'freq_gw' : freq_gw})
    for m in range (19) :
        n = m+2
        fgw, factorn = harmonic_factor(n, e, forb)
        df[f'n={n}'] = interpolate_with_zeros_outside(fgw, factorn, freq_gw) # to be remove after paper publication
        ecc_spectrum += interpolate_with_zeros_outside(fgw, factorn, freq_gw)

    df.to_csv(f'AuxiliaryFiles/eccentricity_impact/ecc_spectrum_m1_{m1}_m2_{m2}_e0_{e0}_') # to be removed after paper publication

    return ecc_spectrum


def harmonic_factor(n, e_array, forb):

    fgw = n*forb

    # psi_e
    denom_psi = np.power(1 - e_array * e_array, 7. / 2.)
    psi_e = (1 + 73. * e_array ** 2 / 74. + 37. * e_array ** 4 / 96.) / denom_psi

    # g_ne
    A = jv(n - 2, n * e_array) - 2 * e_array * jv(n - 1, n * e_array) + 2 / n * jv(n, n * e_array) \
        + 2 * e_array * jv(n + 1, n * e_array) - jv(n + 2, n * e_array)
    B = jv(n - 2, n * e_array) - 2 * jv(n, n * e_array) + jv(n + 2, n * e_array)

    C = jv(n, n * e_array)

    g_ne = (np.power(4. / (n * n), 1. / 3.) *
                np.power(n, 4.) / 32 *
                (np.power(A, 2.) +
                 (1 - e_array * e_array) * np.power(B, 2.) +
                 4 / (3 * (n * n)) * C * C)
        )

    # factor
    factor = np.power(4. / (n * n), 1. / 3.) * g_ne / psi_e

    return fgw, factor


