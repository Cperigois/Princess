import os
import pycbc.waveform
from astropy.cosmology import Planck15
import stochastic.constants as K
import numpy as np
import stochastic.basic_functions as BF
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import jv
from scipy.integrate import quad
from scipy.optimize import fsolve
import stochastic.pix
import warnings


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
    flim = BF.fcut_f(m1=m1, m2=m2, xsi=0, zm=z)

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
        BF.bar_peach(n, size_catalogue)

    return htildSQ




