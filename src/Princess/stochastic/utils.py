from scipy.interpolate import InterpolatedUnivariateSpline
import math
import numpy as np

def Search_Omg(Freq, Omega, freq_ref):
    interp = InterpolatedUnivariateSpline(Freq, Omega)
    out = interp(freq_ref)
    return out

def rho_c(H0) :
    G = 6.674e-8  # cm3 g-1 s-2
    c = 2.99e10  # cm s-1
    return 3. * c * c * H0 * H0 / (8 * math.pi * G)

def Compute_constant(H0) :
    G = 6.674e-8  # cm3 g-1 s-2
    c = 2.99e10  # cm s-1
    yr = 365 * 24 * 3600  # s
    return math.pi * c * c / (rho_c(H0) * 2. * G * yr)

def Cst_snr_bkg(H0) :
    yr = 365 * 24 * 3600  # s
    return 3.*H0*H0*np.sqrt(yr)/(10.*math.pi**2)

