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




def GWk(evt:np.ndarray , type:str, inc:float = None) :
    """
    NOT ADAPTED YET TO THE DATAFRAME FORMAT.
    Compute the contribution of a binary to the background with analytical waveforms from 0909.2867, and 1210.6666.
    Parameters
    ----------
    :param evt (1Darray):
    :param type:
    :param inc:
    :return:
    """

    Mc = evt[0]
    q = evt[1]
    Spin = evt[2]
    z = evt[3]
    zf = evt[4]
    a0 = evt[5]
    ecc = evt[6]
    if inc == 0 :
        Fi = 4./5.
    else :
        Fi = pow(1. + pow(math.cos(float(inc)), 2.), 2.) / 4. + pow(math.cos(float(inc)), 2.)

    f = GS.Freq
    mtot = BF.mass1(Mc, q) + BF.mass2(Mc, q)
    eta = BF.mass1(Mc, q) * BF.mass2(Mc, q) / (math.pow(mtot, 2.))
    Omg = np.zeros(len(f))
    Omg_e0 = np.zeros(len(f))

    if float(ecc) == 1.:  # check that the eccentricity is not 1
        e0 = 0.9999
    else:
        e0 = float(ecc)
    if float(a0) == 0 :
        f0 = 1
    else :
        f0 = 1.e-4 * math.pow(float(a0) + 1, -3. / 2.) * math.pow(mtot, 0.5)
    if f0 < 100000:  # old test now f0 always >1000
        f_lso = 4394.787 / (mtot * (1 + z))
        f_merg = BF.fmerg_f(BF.mass1(Mc, q), BF.mass2(Mc, q), Spin, z)
        f_ring = BF.fring_f(BF.mass1(Mc, q), BF.mass2(Mc, q), Spin, z)
        f_cut = BF.fcut_f(BF.mass1(Mc, q), BF.mass2(Mc, q), Spin, z)
        sigma = BF.sigma_f(BF.mass1(Mc, q), BF.mass2(Mc, q), Spin, z)
        if ((type == 'BNS') or (type == 'BHNS')):
            f_merg = f_lso
            f_ring = 9.e-7
            f_cut = 9.e-7

        # Parameters calculations
        a2 = -323. / 224. + 451. * eta / 168.
        a3 = (27. / 8. - 11. * eta / 6.) * Spin
        e1 = 1.4547 * Spin - 1.8897
        e2 = -1.8153 * Spin + 1.6557
        nu_merg = math.pow(mtot * (1 + z) * (f_merg) * 1.547388618e-5, 1. / 3.)
        nu_ring = math.pow(mtot * (1 + z) * (f_ring) * 1.547388618e-5, 1. / 3.)
        g_merg = math.pow(1. + a2 * nu_merg * nu_merg + a3 * nu_merg * nu_merg * nu_merg, 2.)
        g_ring1 = math.pow(1. + e1 * nu_merg + e2 * nu_merg * nu_merg, 2.)
        g_ring2 = math.pow(1. + e1 * nu_ring + e2 * nu_ring * nu_ring, 2.)
        wm = g_merg / (g_ring1 * f_merg)
        wr = wm * g_ring2 * math.pow(f_ring, -4. / 3.)
        # Doing the interpolation between the observed frequency and the redshift
        if abs(float(zf) - z) >= 0.05:
            z_int = []
            f_int = []
            z2 = float(zf)
            z = z2
            f_pre = math.pow(f0, -8. / 3.)
            #k = 145754753.28 * math.pow(Mc, 5. / 3.)
            while z > float(z):  # graphe temps red freq, verifier sans ecc et sans Q
                tau_z = quad(BF.tau_Myr, z, z2)
                f1 = pow(f_pre, -8. / 3.) - K.Cst * tau_z[0] * pow(Mc * K.M_sun, -5. / 3.) * K.yr * 1.e6
                f_int.append(math.pow(f1, -3. / 8.) / (1. + z))
                z_int.append(z)
                f_pre = pow(f1, -3. / 8.)
                z2 = z
                z = z - 0.001
            int_zf = InterpolatedUnivariateSpline(f_int, z_int)
        z = float(zf)
        en = [e0, e0, e0, e0, e0, e0, e0, e0]
        i = 0
        if f_cut == 9.e-7:  # if the source is a BNS or a NSBH:
            f_cut = f_lso  # turn on frequencies until the last stable orbit
            wr = 0  # No ringdown
        while (f[i] < f_cut):
            n = 2
            while n < 6:
                if (z - z) >= 0.05:
                    z = abs(int_zf(f[i] / n))
                else:
                    z = z
                if f[i] > n * f0 / (1 + z):
                    if type == 'BBH':
                        nu = math.pow(mtot * (1 + z) * f[i] * 1.547388618e-5, 1. / 3.)
                        g_merg = math.pow(1. + a2 * nu * nu + a3 * nu * nu * nu, 2.)
                        g_ring = math.pow(1. + e1 * nu + e2 * nu * nu, 2.)
                    else:
                        g_merg = 1.
                    Dl = Planck15.luminosity_distance(z).value * K.Mpc# check units !!!
                    K2 = pow(Mc * K.M_sun * (1 + z), 5. / 3.) / (Dl * Dl )
                    if n == 2:  # Data for circular orbits
                        if f[i] < f_merg:
                            Omg_e0[i] += g_merg * K.K1 * K2 * math.pow(f[i], 2. / 3.) * Fi
                        elif f[i] < f_ring:
                            Omg_e0[i] += wm * g_ring * K.K1 * K2 * math.pow(f[i], 5. / 3.) * Fi
                        else:
                            Omg_e0[i] += wr * K.K1 * K2 * f[i] * math.pow((f[i]) / (
                                        1 + (f[i] - f_ring) / (sigma / 2.) * (f[i] - f_ring) / (
                                        sigma / 2.)), 2.) * Fi

                    if en[n] > 0.00105:  # If the eccentricity is higher than 0.00105 we take the harmonics into account
                        e = fsolve(BF.fe3, 1.e-7, args=(mtot, float(a0), e0, f[i] * (1 + z) / (float(n))))
                        psi_e = (1 + 73. * e[0] * e[0] / 74. + 37. * e[0] * e[0] * e[0] * e[0] / 96.) / (
                            pow(1 - e[0] * e[0], 7. / 2.))
                        g_ne = math.pow(4. / float(n * n), 1. / 3.) * pow(n, 4.) / 32 * (
                                pow(jv(n - 2, n * e[0]) - 2 * e[0] * jv(n - 1, n * e[0]) + 2 / n * jv(n,
                                                                                                      n * e[
                                                                                                          0]) + 2 *
                                    e[0] * jv(n + 1, n * e[0]) - jv(n + 2, n * e[0]), 2.) + (
                                        1 - e[0] * e[0]) * pow(
                            jv(n - 2, n * e[0]) - 2 * jv(n, n * e[0]) + jv(n + 2, n * e[0]), 2.) + 4 / (
                                        3 * n * n) * jv(n, n * e[0]) * jv(n, n * e[0]))
                        en[n] = e[0]
                        n += 1
                    else:  # If the eccentricity is low we do not calculate harmonics above n = 2
                        psi_e = 1.
                        g_ne = math.pow(4. / float(n * n), 1. / 3.)
                        en[n] = 0
                        n += 6
                    if f[i] < f_merg:
                        Omg[i] += g_merg * K.K1 * K2 * math.pow(f[i], 2. / 3.) * g_ne / psi_e * Fi
                    elif f[i] < f_ring:
                        Omg[i] += wm * g_ring * K.K1 * K2 * math.pow(f[i], 5. / 3.) * g_ne / psi_e * Fi
                    else:
                        Omg[i] += wr * K.K1 * K2 * f[i] * math.pow(
                                (f[i]) / (1 + (f[i] - f_ring) / (sigma / 2.) * (f[i] - f_ring) / (sigma / 2.)),
                                2.) * g_ne / psi_e * Fi
                else:
                    n = n + 1
            i += 1
    return Omg, Omg_e0

def GWk_noEcc(evt, type, inc = None) :

    """
        Parameters
        ----------
        args : numpy array
            Given by the re-built catalogue
        f : numpy array
            Observed frequency range for the calculation. Has to be the same for all sources from a catalogue

        Returns
        -------
        Omg : numpy array
            Size of f with the contribution of the source for each observed frequency
        """

    Mc = evt[0]
    q = evt[1]
    Spin = evt[2]
    z = evt[3]
    zf = z
    if inc == 0 :
        Fi = 4./5.
    else :
        Fi = pow(1. + pow(math.cos(float(inc)), 2.), 2.) / 4. + pow(math.cos(float(inc)), 2.)

    f = GS.Freq
    m1 = BF.mass1(Mc, q)
    m2 = BF.mass2(Mc, q)
    mtot = m1 + m2
    eta = m1 * m2 / (math.pow(mtot, 2.))
    Omg_e0 = np.zeros(len(f))

    f_lso = 4394.787 / (mtot * (1 + z))
    f_merg = BF.fmerg_f(m1,m2 , Spin, z)
    f_ring = BF.fring_f(m1, m2, Spin, z)
    f_cut = BF.fcut_f(BF.mass1(Mc, q), BF.mass2(Mc, q), Spin, z)
    sigma = BF.sigma_f(BF.mass1(Mc, q), BF.mass2(Mc, q), Spin, z)
    if f_merg>1. :
        if ((type == 'BNS') or (type == 'BHNS')):
            f_merg = f_lso
            f_ring = 9.e-7
            f_cut = 9.e-7
        # Parameters calculations
        a2 = -323. / 224. + 451. * eta / 168.
        a3 = (27. / 8. - 11. * eta / 6.) * Spin
        e1 = 1.4547 * Spin - 1.8897
        e2 = -1.8153 * Spin + 1.6557
        nu_merg = math.pow(mtot * (1 + z) * (f_merg) * 1.547388618e-5, 1. / 3.)
        nu_ring = math.pow(mtot * (1 + z) * (f_ring) * 1.547388618e-5, 1. / 3.)
        g_merg = math.pow(1. + a2 * nu_merg * nu_merg + a3 * nu_merg * nu_merg * nu_merg, 2.)
        g_ring1 = math.pow(1. + e1 * nu_merg + e2 * nu_merg * nu_merg, 2.)
        g_ring2 = math.pow(1. + e1 * nu_ring + e2 * nu_ring * nu_ring, 2.)
        wm = g_merg / (g_ring1 * f_merg)
        wr = wm * g_ring2 * math.pow(f_ring, -4. / 3.)
        z = float(zf)
        i = 0
        if f_cut == 9.e-7:  # if the source is a BNS or a NSBH:
            f_cut = f_lso  # turn on frequencies until the last stable orbit
            wr = 0  # No ringdown
        while (f[i] < f_cut):
            n = 2
            if type == 'BBH':
                nu = math.pow(mtot * (1 + z) * f[i] * 1.547388618e-5, 1. / 3.)
                g_merg = math.pow(1. + a2 * nu * nu + a3 * nu * nu * nu, 2.)
                g_ring = math.pow(1. + e1 * nu + e2 * nu * nu, 2.)
            else:
                g_merg = 1.
            Dl = Planck15.luminosity_distance(z).value * K.Mpc# check units !!!
            K2 = pow(Mc * K.M_sun * (1 + z), 5. / 3.) / (Dl * Dl )
            if f[i] < f_merg:
                Omg_e0[i] += g_merg * K.K1 * K2 * math.pow(f[i], 2. / 3.) * Fi
            elif f[i] < f_ring:
                Omg_e0[i] += wm * g_ring * K.K1 * K2 * math.pow(f[i], 5. / 3.) * Fi
            else:
                Omg_e0[i] += wr * K.K1 * K2 * f[i] * math.pow((f[i]) / (
                                            1 + (f[i] - f_ring) / (sigma / 2.) * (f[i] - f_ring) / (
                                            sigma / 2.)), 2.) * Fi
            i+=1
    return Omg_e0


def GWk_no_ecc_pycbcwf(evt, freq, approx, n, size_catalogue, inc_option = 'InCat') :
    """This function calculate the contribution of a binary
        Parameters
        ----------
        args : numpy array
            Given by the re-built catalogue
        f : numpy array
            Observed frequency range for the calculation. Has to be the same for all sources from a catalogue

        Returns
        -------
        Omg : numpy array
            Size of f with the contribution of the source for each observed frequency
        """
    warnings.filterwarnings("ignore")
    flow = int(np.min(freq))
    deltaf = freq[1]-freq[0]
    if inc_option == 'InCat':
        inc = evt.inc
    elif inc_option == 'Rand' :
        inc = np.arccos(np.random.uniform(-1, 1))
    elif inc_option == 'Optimal' :
        inc = 0.
    flim= BF.fcut_f(m1 = evt.m1, m2 = evt.m2, xsi = 0, zm = evt.z).values
    print(flim)search line in graph python
    if flim>flow:
        hptild, hctild = pycbc.waveform.get_fd_waveform(approximant = approx,
                                                        mass1 = evt.m1* (1. + evt.z),
                                                        mass2 = evt.m2 * (1. + evt.z),
                                                        spin1x = 0., spin1y=0., spin1z=evt.chi1,
                                                        spin2x = 0., spin2y=0., spin2z=evt.chi2,
                                                        delta_f=deltaf,
                                                        f_lower=flow,
                                                        inclination= inc,
                                                        distance=evt.Dl, f_ref=20.)
        #hptild = hptild[flow:]
        #hctild = hctild[flow:]
        if len(hptild) < len(freq):
            hptild = np.concatenate((hptild, np.zeros(len(freq) - len(hptild))))
            hctild = np.concatenate((hctild, np.zeros(len(freq) - len(hctild))))
        elif len(hptild) > len(freq):
            hptild = hptild[:len(freq)]
            hctild = hctild[:len(freq)]
        htildSQ = np.array(hptild * np.conjugate(hptild) + hctild * np.conjugate(hctild), dtype=float)
        htildSQ = np.nan_to_num(htildSQ, nan=0.0)
    else :
        htildSQ = 0
        #print('Waveform calculation failed, parameters m1 = {0}, m2 = {1}, and z = {2} hence to a cut frequency below the start of the analysis fcut = {3} < flow = {4}.'.format(evt.m1.values, evt.m2.values, evt.z.values, BF.fcut_f(m1 = evt.m1, m2 = evt.m2, xsi = 0, zm = evt.z).values, flow))

    #Stochastic.pix.bar_peach(n,size_catalogue)

    return htildSQ







