print(f"Loading {__name__}")
import os
import numpy as np
import pandas as pd

from Princess.gwtools.utils import zmaximal
from Princess.gwtools.snr import SNR_source

def horizon(network, SNR_threshold: float = 9., mmin: float = 1., mmax: float = 10000., waveform: str = "IMRPhenomD",
            zmax: float = 150., mratio: float = 1.):
    deltaz = [10, 1, 0.1, 0.01, 0.001]
    Mtot = np.logspace(np.log10(mmin), np.log10(mmax), 100)
    Hori = np.zeros(len(Mtot))
    print(Mtot[0])

    for m in range(len(Mtot)):
        print(m)
        z = 0.001
        m1 = Mtot[m] * mratio / (1 + mratio)
        m2 = m1 / mratio
        zmax_1Hz = np.maximum(zmaximal(m1, m2, 0, 1.2), 0.001)
        for dz in deltaz:
            snr = SNR_threshold + 0.001
            print(snr, ' ', SNR_threshold, ' ', zmax_1Hz, ' ', zmax)
            while ((snr > SNR_threshold) & ((z + dz) < np.minimum(zmax_1Hz, zmax))):
                z = z + dz
                snr_net = 0
                for d in network.compo:
                    snr_net += np.power(SNR_source(d, Mtot[m], z, mratio, waveform), 2.)
                snr = np.sqrt(snr_net)
            z = np.maximum(z - dz, 0.001)
        Hori[m] = z + dz
    output = pd.DataFrame({'Mtot': Mtot, 'Horizon': Hori})
    filename = 'Horizon_' + network.name + '_' + str(mmax) + str(zmax) + '_' + waveform
    if os.path.exists('Horizon') == False:
        os.mkdir('Horizon')
    output.to_csv('Horizon/' + filename + '.dat', sep='\t', index=None)