import os
import numpy as np
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Combi = ['E1E2', 'E2E3', 'E1E3', 'E1CE1', 'E1CE2', 'E2CE1', 'E2CE2', 'E3CE1', 'E3CE2', 'CE1CE2']
    liste_in = '/home/perigois/Documents/Cat_dyn/Design_sens/gammaHL.dat'
    out = pd.DataFrame({'freq': np.arange(3000) + 1})
    g = np.array([])
    with open(liste_in, "r") as liste:
        for line in liste:
            fr, gamma = line.split()
            g = np.append(g,float(gamma))
        out['CE1CE2'] = g

    df = pd.read_csv('gamma_2ET_aligned.txt', index_col = None, sep = '\t')
    print(df.describe())
    out['EL1EL2'] = df.gamma
    df = pd.read_csv('gamma_2ET_2CE.txt', index_col=None, sep='\t')
    print(df.describe())
    out['CE1EL1'] = df.CE1EL1
    out['CE2EL1'] = df.CE2EL1
    out['CE1EL2'] = df.CE1EL2
    out['CE2EL2'] = df.CE2EL2
    out.to_csv('ORF_3G_2ETmisal.dat', index=None, sep='\t')





def designed():
    Combi = ['HL','HV', 'HK','HI', 'LV','LK','LI','VI','IK', 'VK',
             'E1E2', 'E2E3', 'E1E3', 'E1CE1', 'E1CE2','E2CE1','E2CE2','E3CE1','E3CE2', 'CE1CE2' ]
    Namefile = ['HL','HV', 'HK','HI', 'LV','LK','LI','VI','IK', 'VK',
             'E1E2', 'E2E3', 'E1E3', 'HE1', 'LE1','HE2','LE2','HE3','LE3', 'HL' ]
    out = pd.DataFrame({'freq' : np.arange(3000)+1})
    for n in range(len(Combi)) :
        liste_in = '/home/perigois/Documents/Cat_dyn/Design_sens/gamma'+Namefile[n]+'.dat'
        g = np.array([])
        with open(liste_in, "r") as liste:
            for line in liste:
                fr, gamma = line.split()
                g = np.append(g,float(gamma))
        out[Combi[n]] = g
    out.to_csv('ORF.dat', index = None, sep = '\t')