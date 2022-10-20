import numpy as np
import pandas as pd
import Oz.Kst as K # Constants, in CGS
import math
import matplotlib.pyplot as plt
import itertools as iterT
import GreatWizard as GW
import pycbc.psd
import sklearn as skl
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def theta1(a,b):
    return(math.pow(math.cos(b/2.),4.)*g1(a))
def theta2(a,b) :
    return ( math.pow(math.cos(b / 2.), 4.) * g2(a) + g3(a) - math.pow(math.sin(b/2.),4.)*(g2(a)+g1(a)) )
def falpha(a) :
    f = np.array([a*math.cos(a)/(math.pow(a,5.)),
                  math.pow(a,3.)*math.cos(a)/(math.pow(a,5.)),
                  math.sin(a)/(math.pow(a,5.)),
                  a*a*math.sin(a)/(math.pow(a,5.)),
                  math.pow(a,4.)*math.sin(a)/(math.pow(a,5.))
                  ])
    return(f)
def g1(a) :
    fa = falpha(a)
    g = np.array([-9,-6,9,3,1])
    return(5.*np.dot(fa,g)/16.)
def g2(a) :
    fa = falpha(a)
    g = np.array([45,6,-45,9,3])
    return(5.*np.dot(fa,g)/16.)
def g3(a) :
    fa = falpha(a)
    g = np.array([15,-4,-15,9,-1])
    return(5.*np.dot(fa,g)/4.)

if __name__ == '__main__':

    #distances : Sardainia-Netherlands: 1400 km, Hanford-Livinstone: 3000km
    freq = np.linspace(1, 3000, 3000)
    #delta = math.sqrt(2.)/4. # alignement 0: aligned, sqrt(2)/4: antialigned, delta = (sigma1-sigma2)/2
    Delta = 0 # delta = (sigma1+sigma2)/2
    Config = ['aligned', 'misaligned']#, 'CE2EL1', 'CE2EL2']
    delta_l = [0,math.sqrt(2.)/4. ]
    dist_arc = [1400.e5, 1400.e5]#, 8640.e5, 7765.e5]#distance de l'arc entre 2 detecteurs en cm from google earth
    data = pd.DataFrame({'f': freq})
    for conf in range(len(Config)) :
        delta = delta_l[conf]
        beta = dist_arc[conf]/K.R_earth # angle done by the two detector at the earth center in rad
        d = 2.*K.R_earth*math.sin(beta)
        res = np.array([])
        for f in freq :
            a = 2*math.pi*f*d/K.c
            gamma = math.cos(4*delta)*theta1(a,beta) + math.cos(4*Delta)*theta2(a,beta)
            res = np.append(res, gamma)
        data[Config[conf]] = res
        plt.plot(freq, res, label = Config[conf])
    data.to_csv('gamma_2ET.txt', index = None, sep = '\t')
    #HL = pd.read_csv('/home/perigois/Documents/GreatWizardOz/Coba/ET_Sens/ORF/gamma_LH.csv', names = ['f', 'gamma'], header = None, index_col= None)
    #plt.scatter(HL.f, HL.gamma, label = 'LIGO old config checks')
    plt.xlabel(r'$f$ [Hz]', fontsize = 20)
    plt.ylabel(r'$\gamma(f)$', fontsize=20)
    plt.legend(fontsize = 20)
    plt.xlim(1,500)
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    #plt.title('2 ET')
    #plt.xscale('log')
    plt.show()