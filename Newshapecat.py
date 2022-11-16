import os
import numpy as np
import pandas as pd
import math
import sklearn as skl
import Oz.Basic_Functions as BF
import Oz.Getting_Started_Coba as GS
import astropy.cosmology as cosmo
from scipy.interpolate import InterpolatedUnivariateSpline



# Press the green button in the gutter to run the script.
if __name__ == '__main__':


def newshapecat():
    liste = ['BHNS','BNS']
    path = '/home/perigois/Documents/GreatWizardOz/Coba/'
    liste2 = ['Catalog_co_BHNS_fc_Iso_logmet_0.3_magsp_0.1_a_3.0_sn_delayed.dat',
              'Catalog_co_BNS_fc_Iso_logmet_0.3_magsp_0.1_a_3.0_sn_delayed.dat']
    for outname in range(len(liste)) :
        Cata = pd.read_csv('/home/perigois/PycharmProjects/OzGW/Catalogs/'+liste[outname]+'.dat', sep = '\t', index_col = None)
        Cat = pd.read_csv(path+liste2[outname], sep = '\t', index_col = None)
        print(Cat.describe())
        goodcat = Cata.drop(['Mc','q','Xsi'],axis = 1)
        goodcat['m1'] = Cat['m1']
        goodcat['m2'] = Cat['m2']
        goodcat['Dl'] = Cat['Dl']
        goodcat.to_csv('/home/perigois/PycharmProjects/OzGW/Catalogs/'+liste[outname]+'.txt', index = None, sep = '\t')
        print(goodcat.describe())
