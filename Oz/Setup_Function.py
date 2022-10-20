import os
import numpy as np
import pandas as pd
import math
import sklearn as skl
import Oz.Basic_Functions as BF
import Oz.Getting_Started_Coba as GS
import astropy.cosmology as cosmo
from scipy.interpolate import InterpolatedUnivariateSpline
import Oz.SNR as SNR


def MakeCat(model) :
    """Build adequate catalogue for the computation
            Parameters :
            ----------
            path : str
                path to the catalogue
            model : str
                Name of the model, used to setup the output folder
            type : str
                {'BBH' , 'BNS', 'BHNS', 'mix'} type of binary compact objects in the initial catalogue.
                'mix' option is not available !
            Return :
            ret : np.array of str
                list of paths to the catalogues made with this function
    """
    cat = GS.Cat_list_df.loc[model]
    Cata = pd.read_csv(cat.path, sep = '\t', index_col = None)

    ret = np.array([])
    Col = list(Cata.columns)
    Keys = list(GS.Flags.keys())
    if 'flag' in Col :
        Cata = Cata.astype({"flag": str}, errors='ignore')
        for i in range(len(Keys)):
            Cata = Cata[Cata['flag']==Keys[i]]
            key = Keys[i]
            print(Cata)
            if len(Cata['Mc']) != 0 :
                outname = model+'_'+GS.Flags[key]+'.dat'
                mkCat(Cata, outname, model)
                ret  = np.append(ret, outname)
    else :
        outname = model + '.txt'
        check_file = os.path.exists('Catalogs/'+outname)
        if check_file == False : # Check if the catalogue already exist
            print(list(Cata.columns))
            mkCat(Cata, model)
        ret = np.append(ret, outname)

    return ret

def mkCat(Cata, outname, model= None) :
    Col = list(Cata.columns)
    model = GS.Cat_list[model]
    if 'zm' not in Col :
        Cata['zm'] = Cata['z']
    #if 'Mc' not in Col:
    #    Cata['Mc'], Cata['q'] = BF.m1_m2_to_mc_q(Cata['m1'], Cata['m2'])
    if 'm1' not in Col:
        Cata['m1'], Cata['m2'] = BF.mc_q_to_m1_m2(Cata['Mc'], Cata['q'])
    if 'Xsi'  not in Col:
        if model[1]=='BBH' :
            Cata['Xsi'] = BF.Xsi_compil(Cata['m1'], Cata['m2'],Cata['theta1'], Cata['theta2'],Cata['chi1'], Cata['chi2']) #Add differend spin model
        else :
            Cata['Xsi'] = np.zeros(len(Cata['m1']))
    if'Dl' not in Col:
        zm = np.array(Cata['zm'], float)
        dl = np.array([])
        for z in zm :
            dl = np.append(dl, cosmo.Planck15.luminosity_distance(z).value)
        Cata['Dl'] = dl
    if GS.orbit_evo == False :
        OutCat = pd.DataFrame(
            {'Mc': Cata['Mc'], 'q': Cata['q'], 'Xsi': Cata['Xsi'], 'zm': Cata['zm']})
    else :
        OutCat = pd.DataFrame(
            {'m1': Cata['m1'], 'm2': Cata['m2'], 'spinz1': Cata['spinz1'], 'zm': Cata['zm'], 'spinz2': Cata['spinz2'], 'a0': Cata['a0'],
             'e0': Cata['e0']})
    if 'inc' in Col :
        OutCat['inc'] = Cata['inc']
    print(Cata.describe())
    fileexist = os.path.exists(outname+'3G_SNRs.dat')
    if fileexist == False :
        SNRfile = SNR.SNRwf(Cata, outname)
    else :
        SNRfile = outname+'3G_SNRs.dat'
    print('ici')
    for N in GS.Networks:
        #OutCat[N] = SNR.Cat(Cata, Net=N)
        OutCat[N] = SNR.SNR_Net_wf(SNRfile, Net=N)
    OutCat.to_csv('Catalogs/'+outname, sep='\t', index=False)
    truc = OutCat.describe()
    truc.to_csv('Catalogs/Ana_'+outname, sep='\t')










