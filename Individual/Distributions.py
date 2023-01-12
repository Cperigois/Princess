import math
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from Stochastic import Basic_Functions as BF
from Starter.Htild import GWk_noEcc_Pycbcwf

def distrib(Model, parameters = ['m1', 'q', 'z']) :
    """This function extract expected observed parameters distribution from CBCs catalogues, and write it in files.
            ----------
            Model : AstroModel
                Built from the AstroModel class.
            parameters : list
                List of parameters to be analyzed, default ['m1', 'q', 'z']

            Returns
            -------
            """

    cat = pd.dataframe()