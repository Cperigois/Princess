import os
import numpy as np
from Starter.Detection import Detector, Network
from Stochastic import Princess as SP
from Starter.AstroModel import Astromodel


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #Setup and load astro model
    astromodel1 = Astromodel(cat_name='BBH_YSCs', original_cat_path=GS.path_2_cat_folder + GS.Cat, cat_sep="\t",
                             index_column=None, flags=GS.Flags)
    # astromodel1.makeHeader(['Mc','q','zm','zf','t_del','Z','a0','e','flag'])
    # astromodel1.makeCat(flags = Flags)

    print('Astromodel loaded and ready :)')

    astromodel1.compute_SNR()

