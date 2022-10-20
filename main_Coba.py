import os
import numpy as np
import GreatWizard
import pandas as pd
import Oz.Setup_Function as SF
import Oz.Getting_Started_Coba as GS


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Read Catalog and check the columns
    os.system("mkdir Results/")
    os.system("mkdir Catalogs")
    os.system("mkdir Results/Omega")
    os.system("mkdir Results/Omega_e0")
    os.system("mkdir Results/Ana")
    for cat in GS.Cat_list.keys() :
        model = GS.Cat_list[cat]
        Cat_paths = SF.MakeCat( model = cat)
        print(Cat_paths)
        for c in Cat_paths :
            GreatWizard.Oz_pycbc(path = c, model = cat)

