import os
import numpy as np
import GreatWizard
import pandas as pd
import Starter as Start
import Getting_Started as GS


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Read Catalog and check the columns
    os.system("mkdir Results/")
    os.system("mkdir Catalogs")
    os.system("mkdir Results/Omega")
    os.system("mkdir Results/Omega_e0")
    os.system("mkdir Results/Ana")
    print(GS.Cat_list.keys())
    for cat in GS.Cat_list.keys() :
        model = GS.Cat_list[cat]
        Cat_paths = Start.MakeCat(model = cat)
        Cat_paths = Start.MakeCat(cat)
        print(Cat_paths)
        for c in Cat_paths :
            GreatWizard.Oz_pycbc(path = c, model = cat)#, co_type = model[1])

