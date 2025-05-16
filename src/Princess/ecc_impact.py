import os
import pandas as pd
from lalinference.tiger.postproc import fontsize

from Princess.gwtools.utils import compute_ecc_impact
from Princess.Plots.tools import plasma_palette
import matplotlib.pyplot as plt


if __name__ == '__main__':


    # Make sure directories are created
    if not os.path.exists('AuxiliaryFiles/eccentricity_impact'):
        os.mkdir('AuxiliaryFiles/eccentricity_impact')

    #compute_ecc_impact()
    color_palette = plasma_palette(12)

    factor = pd.read_csv('./AuxiliaryFiles/eccentricity_impact/factor_ecc.dat', index_col=None, sep='\t')
    for m in range(9):
        n= m+2
        plt.plot(factor['e'], factor[f'n={n}'], label = f'n = {n}', color = color_palette[m])
    plt.legend()
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.xlabel(r'$e$', fontsize = 24)
    plt.ylabel(r'$(\frac{4}{n^2})^{1/3}\frac{g(n,e)}{\Psi(e)}$', fontsize=24)
    plt.tight_layout()
    plt.savefig('./AuxiliaryFiles/eccentricity_impact/factor_ecc.png')



