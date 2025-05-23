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

    compute_ecc_impact()
    color_palette = plasma_palette(20)
    factor = pd.read_csv('./AuxiliaryFiles/eccentricity_impact/factor_ecc.dat', index_col=None, sep='\t')

    # Définition des harmoniques à afficher dans la légende
    legend_ns = {2, 3, 4, 6, 8, 12, 20}

    # Tracé des courbes
    for m in range(19):
        n = m + 2
        label = f'n = {n}' if n in legend_ns else None
        plt.plot(factor['e'], factor[f'n={n}'], label=label, color=color_palette[m])

    # Ligne de référence horizontale
    plt.axhline(y=0.01, color='gray', linestyle='--', linewidth=1)
    plt.text(0.8, 0.013, 'factor = 0.01', color='gray', fontsize=14)

    # Configuration du graphique
    plt.legend(fontsize=12)
    plt.xlim(0,1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel(r'$e$', fontsize=24)
    plt.ylabel(r'$\left(\frac{4}{n^2}\right)^{1/3}\frac{g(n,e)}{\Psi(e)}$', fontsize=24)
    plt.tight_layout()
    plt.savefig('./AuxiliaryFiles/eccentricity_impact/factor_ecc.png')



