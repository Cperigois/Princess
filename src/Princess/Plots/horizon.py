import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
import json
from tools import plasma_palette, get_project_params
import gwtools.Detector as DET
import gwtools.Network as NET
import Plots.tools as Plttools
os.system('../')
params = json.load(open('./Run/Params.json', 'r'))

for det in params['detector_list'].keys():
    detector = DET.Detector(name=params['detector_list'][det]['name'],
                            configuration=params['detector_list'][det]['configuration'],
                            origin=params['detector_list'][det]['origin'],
                            reference=params['detector_list'][det]['reference'],
                            type=params['detector_list'][det]['type'])

for net in params['network_list'].keys():
    network = NET.Network(name=params['network_list'][net]['name'],
                          compo=params['network_list'][net]['compo'],
                          pic_file=params['network_list'][net]['pic_file'],
                          efficiency=params['network_list'][net]['efficiency'],
                          SNR_thrs=params['network_list'][net]['SNR_thrs'])

    Plttools.horizon_network(network, 150)

color_palette = Plttools.plasma_palette(6)
plt.figure(figsize=(12, 6))
n=0
with PdfPages("Plots/outputs/Horizons.pdf") as pdf:
    for net in params['network_list'].keys():
        df = pd.read_csv(f'horizon_{net}.csv', sep = '\t', index_col = None)
        plt.scatter(df["Mt"], df["z_q_1"], label= "q = 1", color=color_palette[n], markerstyle="P")
        plt.scatter(df["Mt"], df["z_q_0.8"], label= "q = 0.8", color=color_palette[n], markerstyle="X")
        plt.scatter(df["Mt"], df["z_q_0.5"], label= "q = 0.5", color=color_palette[n], markerstyle="*")

        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Total mass [Msun]", fontsize=12)
        plt.ylabel(r"$z$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{net}', fontsize = 12)
        plt.yscale("log")
        #plt.xlim(1, 2500)
        #plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende
        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()
        n+=1

