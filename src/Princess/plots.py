import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
import json
from tools import plasma_palette, get_project_params


#Get GC analysis
GC_ana, GC_param = get_project_params('GC_analysis')
GC_names = {'GC_ng1g_heggie_clusterevolv' : 'A_Evol',
    'GC_ng1g_heggie_clusterevolv_tidal' : 'A_Tidal',
    'GC_ng1g_heggie_noclusterevolv' : 'A_NoEvol',
    'GC_ng1g_oleary_clusterevolv' : 'B_Evol',
    'GC_ng1g_oleary_clusterevolv_tidal' : 'B_Tidal',
    'GC_ng1g_oleary_noclusterevolv' : 'B_NoEvol',
    'GC_ngng_oleary_clusterevolv' : 'C_Evol',
    'GC_ngng_oleary_clusterevolv_tidal' : 'C_Tidal',
    'GC_ngng_oleary_noclusterevolv' : 'C_NoEvol',
    'Field' : 'Field' 
}

#Get Pop III analysis
pop3_ana, pop3_param = get_project_params('PopIII')

#Get AGN analysis
AGN_ana, AGN_param = get_project_params('AGNs')

#Get other channels
OtherC_ana, OtherC_param = get_project_params('OtherChannels')

n= 10

color_palette = plasma_palette(n)

#OTHER CHANNELS

path_other_channels = '/home/perigois/Documents/Catalogs_channel/Results/'

Omg_YSC = pd.read_csv(path_other_channels+'Omega/YSC_Dyn.dat', index_col=None, sep = '\t')
Omg_NSC = pd.read_csv(path_other_channels+'Omega/NSC_Dyn.dat', index_col=None, sep = '\t')
Omg_GC = pd.read_csv(path_other_channels+'Omega/GC_Dyn.dat', index_col=None, sep = '\t')
Omg_Field = pd.read_csv(path_other_channels+'Omega/Field.dat', index_col=None, sep = '\t')

#SENSITIVITIES CE = 40+20km at L and H places respectively

path_pics = "../AuxiliaryFiles/PICs"

PIC_HLV_design = pd.read_csv(path_pics+'/Design_HLVIK_flow_10.txt', delimiter = ' ', names = ['f','PSD_Omg'],index_col = False,header = None)
PIC_O2 = pd.read_csv(path_pics+'/PICs_2G/o2_interpolated.csv', names = ['f','PSD_Omg'],index_col = False,header = None)
PIC_O3 = pd.read_csv(path_pics+'/PICs_2G/o3_interpolated.csv', names = ['f','PSD_Omg'],index_col = False,header = None)
PIC_Aplus = pd.read_csv(path_pics+'/PICs_2G/Aplus_interpolated.csv', names = ['f','PSD_Omg'],index_col = False,header = None)
PIC_HLV_Des = pd.read_csv(path_pics+'/PICs_2G/HLV_design_interpolated.csv', names = ['f','PSD_Omg'],index_col = False,header = None)

PIC_ET = pd.read_csv(path_pics+'/PIC_ET.txt', delimiter = ' ', names = ['f','PSD_Omg'],index_col = False,header = None)
PIC_CE_ET = pd.read_csv(path_pics+'/PIC_CE4020ET.txt', delimiter = ' ', names = ['f','PSD_Omg'],index_col = False,header = None)
PIC_CE = pd.read_csv(path_pics+'/PIC_CE4020.txt', delimiter = ' ', names = ['f','PSD_Omg'],index_col = False,header = None)



with PdfPages("Background.pdf") as pdf:
    # All GC models
    color_index = 0
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in GC_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{GC_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)

        plt.plot(data_bkg["f"], data_bkg["Total"], label=GC_names[model_name], color=color_palette[color_index], linestyle="-")
        color_index = (color_index + 1) % len(color_palette)  # Évite d'aller hors index

    plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
    plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
    plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
    plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
    plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
    plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
    plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
    plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Frequency [Hz]", fontsize=12)
    plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
    plt.xscale("log")
    plt.title('Globular clusters', fontsize = 12)
    plt.yscale("log")
    plt.xlim(1, 2500)
    plt.ylim(1e-14, 1e-7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

    pdf.savefig()  # Enregistre la figure actuelle dans le PDF
    plt.close()

    # All AGN models
    color_index = 1
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in AGN_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{AGN_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)

        plt.plot(data_bkg["f"], data_bkg["Total"], label=model_name, color=color_palette[color_index], linestyle="-")
        color_index = (color_index + 6) % len(color_palette)  # Sécurisation de l'index

    plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
    plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
    plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
    plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
    plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
    plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
    plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
    plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('AGNs', fontsize = 12)
    plt.xlabel("Frequency [Hz]", fontsize=12)
    plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, 2500)
    plt.ylim(1e-14, 1e-7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

    pdf.savefig()  # Enregistre la figure actuelle dans le PDF
    plt.close()

    # Other channels
    color_index = 0
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in OtherC_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{OtherC_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)

        plt.plot(data_bkg["f"], data_bkg["Total"], label=model_name, color=color_palette[color_index], linestyle="-")
        color_index = (color_index + 2) % len(color_palette)  # Sécurisation de l'index

    plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
    plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
    plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
    plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
    plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
    plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
    plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
    plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('Other Channels', fontsize = 12)
    plt.xlabel("Frequency [Hz]", fontsize=12)
    plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, 2500)
    plt.ylim(1e-14, 1e-7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

    pdf.savefig()  # Enregistre la figure actuelle dans le PDF
    plt.close()


    # PopIII
    color_index = 0
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in pop3_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{pop3_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)

        plt.plot(data_bkg["f"], data_bkg["Total"], label=model_name, color=color_palette[color_index], linestyle="-")
        color_index = (color_index + 2) % len(color_palette)  # Sécurisation de l'index

    plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
    plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
    plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
    plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
    plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
    plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
    plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
    plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('PopIII', fontsize = 12)
    plt.xlabel("Frequency [Hz]", fontsize=12)
    plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, 2500)
    plt.ylim(1e-14, 1e-7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

    pdf.savefig()  # Enregistre la figure actuelle dans le PDF
    plt.close()






