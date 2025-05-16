import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
import json
#from tools import plasma_palette, get_project_params


def get_project_params(name) :
    project_params = {'name':name}
    project_params['astro_models_path'] = f'../Run/{name}'
    project_params['catalogs_path'] = f"{project_params['astro_models_path']}/{name}/Catalogs"
    project_params['omega_path'] = f"{project_params['astro_models_path']}/Results/Omega"
    project_params['analysis_path'] = f"{project_params['astro_models_path']}/Results/Analysis"
    params_json = json.load(open(f'../Run/{name}/Params.json', 'r'))
    return project_params, params_json

def plasma_palette(n):
    """
    Generate and plot a palette of `n` colors from the Plasma colormap.

    The generated colors are saved in an image file named `plasma_{n}.png`,
    with indices displayed for clarity.

    Parameters
    ----------
    n : int
        Number of colors to extract from the colormap.

    Returns
    -------
    np.ndarray
        Array of `n` colors from the Plasma colormap.
    """
    cm_plasma = plt.cm.get_cmap('plasma', n)
    colors_pla = cm_plasma(np.linspace(0, 1, n))

    # Create a plot of generated colors
    fig, ax = plt.subplots(figsize=(n, 1.5))  # Ajustement de la taille
    ax.imshow([colors_pla], aspect='auto')

    # Add indexes under each colors
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=8, rotation=90)
    ax.set_yticks([])

    # Sauvegarde de l'image
    filename = f"plasma_{n}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the file to avoid immediate display

    return colors_pla



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


# Plot residual backgrounds
linestyle_array = ['-', ':', '--', '-.']

with PdfPages("Residuals.pdf") as pdf:
    # All GC models
    color_index = 0
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in GC_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{GC_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)
        #LVK
        net_list = ['LVK_O4', 'LVK_O5']
        Total = False
        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index

        plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
        plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
        plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
        plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} LVK', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()

        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        #3G
        net_list = ['ET', '2CE4020', "ET2CE4020"]
        Total = False
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index

        plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
        plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
        plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))            
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} 3G', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()


# All Other channels
    color_index = 0
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in OtherC_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{OtherC_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)
        net_list = ['LVK_O4', 'LVK_O5']
        Total = False
        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index

        plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
        plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
        plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
        plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} LVK', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()

        #3G
        net_list = ['ET', '2CE4020', "ET2CE4020"]
        Total = False
        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index

        plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
        plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
        plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} 3G', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()

    # All Pop
    color_index = 0
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in pop3_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{pop3_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)
        net_list = ['LVK_O4', 'LVK_O5']
        Total = False
        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index

        plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
        plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
        plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
        plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} LVK', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()

        #3G
        net_list = ['ET', '2CE4020', "ET2CE4020"]
        Total = False
        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index

        plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
        plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
        plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} 3G', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()

    # All Pop
    color_index = 0
    plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite

    for model_key, model_params in AGN_param["astro_model_list"].items():
        model_name = model_params["name"]
        data_bkg = pd.read_csv(f"{AGN_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)
        #LVK
        net_list = ['LVK_O4', 'LVK_O5']
        Total = False
        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index

        plt.plot(PIC_HLV_Des["f"], PIC_HLV_Des["PSD_Omg"], label="HLV Des.", color="red", linestyle=":",
             linewidth=1)
        plt.plot(PIC_O2["f"], PIC_O2["PSD_Omg"], label="O2", color="red", linestyle="--", linewidth=1)
        plt.plot(PIC_O3["f"], PIC_O3["PSD_Omg"], label="O3", color="red", linestyle="-", linewidth=1)
        plt.plot(PIC_Aplus["f"], PIC_Aplus["PSD_Omg"], label="A+", color="red", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} LVK', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende

        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()

        #3G
        net_list = ['ET', '2CE4020', "ET2CE4020"]
        Total = False
        color_index = 0
        plt.figure(figsize=(12, 6))  # Double la largeur pour inclure la légende à droite
        for net in net_list : 
            l= 0 #linestyle index
            for thrs in ['Total','8', '20', '50']:
                if thrs == 'Total': 
                    columnname = 'Total'
                    labelcurve = 'Total'
                else :
                    columnname = f"{net}_thrs_{thrs}"
                    labelcurve = f"{net} thrs. {thrs}"
                if net == 'Total':
                    if Total == False : 
                        plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color='black', linestyle='-')
                else :
                    plt.plot(data_bkg["f"], data_bkg[columnname], label=labelcurve , color=color_palette[color_index], linestyle=linestyle_array[l])
                l+=1
            color_index = (color_index + 3) % len(color_palette)  # Évite d'aller hors index
        plt.plot(PIC_ET["f"], PIC_ET["PSD_Omg"], label="ET", color="black", linestyle="--", linewidth=1)
        plt.plot(PIC_CE_ET["f"], PIC_CE_ET["PSD_Omg"], label="ETCE40CE20", color="black", linestyle="-", linewidth=1)
        plt.plot(PIC_CE["f"], PIC_CE["PSD_Omg"], label="CE40CE20", color="black", linestyle="-.", linewidth=1)
        plt.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.ylabel(r"$\Omega_{\rm gw}$", fontsize=16)
        plt.xscale("log")
        plt.title(f'{model_name} 3G', fontsize = 12)
        plt.yscale("log")
        plt.xlim(1, 2500)
        plt.ylim(1e-14, 1e-7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Réserve de l'espace pour la légende
        pdf.savefig()  # Enregistre la figure actuelle dans le PDF
        plt.close()


