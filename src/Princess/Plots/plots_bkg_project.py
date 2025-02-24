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

    #load data

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
GC_data = {}
for model_key, model_params in GC_param["astro_model_list"].items():
        model_name = model_params["name"]
        GC_data[GC_names[model_name]] = pd.read_csv(f"{GC_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)

#Get Pop III analysis
pop3_ana, pop3_param = get_project_params('PopIII')
pop3_data = {}
for model_key, model_params in pop3_param["astro_model_list"].items():
        model_name = model_params["name"]
        pop3_data[model_name] = pd.read_csv(f"{pop3_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)

#Get AGN analysis
AGN_ana, AGN_param = get_project_params('AGNs')
AGN_data = {}
for model_key, model_params in pop3_param["astro_model_list"].items():
        model_name = model_params["name"]
        AGN_data[model_name] = pd.read_csv(f"{AGN_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)


#Get other channels
OtherC_ana, OtherC_param = get_project_params('OtherChannels')
OtherC = {}
for model_key, model_params in pop3_param["astro_model_list"].items():
        model_name = model_params["name"]
        AGN_data[model_name] = pd.read_csv(f"{AGN_ana['omega_path']}/{model_name}.dat", sep="\t", index_col=None)

