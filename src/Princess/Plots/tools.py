import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json



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


def get_project_params(name) :
    project_params = {'name':name}
    project_params['astro_models_path'] = f'../Run/{name}'
    project_params['catalogs_path'] = f"{project_params['astro_models_path']}/{name}/Catalogs"
    project_params['omega_path'] = f"{project_params['astro_models_path']}/Results/Omega"
    project_params['analysis_path'] = f"{project_params['astro_models_path']}/Results/Analysis"
    params_json = json.load(open(f'../Run/{name}/Params.json', 'r'))
    return project_params, params_json


def horizon_network(network, det_list, zmax=200):
    """
    Compute the horizons on the parameter space (Mtot, z) of a network object up to zmax.
    Save the results in a CSV file.

    Parameters
    ----------
    network : Network
        Network to be studied.
    det_list : list
        List of detector objects to be used for the SNR computation.
    zmax : float, optional
        Maximum redshift to explore (default is 200).

    Returns
    -------
    pd.DataFrame
        DataFrame containing total mass and corresponding redshift horizons for different mass ratios.
    """
    from Princess.astrotools.utils import mt_q_to_m1_m2
    from astropy.cosmology import Planck15
    import Princess.gwtools.snr as gwt

    # Define the parameter space
    total_mass = np.logspace(0, 4, 50)
    #mass_ratio = np.array([0.5, 0.7, 0.8, 0.9, 1])
    mass_ratio = np.array([1.])
    output = pd.DataFrame({'Mt': total_mass})

    # Loop over each mass ratio
    for q in mass_ratio:
        horizon = np.array([])  # Array to store horizon values for current mass ratio

        # Convert total mass and mass ratio to component masses
        M1, M2 = mt_q_to_m1_m2(total_mass, q * np.ones(len(total_mass)))

        # Loop over each mass pair
        for m1, m2, mt in zip(M1, M2, total_mass):
            deltaz = 10
            z = 0.0001
            snr = 9  # Initialize with SNR > 8 to enter the loop

            print(f'Calculating horizon for mt = {mt}/{max(total_mass)}, q = {q}')

            # Nested loops for redshift increment and SNR check
            while snr > 8 and z < zmax and deltaz >=0.00001:
                    # Compute luminosity distance for the given redshift
                dl = Planck15.luminosity_distance(z).value

                    # Compute SNR for the current event
                snr_net = gwt.SNR_single(
                    {'m1': m1, 'm2': m2, 'z': z, 'Dl': dl, 'chi1': 0, 'chi2': 0},
                    network=network,
                    det_list=det_list,
                    waveform="IMRPhenomD",
                    freq=np.linspace(1, 2000, 2000)
                )

                # Check if the event is out of frequency band
                #print(snr_net.describe())
                if isinstance(snr_net, float):
                    print(f'Event out of frequency band for z = {z}')
                    snr = 9
                else:
                    snr = snr_net[f'{network.name}_optimal'].values[0]  # Extract scalar SNR value
                    #print(f'Current SNR = {snr} at z = {z}')
                if snr < 8 :
                    z -= deltaz  # Step back
                    deltaz /= 10.0  # Reduce step size
                    snr = 9  # Reset SNR to re-enter the loop
                # Increment redshift
                z += deltaz
                if z <0.000001 :
                    z = 0.0001
                print(f'snr : {snr}, z : {z} , deltaz : {deltaz}')

            # Store the horizon value for the current mass pair
            horizon = np.append(horizon, z)
            print(z)

        # Add the horizon results for the current mass ratio to the output DataFrame
        output[f'z_q_{q}'] = horizon

    # Save the results to a CSV file
    output.to_csv(f'AuxiliaryFiles/Horizons/horizon_{network.name}.csv', sep='\t', index=False)

    return output


def horizon_network_old(network, det_list, zmax = 200) :
    """
    Compute the horizons on the parameter space (Mtot, z) of a network object up to zmax.
    Save the results in a csv file.

    Parameters
    ----------
    network : class network
        Network to be studied.

    Returns
    -------
    dataframe
        Dataframe of two columns, Mt and z
    """

    from Princess.astrotools.utils import mt_q_to_m1_m2
    from astropy.cosmology import Planck15
    import Princess.gwtools.snr as gwt

    total_mass = np.logspace(0, 4, 100)
    mass_ratio = np.array([0.5, 0.7, 0.8, 0.9, 1])
    output = pd.DataFrame({'Mt' : total_mass})

    for q in mass_ratio :
        horizon = np.array([])
        M1,M2 = mt_q_to_m1_m2(total_mass, q*np.ones(len(total_mass)))
        for m1, m2, mt in zip(M1, M2, total_mass):
            deltaz = 10
            z = 0.00001
            snr = 9
            print(f'm1 = {m1}, m2 = {m2}, z = {z}')
            while deltaz>=0.0001 and z<zmax :
                #print(f'apres delta z check snr : {snr} z : {z}')
                while snr > 8 :
                    dl = Planck15.luminosity_distance(z).value
                    snr_net = gwt.SNR_single({'m1': m1, 'm2': m2, 'z': z, 'Dl': dl, 'chi1' : 0, 'chi2' : 0},
                                             network=network, det_list= det_list, waveform = "IMRPhenomD",
                                             freq = np.linspace(1,2000,2000))
                    #print(snr_net.describe())
                    if isinstance(snr_net, float):
                        print(f'Event with m1 = {m1}, m2 = {m2}, z = {z} shows a limit frequency flim = {snr_net}.'
                              f'Therefore it is out of the band for {network.name} network')
                        snr = 9
                    else :
                        snr = snr_net[f'{network.name}_optimal'].values
                        #print(snr)
                    z = z + deltaz
                    #print(f'apres snr <8 snr : {snr} z : {z}')
                else :
                    z= z-deltaz
                    deltaz = deltaz/10.
                    snr = 9
                    #print(f'dans le else snr : {snr} z : {z}')
            horizon = np.append(horizon, z)
        output[f'z_q_{q}'] = horizon
    output.to_csv(f'horizon_{network.name}.csv', sep = '\t', index = None)










