import numpy as np
import pandas as pd

def sample_distribution(distribution, size, **kwargs):
    """Generate samples from a given distribution.

    Parameters:
    - distribution (str): Type of distribution ('uniform' or 'gaussian').
    - size (int): Number of samples.
    - kwargs: Parameters for the chosen distribution.

    Returns:
    - np.array: Generated samples.
    """
    if distribution == "uniform":
        return np.random.uniform(kwargs["low"], kwargs["high"], size)
    elif distribution == "gaussian":
        return np.random.normal(kwargs["mean"], kwargs["std"], size)
    else:
        raise ValueError("Invalid distribution type. Choose 'uniform' or 'gaussian'.")


def generate_population(num_sources, mass_method="m1_m2", mass_distribution="uniform", z_distribution="uniform",
                        mass_params=None, z_params=None):
    """Generate a binary system population with given distributions.

    Parameters:
    - num_sources (int): Number of binary systems to generate.
    - mass_method (str): 'm1_m2' for primary and secondary mass, 'chirp_q' for chirp mass and mass ratio.
    - mass_distribution (str): 'uniform' or 'gaussian' for mass parameters.
    - z_distribution (str): 'uniform' or 'gaussian' for redshift.
    - mass_params (dict): Parameters for mass distribution.
    - z_params (dict): Parameters for redshift distribution.

    Returns:
    - pd.DataFrame: Generated population.
    """
    if mass_params is None or z_params is None:
        raise ValueError("mass_params and z_params must be provided.")

    if mass_method == "m1_m2":
        m1 = sample_distribution(mass_distribution, num_sources, **mass_params)
        m2 = sample_distribution(mass_distribution, num_sources, **mass_params)
        chirp_mass = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)
        q = m2 / m1
    elif mass_method == "chirp_q":
        chirp_mass = sample_distribution(mass_distribution, num_sources, **mass_params)
        q = sample_distribution(mass_distribution, num_sources, **mass_params)
        m1 = chirp_mass * (1 + q) ** (1/5) / q ** (3/5)
        m2 = q * m1
    else:
        raise ValueError("Invalid mass method. Choose 'm1_m2' or 'chirp_q'.")

    z = sample_distribution(z_distribution, num_sources, **z_params)

    df = pd.DataFrame({
        "m1": m1,
        "m2": m2,
        "Mc": chirp_mass,
        "q": q,
        "z": z
    })

    return df

def save_population(population_df, filename):
    """Save the generated population to a .dat file using Pandas.

    Parameters:
    - population_df (pd.DataFrame): Population data.
    - filename (str): Output file name.
    """
    population_df.to_csv(filename, sep='\t', index=False)
    print(f"Catalogue saved as {filename}")