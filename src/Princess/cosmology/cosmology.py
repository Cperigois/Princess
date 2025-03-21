print(f"Loading {__name__}")
import math
import os
import numpy as np
import pandas as pd
import json
import importlib.resources

# Import preset cosmologies
with importlib.resources.open_text("Princess.cosmology", "presets.json") as f:
    preset_cosmologies = json.load(f)

class Cosmology :

    def __init__(self, name: str, Omega_m: float = 0.3, Omega_Lambda: float = 0.7, H0: float = 67):
        """Initializes an AstroModel instance and loads or creates necessary data.

        :param name: (str) Name for the cosmological model.
        :param Omega_m: (float) Refers to Omega_m for a flat LambdaCDM model.
        :param Omega_Lambda: (float) Refers to Omega_Lambda for a flat LambdaCDM model.
        :param H_0: (float): Planck constant in km s−1 Mpc−1
        """


        self.name = name
        self.Omega_m = Omega_m
        self.Omega_L = Omega_Lambda
        self.H0 = H0

    def save(self, reference_paper, model_values, table_values):
        """Save the cosmological model in presets.json."""
        new_model = {
            "H0": self.H0,
            "Omega_m": self.Omega_m,
            "Omega_Lambda": self.Omega_L,
            "reference": reference_paper,
            "model": model_values,
            "table": table_values
        }

        preset_cosmologies[self.name] = new_model

        with open(preset_file, 'w') as f:
            json.dump(preset_cosmologies, f, indent=4)

    @classmethod
    def load(cls, model_name: str):
        """Load an existing model from presets.json or create a new one with default values."""
        if model_name in preset_cosmologies:
            data = preset_cosmologies[model_name]
            return cls(model_name, data["Omega_m"], data["Omega_Lambda"], data["H0"])
        else:
            return cls(model_name)

    def luminosity_distance_table(self, z_max=30, steps=50000):
        """Generate a table of luminosity distance as a function of redshift.

        If a table for this cosmology already exists, it is not recomputed.
        """
        table_filename = f"AuxiliaryFiles/z_dl_table_{self.name}.csv"

        if os.path.exists(table_filename):
            print(f"⚠️ Table {table_filename} already exists. No need to recompute.")
            return pd.read_csv(table_filename)

        print(f"🛠️ Generating redshift-luminosity distance table for {self.name}...")

        z_values = np.linspace(0, z_max, steps)
        d_L_values = [self.luminosity_distance(z) for z in z_values]

        df = pd.DataFrame({"Redshift": z_values, "Luminosity Distance (Mpc)": d_L_values})
        df.to_csv(table_filename, index=False)

        print(f"✅ Table saved as {table_filename}.")
        return df

    def luminosity_distance(self, z):
        """Compute the luminosity distance for a given redshift using numerical integration."""
        c = 299792.458  # Speed of light in km/s
        integral = lambda zp: 1.0 / np.sqrt(self.Omega_m * (1 + zp) ** 3 + self.Omega_L)
        dz = np.linspace(0, z, 1000)
        integral_value = np.trapz([integral(zp) for zp in dz], dz)
        d_C = (c / self.H0) * integral_value  # Comoving distance
        return (1 + z) * d_C  # Luminosity distance

    def object_age(self, z):
        """Compute the age of an object at a given redshift."""
        H0_s = self.H0 / (3.0857e19)  # Convert H0 to s^-1
        integral = lambda zp: 1.0 / ((1 + zp) * np.sqrt(self.Omega_m * (1 + zp) ** 3 + self.Omega_L))
        dz = np.linspace(0, z, 1000)
        integral_value = np.trapz([integral(zp) for zp in dz], dz)
        age = integral_value / H0_s / (3.154e7 * 1e9)  # Convert to Gyr
        return age

    def comoving_volume(self, z):
        """Compute the comoving volume enclosed within a given redshift."""
        D_C = self.comoving_distance(z)  # Comoving distance in Mpc
        V_c = (4 / 3) * math.pi * (D_C ** 3)  # Comoving volume in Mpc^3
        return V_c

    def info(self):
        """Display and return the cosmology model information."""
        info_dict = {
            "Name": self.name,
            "H0": self.H0,
            "Omega_m": self.Omega_m,
            "Omega_Lambda": self.Omega_L,
            "Reference": preset_cosmologies.get(self.name, {}).get("reference", "N/A"),
            "Table": preset_cosmologies.get(self.name, {}).get("table", "N/A")
        }

        print("\n=== Cosmology Model Information ===")
        for key, value in info_dict.items():
            print(f"{key}: {value}")

        return info_dict
    def compute_z(self, dl_array):
        "Computes the redshift from a luminosity distance array."
        self.luminosity_distance_table(self, zmax = 30, steps = 50000)
        table_filename = f"AuxiliaryFiles/z_dl_table_{self.name}.csv"
        df = pd.read_csv(table_filename)
        interpolation = InterpolatedUnivariateSpline(df['Luminosity Distance (Mpc)'], df['Redshift'])
        return interpolation(dl_array)

    def compute_dl(self, z_array):
        "Computes the luminosity distance from a redshift array."
        d_L_values = [self.luminosity_distance(z) for z in z_array]
        return d_L_values