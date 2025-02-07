_**Princess** is program developed by Carole Perigois ([carolperigois@outlook.com](mailto:carolperigois@outlook.com)), with the support of from the European Research Council for the ERC Consolidator grant DEMOBLACK, under contract no. 770017._
<p align="center" width="100%">
    <img width="33%" src="docs/Princess_logo.png">
</p>

# PRINCESS: Prediction of Compact Binaries Observations with Gravitational Waves

## Overview
PRINCESS is a computational tool designed to predict gravitational wave observations from compact binary coalescences (CBCs) in current and future detector networks. The tool combines predictions of individual gravitational wave events and the astrophysical gravitational wave background, leveraging user-provided CBC catalogs.

## Installation and Requirements
PRINCESS is publicly available on GitHub and GitLab and requires the following dependencies:

- Python 3.7 or higher
- PyCBC v1.18 (more recent versions are not compatible)
- Pandas
- NumPy
- SciPy

To verify installation, run the `test.py` script, which checks dependencies and performs a test run using a mini-catalog.

## Repository Structure
```
PRINCESS/
│-- AuxiliaryFiles/        # Contains data related to detectors and past detections
│   ├── LVK data/         # Data from the LIGO-Virgo-KAGRA (LVK) collaboration
│   ├── ORFs/             # Overlap reduction functions for detector networks
│   ├── PICs/             # Power integrated curves
│   ├── PSDs/             # Noise power spectral densities for detectors
│-- Run/
│   ├── getting_started.py  # Initial parameter configuration
│   ├── advanced_params.py  # Advanced computation settings
│   ├── run.py             # Main script executing the full analysis
│-- astrotools/            # Functions related to astrophysical models
│-- gwtools/               # Functions for gravitational wave computations
│-- stochastic/            # Functions for stochastic background computations
│-- README.md
```

## Main Code Files

### `run.py`
This is the primary script for executing a full analysis. It should not be modified. The workflow includes:
- **Initialization**: Reads and structures input parameters.
- **Individual detections**: Computes signal-to-noise ratios (SNR) for each binary event.
- **Background computation**: Computes the gravitational wave background spectrum.
- **Data cleaning**: Removes unnecessary temporary files at the end of execution.

### `Run/getting_started.py`
This file is the entry point for setting up a new project. Users must:
- Define the project folder name.
- Specify input astrophysical models.
- Configure detectors and detector networks.
- Set computation parameters.

### `Run/advanced_params.py`
This file contains additional customizable parameters such as:
- Frequency bounds for different detectors.
- Input catalog formatting options.
- Advanced options for background spectrum calculations.

## Usage
To run an analysis, follow these steps:
1. Edit `getting_started.py` with the desired astrophysical models and detector configurations.
2. Run the main script:
   ```bash
   python run.py
   ```
3. The results will be stored in a folder within `Run/`, named after the project.

## Output Structure
After execution, results are stored in `Run/<project_name>/` with the following directories:
- **Astro_Models/**: Contains transformed input catalogs.
- **Results/Analysis/**: Includes SNR values and detection statistics.
- **Results/Omega/**: Contains the gravitational wave background spectrum.
- **Params.json**: Stores all computation parameters for reference.

## Common Issues
- If `Params.json` is not updating, manually delete `Run/Params.json` before re-running.
- Ensure dependencies are installed correctly by running `test.py`.

## Contact
For issues or feature requests, contact the corresponding author or check the GitHub/GitLab repositories.

---
PRINCESS is an evolving tool, and future updates may enhance its capabilities, including new astrophysical models and detector configurations.

