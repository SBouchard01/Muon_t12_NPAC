# Muon_t12_NPAC
The repository for all the code and plots related to the September 2023 NPAC TL on the muon lifetime.

It was created by Simon BOUCHARD and Pierre MASSON.

## Contents of the repository

### The *spectroscopy* folder
This folder contains the preliminary TL work on gamma spectroscopy. The `calibration.py` file contains some useful functions to calibrate the gamma spectrum. The `spectrum` folder contains the data used in the `Gamma_spectro.ipynb` notebook.

### `muon_software`
This folder contains the code for the digital muon lifetime analysis. It is designed as a self-contained class in the `analysis.py` file, but uses functions from the other files of the folder.

### `muon_hardware`
This folder contains the code for the hardware muon lifetime analysis. Like the `muon_software`, it is designed as a self-contained class in the `analysis.py` file, but uses functions from the other files of the folder.

### The notebooks
Several notebooks are available to illustrate the use of the code. They are located in the main folder of the repository.
They were used to analyze our data (unavailable here due to size issues), but can be used as a template for future analysis.
- `muon_lifetime.ipynb` : this notebook illustrates the use of the `muon_software` class to analyse the digital data.
- `muon_lifetime_electro.ipynb` : this notebook illustrates the use of the `muon_hardware` class to analyse the hardware data.
- `amplifier_behavior.ipynb` : This notebook illustrates the behavior of the different amplifiers used in the hardware analysis.
- `final_plot.ipynb` : This notebook generates the final results of our article, using the best fitting parameters found in the previous notebooks.

### The *images* folder
The relevant plots of our study are stored in this folder.

## Code requirements and usage

This code was written in python and uses the following packages : 
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `iminuit`

### Example of usage
See the [notebooks](#the-notebooks) section


# Citation
If you use this code please cite : 
```
@article{masson2021muon,
  title={Measure of muon lifetime : A comparison between digital and electronic approach},
  author={Masson, Pierre and Bouchard, Simon},
  journal={NPAC TL},
  year={2023}
}
```
