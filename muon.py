import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from calibration import linear, r_squared


def calibrate_TAC(file:str = "data/TAC_calibration_spectrum-15-09-2023.txt",
                  times = np.arange(0, 5, 0.25)+0.150,
                  ADC_range:tuple = (0, 0.95e5),
                  peak_width:float = 2.3,
                  peak_height:float = 60,
                  plot_peaks:bool = False,
                  plot_fit:bool = False):
    """
    Calibrate the TAC by finding the peaks in the spectrum and fitting them to the provided times.
    Note : The default parameters have been optimized for the spectrum "data/TAC_calibration_spectrum-15-09-2023.txt".

    Parameters
    ----------
    file : str, optional
        The file in which the spectrum is stored, as peaks for the different times. Defaults to "data/TAC_calibration_spectrum-15-09-2023.txt".
    times : _type_, optional
        The times **in µs** for which the peaks were detected. Defaults to np.arange(0.25, 5, 0.25)+0.150.
    ADC_range : tuple, optional
        The range of ADC values to consider to avoid fitting on 0 values. Defaults to (0, 0.95e5).
    peak_width : float, optional
        The width of the peaks to identify, used in [find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). Defaults to 2.3.
    peak_height : float, optional
        The minimal height for the peaks to identify, used in [find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). Defaults to 60.

    Raises
    ------
    ValueError
        If the number of peaks found is different from the number of times.
    
    Returns
    -------
    popt : array-like
        The parameters of the linear fit.
    (fig, ax) : tuple
        The figure and axis of the plot. Only returned if there is at least one plot.
    """

    # Open the file and get the data
    content = np.loadtxt(file, skiprows=2, delimiter=";")
    
    # Apply the mask on the ADC range
    mask = np.logical_and(content[:,0] > ADC_range[0], content[:,0] < ADC_range[1])
    ADC = content[:,0][mask] # X value of the peaks (ADC)
    data = content[:,1][mask] # Y value of the peaks (counts)
    
    # Find the peaks in the data (ADC)
    peaks_id, _ = find_peaks(data, height=peak_height, width=peak_width)
    ADC_peaks = ADC[peaks_id] # X value of the peaks (ADC)

    # Check that the number of peaks is the same as the number of times
    if len(ADC_peaks) != len(times):
        raise ValueError(f"The number of peaks found ({len(ADC_peaks)}) is different from the number of times ({len(times)}).")

    # Fit the times to the ADC with a linear function
    popt, pcov = curve_fit(linear, ADC_peaks, times)
    
    r2 = r_squared(times, linear(ADC_peaks, *popt))
    
    # Plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.set_facecolor("white") # Set the background color to white bc jupyter is annoying and makes a transparent background
    
    i = 0
    if plot_peaks:
        ax[i].plot(ADC, data, label="Data")
        ax[i].plot(ADC_peaks, data[peaks_id], "x", label="Peaks")
        ax[i].legend()
        ax[i].set_ylabel("Counts")
        i += 1
        
    if plot_fit:
        ax[i].plot(ADC_peaks, times, "x", label="Data")
        ax[i].plot(ADC_peaks, linear(ADC_peaks, *popt), label="Fit")
        ax[i].legend(loc="lower right")
        ax[i].text(0.05, 0.95, f"y = {popt[0]:.2e}x + {popt[1]:.2f}\nR² = {r2:.4f}", transform=ax[i].transAxes, fontsize=12, verticalalignment='top')
        ax[i].set_xlabel("ADC")
        ax[i].set_ylabel("Time (µs)")
        i += 1
        
    # Remove the last plot if there is no data
    if not plot_peaks or not plot_fit:
        fig.delaxes(ax[i])
    
    if not plot_fit and not plot_peaks:
        plt.close()
        return popt
    
    return popt, (fig, ax)

    
