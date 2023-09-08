import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

#%% Fitted functions
def linear(x, a, b):
        return a*x + b
    
def gauss(x, sigma, mu, A):
    """
    Gaussian function
    
    Parameters
    ----------
    x : array-like
        x values
    
    sigma : float
        Standard deviation
        
    mu : float
        Mean
        
    A : float
        Amplitude
       
    Returns
    -------
    y : array-like
        y values 
    """
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) 


#%% Utils
def r_squared(y, y_fit):
    """
    Compute the R2 value of a fit.
    
    Parameters
    ----------
    y : array-like
        y values
        
    y_fit : array-like
        y values of the fit
        
    Returns
    -------
    R2 : float
        R2 value of the fit
    """
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y - y_fit)**2)
    
    return 1 - ss_res/ss_tot

#%% Find gamma peak(s)

def get_gamma_peaks(data,
                    peak_nb:int=1,
                    peak_height:float=1000,
                    peak_width:float=10,):
    """
    Computes the gamma peaks of the spectrum by fitting a gaussian over the last spectrum peaks. 
    The user needs to give the number of expected gamma peaks.

    Parameters
    ----------
    data : array-like
        The data of the spectrum (x-axis assumed to be in keV)
        
    peak_nb : int, optional
        Number of gamma peaks. Defaults to 1
        
    peak_height : float, optional
        The minimal height for the peaks to identify in the reference spectrum,
        , used in [find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). 
        Defaults to 1000.
        
    peak_width : float, optional
        The width of the peaks to identify, 
        used in [find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). 
        Defaults to 10.

    Returns
    -------
    dict
        Dictionnary of solutions. For each peak, a `gamma_peak_i` key is created, 
        containing `sigma` (the std of the fitted gaussian), 
        `mu` (the mean of the gaussian, or the energy of the peak),
        `A` (the amplitude of the gaussian, can be used to plot it again),
        `R2` (the r squared coefficient)
        
    Note
    ----
    The R2 coefficient might not be really reliable, as it is computed on an arbitrary range of the curve that might include a non-gaussian part.
    """
    
    # First, smooth the signal
    smoothed_data = savgol_filter(data[:, 1], 51, 3)

    # Get the local maximums
    peaks_id, _ = find_peaks(smoothed_data, height=peak_height, width=peak_width)
    peaks = data[peaks_id,0]
    
    # Get the photo-emission peak
    pe_peaks = peaks[-peak_nb:]
    
    sol = {}
    
    for i, pe_peak in enumerate(pe_peaks, start=1):
        pe_value = data[data[:, 0] == pe_peak, 1][0]

        # Fit the gaussian around the peak
        popt, pcov = curve_fit(gauss, data[:, 0], data[:, 1], p0=[1, pe_peak, pe_value], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        
        sigma = popt[0]
        mu = popt[1]
        A = popt[2]
        
        # Compute the r squared
        mask = np.logical_and(data[:, 0] > mu - 3 * sigma, data[:, 0] < mu + 3 * sigma)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, sum(mask))
        y = data[:, 1][mask]
        r_2 = r_squared(y, gauss(x, sigma, mu, A))

        sol[f'gamma_peak_{i}'] = {
            'sigma' : sigma,
            'mu' : mu,
            'A' : A,
            'R2' :  r_2
        }
    
    return sol

def compton_peak(E_gamma):
    """
    Compute the energy of the Compton peak.
    
    Parameters
    ----------
    E_gamma : float
        Energy of the gamma peak in keV
        
    Returns
    -------
    E_peak : float
        Energy of the Compton peak in keV
    """
    me = 511 # keV
    gamma = E_gamma/(me)
    return E_gamma/(1 + 2*gamma)

def compton_edge(E_gamma):
    """
    Compute the energy of the Compton edge.
    
    Parameters
    ----------
    E_gamma : float
        Energy of the gamma peak in keV
        
    Returns
    -------
    E_edge : float
        Energy of the Compton edge in keV
    """
    me = 511 # keV
    gamma = E_gamma/(me)
    return E_gamma * 2*gamma/(1 + 2*gamma)

def resolution(mu, sigma):
    """
    Computes the resolution of a peak

    Parameters
    ----------
    mu : float
        The mean of the gaussian fitted on the peak.
    sigma : float
        The standard deviation of the gaussian fitted on the peak.

    Returns
    -------
    resolution : float
        The resolution of the peak, defined as 
    """
    gaussian_width = 2.355*sigma
    return gaussian_width/mu

#%% Calibration
def calibrate(file, 
              reference_file, 
              peak_height:float=10000,
              peak_width:float=10,
              run_tests:bool=False):
    """
    Calibrate a spectrum using a reference spectrum. Returns the x-axis of the calibrated spectrum in keV.
    
    Parameters
    ----------
    file : str
        Path to the spectrum to calibrate.
        
    reference_file : str
        Path to a Cs137 spectrum, taken with the same setup as the spectrum to calibrate.
        
    peak_height : float, optional
        The minimal height for the peaks to identify in the reference spectrum,
        , used in [find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). 
        Defaults to 1000.
        
    peak_width : float, optional
        The width of the peaks to identify, 
        used in [find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). 
        Defaults to 10.
    
    return_tests : bool, optional
        TODO : If True, prints the tests performed during the calibration. Defaults to False. 
        
    Returns
    -------
    data : array-like
        The calibrated data, with the x-axis in keV
    
    """
    
    # Defaut values for the Cs137 peaks
    x_ray_peak = 31 # keV
    compton_peak = 184 # keV
    gamma_peak = 661 # keV
    
    th_peaks = [x_ray_peak, compton_peak, gamma_peak]
    
    # Load the reference file
    data = np.loadtxt(reference_file, skiprows=2, delimiter=";") 
    
    # Cut the data at x=1000 (to remove a non-physical peak)
    data = data[data[:, 0] < 1e6]
    
    # First, smooth the signal
    smoothed_data = savgol_filter(data[:, 1], 51, 3)
    
    # Get the local maximums
    peaks_id, _ = find_peaks(smoothed_data, height=peak_height, width=peak_width)
    peaks = data[peaks_id,0]
    
    if len(th_peaks) != len(peaks):
        if len(th_peaks) > len(peaks):
            error = f"The number of peaks found ({len(peaks)}) is lower from the number of peaks expected. Try a lower width."
        else:
            error = f"The number of peaks found ({len(peaks)}) is bigger from the number of peaks expected. Try a higher width."
        raise ValueError(error)
    
    # Do a linear fit on the peaks
    popt, pcov = curve_fit(linear, peaks, th_peaks)
    
    # Convert the data
    data[:, 0] = linear(data[:, 0], *popt)
    peaks = linear(peaks, *popt)
    
    # Convert the spectrum to calibrate
    data_to_calibrate = np.loadtxt(file, skiprows=2, delimiter=";")
    data_to_calibrate = data_to_calibrate[data_to_calibrate[:, 0] < 1e6]
    data_to_calibrate[:, 0] = linear(data_to_calibrate[:, 0], *popt)
    
    return np.c_[data_to_calibrate[:, 0], data_to_calibrate[:, 1]]



#%% Full plot (for fun)

def full_plot(data,
              peak_nb:int=1,
              peak_height:float=1000,
              peak_width:float=10):
    
    fig, ax = plt.subplots()
    
    # Unfiltered data : 
    ax.plot(data[:, 0], data[:, 1], alpha=0.5, label="Unfiltered data")

    # Smoothed data
    smoothed_data = savgol_filter(data[:, 1], 51, 3)
    ax.plot(data[:, 0], smoothed_data, label="Smoothed data")

    # Peaks (found)
    peaks_id, _ = find_peaks(smoothed_data, height=peak_height, width=peak_width)
    peaks = data[peaks_id,0]
        
    gamma_peaks = peaks[-peak_nb:]
    for i, gamma_peak in enumerate(gamma_peaks):
        # Gamma peaks
        label = f"Gamma peak ({i+1}) at {gamma_peak:.0f} keV" if len(gamma_peaks) > 1 else f"Gamma peak at {gamma_peak:.0f} keV"
        ax.axvline(gamma_peak, color="red", alpha=0.8, label=label)
        
        # Compton range
        label = f"Compton range ({i+1}) to {compton_edge(gamma_peak):.0f} keV" if len(gamma_peaks) > 1 else f"Compton range to {compton_edge(gamma_peak):.0f} keV"
        ax.axvspan(0, compton_edge(gamma_peak), ymin=-1, ymax=0.4+i/10, alpha=0.2, color="gray", label=label, capstyle="round")

        # Compton peak
        label = f"Compton peak ({i+1}) at {compton_peak(gamma_peak):.0f} keV" if len(gamma_peaks) > 1 else f"Compton peak at {compton_peak(gamma_peak):.0f} keV"
        ax.axvline(compton_peak(gamma_peak), color="gray", alpha=0.3, label=label)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Count")
    
    return fig, ax