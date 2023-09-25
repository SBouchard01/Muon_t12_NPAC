import numpy as np


def rebin(x, height, bins):
    """
    Rebins an histogram in n_bins

    Parameters
    ----------
    x : Array-like
        The positions of the bins. 
    height : Array-like
        The values of the histogram to rebin, of dimension (1, N)
    bins : int
        The number of bins in the histogram
    
    Returns
    -------
    hist : Array-like
        The rebinned histogram
    """
    
    # Get the number of bins
    n_bins = len(x)
    factor = n_bins//bins 
    
    # Create the rebinned histogram
    hist = np.zeros((bins, 2))
    
    # Fill the rebinned histogram
    for i in range(bins):
        hist[i, 0] = np.mean(x[i*factor:(i+1)*factor])
        hist[i, 1] = np.sum(height[i*factor:(i+1)*factor])
        
    return hist


def exp(x, amplitude, tau):
    return amplitude * np.exp(-x/tau)


def linear(x, a, b):
        return a*x + b
    
def chi2(y, y_fit, sigma):
    return np.sum(((y-y_fit)/sigma)**2)

def chi2_reduced(y, y_fit, sigma, n_params):
    return chi2(y, y_fit, sigma)/(len(y)-n_params)
    