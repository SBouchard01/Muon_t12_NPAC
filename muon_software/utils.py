import numpy as np

def rebin(x, height, factor):
    """
    Rebins an histogram in n_bins

    Parameters
    ----------
    x : Array-like
        The positions of the bins. 
    height : Array-like
        The values of the histogram to rebin, of dimension (1, N)
    factor : int
        The factor of division (i.e. how many bins are merged into one)
    
    Returns
    -------
    hist : Array-like
        The rebinned histogram, of dimension (1, N') with N' = ceil(N/factor)
    """
    
    # Get the number of bins
    n_bins = len(x)
    
    # Get the number of bins in the rebinned histogram
    n_bins_rebinned = int(np.ceil(n_bins/factor))
    
    # Create the rebinned histogram
    hist = np.zeros((n_bins_rebinned, 2))
    
    # Fill the rebinned histogram
    for i in range(n_bins_rebinned):
        hist[i, 0] = np.mean(x[i*factor:(i+1)*factor])
        hist[i, 1] = np.sum(height[i*factor:(i+1)*factor])
        
    return hist


def chi2(y, y_fit, sigma):
    return np.sum(((y-y_fit)/sigma)**2)

def chi2_reduced(y, y_fit, sigma, n_params):
    return chi2(y, y_fit, sigma)/(len(y)-n_params)

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

def exp(x, amplitude, tau):
    return amplitude * np.exp(-x/tau)


def linear(x, a, b):
        return a*x + b
    