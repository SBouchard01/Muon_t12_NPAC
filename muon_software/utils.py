import numpy as np

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
    