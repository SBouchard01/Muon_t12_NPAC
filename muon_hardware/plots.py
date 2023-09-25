import numpy as np
import matplotlib.pyplot as plt
from .analysis import muon_lifetime


def x_cut_plot(Object:muon_lifetime, 
               x_cut_list=np.linspace(2, 5, 300),
               bins:int = None):
    """
    Plot the chi2 as a function of the x-cut value.

    Parameters
    ----------
    Object : muon_lifetime
        muon_lifetime object, with the trigger already computed.
    x_cut_list : Array, optional
        List of x-cut values to test in µs. Defaults to np.linspace(0, 2, 500)
    bins : int, optional
        Number of bins to use in the histogram. If None, the bins are automatically 
        set as \sqrt(N) , where N is the number of events. Defaults to None.
        
    Returns
    -------
    fig, ax : matplotlib figure and axis
        Figure and axis of the plot.
    """
    
    # Get the TAC histogram (The bins are automatically computed if none are provided)
    if bins is None:
        bins = int(np.ceil(np.sqrt(len(Object.values_unbinned))))
        
    Object.rebin_hist(bins = bins)

    # Get the chi2 values
    chi2_list = []
    for i in x_cut_list:
        Object.exp_fit(x_cut = i, method='iminuit') 
        chi2 = Object.chi2
        chi2_list.append(chi2)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(x_cut_list, chi2_list, label='$\chi_2$')
    ax.axhline(1, ls='--', color='r', label='Expected $\chi_2$')
    ax.set_xlabel('x-cut value (µs)')
    ax.set_ylabel('$\chi^2$')
    ax.set_title(f'$\chi^2$ as a function of the x-cut value \n(bins = {bins}))')
    ax.legend()
    fig.set_facecolor('white')
    
    return fig, ax



def bin_plot(Object:muon_lifetime,
             bins_list=np.arange(50, 500, 1),
             x_cut:float = 3):
    """
    Plot the chi2 as a function of the bin number.

    Parameters
    ----------
    Object : muon_lifetime
        muon_lifetime object, with the trigger already computed.
    bins_list : Array, optional
        List of bin numbers to test. Defaults to np.arange(50, 500, 1).
    x_cut : float, optional
        x-cut value to use in the fit (in µs). Defaults to 0.64.

    Returns
    -------
    fig, ax : matplotlib figure and axis
        Figure and axis of the plot.
    """

    # Get the chi2 values
    chi2_list = []
    for i in bins_list:
        Object.rebin_hist(bins = i)
        
        Object.exp_fit(x_cut=x_cut, method='iminuit') 
        chi2 = Object.chi2
        chi2_list.append(chi2)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(bins_list, chi2_list, label='$\chi_2$')
    ax.axhline(1, ls='--', c='r', label='Expected $\chi_2$')
    ax.set_xlabel('bin number')
    ax.set_ylabel('$\chi^2$')
    ax.set_title(f'$\chi^2$ as a function of the bin number \n(x-cut = {x_cut} µs)')
    ax.legend()
    fig.set_facecolor('white')
    
    return fig, ax
    