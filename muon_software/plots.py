import numpy as np
from muon_software.analysis import muon_lifetime
import matplotlib.pyplot as plt



def x_cut_plot(Object:muon_lifetime, 
               bins:int = None,
               x_cut_list=np.linspace(0, 2, 500),
               afterpulse_value:float = 0.64):
    """
    Plot the chi2 as a function of the x-cut value.

    Parameters
    ----------
    Object : muon_lifetime
        muon_lifetime object, with the trigger already computed.
    bins : int, optional
        Number of bins in the TAC histogram. If None, the bins are automatically 
        set as \sqrt(N) , where N is the number of events. Defaults to None.
    x_cut_list : Array, optional
        List of x-cut values to test in µs. Defaults to np.linspace(0, 2, 500)
    afterpulse_value : float, optional
        Estimated value of the afterpulse peak (in µs). Defaults to 0.64.

    Returns
    -------
    fig, ax : matplotlib figure and axis
        Figure and axis of the plot.
    """
    
    # Get the TAC histogram (The bins are automatically computed if none are provided)
    if bins is None:
        bin_nb = int(np.ceil(np.sqrt(len(Object.values))))
    else:
        bin_nb = bins
        
    Object.TAC(bins=bin_nb)

    # Get the chi2 values
    chi2_list = []
    for i in x_cut_list:
        Object.exp_fit(x_cut = i, method='iminuit') 
        chi2 = Object.chi2
        chi2_list.append(chi2)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(x_cut_list, chi2_list, label='$\chi_2$')
    ax.axvline(afterpulse_value, ls='--', color='r', label='Afterpulse peak (estimated)')
    ax.set_xlabel('x-cut value (µs)')
    ax.set_ylabel('$\chi^2$')
    ax.set_title(f'$\chi^2$ as a function of the x-cut value \n(bins = {bin_nb})')
    ax.legend()
    fig.set_facecolor('white')
    
    return fig, ax


def bin_plot(Object:muon_lifetime,
             bins_list=np.arange(50, 500, 1),
             x_cut:float = 0.64):
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
        Object.TAC(bins=i)
        
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
    
    
def energy_plot(Object:muon_lifetime,
                bins=100,
                afterpulse_value:float = 1_000):
    """
    Plot the energy spectrum of the events in the detector.

    Parameters
    ----------
    Object : muon_lifetime
        muon_lifetime object, with the trigger already computed.
    bins : int, optional
        Number of bins in the energy histogram. Defaults to 100.
    afterpulse_value : float, optional
        Estimated value of the afterpulse peak (in ADC). Defaults to 1_000.

    Returns
    -------
    fig, ax : matplotlib figure and axis
        Figure and axis of the plot.
    """

    fig, ax = plt.subplots()
    ax.hist(Object.values, bins=bins, histtype='step', label='data')
    ax.axvline(afterpulse_value, ls='--', color='r', label='Afterpulse peak (estimated)')
    
    ax.set_xlabel('Energy (ADC)')
    ax.set_ylabel('Counts')
    ax.legend()
    fig.set_facecolor('white')
    
    return fig, ax