import numpy as np


def TAC_formatting(file:str):
    """
    Takes a TAC output (RHB ASCII file) and returns the histogram on the non-empty range.

    Parameters
    ----------
    file : str
        Path to the RHB ASCII file
        
    Returns
    -------
    data_rebinned : Array-like
        The histogram without the empty range
    """
    
    data = np.loadtxt(file, skiprows=2, delimiter=";") # Load the data
    
    # Remove empty hist part
    nonempty = np.nonzero(data[:,1]) # Get list of indices of nonempty hist values
    max_nonempty = np.max(nonempty) # Get the indice of the last non empty bin
    data = data[:max_nonempty+2] # Remove the empty values after last filled bin
    
    
    return data