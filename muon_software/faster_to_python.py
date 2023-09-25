import numpy as np
import pandas as pd

def faster_formatting(file:str, return_values:bool=False):
    """
    Takes a Faster file and returns the times in ns 
    (with the intrinsic Faster delay)

    Parameters
    ----------
    file : str
        Path to the file that contains the Faster data

    Returns
    -------
    Array-like
        Array of times in ns
    """
    df = pd.read_csv(file, sep='  ', skiprows=2, engine='python', header=None)
    df = df.dropna(axis=1, how='all') # Remove columns with only NaN

    # Change the dataframe labels
    labels = ['DATA NUM', 'TYPE', 'LABEL', 'TIME', 'DELAY', 'VALUE', 'PILEUP']
    if len(df.columns) == len(labels)-1: # i.e. there is no pileup column
        labels = labels[:-1]
    df.columns = labels
    
    # Get the times
    times = df['TIME'].values

    # Remove the last 2 characters ('ns') on each value
    times = np.array([float(t[:-2]) for t in times])

    # Delay value have a format like 'delta_t=00000ns' We want to extract the number
    delay = df['DELAY'].values
    delay = np.array([float(d[8:-2]) for d in delay])
    
    # Measurement values have a format like 'meas=0000' We want to extract the number
    measures = df['VALUE'].values
    measures = np.array([float(m[5:]) for m in measures])
    
    if return_values:
        return times + delay, measures
    
    return times + delay