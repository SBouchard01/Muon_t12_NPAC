import numpy as np

# TODO : maybe convert ADC to voltage ??

def trig(volts, threshold=0.5):
    """
    Takes the voltage and returns a mask for the indices where the voltage is above the threshold.

    Parameters
    ----------
    volts : Array-like
        Array of voltages (in mV)
    threshold : float, optional
        Threshold of the voltage (in mV). Defaults to 0.5.

    Returns
    -------
    Array-like
        Mask of the indices at which the voltage is above the threshold. (i.e. True if it's abovf, False if it's not)
    """
    # Get the times when the voltage is above the threshold
    return volts > threshold



def TAC(measured_times, TAC_range=10_000):
    """
    Takes the measured times and returns the delays that are in the TAC window.

    Parameters
    ----------
    measured_times : Array-like
        Array of measured times in ns
    TAC_range : float, optional
        Range window of the time (in ns) in which we detect an event. Defaults to 10_000.

    Returns
    -------
    final_delays : Array-like
        Array of delays in ns that are in the TAC window.
    """

    # Get the start and stop channels
    Start = measured_times
    Stop = measured_times[1:] # Stop is the next event

    delay = Stop - Start[:-1] # Delay between the start and the stop
    window = np.full((Stop.shape), TAC_range) # The window of the TAC

    mask = delay < window # Get the events that are in the window

    # If an event is True, the next must be False
    for i in range(len(mask)-1): # Needs to be done with a loop !!
        if mask[i]:
            mask[i+1] = False

    final_delays = delay[mask] # Get the delays that are in the window
    
    return final_delays