import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.optimize import curve_fit
from .faster_to_python import faster_formatting
from .utils import exp, r_squared, chi2

class muon_lifetime():
    """
    Class to analyse the muon lifetime data. It takes the data from a Faster fata file and analyzes.
    """
    
    def __init__(self,
                 file:str,
                 error:float = 0,
                 time_unit:str = 'ns'):
        """
        Class to analyse the muon lifetime data.

        Parameters
        ----------
        file : str
            Path to the file containing the data. This must be a Faster data file.
        error : float
            Measurement error on the voltage (in mV). Defaults to 0.
        time_unit : str, optional
            Units of the time in the file. Defaults to 'ns'.

        Raises
        ------
        ValueError
            If the time unit is not recognised.
        """
        
        times, values = faster_formatting(file, return_values=True)
        
        # Set the parameters within the class
        self.times = times
        self.values = values
        self.error = error
        
        # Internal parameters
        expected_flux = 13 # Expected rate of muons per second in the detector
        
        # Convert times in µs
        if time_unit == "s":
            self.times *= 1e6
        elif time_unit == "ms":
            self.times *= 1e3
        elif time_unit == "µs":
            pass
        elif time_unit == "ns":
            self.times *= 1e-3
        else:
            raise ValueError(f"Unrecognised time unit : \'{time_unit}\'. Please use one of the following : \'s\', \'ms\', \'µs\' or \'ns\' ")
        
    
    
    def trigger(self, 
                mode:str = 'manual',
                **kwargs):
        """
        Takes the voltage and returns a mask for the indices where the voltage is above the threshold.
        Also applies the mask on the class data

        Parameters
        ----------
        mode : str, optional
            Mode to use to get the threshold. Can be either 'auto' or 'manual'. Defaults to 'auto'.
        threshold : float, optional
            Threshold of the values (in ADC). Used only if mode is set to 'manual'.
        aqc_time : float, optional
            Acquisition time in seconds. Used only if mode is set to 'auto'.

        Returns
        -------
        Array-like
            Mask of the indices at which the voltage is above the threshold. (i.e. True if it's abovf, False if it's not)
        """
        
        if mode == 'auto':
            acq_time = kwargs['acq_time'] # Acquisition time in seconds
            adc_range = np.linspace(np.min(self.values), np.max(self.values), 10_000)

            Counts = []
            
            for i in range(len(adc_range)):
                n = len(np.where(self.values > adc_range[i])[0])
                Counts.append(n/acq_time)
            
            # Get the x value where the counts is 90% of the max
            threshold_ids = np.where(np.array(Counts) > 13*0.9)[0]
            threshold_id = threshold_ids[-1] if len(threshold_ids) > 0 else 0 # Make sure the threshold is not empty
            threshold = adc_range[threshold_id]
            # print(f"Threshold : {threshold}, {adc_range[threshold]} ADC")
            
        elif mode == 'manual':
            threshold = kwargs['threshold']
            
        else:
            raise ValueError(f"Unrecognised mode : \'{mode}\'. Please use one of the following : \'auto\' or \'manual\' ")
        
        # Get the times when the voltage is above the threshold
        mask = self.values > threshold
        
        self.times = self.times[mask]
        self.values = self.values[mask]
        
        return mask
    
    
    
    def TAC(self, 
            TAC_range:float = 10,
            bins:int = 100):
        """
        Takes the measured times and returns the delays that are in the TAC window.

        Parameters
        ----------
        measured_times : Array-like
            Array of measured times in µs
        TAC_range : float, optional
            Range window of the time (in µs) in which we detect an event. Defaults to 10.
        bins : int, optional
            Number of bins in the delays histogram. Defaults to 100.

        Returns
        -------
        TAC_out : Array-like
            Array of delays in µs that are in the TAC window.
        """

        # Get the start and stop channels
        Start = self.times
        Stop = self.times[1:] # Stop is the next event

        self.delay = Stop - Start[:-1] # Delay between the start and the stop
        window = np.full((Stop.shape), TAC_range) # The window of the TAC

        mask = self.delay < window # Get the events that are in the window

        # If an event is True, the next must be False
        for i in range(len(mask)-1): # Needs to be done with a loop !!
            if mask[i]:
                mask[i+1] = False

        self.TAC_out = self.delay[mask] # Get the delays that are in the window
        
        # Get the histogram of the delays in the window
        self.count, bins = np.histogram(self.TAC_out, bins=bins)
        self.bins_centers = (bins[1:] + bins[:-1])/2
        
        return self.TAC_out
        
        
    
    def exp_fit(self, 
                x_cut:float = 0,
                method:str = 'iminuit',
                p0:list = [1, 5]):
        """
        Fits an exponential on the histogram of the delays in the TAC window.

        Parameters
        ----------
        x_cut : float, optional
            The x value over which the fit is done. This is due to possible detector issues at low x.
            Defaults to 0
        method : str, optional
            Method to use to fit the exponential. Can be either 'scipy' or 'iminuit'. Defaults to 'iminuit'.
        p0 : list, optional
            Initial guess of the parameters of the exponential, for the respective values of the amplitude and the lifetime. Defaults to [1, 5].

        Returns
        -------
        popt : Array-like
            Array of the fitted parameters (amplitude, lifetime) of the exponential.
        perr : Array-like
            Array of the errors on the fitted parameters (amplitude, lifetime) of the exponential.
        """
        
        self.method = method # Store the method used for the fit (for the plot)
        
        # Fit the exp on the histogram
        mask = (self.bins_centers > x_cut) 
        self.x = self.bins_centers[mask]
        self.y = self.count[mask]
        self.yerr = np.sqrt(self.y)
        
        # Remove the empty bins from y in the fit (no physics there !)
        mask = self.y != 0
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.yerr = self.yerr[mask]

        if method == 'scipy':
            self.popt, self.pcov = curve_fit(exp, self.x, self.y, p0=[1, 5])
            self.perr = np.sqrt(np.diag(self.pcov))
            
            # self.chi2 = r_squared(self.y, exp(self.x, *self.popt)) # I cheat here, because I can't estimate the errors on the fit parameters (So i do a worst case estimate)
            ndof = len(self.x) - len(self.popt)
            self.chi2 = chi2(self.y, exp(self.x, *self.popt), self.yerr)/ndof

        elif method == 'iminuit':           
            least_squares = LeastSquares(self.x, self.y, self.yerr, exp)
            
            m = Minuit(least_squares, amplitude=1, tau=5) # Initailize the minimizer
            
            m.migrad()  # Finds minimum of least_squares function
            m.hesse()   # Accurately computes uncertainties
            
            self.popt = m.values
            self.pcov = m.covariance
            self.perr = m.errors
            
            self.chi2 = m.fmin.reduced_chi2
        
        else : 
            raise ValueError(f"Unrecognised method : \'{method}\'. Please use one of the following : \'scipy\' or \'iminuit\' ")
        
        self.tau = self.popt[1]
        self.tau_err = self.perr[1]
            
        return self.popt, self.perr
        
    # TODO : Error computation
    
    def plot_final_results(self,
                           bins:float = None):
        """
        Plots the histogram of the delays in the TAC window and the fitted exponential.

        Parameters
        ----------
        bins : float, optional
            The number of bins to display the histogram. If set to None, 
            the actual histogram bin number will be used. Defaults to None.

        Returns
        -------
        fig, ax
            The axes and figure created.
        """
        
        fig, ax = plt.subplots() 
        
        if bins is None:
            bins = len(self.count) # Defaults to the histogram parameters
        n, bins2, _ = ax.hist(self.TAC_out, bins=bins, label=f'Data ({len(self.TAC_out)} evts)')
        b_ct = (bins2[1:] + bins2[:-1])/2
        plt.errorbar(b_ct, n, yerr=np.sqrt(n), fmt='.', color='k', alpha=0.3)
        
        ax.text(0.5, 0.7, f'$\\tau$ = {self.popt[1]:.2f}±{self.perr[1]:.2f} µs', transform=ax.transAxes)
        
        ax.plot(self.x, exp(self.x, *self.popt), 'r-', label=f'$\\chi^{{2}}_{{{self.method}}}$ = {self.chi2:.2f} \n{np.sum(self.y)} evts')

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Counts')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.set_facecolor('white') # Because jupyter is annoying and saves the figure with a no background
        
        return fig, ax