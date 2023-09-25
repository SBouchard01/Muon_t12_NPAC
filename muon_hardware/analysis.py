import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit.cost import LeastSquares
from iminuit import Minuit
from .TAC_to_python import TAC_formatting
from .calibration import calibrate_TAC
from .utils import rebin, linear, exp, chi2

class muon_lifetime():
    
    
    def __init__(self,
                 file:str,
                 error:float = 0,
                 **kwargs):
        """
        Class to analyse the muon lifetime data.

        Parameters
        ----------
        file : str
            Path to the file containing the data. This must be a Faster data file.
        error : float
            Measurement error on the voltage (in mV). Defaults to 0.
            
        Other parameters
        ----------------
        **kwargs : dict
            All other keyword arguments will be passed on to the `calibrate_TAC` function
        """
        
        # Calibrate the TAC
        self.ADC_hist = TAC_formatting(file)
        ADC_to_time = calibrate_TAC(**kwargs)
        self.times_unbinned = linear(self.ADC_hist[:,0], *ADC_to_time) # Convert ADC to µs
        self.values_unbinned = self.ADC_hist[:, 1]
        
        # Set the parameters within the class
        self.error = error
        
        # Internal parameters
        self.expected_flux = 13 # Expected rate of muons per second in the detector
        
        
        
    def rebin_hist(self, bins:int = 550):
        """
        see utils.rebin()

        Parameters
        ----------
        bins : int, optional
            The number of bins in the new histogram. Defaults to 550.

        Returns
        -------
        data_rebinned
            The rebinned data
        """
        
        # Rebin the histogram
        data_rebinned = rebin(self.times_unbinned, self.values_unbinned, bins)
        self.times = data_rebinned[:,0]
        self.values = data_rebinned[:,1]
        return data_rebinned
    
        
    
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
        mask = (self.times > x_cut) 
        self.x = self.times[mask]
        self.y = self.values[mask]
        self.yerr = np.sqrt(self.y)
        
        # Remove the empty bins from y in the fit (no physics there !)
        mask = self.y != 0
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.yerr = self.yerr[mask]

        if method == 'scipy':
            self.popt, self.pcov = curve_fit(exp, self.x, self.y, p0=p0)
            self.perr = np.sqrt(np.diag(self.pcov))
            
            # self.chi2 = r_squared(self.y, exp(self.x, *self.popt)) # I cheat here, because I can't estimate the errors on the fit parameters (So i do a worst case estimate)
            ndof = len(self.x) - len(self.popt)
            self.chi2 = chi2(self.y, exp(self.x, *self.popt), self.yerr)/ndof

        elif method == 'iminuit':           
            least_squares = LeastSquares(self.x, self.y, self.yerr, exp)
            
            m = Minuit(least_squares, amplitude=p0[0], tau=p0[1]) # Initailize the minimizer
            
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

        plt.plot(self.times, self.values)
        plt.errorbar(self.times, self.values, yerr=np.sqrt(self.values), fmt='.', color='k', alpha=0.3)
        
        ax.text(0.5, 0.7, f'$\\tau$ = {self.popt[1]:.2f}±{self.perr[1]:.2f} µs', transform=ax.transAxes)
        
        ax.plot(self.x, exp(self.x, *self.popt), 'r-', label=f'$\\chi^{{2}}_{{{self.method}}}$ = {self.chi2:.2f} \n{np.sum(self.y)} evts')

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Counts')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.set_facecolor('white') # Because jupyter is annoying and saves the figure with a no background
        
        return fig, ax