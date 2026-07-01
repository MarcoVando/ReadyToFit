"""Basic peak function definitions and model registry.

This module provides the fundamental peak profile functions used in fitting:
- Gaussian: symmetric bell curve
- Lorentzian: Lorentz profile (not currently used in composite models)
- Voigt: convolution of Gaussian and Lorentzian
- Asymmetric Voigt: Voigt with different left/right widths
- Skewed Voigt: Voigt with asymmetric tails

The get_model() function returns function-parameter pairs for each model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile, erf, erfc
# =========================
# Basic peak functions
# =========================
def gaussian(x, A, mu, sigma):
    """Gaussian function.
    Parameters:
    A     : amplitude
    mu    : center
    sigma : width (standard deviation) 
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def lorentzian(x, A, mu, gamma):
    """Lorentzian function.
    Parameters:
    A     : amplitude
    mu    : center
    gamma : width (half-width at half-maximum)
    """
    return A * (gamma**2 / ((x - mu)**2 + gamma**2))
    
def voigt(x, A, mu, sigma, gamma):
    """Voigt function: convolution of Gaussian and Lorentzian.
    Parameters:
    A     : amplitude
    mu    : center  
    sigma : Gaussian width (standard deviation)
    gamma : Lorentzian width (half-width at half-maximum)
    """
    return A * voigt_profile(x - mu, sigma, gamma)


def asym_voigt(x, A, mu, sigma_L, sigma_R, gamma):
    """Asymmetric Voigt function with different left/right Gaussian widths.
    Parameters:
    A       : amplitude
    mu      : center
    sigma_L : Gaussian width on the left side (x < mu)
    sigma_R : Gaussian width on the right side (x > mu)
    gamma   : Lorentzian width (half-width at half-maximum)
    """
    sigma = np.where(x < mu, sigma_L, sigma_R)
    return A * voigt_profile(x - mu, sigma, gamma)

def skew_voigt(x, A, mu, sigma, gamma, alpha):
    """Skewed Voigt function with an additional skewness parameter.
    Parameters:
    A     : amplitude
    mu    : center
    sigma : Gaussian width (standard deviation)
    gamma : Lorentzian width (half-width at half-maximum)
    alpha : skewness parameter (positive for right skew, negative for left skew)    
    """
    base = voigt_profile(x - mu, sigma, gamma)
    skew = 1 + erf(alpha * (x - mu))
    return A * base * skew

def emg_reversed(x, A, mu, sigma, lam):
    """
    Reversed Exponentially Modified Gaussian (left-skewed)
    Parameters:
    A     : amplitude
    mu    : center
    sigma : Gaussian width
    lam   : exponential rate (>0)
    """

    term1 = (lam / 2) * np.exp((lam / 2) * (2*mu + lam*sigma**2 + 2*x))
    term2 = erfc((mu + lam*sigma**2 + x) / (np.sqrt(2) * sigma))

    return A * term1 * term2

def get_model(model_name):    
    if model_name == "gauss":
        return (
            lambda x, A, mu, sigma: gaussian(x, A, mu, sigma),
            ["A", "mu", "sigma"]
        )
    
    elif model_name == "lorentz":
        return (
            lambda x, A, mu, gamma: lorentzian(x, A, mu, gamma),
            ["A", "mu", "gamma"]
        )
    
    elif model_name == "voigt":
        return (
            lambda x, A, mu, sigma, gamma: voigt(x, A, mu, sigma, gamma),
            ["A", "mu", "sigma", "gamma"]
        )

    elif model_name == "asym":
        return (
            lambda x, A, mu, sigma_L, sigma_R, gamma:
                asym_voigt(x, A, mu, sigma_L, sigma_R, gamma),
            ["A", "mu", "sigma_L", "sigma_R", "gamma"]
        )

    elif model_name == "skew":
        return (
            lambda x, A, mu, sigma, gamma, alpha:
                skew_voigt(x, A, mu, sigma, gamma, alpha),
            ["A", "mu", "sigma", "gamma", "alpha"]
        )

    elif model_name == "emg_reversed":
        return (
            lambda x, A, mu, sigma, lam:
                emg_reversed(x, A, mu, sigma, lam),
            ["A", "mu", "sigma", "lam"]
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
