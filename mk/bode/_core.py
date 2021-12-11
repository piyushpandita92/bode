"""
Module defining the priors

Author:
    Murali Krishnan Rajasekharan Pillai

Date:
    12/11/2021
"""

import numpy as np
from scipy.stats import gamma
from scipy.stats import beta

__all__ = ['GammaPrior', 'BetaPrior', 'JeffreysPrior']

class GammaPrior:
    """
    Log probability for a Gamma Prior on the parameters of the GP model.
    """
    def __init__(self, a=8, scale=1):
        self.a = a
        self.scale = scale

    def __call__(self, param):
        """Calculate the log pdf evaluated at x"""
        if np.any(x < 0):
            return -np.inf
        return beta.logpdf(x, a=self.a, b=self.b)

    def sample(self, size=1):
        """Sample Beta random variables"""
        return beta.rvs(a=self.a, b=self.b, size=size)

class BetaPrior:
    """Return the log probability density for a Beta prior on the parameters of the GP model.
    """
    def __init__(self, a=2, b=5):
        self.a = a
        self.b = b
    def __call__(self, param):
        if np.any(param < 0):
            return -np.inf
        return beta.logpdf(param, a=self.a, b=self.b)
    def sample(self, size=1):
        return beta.rvs(a=self.a, b=self.b, size=size)

class JeffreysPrior:
    """
    Return the log probability density for a Jeffreys prior on the parameters of the GP model
    """
    def __call__(self, param):
        if np.any(param < 0):
            return -np.inf
        return -1. * np.log(param)
    def sample(self, size=1):
        raise Exception("[!] Cannot sample from Jeffreys Prior: Choose another prior over this parameter!")
