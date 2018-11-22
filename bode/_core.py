import numpy as np
import math
import sys
from scipy.special import erf
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import expon

__all__ = ['xik', 'bk', 'JeffreysPrior', 'BetaPrior', 'GammaPrior', 'UniformPrior', 'ExponentialPrior', 'ei']

# Defining some frequently used constants
u_k = 1		# upper bound for any dimension
m_k = 0		# lower bound for any dimension
pi = math.pi	

class JeffreysPrior(object):
	"""
	Return the log probability density for a Jeffreys Prior on the parameters of the GP model.
	"""
	def __call__(self, param):
		if np.any(param<0):
			return -np.inf
		return -1 * np.log(param)
	def sample(self, size=1):
		print "Cannot sample from JeffreysPrior! Choose another prior over this parameter."
		sys.exit(1)

class BetaPrior(object):
	"""
	Return the log probability density for a Beta Prior on the parameters of the GP model.
	"""
	def __init__(self, a=2, b=5):
		self.a = a
		self.b = b
	def __call__(self, param):
		if np.any(param<0):
			return -np.inf
		return beta.logpdf(param, a=self.a, b=self.b)
	def sample(self, size=1):
		return beta.rvs(a=self.a, b=self.b, size=size)


class GammaPrior(object):
	"""
	Log probability for a Gamma Prior on the parameters of the GP model.
	"""
	def __init__(self, a=8, scale=1):
		self.a = a
		self.scale = scale
	def __call__(self, param):
		if np.any(param<0):
			return -np.inf
		return gamma.logpdf(param, a=self.a, scale=self.scale)
	def sample(self, size=1):
		return gamma.rvs(a=self.a, scale=self.scale, size=size)

class ExponentialPrior(object):
	"""
	Log probability for a Exponential Prior on the parameters of the GP model.
	"""
	def __init__(self, scale=1):
		self.scale = scale
	def __call__(self, param):
		if np.any(param<0):
			return -np.inf
		return expon.logpdf(param, scale=self.scale)
	def sample(self, size=1):
		return expon.rvs(scale=self.scale, size=size)

class UniformPrior(object):
	"""
	Log probability of a uniform density.
	"""
	def __init__(self, a=0, b=1):
		self.a = a
		self.b = b
	def __call__(self, param):
		if np.any(param<0) or param>self.b or param<self.a:
			return -np.inf
		return -1 * np.log(self.b - self.a)
	def sample(self, size=1):
		return beta.uniform(loc=self.a, scale=self.b, size=size)


def xik(x, ell, ss):
	"""
	:param x:		a dimension of the design being considered
	:param ell:		corresponding lengthscale of the dimension
	"""
	xik = (np.sqrt(pi) / np.sqrt(2)) * ell * (erf((u_k - x) / (np.sqrt(2) * ell)) - erf((m_k - x) / (np.sqrt(2) * ell)))
	return ss * np.prod(xik, axis=1)


def bk(ell, ss):
	"""
	:param ell:		corresponding lengthscale of the dimension
	"""
	zk = 1. / (np.sqrt(2) * ell)
	bk = 2 * np.sqrt(pi) * (ell ** 2) * (zk * erf(zk) + np.exp(-(zk ** 2)) / np.sqrt(pi) - 1. / np.sqrt(pi)) # Piyush derivation
	return ss * np.prod(bk)

def ei(mu, sigma, y_star, mode="min"):
    """
    expected improvement for a minimization problem
    :param x:
    :param m:
    :param sigma:
    :param y_min:
    :return:
    """
    if mode=="min":
    	z = (y_star - mu) / np.sqrt(sigma)
    	return (np.sqrt(sigma)) * norm.pdf(z) + (y_star - mu) * norm.cdf(z)
    elif mode=="max":
    	z = (mu - y_star) / np.sqrt(sigma)
    	return (np.sqrt(sigma)) * norm.pdf(z) + (mu - y_star) * norm.cdf(z)