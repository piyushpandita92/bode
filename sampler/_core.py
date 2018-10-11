import numpy as np
import math


__all__ = ['xik', 'mk', 'bk']

# Defining some frequently used constants
u_k = 1		# upper bound for any dimension
m_k = 0		# lower bound for any dimension
pi = math.pi	

def xik(x, ell):
	"""
	:param x:		a dimension of the design being considered
	:param ell:		corresponding lengthscale of the dimension
	"""
	xik = (np.sqrt(pi) / np.sqrt(2)) * ell * (math.erf((u_k - x) / (np.sqrt(2) * ell)) - math.erf((m_k - x)/(np.sqrt(2)*ell)))
	return xik


def mk(a, b, ell):
	"""
	:param a:		a dimension of a training input 		
	:param b:		a dimension of another training input
	:param ell:		corresponding lengthscale of the dimension
	"""
	_mk = (np.sqrt(pi) / 2.) * ell * np.exp(-(((a - b) ** 2) / (2. * 2 * ell**2))) * (math.erf((2. * u_k 
		- a - b)/(2. * ell)) - math.erf((2. * m_k - a - b)/(2. * ell)))
	return _mk


def bk(ell):
	"""
	:param ell:		corresponding lengthscale of the dimension
	"""
	zk = (m_k - u_k)/(np.sqrt(2) * ell)
	bk = (np.sqrt(pi) / 2.) * 2. * (ell ** 2) * ((-2./np.sqrt(pi)) 
		+ 2 * zk * math.erf(zk) + (2./np.sqrt(pi)) * np.exp(-(zk)**2))
	return bk