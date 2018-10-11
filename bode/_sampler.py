"""
Information acquisition for optimal design of experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
import math
import GPy
from _core import *
import itertools
import time
from pyDOE import *
from copy import copy
from scipy.stats import multivariate_normal
from scipy.stats import norm
start_time = time.time()

__all__ = ['KLSampler']

class KLSampler(object):
	"""
	This class computes the sensitivity of a set of inputs
	by taking the posterior expectation of the var of the 
	corresponding effect functions.
	"""


	def _noise(self):
		noise = np.array([self.model[0].param_array[-1]])
		return noise


	@property
	def noise(self):
		"""
		This returns the estimated noise for the GP model.
		"""
		return self._noise
	

	def _ss(self):
		ss = (np.array([self.model[0].param_array[0]]))
		return ss


	@property
	def ss(self):
		"""
		This returns the signal strength of the GP model.
		"""
		return self._ss

	
	def _lengthscales(self):
		ells = np.array([self.model[0].param_array[i] for i in range(1, self.X.shape[1]+1)])
		return ells


	@property
	def lengthscales(self):
		"""
		This returns the lengthscales corresponding to each input dimension.
		"""
		return self._lengthscales

	
	def w_mat(self, noise=None):
		n = self.X.shape[0]
		if noise is not None:
			l = np.linalg.cholesky(self.kern_mat(self.X, self.X) + noise * np.eye(n))
			
		else:
			l = np.linalg.cholesky(self.kern_mat(self.X, self.X) + self.noise() * np.eye(n))
		w = np.matmul(np.linalg.inv(l).T, np.linalg.inv(l))
		return w


	def alpha(self, noise):
		"""
		This is the apha term defined in the report.
		"""
		if noise is None:
			W = self.w_mat()
		else:
			W = self.w_mat(noise=noise)
		alpha = np.matmul(W, self.Y)
		return alpha
	
	def __init__(self, X, Y, obj_func, noisy, bounds,
		true_func=None,
		model_kern=GPy.kern.Matern32, 
		num_opt_restarts=80,
		num_mc_samples=1000,
		num_quad_points=100,
		energy=0.95,
		nugget=1e-3,
		lengthscale=1.,
		variance=1.,
		kld_tol=1e-2,
		func_name='ex1',
		quad_points=None,
		quad_points_weight=None,
		max_it=50,
		per_sampled=10,
		x_hyp=None
		):
		"""
		:param X:					the inputs of the training data as an array.
		:param Y:					the outputs of the training data as an array.
		:param obj_func:			the objective function object
		:param bounds:				the bounds on the input variables
		:param model_kern:			the kernel of the GP
		:param num_opt_restarts:	the number of optimization restarts while finding the hyper-parameter values for the GP
		:param energy:				the energy of the Truncated GP as a percentage of the sum of the positive eigenvalues
		:param nugget:				the initial value for the noise nugget
		:param lengthscale: 		the initial value for the lengthscale
		:param variance:			the initial value for the noise variance or signal strength
		:param kld_tol:				the stopping tolerance of the algorithm
		:param quad_points:			the quadrature points for the Nystrom approxmiation
		:param quad_points_weight: 	the weights of the quadrature points
		:param max_it:				the maximum number of iterations 
		:param per_sampled:			the ratio of number of data points and the input dimension after which GP model params are optimized
		"""
		assert X.ndim == 2
		self.X = X
		assert Y.ndim == 2
		self.Y = Y
		assert self.X.shape[0] == self.Y.shape[0]
		self.X_u = X
		self.Y_u = Y
		self.X_r = X
		self.Y_r = Y
		self.per_sampled = self.X.shape[1] * per_sampled
		self.dim = self.X.shape[1]
		self.num_obj = self.Y.shape[1]
		self.obj_func = obj_func
		self.true_func = true_func
		self.model_kern = model_kern
		self.nugget = nugget
		self.lengthscale = lengthscale
		self.variance = variance
		self.noisy = noisy
		self.num_opt_restarts = num_opt_restarts
		self.model = self.make_model(self.X, self.Y, it=self.X.shape[0])
		self.model_d = self.make_model(self.X_u, self.Y_u, it=self.X_u.shape[0])
		self.model_r = self.make_model(self.X_r, self.Y_r, it=self.X_r.shape[0])
		self.all_p = {}
		self.num_mc_samples = num_mc_samples
		self.num_quad_points = num_quad_points
		self.energy = energy
		self.x_hyp = x_hyp
		if self.x_hyp:
			self.y_hyp = self.model[0].posterior_samples(self.x_hyp, 1)[0, 0].copy()
		if quad_points is None:
			self.quad_points = np.linspace(0, 1, self.num_quad_points)
			self.quad_points_weight = np.eye(self.num_quad_points)
		else:
			self.quad_points = quad_points
			self.quad_points_weight = quad_points_weight
		self.get_val_vec = self.eig_val_vec(model=self.model)
		self.bounds = bounds
		self.kld_tol = kld_tol
		self.func_name = func_name
		self.max_it = max_it
		
	def kern_mat(self, xi, xj):
		"""
		Computes an ```nxn``` matrix whose elements are the RBF kernel based values for the two
		input arrays. This is the prior covariance.
		:param xi:		array of input(s)
		:param xj:		array of input(s)
		"""
		k = self.model[0].kern.K(xi, xj)
		return k

	def make_model(self, X, Y, it=0):
		"""
		Trains the surrogate model.
		"""
		m = Y.shape[1]
		surrogates = []
		for i in xrange(m):
			# kern = self.model_kern(input_dim=X.shape[1], ARD=True)
			model = GPy.models.GPRegression(X, Y, self.model_kern(input_dim=X.shape[1], ARD=True))
			if it < self.per_sampled:
				model.kern.lengthscale.fix(self.lengthscale)
				model.kern.variance.fix(self.variance)
				model.likelihood.variance.constrain_fixed(self.nugget ** 2)
			else:
				model.likelihood.variance.constrain_fixed(self.nugget ** 2)
				model.optimize_restarts(num_restarts=self.num_opt_restarts, verbose=False)
			if self.noisy:
				self.nugget = model.likelihood.variance ** .5
			# else:
			# 	model.likelihood.variance.constrain_fixed(self.nugget ** 2)
				# model.likelihood.variance.fix(self.nugget**2)
			print model, model.kern.lengthscale
			surrogates.append(model)
		return surrogates

	def eig_func(self, x, w_j, x_d, val_trunc, vec_trunc, model=None):
		"""
		Constructing the eigenfunctions for the given eigenvalues at ```x```.
		"""
		k_x_d_x = (model[0].predict(np.vstack([x_d, np.atleast_2d(x)]), full_cov=True)[1][-1, :-1])[:, None]
		eig_func = (1. / val_trunc) * np.sum(np.multiply(np.multiply(w_j, vec_trunc), k_x_d_x))
		return eig_func

	def eig_val_vec(self, model=None):
		"""
		Eigendecomposition of the ```B``` matrix in equation 15.88 of UQ book chapter.
		"""
		x_d = self.quad_points
		p_x_d = self.quad_points_weight
		K_x_d = model[0].predict(x_d, full_cov=True)[1]
		W_h = np.sqrt(((1. / (np.sum(self.quad_points_weight))) * np.diag(p_x_d)))
		B = np.matmul(np.matmul(W_h, K_x_d), W_h)
		val, vec = np.linalg.eigh(B)
		val[val<0] = 0			   					# taking care of the negative eigenvalues
		idx_sort = np.argsort(-val)
		val_sort = val[idx_sort]
		vec_sort = vec[:, idx_sort]
		tot_val = 1. * (np.cumsum(val_sort)) / np.sum(val_sort)
		idx_dim = min(np.where(tot_val >= self.energy)[0])
		val_trunc = val_sort[:idx_dim + 1, ]
		vec_trunc = vec_sort[:, :idx_dim + 1]
		# phi_x_dx = np.array([np.mean(np.sum(np.multiply(np.multiply(vec_trunc[:, j][:, None], (np.sqrt(((p_x_d / np.sum(self.quad_points_weight)))))[:, None]), K_x_d), axis=0), axis=0) for j in xrange(vec_trunc.shape[1])]) / val_trunc
		phi_x_dx = np.mean(np.multiply(vec_trunc, 1. / (np.sqrt(((p_x_d / np.sum(self.quad_points_weight)))))[:, None]), axis=0)
		return val_trunc, vec_trunc, W_h, x_d, phi_x_dx

	def sample_xi_hyp(self, dim, val_trunc, eig_funcs, m_x, y_hyp):
		"""
		Samples a multivariate random variable conditioned on the data and a 
		hypothetical observation.
		:param m_x:			keep in mind this is the posterior mean conditional 
							on data and a hypothetical observation.
		:param dim:			number of reduced dimensions of the eigenvalues.
		:param val_trunc:	eigenvalues after truncation.
		:param eig_funcs:	eigenvectors after truncation.
		:param y:			hypothetical sampled observation.	
		"""
		sigma_inv = np.multiply(np.matmul(np.sqrt(val_trunc)[:, None], np.sqrt(val_trunc)[None, :]), np.matmul(eig_funcs[:, None], eig_funcs[None, :]))
		sigma_inv_2 = sigma_inv / (self.nugget ** 2)
		sigma_inv_1 = np.eye(dim)
		sigma_3 = np.linalg.inv(sigma_inv_1 + sigma_inv_2) 
		mu_3 = ((y_hyp - m_x)/ (self.nugget**2)) * np.matmul(sigma_3, np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs[:, None]))
		xi = np.random.multivariate_normal(mu_3[:, 0], sigma_3, 1).T
		return xi


	def sample_xi(self, dim):
		"""
		Samples a multivariate centered random variable.
		"""
		mu = np.zeros(dim,)
		sigma = np.eye(dim)
		xi = multivariate_normal.rvs(mu, sigma, 1).T
		return xi


	def obj_est(self, x_grid, x_hyp):
		"""
		Samples a value of the QOI at a given design point.
		"""
		samp = np.zeros(len(x_grid))
		val_trunc, vec_trunc, W_h, x_d, phi_x_dx = self.get_val_vec
		w_j = W_h
		sample_xi  = self.sample_xi(val_trunc.shape[0])
		eig_funcs_f = np.zeros((len(x_grid), len(val_trunc)))
		clock_time = time.time()
		for j in xrange(len(x_grid)):
			x = x_grid[j]
			for i in xrange(eig_funcs_f.shape[1]):
				eig_funcs_f[j, i] = self.eig_func(x, (w_j[w_j>0])[:, None], x_d, val_trunc[i, ], (vec_trunc[:, i])[:, None])
			#print '>... Sampled the eigenfunction at', time.time() - clock_time, 'seconds'
			samp[j, ] =  self.model[0].predict(np.atleast_2d(x))[0][0] + np.sum(np.multiply(np.multiply(sample_xi, (np.sqrt(val_trunc))[:, None]), eig_funcs_f[j, :][:, None])).copy()
		return samp, val_trunc, eig_funcs_f


	def obj_est_hyp(self, x_grid, x_hyp):
		# Repeating the process after adding the hypothetical observation to the data set
		y_hyp = self.y_hyp
		m_x_hyp = self.model[0].predict(x_hyp)[0][0]
		samp_hyp = np.zeros(len(x_grid))
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		eig_funcs_hyp = np.zeros(len(val_trunc))
		eig_funcs_f_hyp = np.zeros((len(x_grid), len(val_trunc)))
		for i in xrange(len(val_trunc)):
			eig_funcs_hyp[i, ] = self.eig_func(x_hyp, (w_j[w_j>0])[:, None], x_d, val_trunc[i, ], (vec_trunc[:, i])[:, None] )
		sample_xi_hyp = self.sample_xi_hyp(val_trunc.shape[0], val_trunc, eig_funcs_hyp, m_x_hyp, y_hyp)
		for j in xrange(len(x_grid)):
			x = x_grid[j]
			for i in xrange(eig_funcs_f_hyp.shape[1]):
				eig_funcs_f_hyp[j, i] = self.eig_func(x, (w_j[w_j>0])[:, None], x_d, val_trunc[i, ], (vec_trunc[:, i])[:, None])
			samp_hyp[j, ] = self.model[0].predict(np.atleast_2d(x))[0][0] + np.sum(np.multiply(np.multiply(sample_xi_hyp, (np.sqrt(val_trunc))[:, None]), (eig_funcs_f_hyp[j, :])[:, None]))
		return samp_hyp, y_hyp, val_trunc, eig_funcs_f_hyp


	def get_phi_x_dx(self, val_trunc, vec_trunc, W_h, x_d, p_x_d):
		phi_x_dx = np.zeros(len(val_trunc))
		w_j = W_h 
		for i in xrange(len(val_trunc)):
			sig = 0
			for j in xrange(x_d.shape[0]):
				x = x_d[j, :][:, None]
				ss = (self.eig_func(x, (w_j[w_j>0])[:, None], x_d, val_trunc[i, ], (vec_trunc[:, i])[:, None])) * p_x_d[j, ]
				sig = sig + ss
			phi_x_dx[i, ] = 1. * sig / x_d.shape[0] 
		return phi_x_dx


	def get_sigma_1(self, val_trunc, phi_x_dx):
		"""
		Returns the mean and the variance of the posterior using the KLE.
		"""
		sigma_1 = np.sum((np.multiply(np.sqrt(val_trunc)[:, None], phi_x_dx[:, None])) ** 2)
		return sigma_1


	def get_params_2(self, val_trunc, eig_funcs_hyp, phi_x_dx):
		"""
		Computes the variance of the distribution conditioned on the hypothetical observation.
		"""
		sigma_inv = np.multiply(np.matmul(np.sqrt(val_trunc)[:, None], np.sqrt(val_trunc)[None, :]), np.matmul(eig_funcs_hyp[:, None], eig_funcs_hyp[None, :])) # using sherman morrison formula
		sigma_3 = np.eye(len(val_trunc)) - (sigma_inv / ((self.nugget ** 2) + (np.matmul(np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs_hyp[:, None]).T, np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs_hyp[:, None])))))
		C = np.multiply(sigma_3, np.matmul(np.multiply(np.sqrt(val_trunc)[:, None], (phi_x_dx)[:, None]), np.multiply(np.sqrt(val_trunc)[:, None], (phi_x_dx)[:, None]).T))
		sigma_2 = np.sum(C)
		mu_3 = np.matmul(sigma_3, np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs_hyp[:, None]))
		mat_coef = np.multiply(np.matmul(sigma_3, np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs_hyp[:, None])), np.multiply(np.sqrt(val_trunc)[:, None], phi_x_dx[:, None]))
		mat_coef_sq = np.matmul(mat_coef, mat_coef.T)
		mu1_mu2_sq_int = np.sum(mat_coef_sq)
		return mu1_mu2_sq_int, sigma_2


	def get_mu_1(self):
		"""
		
		"""
		al = self.alpha(noise=self.nugget**2)
		ek = np.ndarray(self.X.shape[0])[:, None]
		ells = self.lengthscales()
		s = self.ss()
		for i in xrange(self.X.shape[0]):
			e = 1.
			for j in xrange(self.X.shape[1]):
				e = e * xik(self.X[i, j], ells[j])
			ek[i, ] = e
		mu_1 =  s * np.sum(np.multiply(al, ek)) 
		return mu_1


	def get_mu_comp(self, model, X, Y):
		"""
		Computes the predicted mean for different sampling schemes.
		"""
		al = np.matmul(model.posterior.woodbury_inv, Y)
		ek = np.ndarray(X.shape[0])[:, None]
		ells = model.param_array[1: -1]
		s = model.param_array[0]
		for i in xrange(X.shape[0]):
			e = 1.
			for j in xrange(X.shape[1]):
				e = e * xik(X[i, j], ells[j])
			ek[i, ] = e
		mu_us =  s * np.sum(np.multiply(al, ek)) 
		return mu_us

	def get_sigma_comp(self, model):
		"""
		Variance of the QoI for comparable sampling methods.
		"""
		val_trunc, vec_trunc, W_h, x_d, phi_x_dx = self.eig_val_vec(model=model)
		return np.sum((np.multiply(np.sqrt(val_trunc)[:, None], phi_x_dx[:, None])) ** 2)


	def get_mu_1_minus_mu_2(self, val_trunc, eig_funcs, phi_x_dx, m_x, y_hyp):
		"""
		This method is not needed anymore.
		"""
		dim = len(val_trunc)
		mu = (1. * (y_hyp - m_x)  / np.multiply(np.sqrt(val_trunc), eig_funcs))
		mu_2 = np.sum(np.multiply(np.sqrt(val_trunc), np.multiply(mu, phi_x_dx))) 
		return mu_2


	def get_eig_funcs_hyp(self, x_hyp, w_j, x_d, val_trunc, vec_trunc, model=None):
		"""
		Computes the values of the eigenfunctions at a point ```x_hyp```.
		"""
		eig_funcs_hyp = np.zeros(len(val_trunc))
		k_x_d_x = (model[0].predict(np.vstack([x_d, np.atleast_2d(x_hyp)]), full_cov=True)[1][-1, :-1])[:, None]
		eig_funcs_hyp = np.sum(np.multiply(vec_trunc, np.multiply((w_j[w_j>0])[:, None], k_x_d_x)), axis=0) / val_trunc
		return eig_funcs_hyp


	def avg_kld_mean(self, x_hyp, val_trunc, vec_trunc, W_h, x_d, phi_x_dx, model=None):
		"""
		Take samples from the posterior for a hypothetical point and
		compute the average Kullbeck Liebler (KL) Divergence from the 
		augmented posterior to the current posterior.	
		For the mean of a black box function as the quantity of interest,
		the above distributions are Gaussian with known means and 
		computed variances.
		"""
		# These remain constant for a single optimization iteration
		w_j = W_h 
		m_x = self.model[0].predict(np.atleast_2d(x_hyp))[0][0]
		sigma_x = model[0].predict(np.atleast_2d(x_hyp), full_cov=True)[1][0][0]
		eig_funcs_hyp = self.get_eig_funcs_hyp(x_hyp, w_j, x_d, val_trunc, vec_trunc, model=model)
		sigma_1 = self.get_sigma_1(val_trunc, phi_x_dx)
		mu1_mu2_sq_int, sigma_2 = self.get_params_2(val_trunc, eig_funcs_hyp, phi_x_dx)
		kld = (1. * np.log(np.sqrt(sigma_1) / np.sqrt(sigma_2))) + (sigma_2 / (2. * sigma_1)) +  ((mu1_mu2_sq_int * (sigma_x)) /((self.nugget ** 4) * 2 * sigma_1)) - 0.5
		# print '>... ekld computed for x = ', x_hyp, '>... ekld = ', kld
		return kld


	def update_XY(self, x_best, y_obs):
		"""
		Augment the observed set with the newly added design and the
		corresponding function value.
		"""
		self.X = np.vstack([self.X, np.atleast_2d([x_best])])
		self.Y = np.vstack([self.Y, np.atleast_2d([y_obs])])


	def update_comp_models(self):
		x_grid = lhs(self.X.shape[1], 1000)
		# x_grid = design.latin_center(10, self.X.shape[1])
		pred_var = np.array([self.model_d[0].predict(np.atleast_2d(x))[1][0, 0] for x in x_grid])
		self.X_u = np.vstack([self.X_u, np.atleast_2d(x_grid[np.argmax(pred_var), :])])
		X_u_new = x_grid[np.argmax(pred_var)]
		self.Y_u = np.vstack([self.Y_u, np.atleast_2d(self.obj_func(X_u_new))])
		self.model_d = self.make_model(X=self.X_u, Y=self.Y_u, it=self.X_u.shape[0])
		X_r_new = x_grid[np.random.choice(len(x_grid)), :]
		self.X_r = np.vstack([self.X_r, np.atleast_2d(X_r_new)])
		self.Y_r = np.vstack([self.Y_r, np.atleast_2d(self.obj_func(X_r_new))])
		self.model_r = self.make_model(X=self.X_r, Y=self.Y_r, it=self.X_r.shape[0])

	def make_plots(self, it, kld, X_design, x_best, y_obs, model=None):
		# matplotlib.use('PS')
		sns.set_style("white")
		sns.set_context("paper")
		n = self.X.shape[0]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		x_grid = np.linspace(0, 1, 1000)[:, None]
		if self.true_func:
			y_grid = np.array([self.true_func(x_grid[i]) for i in xrange(x_grid.shape[0])])
			ax1.plot(x_grid, y_grid, '--' , c=sns.color_palette()[2], label='true function')
		y_pos = model[0].posterior_samples_f(x_grid, 1000, full_cov=True)
		y_m = np.percentile(y_pos, 50, axis=1)
		y_l = np.percentile(y_pos, 2.5, axis=1)
		y_u = np.percentile(y_pos, 97.5, axis=1)
		ax1.plot(x_grid, y_m, '-', c=sns.color_palette()[3], linewidth=2.0, label='posterior mean')
		ax1.fill_between(x_grid[:, 0], y_l, y_u, color=sns.xkcd_palette(["orange"]), label='uncertainty bands')
		if it==self.max_it-1:
			ax1.scatter(x_best, y_obs, s=50, c=sns.color_palette()[4])
			ax1.scatter(self.X[:, 0], self.Y[:, 0], s=50, c=sns.color_palette()[4])
		else:
			ax1.scatter(x_best, y_obs, marker='D', s=30, c=sns.color_palette()[1], label='latest experiment')
			ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='o', s=50, c=sns.color_palette()[4], label='observed data')
		idx = np.argsort(X_design[:, ], axis=0)[:, 0]
		ax2.plot(X_design[idx[:]], kld[idx[:]], c=sns.color_palette()[5], label='EKLD')
		# ax1.scatter(x_grid[np.argmax(pred_var_kl), 0], self.obj_func(x_grid[np.argmax(pred_var_kl)]), s=50, marker='o', c='yellow') # maximum variance point
		# ax2.plot(x_grid[:, 0], pred_var_kl/max(pred_var_kl)/1., c=sns.color_palette()[2], label='current uncertainty')
		# Plotting uncertainty sampling status
		# pred_var = np.array([self.model_d[0].predict(np.atleast_2d(x))[1][0, 0] for x in x_grid])
		# ax2.plot(x_grid[:, 0], pred_var/max(pred_var)/1., '--', c=sns.color_palette()[4], label='uncertainty sampling (US)')
		# y_d_pos = self.model_d[0].posterior_samples_f(x_grid, 1000, full_cov=True)
		# y_d_m = np.percentile(y_d_pos, 50, axis=1)
		# ax1.plot(x_grid, y_d_m, '-', c=sns.color_palette()[4], label='posterior mean US')
		# idx_us = np.argmax(pred_var)
		# ax1.scatter(x_grid[np.argmax(pred_var), 0], self.obj_func(x_grid[np.argmax(pred_var)]), marker='o', c='blue', label='US sample')
		# This is an ad hoc step here , not yet final if we need this.
		# if it==self.max_it - 1:
		# 	ax1.scatter(self.X_u, self.Y_u, marker='o', c='blue', label='US samples')
		# ax1.plot(self.quad_points, self.quad_points_weight, '--', c='black', label='weights')
		ax1.set_xlabel('x', fontsize=16)
		ax2.set_ylabel('EKLD', fontsize=16)
		ax2.set_ylim(0, 1)
		ax1.set_ylabel('objective', fontsize=16)
		ax1.set_xlim(self.bounds[0])
		ax1.tick_params(axis='both', which='both', labelsize=16)
		ax2.tick_params(axis='both', which='both', labelsize=16)
		plt.xticks(fontsize=16)
		plt.savefig(self.func_name + '_kld_' + str(it).zfill(len(str(self.max_it))) +'.pdf')
		plt.clf()

	def optimize(self, num_designs=1000, verbose=0, plots=0, comp=False):
		rel_kld = np.zeros(self.max_it)
		kld_all = np.ndarray((self.max_it, num_designs))
		mu_qoi = []
		sigma_qoi = []
		if comp:
			mu_us = []
			mu_rs = []
			sigma_us = []
			sigma_rs = []
		for i in xrange(self.max_it):
			print 'iteration no. ', i
			X_design = lhs(self.X.shape[1], num_designs)
			kld = np.zeros(X_design.shape[0])
			val_trunc, vec_trunc, W_h, x_d, phi_x_dx = self.get_val_vec
			print '>... current mean and variance of the QoI', self.get_mu_1(), self.get_sigma_1(val_trunc, phi_x_dx)
			mu_qoi.append(self.get_mu_1())
			sigma_qoi.append(self.get_sigma_1(val_trunc, phi_x_dx))
			if comp:
				mu_us.append(self.get_mu_comp(self.model_d[0], self.X_u, self.Y_u))
				sigma_us.append(self.get_sigma_comp(self.model_d))
				mu_rs.append(self.get_mu_comp(self.model_r[0], self.X_r, self.Y_r))
				sigma_rs.append(self.get_sigma_comp(self.model_r))
			for j in xrange(X_design.shape[0]):
				if verbose>0:
					print "> ... computing the EKLD for design no.", j
				kld[j] = self.avg_kld_mean(X_design[j, :], val_trunc, vec_trunc, W_h, x_d, phi_x_dx, model=self.model)
				kld_all[i, j] = kld[j]
			idx_best = np.argmax(kld)
			rel_kld[i, ] = max(kld)
			if verbose>0:
				print '>... maximum EKLD: ', max(kld)
			kld_all[i, :] = 1. * kld_all[i, :]
			x_best = X_design[idx_best, ]
			if verbose>0:
				print '>... run the next experiment at design: ', x_best
			y_obs = self.obj_func(x_best)
			if verbose>0:
				print '>... simulated the output at the selected design: ', y_obs
			if plots>0:
				self.make_plots(i, kld, X_design, x_best, y_obs, model=self.model)
			self.update_XY(x_best, y_obs)
			self.update_comp_models()
			if verbose>0:
				print '>... reconstructing surrogate model(s)'
			self.model = self.make_model(self.X, self.Y, it=self.X.shape[0])
			self.get_val_vec = self.eig_val_vec(model=self.model) # Generate different eigenvalues and eigenvectors as new data arrives
			if i == self.max_it-1:
				mu_qoi.append(self.get_mu_1())
				sigma_qoi.append(self.get_sigma_1(self.get_val_vec[0], self.get_val_vec[4]))
				if comp:
					mu_us.append(self.get_mu_comp(self.model_d[0], self.X_u, self.Y_u))
					sigma_us.append(self.get_sigma_comp(self.model_d))
					mu_rs.append(self.get_mu_comp(self.model_r[0], self.X_r, self.Y_r))
					sigma_rs.append(self.get_sigma_comp(self.model_r))
			if (max(kld) / max(rel_kld)) < self.kld_tol:
				print '>... relative ekld below specified tolerance ... stopping optimization now.'
				break
		if comp:
			return self.X, self.Y, self.Y_u, kld_all, X_design, mu_qoi, sigma_qoi, (mu_us, sigma_us, mu_rs, sigma_us)
		return self.X, self.Y, self.Y_u, kld_all, X_design, mu_qoi, sigma_qoi
