"""
Information acquisition for optimal design of experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
import math
import GPy
from pyDOE import *
from ._core import *
import itertools
import time
from copy import copy
from scipy.stats import multivariate_normal
from scipy.stats import norm
import emcee
import pdb
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


	def __init__(self, X, Y, x_hyp, obj_func, noisy, bounds,
		true_func=None,
		model_kern=GPy.kern.Matern32,
		num_opt_restarts=80,
		num_mc_samples=1000,
		num_quad_points=100,
		energy=0.95,
		nugget=1e-3,
		lengthscale=1.,
		variance=1.,
		N_avg=1000,
		kld_tol=1e-2,
		func_name='ex1',
		quad_points=None,
		quad_points_weight=None,
		max_it=50,
		ekld_nugget=1e-3,
		per_sampled=10,
		mcmc_acc_low=0.33,
		mcmc_acc_upp=0.7,
		mcmc_model=False,
		mcmc_steps=500,
		mcmc_final=.3,
		ego_init_perc=.2,
		mcmc_chains=10,
		mcmc_burn=100,
		mcmc_thin=30,
		mcmc_parallel=False,
		ego_iter=50,
		initialize_from_prior=True,
		variance_prior=GammaPrior(a=8, scale=1),
		lengthscale_prior=BetaPrior(a=2, b=5),
		noise_prior=JeffreysPrior(),
		mcmc_model_avg=50
		):
		"""
		:param X:		the inputs of the training data as an array.
		:param Y:		the outputs of the training data as an array.
		:param idx: 	set of indicies for which the
						effect function is needed.
		:param all:		if all lower level indicies are needed as well.
		"""
		assert X.ndim == 2
		self.X = X
		assert Y.ndim == 2
		self.Y = Y
		assert self.X.shape[0] == self.Y.shape[0]
		self.X_u = X
		self.Y_u = Y
		self.per_sampled = self.X.shape[1] * per_sampled
		self.dim = self.X.shape[1]
		self.num_obj = self.Y.shape[1]
		self.obj_func = obj_func
		self.true_func = true_func
		self.model_kern = model_kern
		self.nugget = nugget
		self.ekld_nugget = ekld_nugget
		self.lengthscale = lengthscale
		self.variance = variance
		self.noisy = noisy
		self.num_opt_restarts = num_opt_restarts
		self.mcmc_model = mcmc_model
		self.mcmc_steps = mcmc_steps
		self.mcmc_final = mcmc_final
		self.mcmc_chains = mcmc_chains
		self.mcmc_burn = mcmc_burn
		self.mcmc_thin = mcmc_thin
		self.mcmc_model_avg = mcmc_model_avg
		assert (self.mcmc_steps - self.mcmc_burn) / self.mcmc_thin >= (self.mcmc_model_avg / self.mcmc_chains)
		self.mcmc_parallel = mcmc_parallel
		self.mcmc_acc_low = mcmc_acc_low
		self.mcmc_acc_upp = mcmc_acc_upp
		self.variance_prior = variance_prior
		self.lengthscale_prior = lengthscale_prior
		self.noise_prior = noise_prior
		self.initialize_from_prior = initialize_from_prior
		self.ego_iter = ego_iter
		self._ego_init = ego_init_perc * self.ego_iter
		self._ego_seq = (1 - ego_init_perc) * self.ego_iter
		self.model = self.make_model(self.X, self.Y, it=0, mcmc=self.mcmc_model)
		self.model_d = self.make_model(self.X_u, self.Y_u, it=0, mcmc=self.mcmc_model)
		self.all_p = {}
		self.num_mc_samples = num_mc_samples
		self.num_quad_points = num_quad_points
		self.energy = energy
		self.x_hyp = x_hyp
		if self.x_hyp:
			if not self.mcmc_model:
				self.y_hyp = self.model[0].posterior_samples(self.x_hyp, 1)[0, 0].copy()
		if quad_points is None:
			self.quad_points = np.linspace(0, 1, self.num_quad_points)
			self.quad_points_weight = np.eye(self.num_quad_points)
		else:
			self.quad_points = quad_points
			self.quad_points_weight = quad_points_weight
		if not self.mcmc_model:
			self.get_val_vec = self.eig_val_vec(model=self.model)
		self.N_avg = N_avg
		self.bounds = bounds
		self.kld_tol = kld_tol
		self.__name__ = func_name
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

	def get_log_prior(self, param):
		"""
		Returns the sum log-pdf of the parameters
		"""
		var_log_prior = self.variance_prior(param[0])
		ell_log_prior = 0
		for j in range(self.X.shape[1]):
			ell_log_prior += self.lengthscale_prior(param[j+1])
		if self.noisy:
			noise_log_prior = self.noise_prior(param[-1])
			return ell_log_prior + var_log_prior + noise_log_prior
		else:
			return ell_log_prior + var_log_prior

	def get_likelihood(self, param, model, X, Y):
		"""
		Log likelihood of the model
		"""
		model.kern.variance = param[0]
		if self.noisy:
			model.kern.lengthscale = param[1: -1]
			model.likelihood.variance = param[-1] ** 2
		else:
			model.kern.lengthscale = param[1:]
			model.likelihood.variance = self.nugget ** 2
		return model.log_likelihood()


	def lnprob(self, param, model, X, Y):
		if np.any(param<0):
			return -np.inf
		return self.get_likelihood(param, model, X, Y) + self.get_log_prior(param)


	def make_model(self, X, Y, it=0, mcmc=False, last_model=None, nugget=None):
		"""
		Trains the surrogate model.
		"""
		m = Y.shape[1]
		surrogates = []
		for i in range(m):
			if mcmc:
				model = GPy.models.GPRegression(X, Y, self.model_kern(input_dim=X.shape[1], ARD=True))
				if self.noisy:
					ndim, nchains = X.shape[1] + 2, self.mcmc_chains
					if it==0:
						if self.initialize_from_prior:
							init_pos = [np.hstack([self.variance_prior.sample(size=1), self.lengthscale_prior.sample(size=X.shape[1]), self.noise_prior.sample(size=1)]) for j in range(nchains)]
						else:
							init_pos = [np.hstack([self.variance * np.random.rand(1), self.lengthscale * np.random.rand(X.shape[1]), self.nugget * np.random.rand(1)]) for j in range(nchains)]
					else:
						init_pos = [last_model[0][(i + 1) * (self.mcmc_model_avg // self.mcmc_chains) - 1, :] for i in range(self.mcmc_chains)]
				else:
					ndim, nchains = X.shape[1] + 1, self.mcmc_chains
					if it==0:
						if self.initialize_from_prior:
							init_pos = [np.hstack([self.variance_prior.sample(size=1), self.lengthscale_prior.sample(size=X.shape[1])]) for j in range(nchains)]
						else:
							init_pos = [np.hstack([self.variance * np.random.rand(1), self.lengthscale * np.random.rand(X.shape[1]), self.nugget * np.random.rand(1)]) for j in range(nchains)]
					else:
						# pdb.set_trace()
						init_pos = [last_model[0][(i + 1) * (self.mcmc_model_avg // self.mcmc_chains) - 1, :] for i in range(self.mcmc_chains)]

				sampler = emcee.EnsembleSampler(nchains, ndim, self.lnprob, args=(model, X, Y))
				sampler.run_mcmc(init_pos, self.mcmc_steps)
				print('>... acceptance ratio(s):', sampler.acceptance_fraction)
				samples_thin = sampler.chain[:, self.mcmc_burn:self.mcmc_steps:self.mcmc_thin, :]
				surrogates.append(samples_thin[:, - int(self.mcmc_model_avg / self.mcmc_chains):, :].reshape((-1, ndim)))
				return surrogates
			else:
				# try:
				model = GPy.models.GPRegression(X, Y, self.model_kern(input_dim=X.shape[1], ARD=True))
				model.likelihood.variance.constrain_fixed(self.ekld_nugget ** 2)
				model.optimize_restarts(num_restarts=self.num_opt_restarts, verbose=False)
				# except:
				# 	model = GPy.models.GPRegression(X, Y, self.model_kern(input_dim=X.shape[1], ARD=True))
				# 	model.likelihood.variance.constrain_fixed(self.ekld_nugget ** 2)
				# 	model.optimize_restarts(num_restarts=self.num_opt_restarts, verbose=False)
				# return model
			# print model, model.kern.lengthscale
			surrogates.append(model)
		return surrogates


	def eig_func(self, x, w_j, x_d, val_trunc, vec_trunc, model=None):
		"""
		Constructing the eigenfunctions for the given eigenvalues at ```x```.
		"""
		k_x_d_x = (model[0].predict(np.vstack([x_d, np.atleast_2d(x)]), include_likelihood=False, full_cov=True)[1][-1, :-1])[:, None]
		eig_func = (1. / val_trunc) * np.sum(np.multiply(np.multiply(w_j, vec_trunc), k_x_d_x))
		return eig_func


	def eig_val_vec(self, model=None):
		"""
		Eigendecomposition of the ```B``` matrix in equation 15.88 of UQ book chapter.
		"""
		x_d = self.quad_points
		p_x_d = self.quad_points_weight
		K_x_d = model.predict(x_d, full_cov=True, include_likelihood=False)[1]
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
		phi_x_dx = np.array([np.mean(np.sum(np.multiply(np.multiply(vec_trunc[:, j][:, None], (np.sqrt(((p_x_d / np.sum(self.quad_points_weight)))))[:, None]), K_x_d), axis=0), axis=0) for j in range(vec_trunc.shape[1])]) / val_trunc
		# phi_x_dx = np.mean(np.multiply(vec_trunc, (np.sqrt(((p_x_d / np.sum(self.quad_points_weight)))))[:, None]), axis=0)
		# phi_x_dx = self.get_phi_x_dx(val_trunc, vec_trunc, W_h, x_d, p_x_d)
		return val_trunc, vec_trunc, W_h, x_d, phi_x_dx


	def sample_xi_hyp(self, dim, val_trunc, eig_funcs, m_x, y_hyp, model):
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
		sigma_inv_2 = sigma_inv / (model[0].likelihood.variance)
		sigma_inv_1 = np.eye(dim)
		sigma_3 = np.linalg.inv(sigma_inv_1 + sigma_inv_2)
		mu_3 = ((y_hyp - m_x)/ (model.likelihood.variance)) * np.matmul(sigma_3, np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs[:, None]))
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
		for j in range(len(x_grid)):
			x = x_grid[j]
			for i in range(eig_funcs_f.shape[1]):
				eig_funcs_f[j, i] = self.eig_func(x, (w_j[w_j>0])[:, None], x_d, val_trunc[i, ], (vec_trunc[:, i])[:, None])
			#print '>... Sampled the eigenfunction at', time.time() - clock_time, 'seconds'
			samp[j, ] =  self.model[0].predict(np.atleast_2d(x), include_likelihood=False)[0][0] + np.sum(np.multiply(np.multiply(sample_xi, (np.sqrt(val_trunc))[:, None]), eig_funcs_f[j, :][:, None])).copy()
		return samp, val_trunc, eig_funcs_f


	def obj_est_hyp(self, x_grid, x_hyp):
		# Repeating the process after adding the hypothetical observation to the data set
		y_hyp = self.y_hyp
		m_x_hyp = self.model[0].predict(x_hyp, include_likelihood=False)[0][0]
		samp_hyp = np.zeros(len(x_grid))
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		eig_funcs_hyp = np.zeros(len(val_trunc))
		eig_funcs_f_hyp = np.zeros((len(x_grid), len(val_trunc)))
		for i in range(len(val_trunc)):
			eig_funcs_hyp[i, ] = self.eig_func(x_hyp, (w_j[w_j>0])[:, None], x_d, val_trunc[i, ], (vec_trunc[:, i])[:, None] )
		sample_xi_hyp = self.sample_xi_hyp(val_trunc.shape[0], val_trunc, eig_funcs_hyp, m_x_hyp, y_hyp, self.model)
		for j in range(len(x_grid)):
			x = x_grid[j]
			for i in range(eig_funcs_f_hyp.shape[1]):
				eig_funcs_f_hyp[j, i] = self.eig_func(x, (w_j[w_j>0])[:, None], x_d, val_trunc[i, ], (vec_trunc[:, i])[:, None])
			samp_hyp[j, ] = self.model[0].predict(np.atleast_2d(x), include_likelihood=False)[0][0] + np.sum(np.multiply(np.multiply(sample_xi_hyp, (np.sqrt(val_trunc))[:, None]), (eig_funcs_f_hyp[j, :])[:, None]))
		return samp_hyp, y_hyp, val_trunc, eig_funcs_f_hyp

	def get_params_2(self, model, X, Y, x_hyp):
		ells = model.kern.lengthscale
		ss = model.kern.variance
		sigma_1 = self.get_sigma_1(model, X, Y)
		ek = xik(X, ells, ss)[:, None]
		k_X_x_hyp = model.kern.K(X, np.atleast_2d(x_hyp))
		k_x_x_hyp = model.predict(np.atleast_2d(x_hyp), full_cov=False, include_likelihood=True)[1]
		xi_x_hyp = xik(np.atleast_2d(x_hyp), ells, ss)
		v_x_hyp =  xi_x_hyp - np.matmul(np.matmul(ek.T, model.posterior.woodbury_inv), k_X_x_hyp)
		sigma_2 = sigma_1 - (v_x_hyp ** 2) / k_x_x_hyp
		mu1_mu2_sq_int = (v_x_hyp) ** 2
		return mu1_mu2_sq_int.item(), sigma_2.item() # All scalars now

	def get_sigma_1(self, model, X, Y):
		ells = model.kern.lengthscale
		ss = model.kern.variance
		sigma_0 = bk(ells, ss)
		ek = xik(X, ells, ss)[:, None]
		sigma_1 = sigma_0 - np.matmul(np.matmul(ek.T, model.posterior.woodbury_inv), ek)
		return sigma_1.item() 			#	 Scalar


	def get_mu_1(self, model, X, Y):
		"""
		Mean of the QoI.
		"""
		al = np.matmul(model.posterior.woodbury_inv, Y)
		ells = model.kern.lengthscale
		ss = model.kern.variance
		ek = xik(X, ells, ss)[:, None]
		mu_1 = np.matmul(al.T, ek)
		return mu_1.item()	 			#	Scalar


	def get_eig_funcs_hyp(self, x_hyp, w_j, x_d, val_trunc, vec_trunc, model=None):
		"""
		Computes the values of the eigenfunctions at a point ```x_hyp```.
		"""
		eig_funcs_hyp = np.zeros(len(val_trunc))
		k_x_d_x = (model.predict(np.vstack([x_d, np.atleast_2d(x_hyp)]), full_cov=True, include_likelihood=False)[1][-1, :-1])[:, None]
		eig_funcs_hyp = np.sum(np.multiply(vec_trunc, np.multiply((w_j[w_j>0])[:, None], k_x_d_x)), axis=0) / val_trunc
		return eig_funcs_hyp

	def get_mu_sigma(self, model, X, Y):
		if self.mcmc_model:
			mu_1 = 0
			sigma_1 = 0
			params = self.model[0]
			for k in range(params.shape[0]):
				mcmc_model = self.make_mcmc_model(params[k, :], X, Y)
				val_trunc, vec_trunc, W_h, x_d, phi_x_dx = self.eig_val_vec(model=mcmc_model)
				mu_1 += self.get_mu_1(mcmc_model, X, Y)
				sigma_1 += self.get_sigma_1(mcmc_model, X, Y)
			return mu_1 / params.shape[0], sigma_1 / params.shape[0]
		else:
			val_trunc, vec_trunc, W_h, x_d, phi_x_dx = self.eig_val_vec(model=model)
			return self.get_mu_1(model, X, Y), self.get_sigma_1(val_trunc, phi_x_dx)

	def make_mcmc_model(self, param, X, Y):
		"""
		build the GP model for the given parameters
		"""
		model = GPy.models.GPRegression(X, Y, self.model_kern(input_dim=X.shape[1], ARD=True))
		if self.noisy:
			model.kern.variance.fix(param[0])
			model.kern.lengthscale.fix(param[1: -1])
			model.likelihood.variance.fix(param[-1] ** 2)
		else:
			model.kern.variance.fix(param[0])
			model.kern.lengthscale.fix(param[1:])
			model.likelihood.variance.constrain_fixed(self.nugget ** 2)
		return model

	def avg_kld_mean(self, x_hyp, X, Y, model=None):
		"""
		Take samples from the posterior for a hypothetical point and
		compute the average Kullbeck Liebler (KL) Divergence from the
		augmented posterior to the current posterior.
		For the mean of a black box function as the quantity of interest,
		the above distributions are Gaussian with known means and
		computed variances.
		"""
		# These remain constant for a single optimization iteration
		m_x = model.predict(np.atleast_2d(x_hyp), include_likelihood=False)[0][0]
		sigma_x = model.predict(np.atleast_2d(x_hyp), include_likelihood=False, full_cov=True)[1][0][0]
		sigma_1 = self.get_sigma_1(model, X, Y)
		# print sigma_1
		mu1_mu2_sq_int, sigma_2 = self.get_params_2(model, X, Y, x_hyp)
		# print sigma_2
		kld = (1. * np.log(np.sqrt(sigma_1) / np.sqrt(sigma_2))) + (sigma_2 / (2. * sigma_1)) +  ((mu1_mu2_sq_int) / (sigma_1) / ((sigma_x + model.likelihood.variance))) * 0.5 - 0.5
		return kld

	def mcmc_kld(self, x_hyp, model):
		"""
		MCMC averaged value of the EKLD.
		"""
		params = model[0]
		kld_j = np.ndarray((params.shape[0], 1))
		for i in range(params.shape[0]):
			mcmc_model = self.make_mcmc_model(params[i, :], self.X, self.Y)
			kld_j[i] =  self.avg_kld_mean(x_hyp, self.X, self.Y, model=mcmc_model)
		return np.mean(np.log(kld_j)), np.var(np.log(kld_j))

	def update_XY(self, x_best, y_obs):
		"""
		Augment the observed set with the newly added design and the
		corresponding function value.
		"""
		self.X = np.vstack([self.X, np.atleast_2d(x_best)])
		self.Y = np.vstack([self.Y, np.atleast_2d(y_obs)])

	def update_comp_models(self, it):
		"""
		updates the US and RS GPs.
		"""
		x_grid = lhs(self.X.shape[1], 1000)
		if self.mcmc_model:
			params = self.model_d[0]
			pred_var = np.ndarray((x_grid.shape[0], params.shape[0]))
			for i in range(params.shape[0]):
				mcmc_model_d = self.make_mcmc_model(params[i, :], self.X_u, self.Y_u)
				pred_var[:, i] = np.array([mcmc_model_d.predict(np.atleast_2d(x), include_likelihood=False)[1][0, 0] for x in x_grid])
			pred_var = pred_var.mean(axis=1)
			self.X_u = np.vstack([self.X_u, np.atleast_2d(x_grid[np.argmax(pred_var), :])])
			X_u_new = x_grid[np.argmax(pred_var)]
			self.Y_u = np.vstack([self.Y_u, np.atleast_2d(self.obj_func(X_u_new))])
			self.model_d = self.make_model(X=self.X_u, Y=self.Y_u, it=it, mcmc=self.mcmc_model, last_model=self.model_d)
		else:
			pred_var = np.array([self.model_d.predict(np.atleast_2d(x), include_likelihood=False)[1][0, 0] for x in x_grid])
			self.X_u = np.vstack([self.X_u, np.atleast_2d(x_grid[np.argmax(pred_var), :])])
			X_u_new = x_grid[np.argmax(pred_var)]
			self.Y_u = np.vstack([self.Y_u, np.atleast_2d(self.obj_func(X_u_new))])
			self.model_d = self.make_model(X=self.X_u, Y=self.Y_u, it=it, mcmc=self.mcmc_model)

	def make_plots(self, it, kld, X_design, x_best, y_obs, model=None, ekld_model=None, comp_plots=False):
		sns.set_style("white")
		sns.set_context("paper")
		n = self.X.shape[0]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		idx = np.argsort(X_design[:, ], axis=0)[:, 0]
		x_grid = X_design[idx[:]]
		if self.true_func:
			y_grid = np.array([self.true_func(x_grid[i]) for i in range(x_grid.shape[0])])
			true = ax1.plot(x_grid, y_grid, '-' , c=sns.color_palette()[0], linewidth=4.0, label='true function')
		if self.mcmc_model:
			params = model[0]
			mcmc_model = self.make_mcmc_model(params[0, :], self.X, self.Y)
			y_pos = mcmc_model.posterior_samples_f(x_grid, 500, full_cov=True)[:,0,:]
			for i in range(1, params.shape[0]):
				mcmc_model = self.make_mcmc_model(params[i, :], self.X, self.Y)
				y_pos = np.hstack([y_pos, mcmc_model.posterior_samples_f(x_grid, 500, full_cov=True)[:,0,:]])
		else:
			y_pos = model.posterior_samples_f(x_grid, 1000, full_cov=True)[:,0,:]
		# pdb.set_trace()
		y_m = np.percentile(y_pos, 50, axis=1)
		y_l = np.percentile(y_pos, 2.5, axis=1)
		y_u = np.percentile(y_pos, 97.5, axis=1)
		obj = ax1.plot(x_grid, y_m, '--', c=sns.color_palette()[1], linewidth=3.0, label='physical response GP', zorder=3)
		ax1.fill_between(x_grid[:, 0], y_l, y_u, color=sns.color_palette()[1], alpha=0.25, zorder=3)
		if self.mcmc_model:
			idx = np.argsort(X_design[:, ], axis=0)[:, 0]
			y_ekld_pos = np.exp(ekld_model[0].posterior_samples_f(X_design, 1000, full_cov=True)[:,0,:])
			y_ekld_m = np.percentile(y_ekld_pos, 50, axis=1)
			y_ekld_l = np.percentile(y_ekld_pos, 2.5, axis=1)
			y_ekld_u = np.percentile(y_ekld_pos, 97.5, axis=1)
			ekld = ax2.plot(X_design[idx[:]], y_ekld_m[idx[:]], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], label='EKLD GP', zorder=5)
			ax2.fill_between(X_design[idx[:], 0], y_ekld_l[idx[:]], y_ekld_u[idx[:]], color=sns.color_palette()[2], alpha=0.25, zorder=5)
		else:
			idx = np.argsort(X_design[:, ], axis=0)[:, 0]
			ax2.plot(X_design[idx[:]], kld[idx[:]], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], label='EKLD')
		if it==self.max_it-1:
			ax1.scatter(x_best, y_obs, marker='X', s=80, c='black', zorder=10)
			dat = ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
		else:
			obs = ax1.scatter(x_best, y_obs, marker='D', s=80, c=sns.color_palette()[3], label='latest experiment', zorder=10)
			dat = ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
		if comp_plots:
			# Now we make the plots for US
			params = self.model_d[0]
			pred_var = np.ndarray((x_grid.shape[0], params.shape[0]))
			for i in range(params.shape[0]):
				mcmc_model = self.make_mcmc_model(params[i, :], self.X_u, self.Y_u)
				pred_var[:, i] = np.array([mcmc_model.predict(np.atleast_2d(x), include_likelihood=False)[1][0, 0] for x in x_grid])
			pred_var = pred_var.mean(axis=1)
			ax2.plot(x_grid, pred_var / max(pred_var), linestyle=':', linewidth=4, color='black', label='uncertainty sampling')
			ax1.scatter(x_grid[np.argmax(pred_var), :], self.obj_func(x_grid[np.argmax(pred_var), :]), marker='*', color='red', s=40)
			ax1.scatter(self.X_u, self.Y_u, marker='X', color='green', s=40)
		ax1.set_xlabel('$x$', fontsize=16)
		ax2.set_ylabel('$G(x)$', fontsize=16)
		ax2.set_ylim(0, 1)											# This fixing of the limits can be a bit tricky
		lines, labels = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc=9, fontsize=12)
		plt.xticks(fontsize=16)
		ax1.tick_params(axis='both', which='both', labelsize=16)
		ax2.tick_params(axis='both', which='both', labelsize=16)
		ax2.spines['right'].set_color(sns.color_palette()[2])
		ax2.yaxis.label.set_color(sns.color_palette()[2])
		ax2.tick_params(axis='y', colors=sns.color_palette()[2])
		ax1.set_ylabel('$f(x)$', fontsize=16)
		ax1.set_xlim(self.bounds[0])
		plt.savefig(self.__name__ + '_kld_' + str(it+1).zfill(len(str(self.max_it))) +'.png', dpi=(900), figsize=(3.25, 3.25))
		plt.clf()


	def optimize(self, num_designs=1000, verbose=0, plots=0, comp=False, comp_plots=False):
		rel_kld = np.zeros(self.max_it)
		if self.mcmc_model:
			kld_all = np.ndarray((self.max_it, num_designs))
		else:
			kld_all = np.ndarray((self.max_it, num_designs))
		mu_qoi = []
		sigma_qoi = []
		models = []
		if comp:
			mu_us = []
			sigma_us = []
			models_us = []
		for i in range(self.max_it):
			print('iteration no. ', i + 1)
			X_design = lhs(self.X.shape[1], num_designs, criterion='center')
			kld = np.zeros(X_design.shape[0])
			mu, sigma = self.get_mu_sigma(self.model, self.X, self.Y)
			mu_qoi.append(mu)
			sigma_qoi.append(sigma)
			models.append(self.model)
			models_us.append(self.model_d)
			print('>... current mean and variance of the QoI for EKLD', mu, sigma)
			if comp:
				mu_qoi_us, sigma_qoi_us = self.get_mu_sigma(self.model_d, self.X_u, self.Y_u)
				mu_us.append(mu_qoi_us)
				sigma_us.append(sigma_qoi_us)
				print('>... current mean and variance of the QoI for US', mu_qoi_us, sigma_qoi_us)
			if self.mcmc_model:
				num_lhs_ego = int(self._ego_init)
				num_seq_ego = int(self._ego_seq)
				ekld_mu = np.ndarray((num_lhs_ego, 1))
				ekld_var = np.ndarray((num_lhs_ego, 1))
				ego_lhs = lhs(self.X.shape[1], num_lhs_ego, criterion='center')
				print('>... computing the EKLD for the initial EGO designs.')
				for it in range(num_lhs_ego):
					ekld_mu[it, ], ekld_var[it, ] = self.mcmc_kld(ego_lhs[it, :], model=self.model)
				ego_model = self.make_model(ego_lhs, ekld_mu, mcmc=False)
				print('>... done.')
				for _ in range(num_seq_ego):
					X_design = lhs(self.X.shape[1], num_designs)
					ego_max = max(ego_model[0].predict(ego_lhs, full_cov=False, include_likelihood=False)[0])
					mu_ekld, sigma_ekld = ego_model[0].predict(X_design, full_cov=False, include_likelihood=False)
					ei_ekld = ei(mu_ekld, sigma_ekld, ego_max, mode="max")
					x_best_ego = X_design[np.argmax(ei_ekld), :]
					y_obs_ego, y_var_ego = self.mcmc_kld(np.atleast_2d(x_best_ego), model=self.model)
					# print x_best_ego, y_obs_ego
					ego_lhs = np.vstack([ego_lhs, np.atleast_2d(x_best_ego)])
					ekld_mu = np.vstack([ekld_mu, np.atleast_2d(y_obs_ego)])
					ekld_var = np.vstack([ekld_var, np.atleast_2d(y_var_ego)])
					print('>... reconstructing EKLD surrogate model.')
					ego_model = self.make_model(ego_lhs, ekld_mu, mcmc=False)
					print('>... done.')
			else:
				val_trunc, vec_trunc, W_h, x_d, phi_x_dx = self.eig_val_vec(model=self.model)
				for j in range(X_design.shape[0]):
					if verbose>0:
						print("> ... computing the EKLD for design no.", j)
					kld[j] = self.avg_kld_mean(X_design[j, :], val_trunc, vec_trunc, W_h, x_d, phi_x_dx, model=self.model)
					kld_all[i, j] = kld[j]
			if self.mcmc_model:
				idx_best = np.argmax(ekld_mu)
				x_best = ego_lhs[idx_best, ]
				kld = np.exp(mu_ekld[:, 0]) 	# Applying the transformation here
				rel_kld[i, ] = max(np.exp(mu_ekld))
				kld_all [i, :] = np.exp(mu_ekld[:, 0])
				if verbose>0:
					print('>... maximum EKLD', max(np.exp(ekld_mu)))
			else:
				idx_best = np.argmax(kld)
				rel_kld[i, ] = max(kld)
				kld_all[i, :] = 1. * kld_all[i, :]
				x_best = X_design[idx_best, ]
				if verbose>0:
					print('>... maximum EKLD: ', max(kld))
			if verbose>0:
				print('>... run the next experiment at design: ', x_best)
			y_obs = self.obj_func(x_best)
			if verbose>0:
				print('>... simulated the output at the selected design: ', y_obs)
			if plots>0:
				if self.mcmc_model:
					self.make_plots(i, np.exp(mu_ekld), X_design, x_best, y_obs, model=self.model, ekld_model=ego_model, comp_plots=comp_plots)
				else:
					self.make_plots(i, kld, X_design, x_best, y_obs, model=self.model, comp_plots=comp_plots)
			self.update_XY(x_best, y_obs)
			if comp:
				self.update_comp_models(it=i+1)
			if verbose>0:
				print('>... reconstructing surrogate model(s)')
			self.model = self.make_model(self.X, self.Y, it=i+1, mcmc=self.mcmc_model, last_model=self.model)
			if not self.mcmc_model:
				self.get_val_vec = self.eig_val_vec(model=self.model) # Generate different eigenvalues and eigenvectors as new data arrives
			if i == self.max_it-1:
				mu, sigma = self.get_mu_sigma(self.model, self.X, self.Y)
				mu_qoi.append(mu)
				sigma_qoi.append(sigma)
				models.append(self.model)
				if comp:
					mu_qoi_us, sigma_qoi_us = self.get_mu_sigma(self.model_d, self.X_u, self.Y_u)
					mu_us.append(mu_qoi_us)
					sigma_us.append(sigma_qoi_us)
					models_us.append(self.model_d)
			if (max(kld) / max(rel_kld)) < self.kld_tol:
				print('>... relative ekld below specified tolerance ... stopping optimization now.')
				break
		if comp:
			if self.mcmc_model:
				return self.X, self.Y, self.X_u, kld_all, X_design, mu_qoi, sigma_qoi, (mu_us, sigma_us, models, models_us)
			else:
				return self.X, self.Y, self.X_u, kld_all, X_design, mu_qoi, sigma_qoi, (mu_us, sigma_us)
		return self.X, self.Y, self.Y_u, kld_all, X_design, mu_qoi, sigma_qoi
