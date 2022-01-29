import matplotlib
matplotlib.use('agg')
import numpy as np
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math
import os
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pdb
import GPy
import time
import pickle
import itertools
from bode import *
# from cycler import cycler
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import beta
from pyDOE import *


class Ex2Func(object):
    def __init__(self, sigma_noise=lambda x: 0.5, mu1=0, sigma1=1, mu2=0.5, sigma2=1):
        self.sigma_noise = sigma_noise
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2

    def __call__(self, x):
    	return norm.pdf(x[0], loc=mu1, scale=sigma1) + norm.pdf(x[0], loc=mu2, scale=sigma2)
    	# return (1./np.sqrt(2 * math.pi)/self.sigma) * (np.exp(-((((x[0]-self.mu)/self.sigma)**2)/2))) + self.sigma_noise(x[0]) * np.random.randn()


if __name__=='__main__':
	np.random.seed(1263)
	n = 3
	dim = 1
	n_true = 1000000
	noise = 0
	noise_true = 0
	sigma_noise = eval('lambda x: ' + str(noise))
	sigma_true = eval('lambda x: ' + str(noise_true))
	mu1 = 0.2
	sigma1 = 0.05
	mu2 = 0.8
	sigma2 = 0.05
	objective_true = Ex2Func(sigma_noise=sigma_true, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)
	objective = Ex2Func(sigma_noise=sigma_noise, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)
	X_true = lhs(dim, samples=n_true, criterion='center')
	Y_true = np.array([objective(x) for x in X_true])[:, None]
	true_mean = np.mean(Y_true)
	print('true E[f(x)]: ', true_mean)
	num_it = 15
	X_init = lhs(dim, samples=n, criterion='center')
	Y_init = np.array([objective(x) for x in X_init])[:, None]
	out_dir = 'mcmc_ex2_n={0:d}_it={1:d}'.format(n, num_it)
	if os.path.isdir(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	num_samp = 30
	samp_pts = 100
	X_samp = np.linspace(0, 1, samp_pts)
	y_samp = np.zeros((num_samp, samp_pts))
	y_samp_hyp = np.zeros((num_samp, samp_pts))
	x_hyp = np.array([[.6]])
	num_quad_points = 500
	# quad_points = uniform.rvs(0, 1, size=num_quad_points)
	# quad_points_weight = uniform.pdf(quad_points)
	quad_points = lhs(dim, num_quad_points)
	# quad_points = np.linspace(0, 1, num_quad_points)[:, None] # Linearly space points
	# quad_points_weight = 1. / np.sqrt(1. - (quad_points[:, 0] ** 2)) # Right side heavy
	# quad_points_weight = 1./ np.sqrt(quad_points[:,0])			   # Left side heavy
	# quad_points_weight = 1./np.sqrt(abs(quad_points-0.5)) 		 	# Middle heavy
	quad_points_weight = np.ones(num_quad_points)
	kls = KLSampler(X_init, Y_init, x_hyp,
		model_kern=GPy.kern.RBF,
		bounds=[(0, 1)] * X_init.shape[1],
		obj_func=objective,
		true_func=objective_true,
		noisy=False,
		energy=0.95,
		nugget=1e-3,
		lengthscale=.2,
		variance=1.,
		kld_tol=1e-3,
		func_name=os.path.join(out_dir, 'ex2'),
		num_quad_points=num_quad_points,
		quad_points=quad_points,
		quad_points_weight=quad_points_weight,
		max_it=num_it,
		per_sampled=30,
		mcmc_model=True,
		mcmc_chains=8,
		mcmc_steps=5000,
		mcmc_burn=100,
		mcmc_model_avg=80,
		mcmc_thin=20,
		ego_iter=20,
		mcmc_parallel=8,
		initialize_from_prior=True,
		variance_prior=GammaPrior(a=1, scale=2),
		lengthscale_prior=ExponentialPrior(scale=0.7),
		noise_prior=JeffreysPrior())
	X, Y, X_u, kld, X_design, mu_qoi, sigma_qoi, comp_log = kls.optimize(num_designs=1000, verbose=1, plots=1, comp=True)
	np.save(os.path.join(out_dir,'X.npy'), X)
	np.save(os.path.join(out_dir,'Y.npy'), Y)
	np.save(os.path.join(out_dir,'X_u.npy'), X_u)
	np.save(os.path.join(out_dir,'kld.npy'), kld)
	np.save(os.path.join(out_dir,'mu_qoi.npy'), mu_qoi)
	np.save(os.path.join(out_dir,'sigma_qoi.npy'), sigma_qoi)
	with open(os.path.join(out_dir, "comp.pkl"), "wb") as f:
		pickle.dump(comp_log, f)
	kld_max = np.ndarray(kld.shape[0])
	err = np.zeros(num_it + 1) # Error in the QoI being estimated after each iteration
	for i in range(kld.shape[0]):
		kld_max[i] = max(kld[i, :])
	plt.plot(np.arange(len(kld_max)), kld_max/max(kld_max), color=sns.color_palette()[1])
	plt.xlabel('iterations', fontsize=16)
	plt.ylabel('relative maximum $G(x)$', fontsize=16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.savefig(os.path.join(out_dir,'ekld.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	size = 10000
	x = np.ndarray((size, len(mu_qoi)))
	x_us = np.ndarray((size, len(mu_qoi)))
	for i in range(len(mu_qoi)):
		x[:, i] = norm.rvs(loc=mu_qoi[i], scale=sigma_qoi[i] ** .5, size=size)
		x_us[:, i] = norm.rvs(loc=comp_log[0][i], scale=comp_log[1][i] ** .5, size=size)
	bp_ekld = plt.boxplot(x, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	pos = np.arange(n, n + len(mu_qoi))
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of $Q$', linewidth=4)
	plt.xlabel('no. of samples', fontsize=16)
	plt.ylabel('$Q$', fontsize=16)
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=16)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(out_dir, 'box.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	sns.distplot(norm.rvs(loc=mu_qoi[0], scale=sigma_qoi[0] ** .5, size=size), color=sns.color_palette()[1], label='initial distribution of $Q$', norm_hist=True)
	sns.distplot(norm.rvs(loc=mu_qoi[-1], scale=sigma_qoi[-1] ** .5, size=size), color=sns.color_palette()[0], label='final distribution of $Q$', norm_hist=True)
	plt.scatter(true_mean, 0, c=sns.color_palette()[2], label='true value of $Q$')
	plt.legend(fontsize=12)
	plt.xlabel('$Q$', fontsize=16)
	plt.ylabel('$p(Q| \mathrm{data})$', fontsize=16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.savefig(os.path.join(out_dir, 'dist.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of $Q$', linewidth=4)
	plt.plot(pos, mu_qoi, '-o',  color=sns.color_palette()[1], label='EKLD', markersize=10)
	plt.plot(pos, comp_log[0], '-*',  color=sns.color_palette()[2], label='uncertainty sampling', markersize=10)
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('no. of samples', fontsize=16)
	plt.ylabel('$\mathbb{E}[Q]$', fontsize=16)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(out_dir, 'comparison.png'), dpi=(900), figsize=(3.25, 3.25))
	quit()
