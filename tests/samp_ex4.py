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
import itertools
from sampler import *
from cycler import cycler
import pickle
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import cauchy
from pyDOE import *


class Ex1Func(object):
    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
	    """
	    Bilionis:	Where did you find this function?
	    PP:	from Knowles et al. ref 16 of the paper.
	    """
	    g = 100. * (((x[1:6] - 0.5) ** 2 - np.cos(2. * np.pi * (x[1:6] - 0.5))).sum() + 5.)
	    k = 0.5 * (1-x[0]) * (g + 1.)
	    return (k - 150.) / 80. 

class Ex2Func(object):
    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
	    """
	    Bilionis:	Where did you find this function?
	    PP:	from Knowles et al. ref 16 of the paper.
	    """
	    y = 10 * math.sin(np.pi * x[0] * x[1]) + (20 * ((x[2] - 5) ** 2)) + 10 * x[3] + 5 * x[4] 
	    return (y - 400) / 50.

if __name__ == '__main__':
	np.random.seed(1223)
	# Compute the true mean first
	n = 20
	dim = 5
	noise = 0
	noise_true = 0
	sigma = eval('lambda x: ' + str(noise))
	sigma_true = eval('lambda x: ' + str(noise_true))
	objective_true = Ex2Func(sigma=sigma_true)
	objective = Ex2Func(sigma=sigma)
	X_true = lhs(dim, 100000)
	Y_true = np.array([objective(x) for x in X_true])[:, None]
	# true_mean = -0.17925294951639562
	true_mean = np.mean(Y_true)
	print 'true E[f(x)]: ', true_mean
	X_init = lhs(dim, n)
	Y_init = np.array([objective(x) for x in X_init])[:, None]
	num_quad_points = 500
	quad_points = lhs(dim, num_quad_points)
	quad_points_weight = np.ones(num_quad_points)
	num_it = 45
	out_dir = 'mcmc_ex4_n={0:d}_it={1:d}'.format(n, num_it)
	if os.path.isdir(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	# quad_points = uniform.rvs(0, 1, size=num_quad_points)
	# quad_points_weight = uniform.pdf(quad_points)
	# quad_points = norm.rvs(mu, sigma, size=num_quad_points)
	# quad_points_weight = norm.pdf(quad_points, mu, sigma)
	# quad_points = np.linspace(0.01, .99, num_quad_points)[:, None] 			# Linearly space points
	# quad_points_weight = 1. / np.sqrt(1. - (quad_points[:, 0] ** 2)) 			# Right side heavy
	# quad_points_weight = 1. / np.sqrt(quad_points[:, 0])			   			# Left side heavy
	# quad_points_weight = 1. / np.sqrt(abs(quad_points-0.5)) 		 			# Middle heavy
	x_hyp = np.array([[0.6]])
	kls = KLSampler(X_init, Y_init, x_hyp, 
		model_kern=GPy.kern.RBF, 
		bounds=[(0,1)] * X_init.shape[1], 
		obj_func=objective,
		true_func=objective_true,
		noisy=False,
		nugget=1e-3,
		lengthscale=0.3,
		variance=1.,
		kld_tol=1e-5,
		func_name=os.path.join(out_dir,'ex4'),
		energy=0.95,
		num_quad_points=num_quad_points,
		quad_points=quad_points,
		quad_points_weight=quad_points_weight,
		max_it=num_it,
		per_sampled=20,
		num_opt_restarts=100,
		mcmc_model=True,
		mcmc_chains=14,
		mcmc_model_avg=56, 
		mcmc_steps=1000,
		mcmc_burn=200,
		mcmc_thin=20,
		mcmc_parallel=8,
		ego_iter=50,
		variance_prior=GammaPrior(a=1, scale=2),
		lengthscale_prior=ExponentialPrior(scale=0.7),
		noise_prior=JeffreysPrior())
	X, Y, X_u, kld, X_design, mu_qoi, sigma_qoi, comp_log = kls.optimize(num_designs=10000, verbose=0, plots=0, comp=True)
	np.save(os.path.join(out_dir,'X.npy'), X)
	np.save(os.path.join(out_dir,'Y.npy'), Y)
	np.save(os.path.join(out_dir,'X_u.npy'), X_u)
	np.save(os.path.join(out_dir,'kld.npy'), kld)
	np.save(os.path.join(out_dir,'mu_qoi.npy'), mu_qoi)
	np.save(os.path.join(out_dir,'sigma_qoi.npy'), sigma_qoi)
	with open(os.path.join(out_dir, "comp.pkl"), "wb") as f:
		pickle.dump(comp_log, f)
	kld_max = np.ndarray(kld.shape[0])
	for i in xrange(kld.shape[0]):
		kld_max[i] = max(kld[i, :])
	plt.plot(np.arange(len(kld_max)), kld_max / max(kld_max), color=sns.color_palette()[1])
	plt.xlabel('iterations', fontsize=16)
	plt.ylabel('relative maximum EKLD', fontsize=16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.savefig(os.path.join(out_dir, 'ekld.pdf'))
	plt.clf()
	size = 10000
	x = np.ndarray((size, len(mu_qoi)))
	x_us = np.ndarray((size, len(mu_qoi)))
	for i in xrange(len(mu_qoi)):
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