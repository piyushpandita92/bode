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
import pickle
from bode._sampler import *
# from cycler import cycler
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import cauchy
from pyDOE import *


class Ex1Func(object):
    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
        x = 6 * x
        return (4 * (1. - np.sin(x[0] + 8 * np.exp(x[0] - 7.))) \
                -10.) / 5. + self.sigma(x[0]) * np.random.randn()


if __name__=='__main__':
	np.random.seed(1333)
	# wire-problem true_mean=0.1736164
	# wire-problem true_mean=0.2694
	n = 3
	n_true = 100000
	dim = 1
	noise = 0
	noise_true = 0
	sigma = eval('lambda x: ' + str(noise))
	sigma_true = eval('lambda x: ' + str(noise_true))
	objective_true = Ex1Func(sigma=sigma_true)
	objective = Ex1Func(sigma=sigma)
	X_init = lhs(dim, samples=n, criterion='center')
	Y_init = np.array([objective(x) for x in X_init])[:, None]
	X_true = lhs(dim, samples=n_true, criterion='center')
	Y_true = np.array([objective(x) for x in X_true])[:, None]
	true_mean = np.mean(Y_true)
	print(true_mean)
	quit()
	print('true E[f(x)]: ', true_mean)
	num_quad_points = 500
	# quad_points = np.linspace(0, 1, num_quad_points)[:, None]
	quad_points = lhs(dim, num_quad_points)
	quad_points_weight = np.ones(num_quad_points)
	mu = 0.5
	sigma = 0.2
	num_it = 1
	out_dir = 'mcmc_ex1_n={0:d}_it={1:d}'.format(n, num_it)
	if os.path.isdir(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	# quad_points = uniform.rvs(0, 1, size=num_quad_points)
	# quad_points_weight = uniform.pdf(quad_points)
	# quad_points = norm.rvs(mu, sigma, size=num_quad_points)
	# quad_points_weight = norm.pdf(quad_points, mu, sigma)
	# quad_points = np.linspace(0.01, .99, num_quad_points)[:, None] 			# Linearly space points
	# quad_points_weight = 1. / np.sqrt(1. - (quad_points[:, 0] ** 2)) 	# Right side heavy
	# quad_points_weight = 1./ np.sqrt(quad_points[:, 0])			   		# Left side heavy
	# quad_points_weight = 1./np.sqrt(abs(quad_points-0.5)) 		 	# Middle heavy
	idx_quad = np.argsort(quad_points, axis=0)
	x_hyp = np.array([[.6]])
	kls = KLSampler(X_init, Y_init, x_hyp=False,
		model_kern=GPy.kern.RBF,
		bounds=[(0,1)] * X_init.shape[1],
		obj_func=objective,
		true_func=objective_true,
		noisy=False,
		nugget=1e-3,
		lengthscale=0.2,
		variance=1.,
		kld_tol=1e-6,
		func_name=os.path.join(out_dir,'ex1'),
		energy=0.95,
		num_quad_points=num_quad_points,
		quad_points=quad_points,
		quad_points_weight=quad_points_weight,
		max_it=num_it,
		per_sampled=20,
		mcmc_model=True,
		mcmc_chains=6,
		mcmc_model_avg=60,	#should be a multiple of number of chains
		mcmc_steps=1000,
		mcmc_burn=200,
		mcmc_thin=20,
		mcmc_parallel=8,
		ego_iter=20,
		initialize_from_prior=True,
		variance_prior=GammaPrior(a=1, scale=2),
		lengthscale_prior=ExponentialPrior(scale=0.7),
		noise_prior=JeffreysPrior())
	X, Y, X_u, kld, X_design, mu_qoi, sigma_qoi, comp_log = kls.optimize(num_designs=1000, verbose=1, plots=1, comp=True, comp_plots=False)
	np.save(os.path.join(out_dir,'X.npy'), X)
	np.save(os.path.join(out_dir,'Y.npy'), Y)
	np.save(os.path.join(out_dir,'X_u.npy'), X_u)
	np.save(os.path.join(out_dir,'kld.npy'), kld)
	np.save(os.path.join(out_dir,'mu_qoi.npy'), mu_qoi)
	np.save(os.path.join(out_dir,'sigma_qoi.npy'), sigma_qoi)
	with open(os.path.join(out_dir, "comp.pkl"), "wb") as f:
		pickle.dump(comp_log, f)
	kld_max = np.ndarray(kld.shape[0])
	for i in range(kld.shape[0]):
		kld_max[i] = max(kld[i, :])
	plt.plot(np.arange(len(kld_max)), kld_max / max(kld_max), color=sns.color_palette()[1])
	plt.xlabel('iterations', fontsize=16)
	plt.ylabel('relative maximum $G(x)$', fontsize=16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.savefig(os.path.join(out_dir,'ekld.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	size = 10000
	x = np.ndarray((size, len(mu_qoi)))
	x_us = np.ndarray((size, len(mu_qoi)))
	x_rs = np.ndarray((size, len(mu_qoi)))
	pos = np.arange(n, n + len(mu_qoi))
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the $Q$', linewidth=4)
	for i in range(len(mu_qoi)):
		x[:, i] = norm.rvs(loc=mu_qoi[i], scale=sigma_qoi[i] ** .5, size=size)
		x_us[:, i] = norm.rvs(loc=comp_log[0][i], scale=comp_log[1][i] ** .5, size=size)
	bp_ekld = plt.boxplot(x, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	plt.xlabel('no. of samples', fontsize=16)
	plt.ylabel('$Q$', fontsize=16)
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=16)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(out_dir, 'box.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	sns.distplot(norm.rvs(loc=mu_qoi[0], scale=sigma_qoi[0] ** .5, size=size), color=sns.color_palette()[1], label='initial distribution of $Q$', norm_hist=True)
	sns.distplot(norm.rvs(loc=mu_qoi[-1], scale=sigma_qoi[-1] ** .5, size=size), hist=True, color=sns.color_palette()[0], label='final distribution of $Q$', norm_hist=True)
	plt.scatter(true_mean, 0, c=sns.color_palette()[2], label='true value of the $Q$')
	plt.legend(fontsize=12)
	plt.xlabel('$Q$', fontsize=16)
	plt.ylabel('$p(Q|\mathrm{data})$', fontsize=16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.savefig(os.path.join(out_dir, 'dist.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	# Comparison plot
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the $Q$', linewidth=4)
	plt.plot(pos, mu_qoi, '-o',  color=sns.color_palette()[1], label='EKLD', markersize=10)
	plt.plot(pos, comp_log[0], '-*',  color=sns.color_palette()[2], label='uncertainty sampling', markersize=10)
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('no. of samples', fontsize=16)
	plt.ylabel('$\mathbb{E}[Q]$', fontsize=16)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(out_dir, 'comparison.png'), dpi=(900), figsize=(3.25, 3.25))
	quit()
