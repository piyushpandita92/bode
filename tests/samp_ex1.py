import numpy as np
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
import matplotlib.pyplot as plt
import math
import os
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pdb
import GPy
import time
import itertools
import design
import bode
from cycler import cycler
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
                -10.)/5. + self.sigma(x[0]) * np.random.randn()


if __name__=='__main__':
	np.random.seed(1223)
	n = 5
	noise = 0
	noise_true = 0
	sigma = eval('lambda x: ' + str(noise))
	sigma_true = eval('lambda x: ' + str(noise_true))
	objective_true = Ex1Func(sigma=sigma_true)
	objective = Ex1Func(sigma=sigma)
	X_init = design.latin_center(n, 1)
	Y_init = np.array([objective(x) for x in X_init])[:, None]	
	num_quad_points = 500
	quad_points = np.linspace(0, 1, num_quad_points)[:, None]
	quad_points_weight = np.ones(num_quad_points)
	mu = 0.5
	sigma = 0.2
	out_dir = 'ex1_n={0:d}_sigma={1:s}'.format(n, str(noise))
	if os.path.isdir(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	idx_quad = np.argsort(quad_points, axis=0)
	num_it = 50
	x_hyp = np.array([[.6]])
	kls = bode.KLSampler(X_init, Y_init, x_hyp, 
		model_kern=GPy.kern.RBF, 
		bounds=[(0,1)] * X_init.shape[1], 
		obj_func=objective, 
		true_func=objective_true,
		noisy=False,
		nugget=1e-3,
		lengthscale=0.2,
		kld_tol=1e-4,
		func_name=os.path.join(out_dir,'ex1'),
		energy=0.95,
		num_quad_points=num_quad_points,
		quad_points=quad_points,
		quad_points_weight=quad_points_weight,
		max_it=num_it,
		per_sampled=10)
	X, Y, Y_u, kld, X_design, mu_qoi, sigma_qoi, comp_log = kls.optimize(num_designs=500, verbose=1, plots=1, comp=True)
	np.save(os.path.join(out_dir,'X.npy'), X)
	np.save(os.path.join(out_dir,'Y.npy'), Y)
	np.save(os.path.join(out_dir,'Y_u.npy'), Y_u)
	np.save(os.path.join(out_dir,'kld.npy'), kld)
	np.save(os.path.join(out_dir,'mu_qoi.npy'), mu_qoi)
	np.save(os.path.join(out_dir,'sigma_qoi.npy'), sigma_qoi)
	kld_max = np.ndarray(kld.shape[0])
	true_mean = -1.35997716767 # Approxmiate True Mean 
	for i in xrange(kld.shape[0]):
		kld_max[i] = max(kld[i, :])
	plt.plot(np.arange(len(kld_max)), kld_max/max(kld_max), color=sns.color_palette()[1])
	plt.xlabel('iterations')
	plt.ylabel('relative maximum EKLD')
	plt.savefig(os.path.join(out_dir,'ekld.pdf'))
	plt.clf()
	size = 10000
	x = np.ndarray((size, len(mu_qoi)))
	x_us = np.ndarray((size, len(mu_qoi)))
	x_rs = np.ndarray((size, len(mu_qoi)))
	for i in xrange(len(mu_qoi)):
		x[:, i] = norm.rvs(loc=mu_qoi[i], scale=sigma_qoi[i] ** .5, size=size)
		x_us[:, i] = norm.rvs(loc=comp_log[0][i], scale=comp_log[1][i] ** .5, size=size)
		x_rs[:, i] = norm.rvs(loc=comp_log[2][i], scale=comp_log[3][i] ** .5, size=size)
	bp_ekld = plt.boxplot(x, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	pos = np.arange(n, n + len(mu_qoi))
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the QoI')
	# plt.setp(bp_ekld['boxes'], color='black')
	# plt.setp(bp_ekld['whiskers'], color='black')
	# plt.setp(bp_ekld['caps'], color='black')
	# plt.setp(bp_ekld['medians'], color='blacksns.color_palette()[1])
	# plt.setp(bp_ekld['fliers'], color=sns.color_palette()[1], marker='o')
	plt.xlabel('no. of samples')
	plt.ylabel('QoI')
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=8)
	plt.legend()
	plt.savefig(os.path.join(out_dir, 'box.pdf'))
	plt.clf()
	sns.distplot(norm.rvs(loc=mu_qoi[0], scale=sigma_qoi[0] ** .5, size=size), color=sns.color_palette()[1], label='initial distribution of QoI', norm_hist=True)
	sns.distplot(norm.rvs(loc=mu_qoi[-1], scale=sigma_qoi[-1] ** .5, size=size), hist=True, color=sns.color_palette()[0], label='final distribution of QoI', norm_hist=True)
	# plt.scatter(np.mean(Y_u), 0, c=sns.color_palette()[2], label='uncertainty sampling mean')
	plt.scatter(true_mean, 0, c=sns.color_palette()[2], label='true mean')
	# plt.xticks([mu_qoi[-1]])
	plt.legend()
	plt.xlabel('QoI')
	plt.ylabel('p(QoI)')
	plt.savefig(os.path.join(out_dir, 'dist.pdf'))
	plt.clf()
	# Comparison plot
	bp_ekld = plt.boxplot(x, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	plt.setp(bp_ekld['boxes'], color=sns.color_palette()[1])
	plt.setp(bp_ekld['whiskers'], color=sns.color_palette()[1])
	plt.setp(bp_ekld['caps'], color=sns.color_palette()[1])
	plt.setp(bp_ekld['medians'], color=sns.color_palette()[1])
	# ekld_fl = plt.setp(bp_ekld['fliers'], color=sns.color_palette()[1], marker='o')
	bp_us = plt.boxplot(x_us, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	plt.setp(bp_us['boxes'], color=sns.color_palette()[2])
	plt.setp(bp_us['whiskers'], color=sns.color_palette()[2])
	plt.setp(bp_us['caps'], color=sns.color_palette()[2])
	plt.setp(bp_us['medians'], color=sns.color_palette()[2])
	# us_fl = plt.setp(bp_us['fliers'], color=sns.color_palette()[2], marker='x')
	bp_rs = plt.boxplot(x_rs, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	plt.setp(bp_rs['boxes'], color=sns.color_palette()[3])
	plt.setp(bp_rs['whiskers'], color=sns.color_palette()[3])
	plt.setp(bp_rs['caps'], color=sns.color_palette()[3])
	plt.setp(bp_rs['medians'], color=sns.color_palette()[3])
	# rs_fl = plt.setp(bp_rs['fliers'], color=sns.color_palette()[3], marker='*')
	hekld, = plt.plot([1, 1], color=sns.color_palette()[1])
	hus, = plt.plot([1, 1], color=sns.color_palette()[2])
	hur, = plt.plot([1, 1], color=sns.color_palette()[3])
	# plt.scatter(pos, comp_log[0], s=40, marker='x', color=sns.color_palette()[2], label='uncertainty sampling')
	# plt.scatter(pos, comp_log[2], s=30, marker='*', color=sns.color_palette()[3], label='random sampling')
	# plt.scatter(pos, mu_qoi, s=20, marker='o', color=sns.color_palette()[1], label='EKLD')
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the QoI')
	plt.xlabel('no. of samples')
	plt.ylabel('QoI')
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=4)
	plt.legend((hekld, hus, hur), ('EKLD', 'uncertainty sampling', 'random sampling'))
	hekld.set_visible(False)
	hus.set_visible(False)
	hur.set_visible(False)
	plt.ylim(np.min(np.vstack([x, x_us, x_rs])), np.max(np.hstack([x, x_us, x_rs])))
	plt.savefig(os.path.join(out_dir, 'comparison.pdf'))
	quit()
