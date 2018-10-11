import numpy as np
import seaborn as sns
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
from klsamp import *
from cycler import cycler
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import beta


class Ex1Func(object):
    def __init__(self, sigma_noise=lambda x: 0.5, mu=0, sigma=1):
        self.sigma_noise = sigma_noise
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
    	return norm.pdf(x[0], loc=self.mu, scale=self.sigma)


if __name__=='__main__':
    np.random.seed(435)
    n = 4
    noise = 0
    noise_true = 0
    sigma_noise = eval('lambda x: ' + str(noise))
    sigma_true = eval('lambda x: ' + str(noise_true))
    mu = 0.5
    sigma = 0.03
    objective_true = Ex1Func(sigma_noise=sigma_true, mu=mu, sigma=sigma)
    objective = Ex1Func(sigma_noise=sigma_noise, mu=mu, sigma=sigma)
    X_init = design.latin_center(n, 1)
    Y_init = np.array([objective(x) for x in X_init])[:, None]
    out_dir = 'klsamp_n={0:d}_sigma={1:s}'.format(n, str(noise))
    num_samp = 30
    samp_pts = 100
    X_samp = np.linspace(0, 1, samp_pts)
    y_samp = np.zeros((num_samp, samp_pts))
    y_samp_hyp = np.zeros((num_samp, samp_pts))
    x_hyp = np.array([[0.6]])
    num_quad_points = 100
    num_it = 50
    quad_points = np.linspace(0, 1., num_quad_points)[:, None] # Linearly spaced points
    quad_points_weight = np.ones(num_quad_points)
    kls = KLSampler(X_init, Y_init, x_hyp, 
        model_kern=GPy.kern.RBF, 
        bounds=[(0, 1)] * X_init.shape[1], 
        obj_func=objective, 
        true_func=objective_true,
        noisy=False,
        energy=0.95,
        nugget=1e-3,
        kld_tol=1e-3,
        func_name='ex3',
        num_quad_points=num_quad_points,
        quad_points=quad_points,
        quad_points_weight=quad_points_weight,
        max_it=num_it)
    X, Y, Y_u, kld, X_design, mu_qoi, sigma_qoi = kls.optimize(num_designs=200, verbose=1, plots=1)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    np.save('Y_u.npy', Y_u)
    np.save('kld.npy', kld)
    np.save('mu_qoi', np.array(mu_qoi))
    np.save('sigma_qoi', np.array(sigma_qoi))
    kld_max = np.ndarray(kld.shape[0])
    err = np.zeros(num_it + 1) # Error in the QoI being estimated after each iteration
    true_mean = 1.
    for i in xrange(Y.shape[0]-n):
        err[i] = (abs(np.mean(Y[:n+i, :]) - true_mean)/abs(true_mean))*100
    for i in xrange(kld.shape[0]):
        kld_max[i] = max(kld[i, :])
    plt.plot(np.arange(len(kld_max)), kld_max/max(kld_max), color=sns.color_palette()[1])
    plt.xlabel('iterations')
    plt.ylabel('relative maximum KLD')
    plt.savefig('kld.pdf')
    plt.clf()
    plt.plot(np.arange(len(err)), err, color=sns.color_palette()[1])
    plt.xlabel('iterations')
    plt.ylabel('relative absolute error in the QoI')
    plt.savefig('doe.pdf')
    plt.clf()
    quit()
