"""
Information acquisition for optimal design of experiments

Author:
    Murali Krishnan Rajasekharan Pillai

Date:
    11/12/2021
"""

import GPy
from bode._core import *

class KLSampler(object):
    """
    This class computes the sensitivity of a set of inputs by taking the posterior expectation of the var of the corresponding effect function
    """
    def __init__(self, X, Y, obj_func, noisy,
        per_sampled=10,
        true_func=None,
        model_kern=GPy.kern.Matern32,
        nugget=1e-3,
        ekld_nugget=1e-3,
        lengthscale=1.,
        variance=1.,
        num_opt_restarts=80,
        mcmc_model=False,
        mcmc_steps=500,
        mcmc_final=0.3,
        mcmc_chains=10,
        mcmc_burn=100,
        mcmc_thin=30,
        mcmc_model_avg=50,
        mcmc_parallel=False,
        mcmc_acc_low=0.3,
        mcmc_acc_upp=0.7,
        variance_prior=GammaPrior(a=8, scale=1),
        lengthscale_prior=BetaPrior(a=2, b=5),
        noise_prior=JeffreysPrior(),
        initialize_from_prior=True,
        ego_iter=50,
        ego_init_perc=0.2):
        """
        Arguments:
        ----------
        X           :       np.array

        Y           :       np.array

        obj_func    :
            A callable objective function for BODE
        noisy       :
            Indicator for noisy
        per_sampled :

        true_func   :

        model_kern  :

        nugget      :

        ekld_nugget :

        lengthscale :

        variance    :

        num_opt_restarts    :

        mcmc_model  :

        mcmc_steps  :

        mcmc_final  :

        mcmc_chains :

        mcmc_burn   :

        mcmc_thin   :

        mcmc_model_avg  :

        mcmc_parallel   :

        mcmc_acc_low    :

        mcmc_acc_upp    :

        variance_prior  :

        lengthscale_prior   :

        noise_prior     :

        initialize_from_prior :

        ego_iter    :

        ego_init_perc   :
        """
        assert X.ndim == 2
        self.X = X
        assert Y.ndim == 2
        self.Y = Y
        assert X.shape[0] == Y.shape[0], "Number of samples not the same.."

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
        self._ego_seq = (1. - ego_init_perc) * self.ego_iter

        pass
