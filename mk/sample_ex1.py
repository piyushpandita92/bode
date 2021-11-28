"""
Re-implementation of Example1 of BODE for understanding
Author:
    Murali Krishnan Rajasekharan Pillai

Date:
    11/26/2021
"""

import numpy as np
import pyDOE
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
plt.rcParams['agg.path.chunksize'] = 1000

class Ex1Func():
    """
    Example 1 :
    """
    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
        x = 6. * x
        fx = (4. * (1. - np.sin(x + 8. * np.exp(x - 7.))) \
            - 10.) / 5. + self.sigma(x) * np.random.randn()
        return fx

if __name__ == '__main__':
    np.random.seed(1333)
    n = 3
    n_true = 100
    dim = 1
    noise = 0
    noise_true = 0
    sigma = lambda x: noise
    sigma_true = lambda x: noise_true
    objective_true = Ex1Func(sigma=sigma_true)
    objective = Ex1Func(sigma=sigma)

    X_init = pyDOE.lhs(dim, samples=n, criterion='center')
    Y_init = objective(X_init)
    X_true = pyDOE.lhs(dim, samples=n_true, criterion='center')
    X_idx = np.argsort(X_true, axis=0)
    Y_true = objective(X_true)
    print(idx)
    print(Y_true[idx].reshape(-1,1))
    true_mean = Y_true.mean()
    print(r'true E[f(x)]: ', true_mean)
    # fig, ax = plt.subplots(figsize=(12, 9))
    # ax.plot(X_init, Y_init, 'x', label='Init')
    # ax.plot(X_true[X_idx].reshape(-1,1), Y_true[X_idx].reshape(-1,1), label='True')
    # plt.show()
    # plt.tight_layout()

    num_quad_points = 500
    quad_points = pyDOE.lhs(dim, num_quad_points)
