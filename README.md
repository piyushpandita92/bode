# bode
Bayesian Optimal Design of Experiments 
This module is called bode

Bayesian Optimal Design of Experiments for Inferring the Expected Value of a Black-box Function

 
It needs the support of the following PYTHON packages.
1. pyDOE 
2. GPy (version 1.9.2, mandatory)
3. matplotlib(version 2.0.0, best vizualization)
4. seaborn (version 0.7.1, best vizualization)
5. tqdm

To install the package do the following:
pip install git+git://github.com/piyushpandita92/bode.git  

or clone the repository and run python setup.py install.

Import the package like as follows:
 ```import bode```

The simple examples samp_ex1.py, samp_ex2.py provide a self explanatory overview of using bode.
This code works for estimating/inferring the expectation of a function (so the user would have to include that in their function object).

The user mainly needs to specify the objective function ```obj_func``` as an object, number of iterations (samples to be collected depending on the budget) ```max_it```, number of designs of the discretized input space (for calculating the value of the EKLD criterion) ```X_design```. 

Note: The methodology should be used with the inputs transformed to [0, 1]^{d} cube and outputs roughly normalized to a standard normal.

For sequential design  (one suggested design/experiment at a time):
Running the code: the examples in the ```tests``` directory can be called from the command line with a set of arguments as follows: python tests/ex1.py .


After each iteration a plot depicting the state of the function is generated for 1d problems, this can be controlled by a make_plots flag.  

More documentation to follow.
