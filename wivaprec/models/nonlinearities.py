# To-do: Create "nonlinearity" class that includes:
# * f
# * f_inverse
# * f_prime
# * domain
# * range

# To-do: Add a few other nonlinearities.

from autograd import numpy as np


def tanh_plus(x):
    """tanh-like function with infinite range (guaranteeing existence of inverse)"""
    return np.sign(x) * np.log(np.abs(x) + 1)


def tanh_plus_inverse(y):
    """Inverse of tanh_plus"""
    return np.sign(y) * (np.exp(np.abs(y)) - 1)
