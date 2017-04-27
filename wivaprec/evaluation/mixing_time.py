# Estimates of mixing time
# To-do's:
# * Estimate "baseline" mixing times from "brute-force simulations" (GHMC, Random-Walk Metropolis-Hastings, NUTS)
# * Make sure we cover all relevant modes


import pyemma


def estimate_mixing_time_tica(x):
    """Return an estimate of the mixing time of a stochastic process,
    given a trajectory sampled from that process, assuming the slowest relaxation process
    is a linear combination of the input dimensions.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_samples, n_dim)
        Trajectory of MCMC samples

    Returns
    -------
    mixing_time : float
        Stochastic lower bound on the mixing time of the underlying process,
        by estimating the second largest eigenvalue of the MCMC propagator using tICA

    References
    ----------
    Identification of slow molecular order parameters for Markov model construction
    https://arxiv.org/abs/1302.6614
    """

    tica = pyemma.coordinates.tica(x)
    return tica.timescales[0]


def estimate_mixing_time_msm(x):
    """


    Parameters
    ----------
    x

    Returns
    -------

    """
    raise(NotImplementedError())
