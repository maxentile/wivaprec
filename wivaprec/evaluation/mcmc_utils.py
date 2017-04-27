# Utilities for testing improvements in MCMC efficiency

# To-do: Add HMC / GHMC.

# To-do: Stronger baselines: adaptive MCMC, affine-invariant ensemble sampler, ensemble preconditioning

import numpy as np
from tqdm import tqdm


def metropolis_hastings(x_0, target_log_density_fxn, proposal_sigma=1.0, n_steps=10000):
    """Simple random-walk metropolis-hastings."""
    xs = [x_0]
    log_p_old = target_log_density_fxn(xs[-1])
    n_accept = 0.0
    for i in tqdm(range(n_steps)):
        x_prop = xs[-1] + np.random.randn(len(xs[-1])) * proposal_sigma
        log_p_prop = target_log_density_fxn(x_prop)

        mh_ratio = np.exp(log_p_prop - log_p_old)

        if np.random.rand() < mh_ratio:
            xs.append(x_prop)
            log_p_old = log_p_prop
            n_accept += 1
        else:
            xs.append(xs[-1])

    acc_rate = 1.0 * n_accept / n_steps

    return np.array(xs), acc_rate


def evaluate_on_mcmc(model, target_log_q, proposal_sigma=1.0):
    """Evaluate a change-of-variables transformation by how much it speeds up random-walk
    Metropolis-Hastings."""
    theta = model.optimization_history[-1][0]

    def preconditioned_target_log_q(x):
        """target_log_q(f(x)) - |log jacobian(f)(x)|"""
        y, _, log_det_jac = model.forward_pass(x, theta)
        log_q = target_log_q(y)
        return (log_q + log_det_jac)

    x_0 = model.sample_z0(1).flatten()

    xs_naive, acc_rate_naive = metropolis_hastings(x_0, target_log_q, proposal_sigma=proposal_sigma)
    xs, acc_rate = metropolis_hastings(x_0, preconditioned_target_log_q, proposal_sigma=proposal_sigma)

    print(acc_rate, acc_rate_naive)
    return xs_naive, xs
