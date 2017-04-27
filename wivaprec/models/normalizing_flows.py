from autograd import grad, elementwise_grad
from autograd import numpy as np
from autograd.optimizers import adam

from nonlinearities import tanh_plus


# TODO : Add "scaling" layer

class NormalizingFlow():
    """
    References
    ----------
    Variational inference with normalizing flows http://jmlr.org/proceedings/papers/v37/rezende15.pdf
    """

    def __init__(self, K, n_dim, h=tanh_plus):
        self.K, self.n_dim = K, n_dim
        self.n_params = K * (2 * n_dim + 1)
        self.h = h
        self.h_prime = elementwise_grad(self.h)

    def psi(self, z, w, b):
        return np.dot(self.h_prime(np.dot(z, w) + b), w.T)

    def f(self, z, w, b, u):
        return z + np.dot(self.h(np.dot(z, w) + b), u.T)

    def log_det_jac(self, z, w, b, u):
        return np.log(np.abs(1 + np.dot(self.psi(z, w, b), u)))

    def unpack_params(self, params):
        assert (params.shape == (self.n_params,))
        params_per_layer = (2 * self.n_dim + 1)
        layers = []

        ws, us, bs = [], [], []
        for i in range(self.K):
            theta = params[i * (params_per_layer):(i + 1) * params_per_layer]
            w, u, b = theta[:self.n_dim], theta[self.n_dim:2 * self.n_dim], theta[-1]

            if self.h == np.tanh:  # if h has bounded domain, ensure invertibility
                wu = np.dot(w, u)
                u += (np.log(1 + np.exp(wu)) - 1 - wu) * w / np.linalg.norm(w)

            layers.append({"w": np.reshape(w, (self.n_dim, 1)), "u": np.reshape(u, (self.n_dim, 1)), "b": b})
        return layers

    def reshape_inputs(self, z):
        """If z is a single instance (shape (D,)), reshape it to (1, D)"""
        if len(z.shape) != 2: z = np.reshape(z, (1, len(z)))
        assert (z.shape[1] == self.n_dim)
        return z

    def forward_pass(self, z, params):
        """Given samples z ~ q_0, return samples zk ~ q_K, and their normalized log densities."""
        layers = self.unpack_params(params)
        z = self.reshape_inputs(z)
        log_q0s = self.log_q0(z)
        layer_log_det_jacs = []

        for i in range(self.K):
            w, b, u = layers[i]["w"], layers[i]["b"], layers[i]["u"]
            layer_log_det_jacs.append(self.log_det_jac(z, w, b, u))
            z = self.f(z, w, b, u)
        layer_log_det_jacs = np.array(layer_log_det_jacs)
        log_det_jacs = np.sum(layer_log_det_jacs, 0).flatten()
        log_qks = log_q0s - log_det_jacs

        return z, log_qks, log_det_jacs

    def log_q0(self, z):
        """Normal"""
        z = self.reshape_inputs(z)
        log_q0s = -0.5 * (self.n_dim * np.log(2 * np.pi) + np.sum(z ** 2, 1))
        assert (log_q0s.shape == (len(z),))
        return log_q0s

    def sample_z0(self, n_samples):
        """Normal"""
        return np.random.randn(n_samples, self.n_dim)

    def sample_zK(self, n_samples, theta):
        """Push samples from q_0 through a forward pass"""

        zks, _, _ = self.forward_pass(self.sample_z0(n_samples), theta)
        return zks

    def variational_objective(self, params, z0_samples, target_log_q, beta=1.0):
        """Unbiased estimate of the annealed variational free energy,
            F = <log_qK>_q0 - beta <target_log_q>_q0"""
        zks, log_qks, _ = self.forward_pass(z0_samples, params)
        return np.mean(log_qks) - beta * np.mean([target_log_q(z) for z in zks])

    def reparameterization_gradient(self, params, target_log_q, n_samples=10, beta=1.0):
        """Unbiased estimate of gradient of variational free energy."""
        z0_samples = self.sample_z0(n_samples)
        objective = lambda params: self.variational_objective(params, z0_samples, target_log_q, beta)
        return grad(objective)(params)

    def progress_logger_factory(self, target_log_q, n_samples_per_report=100, report_interval=10):
        """Construct progress logger"""

        def log_progress(x, i, g):
            if i % report_interval == 0:
                free_energy = self.variational_objective(x, self.sample_z0(n_samples_per_report), target_log_q)
                self.optimization_history.append((x, free_energy))
                initial_F = self.optimization_history[0][1]
                print("Iteration {}:\n\tVariational free energy: {:.3f}\n\tImprovement over start: {:.3f}".format(i,
                                                                                                                  free_energy,
                                                                                                                  initial_F - free_energy))

        return log_progress

    def fit(self, target_log_q, init_params=None, n_iter=1000, n_samples=1,
            n_samples_per_report=10, report_interval=10, step_size=0.01, annealed=True):
        """Fit normalizing flow to target_log_q by minimizing variational free energy"""
        if annealed:
            beta = np.linspace(0.001, 1.0, n_iter)
        else:
            beta = np.ones(n_iter)

        if init_params == None:
            init_params = 0.01 * np.random.randn(self.n_params)

        self.optimization_history = []
        progress_log_callback = self.progress_logger_factory(target_log_q, n_samples_per_report, report_interval)

        reparam_gradient = lambda params, i: self.reparameterization_gradient(params, target_log_q, n_samples, beta[i])

        self.params = adam(grad=reparam_gradient, init_params=init_params, step_size=step_size,
                           callback=progress_log_callback, num_iters=n_iter)
