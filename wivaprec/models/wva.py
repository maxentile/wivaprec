from autograd import grad, jacobian
from autograd import numpy as np
from autograd.optimizers import adam
from autograd.scipy.stats import multivariate_normal

from wivaprec.models.utilities import clip_gradients


class WildVariationalApproximation():
    """Expresses a complicated distribution as a reparameterization of a Gaussian."""

    def __init__(self, n_dim, model):
        """Accepts an integer `n_dim` and a `model` object that supports:
        * transform(params, x) --> y
        * inv_transform(params, y) --> x
        * gradients of transform and inv_transform.
        """
        self.n_dim, self.model = n_dim, model

    def reshape_inputs(self, z):
        """If z is a single instance (shape (D,)), reshape it to (1, D)"""
        if len(z.shape) != 2: z = np.reshape(z, (1, len(z)))
        return z

    def forward_pass(self, z, params):
        """Passes z through model.transform, returning samples and their log densities."""
        z = self.reshape_inputs(z)
        log_q0s = self.log_q0(z)

        f = lambda z: self.model.transform(params, z)
        zK = f(z)
        jac = jacobian(f)(z)
        jacs = [jac[i, :, i, :] for i in range(len(jac))]

        slogdets = [np.linalg.slogdet(jac) for jac in jacs]
        assert ([s[0] == 1 for s in slogdets])  # assert these are all positive...
        log_det_jacs = np.array([s[1] for s in slogdets])
        log_qks = log_q0s - log_det_jacs

        return zK, log_qks, log_det_jacs

    def reverse_pass(self, z, params):
        """Passes z through model.inv_transform, returning samples and their log densities."""
        zK = self.reshape_inputs(z)

        f = lambda z: self.model.inv_transform(params, z)
        z0 = f(zK)
        jac = jacobian(f)(zK)
        jacs = [jac[i, :, i, :] for i in range(len(jac))]

        slogdets = [np.linalg.slogdet(jac) for jac in jacs]
        assert ([s[0] == 1 for s in slogdets])  # assert these are all positive...
        log_det_jacs = np.array([s[1] for s in slogdets])
        log_q0s = self.log_q0(z0) - log_det_jacs

        return z0, log_q0s, log_det_jacs

    def log_q0(self, z):
        """Normal log density."""
        z = self.reshape_inputs(z)
        return multivariate_normal.logpdf(z, mean=np.zeros(self.n_dim))

    def sample_z0(self, n_samples, i=-1):
        """Normal samples."""
        return np.random.randn(n_samples, self.n_dim)

    def sample_zK(self, n_samples, params):
        """Draw samples from model."""
        return self.model.transform(params, self.sample_z0(n_samples))

    def variational_objective(self, params, z0_samples, target_log_q, beta=1.0):
        """Unbiased estimate of the annealed variational free energy,
            F = <log_q> - beta <target_log_q>"""
        zks, log_qks, _ = self.forward_pass(z0_samples, params)
        return np.mean(log_qks) - beta * np.mean([target_log_q(z) for z in zks])

    def reparameterization_gradient(self, params, target_log_q, n_samples=10, beta=1.0):
        """Unbiased estimate of gradient of variational free energy."""
        z0_samples = self.sample_z0(n_samples)
        objective = lambda params: self.variational_objective(params, z0_samples, target_log_q, beta)
        g = grad(objective)(params)
        return g

    def progress_logger_factory(self, target_log_q, n_samples_per_report=100, report_interval=10, verbose=True):
        """Construct progress logger."""

        def log_progress(x, i, g):
            if (i % report_interval == 0):
                free_energy = self.variational_objective(x, self.sample_z0(n_samples_per_report), target_log_q)
                self.optimization_history.append((x, free_energy))
                initial_F = self.optimization_history[0][1]
                if verbose:
                    print("Iteration {}:\n\tVariational free energy: {:.3f}".format(i, free_energy))
                    if (i > 0): print("\tImprovement over start: {:.3f}".format(initial_F - free_energy))
                    print("\tlog(Gradient norm): {:.3f}".format(np.log(np.linalg.norm(g))))

        return log_progress

    def fit(self, target_log_q, n_iter=1000, n_samples=1,
            n_samples_per_report=10, report_interval=10, annealed=True, step_size=0.01, l2_penalty=1.0):
        """Optimize with parameters of self.model to minimize the variational free energy
        between target_log_q and the distribution of y = model.transform(x), x ~ N(0,1)"""
        if annealed:
            beta = np.linspace(0.01, 1.0, n_iter)
        else:
            beta = np.ones(n_iter)

        self.optimization_history = []
        progress_log_callback = self.progress_logger_factory(target_log_q, n_samples_per_report, report_interval)

        normalization = lambda params: l2_penalty * np.sum(np.abs(params) ** 2)

        reparam_gradient = lambda params, i: clip_gradients(
            self.reparameterization_gradient(params, target_log_q, n_samples, beta[i]), 1) + grad(normalization)(params)

        self.params = adam(grad=reparam_gradient, init_params=self.model.params, step_size=step_size,
                           callback=progress_log_callback, num_iters=n_iter)
