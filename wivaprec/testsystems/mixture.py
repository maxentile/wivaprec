from autograd import numpy as np
from autograd.scipy.misc import logsumexp


# Related question: we have better ways of learning mixture models, I think. Is there a way to
#   turn a mixture model into a change of variables? Related task: annealing from a single gaussian to a
# I guess, the question is: if I have a Gaussian mixture model in hand (means, covariance matrices, mixture weights)
#   can we write down a reparameterizable version of the Gaussian mixture model --
#   Can I write down a function that turns from a Gaussian, into samples from the Gaussian mixture model?
#       (GMM: multinomial distribution on mixture_label, Multivariate Gaussian conditioned on mixture_label)
# ...
# Oor, is there a convenient way to learn a generator network that produces samples from a mixture model whose
# parameters you know...
# I guess we can find out!


class MixtureModel():
    def __init__(self, locations, scales):
        pass


def perturbed_bimodal_distribution(z, exponent=2, perturbation_magnitude=1.0, perturbation_frequency=10, separation=5.0,
                                   offset=0.0
                                   ):
    """Log unnormalized density function with two isolated "modes" (regions of high density)
     and a highly multimodal perturbation.

    q(z) = q_0(z) * q_1(z) * perturbation(z)
    where
        q_0(z) = exp(-(z - mu_0)^exponent)
        q_1(z) = exp(-(z - mu_1)^exponent)
        perturbation(z) = exp(perturbation_magnitude * sum(sin(perturbation_frequency * z)))
    and mu_0 - mu_1 = separation, and sin is elementwise.

    When exponent=2, perturbation_magnitude=0, then each mixture component is Gaussian.

    Mimics a multiscale structure expected in molecular mechanics models, where there may be a small number of
    "metastable states" separated by a large energy barrier, and a large number of smaller modes within each metastable state.

    Parameters
    ----------
    z : numpy.ndarray of shape (n_samples, n_dim)
        Evaluate the log density at these points

    exponent : float
        Exponent defining each mixture component.

    perturbation_magnitude : float
        perturbation(z) = perturbation_magnitude * sum(sin(perturbation_frequency * z))

    perturbation_frequency : float
        perturbation(z) = perturbation_magnitude * sum(sin(perturbation_frequency * z))

    separation : float
        How far away are the means of the mixture components?

    Returns
    -------
    log_qs : numpy.ndarray of shape (n_samples,)
        Unnormalized log densities of input samples
    """

    modes = (separation / 2) * np.array([-1.0, 1.0 + offset])
    if len(z.shape) != 2: z = np.reshape(z, (1, len(z)))

    mixture_probs = np.array([np.sum(np.abs(z - mode) ** exponent, 1) for mode in modes])
    log_qs = logsumexp(- mixture_probs, 0) + perturbation_magnitude * np.sum(np.sin(perturbation_frequency * z), 1)
    return log_qs


    # To-do: Multi-component mixture generalization of above: instead of just two modes,
    #   include several modes.

    # To-do: Systematic screen over problem instances (vary each parameter)

    # To-do: Problem class sampler: define a distribution over problem instances (e.g. sample mixture component
    #   means from a Student-t distribution, ...)

    # To-do: Locally ill-conditioned distributions (e.g. donuts)
