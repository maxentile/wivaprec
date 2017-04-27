from autograd import numpy as np

from wivaprec.evaluation.multimodality import check_if_samples_cover_both_sides_of_origin
from wivaprec.models import NormalizingFlow
from wivaprec.testsystems import perturbed_bimodal_distribution

if __name__ == "__main__":
    np.random.seed(0)

    # let's see how sensitive the result is to the relative distance to the origin of the
    # two modes

    # to-do: multiple repeats, with different random starting conditions!

    for offset in [0, 0.01, 0.1, 0.5, 1.0]:
        print("Offset: {}".format(offset))
        target_log_q = lambda z: perturbed_bimodal_distribution(z, offset=offset)

        # let's try this in a range of dimensionalities..
        for D in [2, 5, 10, 50]:
            print("Dimensionality: {}".format(D))
            flow = NormalizingFlow(K=1, n_dim=D)
            flow.fit(target_log_q, n_samples=100, n_iter=1001, report_interval=500, n_samples_per_report=100)
            theta = flow.optimization_history[-1][0]
            samples, log_qks, log_det_jacs = flow.forward_pass(flow.sample_z0(1000), theta)

            evenness = check_if_samples_cover_both_sides_of_origin(samples)
            print("Evenness: {:.3f}  (ideal: 0.5)".format(evenness))

            # also, is this likely to be a problem only with a small number of samples?
