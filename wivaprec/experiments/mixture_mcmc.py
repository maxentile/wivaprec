from autograd import numpy as np

from wivaprec.evaluation.multimodality import check_if_samples_cover_both_sides_of_origin
from wivaprec.models import NormalizingFlow
from wivaprec.testsystems import perturbed_bimodal_distribution

if __name__ == "__main__":
    np.random.seed(0)

    # let's try this in a range of dimensionalities..
    for D in [2, 5, 10, 50]:
        print("Dimensionality: {}".format(D))
        flow = NormalizingFlow(K=1, n_dim=D)
        target_log_q = perturbed_bimodal_distribution
        flow.fit(target_log_q, n_samples=1, n_iter=1001, report_interval=500, n_samples_per_report=100)
        theta = flow.optimization_history[-1][0]
        samples, log_qks, log_det_jacs = flow.forward_pass(flow.sample_z0(1000), theta)

        evenness = check_if_samples_cover_both_sides_of_origin(samples)
        print("Evenness: {:.3f}  (ideal: 0.5)".format(evenness))
