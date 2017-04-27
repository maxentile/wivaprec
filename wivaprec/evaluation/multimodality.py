# To-do: add a way to quantify whether the variational procedure is dropping important modes.

# For a mixture model, this should be easy: we just draw samples from qK, assign them to their
# most likely mixture component. We then see how many mixture components are hit...

from autograd import numpy as np


def check_if_samples_cover_both_sides_of_origin(samples):
    """For one of the mixture distributions I had constructed, the modes are on
    opposite sides of the origin, so I can tell how well the generated samples
    are falling into one or the other mode...
    """
    left, right = np.sum(samples > 0), np.sum(samples < 0)
    if min(left, right) == 0:
        print("Completely dropped a mode!")
    return 1.0 * min(left, right) / (left + right)
