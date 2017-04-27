# Random model-related utilities: gradient clipping, etc.
from autograd import numpy as np


def clip_gradients(g, thresh=100):
    """If the magnitude of any entries of g exceeds the threshold,
    reduce them to the threshold.
    """
    bad_indices = np.abs(g) > thresh
    g[bad_indices] /= (np.abs(g)[bad_indices] / thresh)
    return g
