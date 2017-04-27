# Tinkering with methods for improving the reparameterization gradient estimate:
# * Importance weighting
# * Correlated z0 samples
#   * Instead of each batch z0 being drawn i.i.d. from q0, have them be correlated with previous batches
#   * This should hopefully make optimization easier?
