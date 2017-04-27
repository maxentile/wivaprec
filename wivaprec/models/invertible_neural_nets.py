from autograd import numpy as np

from nonlinearities import tanh_plus, tanh_plus_inverse


class InvertibleNeuralNet():
    """MLP where input_dim=output_dim=hidden_layer_dims.

    Convenient class of parameterized differentiable functions that supports differentiable
    `transform` and `inv_transform` operations.
    """

    def __init__(self, n_dim, n_hidden_layers=2,
                 nonlinearity=tanh_plus, inv_nonlinearity=tanh_plus_inverse):
        """Initialize model. Trusts user to define inv_nonlinearity properly."""
        self.n_dim, self.n_hidden_layers, self.nonlinearity = n_dim, n_hidden_layers, nonlinearity
        self.inv_nonlinearity = inv_nonlinearity
        self.initialize()

    def initialize(self):
        """Weight matrices are initialized as small perturbations of the identity."""
        # """Weight matrices are initialized with small random weights."""
        init_scale = 0.01
        n = self.n_dim
        params = [(init_scale * np.random.randn(n, n) + np.eye(n),  # weight matrix
                   init_scale * np.random.randn(n))  # bias vector
                  for _ in range(self.n_hidden_layers + 1)]
        self.params = self.flatten_params(params)
        self.num_params = len(self.params)

    def transform(self, params, inputs):
        """Forward pass"""
        params = self.unpack_params(params)
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = self.nonlinearity(outputs)
        return outputs

    def inv_transform(self, params, outputs):
        """Reverse pass"""
        params = self.unpack_params(params)
        W, b = params[-1]
        inputs = np.dot(outputs - b, np.linalg.inv(W))
        for W, b in params[::-1][1:]:
            inputs = np.dot(self.inv_nonlinearity(inputs) - b, np.linalg.inv(W))
        return inputs

    def flatten_params(self, params):
        """Flatten list of (W,b) pairs into param vector"""
        flat_params = np.hstack([np.hstack([W.flatten(), b]) for (W, b) in params])
        # assert (len(flat_params) == self.num_params)
        return flat_params

    def unpack_params(self, params):
        """Unpack param vector into list of (W,b) pairs."""
        # assert (len(params) == self.num_params)
        structured_params = []
        n = self.n_dim
        params_per_layer = n ** 2 + n
        for i in range(self.n_hidden_layers + 1):
            layer_params = params[params_per_layer * i:params_per_layer * (i + 1)]
            W = layer_params[:n ** 2].reshape((n, n))
            b = layer_params[n ** 2:]
            structured_params.append((W, b))
        return structured_params
