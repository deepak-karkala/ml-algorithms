import numpy as np
import copy

class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape
    
    def layer_name(self):
        return self.__class__.__name__
    
    def n_parameters(self):
        return 0
    
    def forward_pass(self, x, training):
        raise NotImplementedError()
    
    def backward_pass(self, grad):
        raise NotImplementedError()
    
    def output_shape(self, x):
        raise NotImplementedError()


class Dense(Layer):
    """
    Fully connected NN layer
    Parameters
    ----------
    n_units: int
        Number of input units
    input_shape: tuple
        For dense layers a single digit specifying
        the number of features of the input.
    """
    def __init__(self, n_units, input_shape=None, trainable=True):
        self.n_units = n_units
        # Input shape
        #   First layer: set explicitly based on input
        #   Next layers: set based on n_units in prev layer
        self.input_shape = input_shape
        self.trainable = trainable
        self.W = None
        self.b = None
        # Save input for backward pass
        self.layer_input = None
    
    def initialize(self, optimizer):
        limit = 1/np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.b = np.zeros((1, self.n_units))
        self.W_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def forward_pass(self, X, training):
        self.layer_input = X
        # X: (N_batch x N_units_input)
        # W: (N_units_input x N_units_output)
        return X @ self.W + self.b
    
    def backward_pass(self, grad_op):
        W = self.W
        if self.trainable:
            # Gradient wrt layer parameters
            grad_W = self.layer_input.T @ grad_op
            grad_b = np.sum(grad_op, axis=0, keepdims=True)

            # Update parameters based on optimizer rule
            self.W = self.W_opt.update(self.W, grad_W)
            self.b = self.b_opt.update(self.b, grad_b)
        
        # Gradient wrt output of prev layer (=input of present layer)
        # Calculated based on the weights used during the forward pass
        grad_op = grad_op @ W.T
        return grad_op
    
    def output_shape(self):
        return (self.n_units,)


# Activation layer
class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, activation_func, trainable=True):
        self.activation_func = activation_func
        self.trainable = trainable
    
    def layer_name(self):
        return self.activation_func.__class__.__name__
    
    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)
    
    def backward_pass(self, grad_op):
        return grad_op * self.activation_func.gradient(self.layer_input)
    
    def output_shape(self):
        return self.input_shape
