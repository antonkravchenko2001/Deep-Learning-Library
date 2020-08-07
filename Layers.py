import numpy as np

"""general class Layer"""


class Layer:
    def __init__(self):
        self.input = None
        self.output = None


"""Dense Layer. Here outputs are multiplied by weight matrix"""


class Dense(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.v = np.zeros((input_size, output_size))
        self.bias = np.random.rand(1, output_size) - 0.5
        """change of weight which will be updated for each sample"""
        self.weights = np.random.rand(input_size, output_size) - 0.5
        """change of bias which will be updated for each sample"""
        self.bias_decrement = np.zeros((1, output_size))
        self.weight_decrement = np.zeros((input_size, output_size))

    """forward propagation"""
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    """back propagation"""
    def back_propagation(self, output_error, lr):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weight_decrement += lr * weights_error
        self.bias_decrement += lr * output_error
        return input_error


"""Activation Layer. Here activation function is applied to output of previous layer"""


class Activation(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    """forward propagation"""

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation.function(self.input)
        return self.output

    """back propagation"""
    def back_propagation(self, output_error, lr=1):
        result = self.activation.prime(self.input)
        a = np.dot(result, output_error.T).T
        return a

