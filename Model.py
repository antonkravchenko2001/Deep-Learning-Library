import numpy as np
from Activation_Functions import tanh, sigmoid, relu, log_cost, softmax
from Layers import Dense, Activation

"""Class Model"""


class Model:
    l = []

    """add layer to Model"""

    @classmethod
    def add(cls, *args):
        for arg in args:
            cls.l.append(arg)

    """train Model"""

    @classmethod
    def train(cls, x, y, cost=log_cost, lr=0.01, beta=0.9, epochs=100, lr_scaler=None):
        for i in range(epochs):
            if lr_scaler:
                lr = lr_scaler(i)
            loss = 0
            for j in range(x.shape[0]):
                output = x[j]
                # forward propogation
                for layer in cls.l:
                    output = layer.forward_propagation(output)
                loss += (cost.function(y[j], output) + cost.function(1 - y[j], 1 - output)) / x.shape[0]
                error = cost.prime(y[j], output) - cost.prime(1 - y[j], 1 - output)
                # back propogation
                for layer in reversed(cls.l):
                    error = layer.back_propagation(error, lr)
            # weight update
            for layer in cls.l:
                if not isinstance(layer, Activation):
                    layer.v = layer.v * beta + layer.weight_decrement / x.shape[0]
                    layer.weights -= layer.v
                    layer.bias -= layer.bias_decrement / x.shape[0]
                    layer.weight_decrement = np.zeros_like(layer.weight_decrement)
                    layer.bias_decrement = np.zeros_like(layer.bias_decrement)
            print(i + 1, loss)

    """predict output"""

    @classmethod
    def predict(cls, x):
        prediction = np.zeros((x.shape[0],))
        for j in range(x.shape[0]):
            output = x[j]
            for layer in cls.l:
                output = layer.forward_propagation(output)
            output = np.argmax(output)
            prediction[j] = output
        return prediction

    """evaluate on test set"""

    @classmethod
    def evaluate(cls, x, y):
        c = 0
        prediction_array = np.zeros((x.shape[0], 1, 10))
        for j in range(x.shape[0]):
            output = x[j]
            for layer in cls.l:
                output = layer.forward_propagation(output)
            prediction_array[j] = output
            if np.argmax(output) == y[j]:
                c += 1
        accuracy = c / y.size
        print(accuracy)







