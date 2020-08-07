import numpy as np
from scipy import special
"""softmax fucntion"""


def softmax_(x):
    return special.softmax(x)


"""softmax derivative"""


def softmax_prime_(x):
    y = np.zeros((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            if i == j:
                y[i, i] = softmax_(x)[0][i]*(1-softmax_(x)[0][i])
            else:
                y[i, j] = -softmax_(x)[0][i]*softmax_(x)[0][j]
    return y


"""sigmoid"""


def sigmoid_(x):
    return special.expit(x)


"""sigmoid derivative"""


def sigmoid_prime_(x):
    y = np.zeros((x.size, x.size))
    idx = np.arange(x.shape[1])
    y[idx, idx] = sigmoid_(x)*(1-sigmoid_(x))
    return y


"""relu function"""


def relu_(x):
    x[0][x[0] <= 0] = 0
    return x


"""relu drivative"""


def relu_prime_(x):
    x[0][x[0] <= 0] = 0
    x[0][x[0] > 0] = 1
    idx = np.arange(x.size)
    y = np.zeros((x.size, x.size))
    y[idx, idx] = x[0]
    return y


"""tanh"""


def tanh_(x):
    return np.tanh(x)


"""tanh drivative"""


def tanh_prime_(x):
    y = np.zeros((x.size, x.size))
    idx = np.arange(x.shape[1])
    y[idx, idx] = 1-np.tanh(x[0])**2
    return y


"""cost function for cross entropy"""


def log_cost_(y, prediction):
    eps = 1e-15
    for i in range(prediction.shape[1]):
        prediction[0][i] = max(prediction[0][i], eps)
        prediction[0][i] = min(prediction[0][i], 1 - eps)
    return -np.sum(y*np.log(prediction))


"""derivative of cost fucntion for cross entropy"""


def log_cost_prime(y, prediction):
    eps = 1e-15
    for i in range(prediction[0].size):
        prediction[0][i] = max(prediction[0][i], eps)
        prediction[0][i] = min(prediction[0][i], 1 - eps)
    return -y * 1/prediction


"""mse cost function"""


def mse_cost_(y, prediction):
    return (y - prediction)**2


"""derivative of mse"""


def mse_cost_prime(y, prediction):
    return 2*(y-prediction)


class ActivationFunc:
    pass


class Tanh(ActivationFunc):
    def __init__(self):
        self.function = tanh_
        """prime stands for derivative of a fucntion"""
        self.prime = tanh_prime_


class Relu(ActivationFunc):
    def __init__(self):
        self.function = relu_
        self.prime = relu_prime_


class Sigmoid(ActivationFunc):
    def __init__(self):
        self.function = sigmoid_
        self.prime = sigmoid_prime_


class Softmax(ActivationFunc):
    def __init__(self):
        self.function = softmax_
        self.prime = softmax_prime_


class CostFunc:
    pass


class LogCost(CostFunc):
    def __init__(self):
        self.function = log_cost_
        self.prime = log_cost_prime


class MseCost(CostFunc):
    def __init__(self):
        self.function = mse_cost_
        self.prime = mse_cost_prime


"""instances of ActivationFunc class"""


tanh = Tanh()
sigmoid = Sigmoid()
relu = Relu()
log_cost = LogCost()
softmax = Softmax()
