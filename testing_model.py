import tensorflow as tf
from Layers import *
from Activation_Functions import *
from Model import *
"""loading MNIST"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255, x_test/255
x_train = x_train.reshape(-1, 1, 28*28)
x_test = x_test.reshape(-1, 1, 28*28)
y_train = tf.keras.utils.to_categorical(y_train)

"""creating change function to manually change lr"""


def change(i):
    if i < 20:
        return 0.1
    elif 19 < i < 40:
        return 0.3
    else:
        return 0.7


"""create Neural Network"""


FeedForward = Model()
Model.add(Dense(28*28, 128))
Model.add(Activation(activation=relu))
Model.add(Dense(128, 100))
Model.add(Activation(activation=tanh))
Model.add(Dense(100, 10))
Model.add(Activation(activation=softmax))
Model.train(x_train[0:3000], y_train[0:3000], lr=0.1, epochs=100, lr_scaler=change)
Model.evaluate(x_test[0:200], y_test[0:200])
