import numpy as np


class Network():
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

    def feedforward(self, a):
        """
        :param a: initial numpy array of inputs (activations)
        :return: final output of network
        """
        for w, b in zip(self.weights, self.biases):
            a = Network.sigmoid(np.dot(w, a) + b)
        return a

    @staticmethod
    def sigmoid(z):
        return 1./(1. + np.exp(-z))

    @staticmethod
    def d_sigmoid(z):
        return Network.sigmoid(z) * (1. - Network.sigmoid(z))

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        implements stochastic gradient descent. if optional test_data is provided we also print progress after each epoch.
        helps to track progress but slows down SGD a lot
        :param training_data: tuples of (x, y) where y is the label/output associated with x
        :param epochs: number of runs
        :param mini_batch_size: batch size for SGD
        :param eta: learning rate
        :param test_data:
        """
        training_data_size = len(training_data)
        for i in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, training_data_size, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(i))


    def update_mini_batch(self, mini_batch, eta):
        """

        :param mini_batch:
        :param eta:
        :return:
        """

        pass

    def backprop(self):
        pass

    def evaluate(self, test_data):
        pass