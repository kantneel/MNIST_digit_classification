import numpy as np
import data_classification_utils as dcu
import math


class SigmoidNeuron(object):
    def __init__(self, numFeatures):
        """numFeatures: Number of features."""
        self.numFeatures = numFeatures
        self.weights = np.random.rand(numFeatures)

    def sigmoid(self, z):
        """Function that maps dot product values to activation level. -infinity
        to infinity -> 0 to 1. Also just the probability that the sample is in
        the positive class."""
        return 1.0 / (1 + math.exp(-2*z))

    def loss(self, sample, label, w):
        """sample: np.array of shape(1, numFeatures).
        label:  the correct label of the sample 
        w:      the weight vector under which to calculate loss"""
        z = np.dot(w, sample)

        if label == True:
            return math.log(1.0 + math.exp(-2*z))
        else:
            return math.log(1.0 + math.exp(2*z))

    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: True if activation is greater than 0.5."""

        value = self.sigmoid(np.dot(self.weights, sample))
        if value > 0.5:
            return True
        return False
    
    def train_single(self, sample, alpha, label):
        """Performs stochastic gradient descent on a single sample ->
        subtracts alpha * gradient form the current set of weights."""

        def helper(weights):
            return self.loss(sample, label, weights)
        
        self.weights = self.weights - alpha * dcu.gradient(helper, self.weights)

    def train(self, samples, alpha, labels):
        """Performs one iteration of stochastic gradient descent 
        by going through samples one at a time."""

        for i in range(samples.shape[0]):
            self.train_single(samples[i], alpha, labels[i])