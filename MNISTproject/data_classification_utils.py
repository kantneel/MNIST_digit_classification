import numpy as np
import math
import matplotlib.pyplot as plt

def gradient(f, w):
    """Implements the finite difference approximation for gradients."""

    gradList = []
    for i in range(w.size):
        copy = np.copy(w)
        copy[i] = copy[i] + 0.01
        gradList.append((f(copy) - f(w)) / 0.01)

    return np.asarray(gradList)

def display_digit_features(weights, bias):
    """Visualizes a set of weight vectors for each digit."""

    feature_matrices = []
    for i in range(10):
        feature_matrices.append(convert_weight_vector_to_matrix(weights[i, :], 28, 28, bias))

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(feature_matrices[i], cmap='gray')
        
    plt.show()

def apply_bias(samples):
    """Adds a bias of 1 to the first feature."""

    return np.hstack([samples, np.ones((len(samples), 1))])

def simple_image_featurization(image):
    """Converts an image to a numpy vector of shape (1, w * h), where w is the
        width of the image, and h is the height."""

    imageVector = []
    for i in range(28):
        for j in range(28):
            imageVector.append(image.getPixel(j, i))
    return np.asarray(imageVector).astype(float)


def zero_one_loss(classifier, samples, labels):
    """classifier: The classifier under test.
    sample: The samples under test, should be a numpy array of shape (numSamples, numFeatures).
    label: The correct labels of the samples under test.

    Returns the fraction of samples that are wrong."""

    def zero_one_loss_ss(classifier, sample, label):
        """Helper function to give losses for individual samples."""

        classifierLabel = classifier.classify(sample)
        if classifierLabel == label:
            return 0
        return 1

    total = samples.shape[0]
    wrong = 0
    for i in range(total):
        wrong += zero_one_loss_ss(classifier, samples[i], labels[i])

    return float(wrong) / total


def convert_weight_vector_to_matrix(weight_vector, w, h, bias):
    """weight_vector: The weight vector to transformed into a matrix.
    w, h: width and height of image
    bias: boolean for bias feature

    Returns a w x h array for an image. THe weight vector corresponds to 
    reading the image row by row."""

    matrix = [[] for i in range(h)]
    weightInd = 0
    for i in range(h):
        for j in range(w):
            matrix[i].append(weight_vector[weightInd])
            weightInd += 1

    return np.asarray(matrix)