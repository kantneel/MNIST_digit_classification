import numpy as np
import neuron
import samples
import data_classification_utils as dcu

"""Hyperparameters for training neurons"""
alpha = 0.01
bias = False
num_times_to_train = 10
num_train_examples = 500

def get_neuron_training_data():
	training_data = samples.loadDataFile("digitdata/trainingimages.txt", num_train_examples, 28, 28)
	training_labels = np.array(samples.loadLabelsFile("digitdata/traininglabels.txt", num_train_examples))
	training_labels = training_labels == 3

	featurized_training_data = np.array(map(dcu.simple_image_featurization, training_data))
	return training_data, featurized_training_data, training_labels

def get_neuron_test_data():
	test_data = samples.loadDataFile("digitdata/testimages.txt", 1000, 28,28)
	test_labels = np.array(samples.loadLabelsFile("digitdata/testlabels.txt", 1000))
	test_labels = test_labels == 3	

	featurized_test_data = np.array(map(dcu.simple_image_featurization, test_data))
	return test_data, featurized_test_data, test_labels



"""Works by training neuron num_times_to_train times over num_train_examples examples.
This test only tries to measure the accuracy on classifying the digit 3 as such, since the classifier is binary"""

raw_training_data, featurized_training_data, training_labels = get_neuron_training_data()
raw_test_data, featurized_test_data, test_labels = get_neuron_test_data()

theNeuron = neuron.SigmoidNeuron(784)
if bias:
	dcu.apply_bias(featurized_training_data)
for i in range(num_times_to_train):
	theNeuron.train(featurized_training_data, alpha, training_labels)

training_accuracy = (1 - dcu.zero_one_loss(theNeuron, featurized_training_data, training_labels)) * 100
test_accuracy = (1 - dcu.zero_one_loss(theNeuron, featurized_test_data, test_labels)) * 100

print('Final training accuracy: ' + str(training_accuracy) + '% correct')
print("Test accuracy: " + str(test_accuracy) + '% correct')