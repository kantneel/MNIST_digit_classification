import numpy as np
import perceptron
import samples
import data_classification_utils as dcu

"""Hyperparameters for training a perceptron"""
bias = False
num_times_to_train = 10
num_train_examples = 3000

def get_perceptron_training_data():
	training_data = samples.loadDataFile("digitdata/trainingimages.txt", num_train_examples, 28, 28)
	training_labels = map(str, samples.loadLabelsFile("digitdata/traininglabels.txt", num_train_examples))

	featurized_training_data = np.array(map(dcu.simple_image_featurization, training_data))
	return training_data, featurized_training_data, training_labels

def get_perceptron_test_data():
	test_data = samples.loadDataFile("digitdata/testimages.txt", 1000, 28,28)
	test_labels = map(str, samples.loadLabelsFile("digitdata/testlabels.txt", 1000))

	featurized_test_data = np.array(map(dcu.simple_image_featurization, test_data))
	return test_data, featurized_test_data, test_labels


"""Works by training neuron num_times_to_train times over num_train_examples examples"""

raw_training_data, featurized_training_data, training_labels = get_perceptron_training_data()
raw_test_data, featurized_test_data, test_labels = get_perceptron_test_data()

digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
testPerceptron = perceptron.Perceptron(digits, 784)
trainingData = get_perceptron_training_data()
testData = get_perceptron_test_data()
if bias:
	dcu.apply_bias(trainingData[1])
for i in range(num_times_to_train):
	testPerceptron.train(trainingData[1], trainingData[2])

training_accuracy = (1 - dcu.zero_one_loss(testPerceptron, trainingData[1], trainingData[2])) * 100
test_accuracy = (1 - dcu.zero_one_loss(testPerceptron, testData[1], testData[2])) * 100
print('Final training accuracy: ' + str(training_accuracy) + '% correct')
print("Test accuracy: " + str(test_accuracy) + '% correct')
dcu.display_digit_features(testPerceptron.categoryWeights, bias)