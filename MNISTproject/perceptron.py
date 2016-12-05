import numpy as np

class Perceptron(object):
	def __init__(self, categories, numFeatures):
		"""categories: list of strings 
		   numFeatures: int"""
		self.categories = categories
		self.numFeatures = numFeatures
		self.categoryWeights = np.random.rand(len(self.categories), self.numFeatures)

	def classify(self, sample):
		"""sample: np.array of shape (1, numFeatures)
		   returns: category with highest score"""

		bestScore = -100000000
		bestLabel = None
		for i in range(self.categoryWeights.shape[0]):
			score = np.dot(sample, self.categoryWeights[i])
			if score > bestScore:
				bestScore = score
				bestLabel = self.categories[i]

		return bestLabel

	def train(self, samples, labels):
		"""samples: np.array of shape (numFeatures, numSamples)
		   labels: list of numSamples strings, which must all be in self.categories 
		   Trains perceptron weights by iterating over samples one at a time."""

		for i in range(samples.shape[0]):
			bestScore = -10000000
			bestLabel = None
			for j in range(self.categoryWeights.shape[0]):
				score = np.dot(samples[i], self.categoryWeights[j])
				if score > bestScore:
					bestScore = score
					bestLabel = self.categories[j]
			if bestLabel != labels[i]:
				correctLabelInd = self.categories.index(labels[i])
				badLabelInd = self.categories.index(bestLabel)
				self.categoryWeights[correctLabelInd] = np.add(self.categoryWeights[correctLabelInd], samples[i])
				self.categoryWeights[badLabelInd] = np.add(self.categoryWeights[badLabelInd], -1 * samples[i])