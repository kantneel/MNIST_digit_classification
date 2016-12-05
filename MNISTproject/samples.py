import os

IMG_HEIGHT = 0
IMG_WIDTH = 0

class Image(object):
	"""
	A pixel-level encoding of a digit image as 
	given by the MNIST dataset.

	Each digit is 28x28 pixels, and each pixel can take the following values
	which represent the intensity of the pixel:
	  0: no edge (blank)
	  1: gray pixel (+) 
	  2: black pixel (#)

	Pixel data is stored in a 2-dimensional array, with the first argument being
	the column (0 is left), and second being row (0 is bottom).

	"""
	def __init__(self, data,width,height):
		"""
		Create a new object from file input (standard MNIST encoding).
		"""
		IMG_HEIGHT = height
		IMG_WIDTH = width
		self.height = IMG_HEIGHT
		self.width = IMG_WIDTH
		if data == None:
			data = [[' ' for i in range(IMG_WIDTH)] for j in range(IMG_HEIGHT)]
		self.pixels = arrayInvert(convertToInteger(data))

	def getPixel(self, column, row):
		"""
		Returns the value of the pixel at column, row as 0, or 1.
		"""
		return self.pixels[column][row]

	def getPixels(self):
		"""
		Returns all pixels as a list of lists.
		"""
		return self.pixels

def arrayInvert(array):
	"""
	Inverts a matrix stored as a list of lists.
	"""
	result = [[] for i in array]
	for outer in array:
		for inner in range(len(outer)):
			result[inner].append(outer[inner])
	return result

def loadDataFile(filename, n, width, height):
	"""
	Reads n data images from a file and returns a list of Image objects.

	(Return less then n items if the end of file is encountered)."""
	IMG_WIDTH = width
	IMG_HEIGHT = height
	fin = readlines(filename)
	fin.reverse()
	items = []
	for i in range(n):
		data = []
		for j in range(height):
			data.append(list(fin.pop()))
		if len(data[0]) < IMG_WIDTH - 1:
			# we encountered end of file...
			print "Finishing with %d examples (maximum)" % i
			break
		items.append(Image(data,IMG_WIDTH,IMG_HEIGHT))
	return items

def readlines(filename):
	"""Opens a file and reads it"""
	if(os.path.exists(filename)):
		return [l[:-1] for l in open(filename).readlines()]
	else:
		print("Error: file not found.")

def loadLabelsFile(filename, n):
	"""Reads n labels from a file and returns a list of integers."""
	fin = readlines(filename)
	labels = []
	for line in fin[:min(n, len(fin))]:
		if line == '':
			break
		labels.append(int(line))
	return labels

def IntegerConversionFunction(character):
	"""Helper function for file reading."""

	if(character == ' '):
		return 0
	elif(character == '+'):
		return 1
	elif(character == '#'):
		return 2

def convertToInteger(data):
	"""Helper function for file reading."""
	
	if type(data) != type([]):
		return IntegerConversionFunction(data)
	else:
		return map(convertToInteger, data)
