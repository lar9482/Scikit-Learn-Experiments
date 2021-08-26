import matplotlib.pyplot as plt
import os
import random

class graph:
	def __init__(self, modelName, path):
		self.title = modelName
		self.directory = path
		self.allMarkers = ['^', 's', 'p', 'h', 'x', 'd']
		self.allColors = ['b', 'g', 'r', 'c', 'm', 'y']
		self.usedColors = {}
		self.usedMarkers = {}
		self.initialize_Graph()
		
		
	def initialize_Graph(self):
		plt.xlabel('failure_numbers')
		plt.ylabel('failure_times')
		plt.title(self.title)

	def _select_color_and_marker(self):
		colorValue = ''
		while ((colorValue not in self.usedColors)):
			colorValue = random.choice(self.allColors)
		self.allColors[colorValue] = ' '

		markerValue = ''
		while(markerValue not in self.usedMarkers):
			markerValue = random.choice(self.allMarkers)
		self.allMarkers[markerValue] = ' '

		return colorValue, markerValue

	def _select_color(self):
		colorValue = ''
		while (colorValue not in self.usedColors):
			colorValue = random.choice(self.allColors)
		self.allColors[colorValue] = ' '

		return colorValue

	def build_continuous_graph(self, input_set, output_set, name):
		#Creates plots for the test and predicted data
		colorValue = random.choice(self.allColors)
		continuousGraph = plt.plot(input_set, output_set, color = colorValue, linestyle = "solid", label = name)
		# discreteGraph = plt.plot(input_set, output_set)
		# testDiscrete = plt.plot(input_set, testing_set, "oc") #plots discrete values
		# predictDiscrete = plt.plot(input_set, actual_set, "or")

	def build_discrete_graph(self, input_set, output_set, name):
		colorValue, markerValue = random.choice(self.allColors), random.choice(self.allMarkers)
		discreteGraph = plt.plot(input_set, output_set, color = colorValue, linestyle = "None", marker = markerValue, label = name)

	def build_legend(self):
		# print(tuple(self.graphList))
		# print(self.labelList)
		# legend = plt.legend(tuple(self.graphList), self.labelList, loc = 'upper right')
		# plt.gca().add_artist(legend)
		plt.legend()

	def save_graph(self):
		plt.savefig(os.path.join(self.directory, self.title))
		self.clear_graph()

	def clear_graph(self):
		plt.clf()

	