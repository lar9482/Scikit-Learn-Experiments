import os
import matplotlib.pyplot as plt

class graph:
	def __init__(self, test_set, predict_set, modelName, names):
		self.title = modelName
		self.title = self.title + names
		
		self.time = [i for i in range(0, len(predict_set))]
		self.test = [test_set[i][0] for i in range(0, len(predict_set))]
		self.predict = [predict_set[i] for i in range(0, len(predict_set))]

	def build_graph(self):
		#Creates plots for the test and predicted data
		testContinuous = plt.plot(self.time, self.test, label = 'test_data', linestyle = 'solid')
		predictedContinuous = plt.plot(self.time, self.predict, label = 'predicted_data', linestyle = '--')
		testDiscrete = plt.plot(self.time, self.test, "oc") #plots discrete values
		predictDiscrete = plt.plot(self.time, self.predict, "or")
		
		#Creates the title and labels
		# plt.xlabel('index')
		plt.ylabel('fail_times')
		plt.title(self.title)

		#Creates the legends of the lines
		testLegend = plt.legend(handles = testContinuous, loc='upper right')
		predictLegend = plt.legend(handles = predictedContinuous, loc='lower right')
		plt.gca().add_artist(testLegend)
		plt.gca().add_artist(predictLegend)


	def save_graph(self):
		self.build_graph()
		plt.savefig(os.path.join(self.directory, self.title))

	def clear_graph(self):
		plt.clf()

	
