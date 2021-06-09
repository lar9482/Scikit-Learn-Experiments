import matplotlib.pyplot as plt

class graph:
	def __init__(self, test_set, predict_set, set_name, trainingModel, penality, maxIteration, solver, RMSE, F_Score, AIC):
		self.title = set_name+ ' ' +trainingModel+ ' ' +penality+ ' ' +str(maxIteration)+ ' ' +solver
		self.time = [i for i in range(0, len(predict_set))]
		self.test = [test_set[i][0] for i in range(0, len(predict_set))]
		self.predict = [predict_set[i] for i in range(0, len(predict_set))]

		self.RMSE = RMSE
		self.F_Score = F_Score
		self.AIC = AIC

	def build_graph(self):
		#Creates plots for the test and predicted data
		testContinuous = plt.plot(self.time, self.test, label = 'test_data', linestyle = 'solid')
		predictedContinuous = plt.plot(self.time, self.predict, label = 'predicted_data', linestyle = '--')
		testDiscrete = plt.plot(self.time, self.test, "oc") #plots discrete values
		predictDiscrete = plt.plot(self.time, self.predict, "or")
		
		#Creates the title and labels
		plt.xlabel('index')
		plt.ylabel('fail_times')
		plt.title(self.title)

		#Creates the legends of the lines
		testLegend = plt.legend(handles = testContinuous, loc='upper right')
		predictLegend = plt.legend(handles = predictedContinuous, loc='lower right')
		plt.gca().add_artist(testLegend)
		plt.gca().add_artist(predictLegend)

		#Displays a table with the RMSE, F-Score, and AIC.
		
	def build_statistics_table(self):
		colWidth = 1
		rowHeight = 2.5
		fontSize = 10

		CellText = [['RMSE', self.RMSE], ['F-Score', self.F_Score], ['AIC', self.AIC]]
		Loc = 'bottom'
		table = plt.table(cellText = CellText, loc = Loc)
		table.scale(colWidth, rowHeight)
		table.auto_set_font_size(False)
		table.set_fontsize(fontSize)
		#Makes room for the table
		plt.subplots_adjust(left=0.1, bottom=0.3)

	def save_graph(self):
		self.build_graph()
		self.build_statistics_table()
		plt.savefig(self.title)

	def clear_graph(self):
		plt.clf()

	