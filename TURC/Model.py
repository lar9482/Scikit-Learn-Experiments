import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

class Model:
	def __init__(self, dataSetName, sheetName, testSize = 0.2):
		self.dataSet = pd.read_excel(dataSetName, sheetName)
		self.fail_indexes = np.array(self.dataSet['FN'])
		self.fail_times = np.array(self.dataSet['FT'])
		
		self.scale_data()

		self.index_train, self.index_test, self.fail_train, self.fail_test = train_test_split(self.fail_indexes, self.fail_times, test_size = testSize)
		self.index_predicted = np.zeros(len(self.index_test), dtype = int)
		self.fail_predicted = np.zeros(len(self.fail_test), dtype = int)

		self.reshape_data()

		self.Model = None

		self.RMSE = 0.00
		self.F_Score = 0.00
		self.AIC = 0.00

	#This function cleans up the data
	#Essentially, it scales the data to have zero mean and variance
	def scale_data(self):
		self.fail_indexes = self.fail_indexes.reshape(-1, 1)
		self.fail_times = self.fail_times.reshape(-1, 1)

		scaler_index = StandardScaler().fit(self.fail_indexes)
		scaler_fail = StandardScaler().fit(self.fail_times)

		self.fail_indexes = scaler_index.transform(self.fail_indexes)
		self.fail_times = scaler_fail.transform(self.fail_times)

		#The fail_times data is encoded as floating points(not present in the SYS1 file)
		#This encodes the fail_times data as ints
		label_encoder = preprocessing.LabelEncoder()
		self.fail_indexes = label_encoder.fit_transform(self.fail_indexes)
		self.fail_times = label_encoder.fit_transform(self.fail_times)

	#In case the data comes in 1-D arrays, this method rearranges the data into 2-D arrays.
	#Scikit-learn only trains on 2-D arrays.
	def reshape_data(self):
		#index:X(input), fail:Y(output)
		self.index_train = self.index_train.reshape(-1, 1)
		self.index_test = self.index_test.reshape(-1, 1)
		self.fail_test = self.fail_test.reshape(-1, 1)

	def train_logistic_regression(self, Penalty = 'l2', maxIterations = 1000, Solver = 'lbfgs'):
		self.Model = LogisticRegression(penalty = Penalty, max_iter = maxIterations, solver = Solver)
		self.Model.fit(self.index_train, self.fail_train)

	def predict_logistic_regression(self, testSet):
		if (testSet == 'index'):
			print(self.index_test)
			print('Predicted output on index_test:')
			self.index_predicted = self.Model.predict(self.index_test)
			print(self.index_predicted)

		elif (testSet == 'fail'):
			print(self.fail_test)
			print('Predicted output on fail_test:')
			self.fail_predicted = self.Model.predict(self.fail_test)
			print(self.fail_predicted)

	def plot_comparison(self, test_set, predict_set, set_name, trainingModel, penality, maxIteration, solver):
		length = len(predict_set)

		x1 = [i for i in range(0, length)]
		y1 = [test_set[i][0] for i in range(0, length)]

		x2 = [i for i in range(0, length)]
		y2 = [predict_set[i] for i in range(0, length)]

		#Creates plots for the test and predicted data
		testContinuous = plt.plot(x1, y1, label = 'test_data', linestyle = 'solid')
		predictedContinuous = plt.plot(x2, y2, label = 'predicted_data', linestyle = '--')
		testDiscrete = plt.plot(x1, y1, "oc") #plots discrete values
		predictDiscrete = plt.plot(x2, y2, "or")
		#title = 'DataSet: ' + set_name + ', Model: ' + trainingModel + ', penality: ' + penality + ', maxIteration: ' + str(maxIteration)+', Algorithm: ' + solver
		title = set_name+ ' ' +trainingModel+ ' ' +penality+ ' ' +str(maxIteration)+ ' ' +solver
		#Creates the title and labels
		plt.xlabel('index')
		plt.ylabel('fail_times')
		plt.title(title)

		#Creates the legends of the lines
		testLegend = plt.legend(handles = testContinuous, loc='upper right')
		predictLegend = plt.legend(handles = predictedContinuous, loc='lower right')
		plt.gca().add_artist(testLegend)
		plt.gca().add_artist(predictLegend)

		#Displays a table with the RMSE, F-Score, and AIC.
		self.display_statistics()
		plt.savefig(title)
		
	def display_statistics(self):
		self.calculate_RMSE(self.fail_test, self.fail_predicted)
		self.calculate_F_Score(self.fail_test, self.fail_predicted)
		self.calculate_AIC()
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

	def graph_results(self, test_set, set_name, trainingModel, penality, maxIteration, solver):
		if (test_set == 'index'):
			self.plot_comparison(self.index_test, self.index_predicted, set_name, trainingModel, penality, maxIteration, solver)
		elif(test_set == 'fail'):
			self.plot_comparison(self.fail_test, self.fail_predicted, set_name, trainingModel, penality, maxIteration, solver)

	def calculate_RMSE(self, test_set, predicted_set):
		self.RMSE = mean_squared_error(test_set.reshape(1, -1)[0], predicted_set, squared = False)

	def calculate_F_Score(self, test_set, predicted_set):
		
		self.F_Score = f1_score(test_set.reshape(1, -1)[0], predicted_set, average = None, zero_division = 1)
		if (isinstance(self.F_Score, (float)) is False):
			self.F_Score = 'Zero_Division detected'

	def calculate_AIC(self, numParams = 3):
		if (self.RMSE == 0):
			print('calculate RMSE first')
		else:
			self.AIC = 2*(numParams - math.log(self.RMSE))

		

dataSets = ['SYS1', 'SYS2', 'SYS3', 'CSR1', 'CSR2']
algoNames = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalities = ['l1', 'l2', 'elasticnet', 'none']

for i in dataSets:
	model = Model('model_data.xlsx', i)
	for j in algoNames:
		for l in penalities:
			max_iteration = 2000
			try:
				model.train_logistic_regression(l, max_iteration, j)
			except:
				print('Failed to fit %s using %s' % (l, j))
				continue

			while max_iteration <= 20000:
				model.train_logistic_regression(l, max_iteration, j)
				model.predict_logistic_regression('fail')
				model.graph_results('fail', i, 'Logistic-Regression', l, max_iteration, j)
				max_iteration = max_iteration + 2000
