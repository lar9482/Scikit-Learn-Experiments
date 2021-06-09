import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import graph
from graph import *

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

			#Calculating statistics on this computation
			self.calculate_statistics()

		elif (testSet == 'fail'):
			print(self.fail_test)
			print('Predicted output on fail_test:')
			self.fail_predicted = self.Model.predict(self.fail_test)
			print(self.fail_predicted)

			self.calculate_RMSE(self.fail_test, self.fail_predicted)
			self.calculate_F_Score(self.fail_test, self.fail_predicted)
			self.calculate_AIC()

	def graph_results(self, set_name, trainingModel, penality, maxIteration, solver):
		try:
			Graph = graph(self.fail_test, self.fail_predicted, set_name, trainingModel, penality, maxIteration, solver, self.RMSE, self.F_Score, self.AIC)
			Graph.save_graph()
			Graph.clear_graph()
		except:
			print('Params are not right')

	def calculate_RMSE(self, test_set, predicted_set):
		self.RMSE = mean_squared_error(test_set.reshape(1, -1)[0], predicted_set, squared = False)

	def calculate_F_Score(self, test_set, predicted_set):
		
		self.F_Score = f1_score(test_set.reshape(1, -1)[0], predicted_set, average = 'weighted', zero_division = 0)
		if (isinstance(self.F_Score, (float)) is False):
			self.F_Score = 'Zero_Division detected'

	def calculate_AIC(self, numParams = 3):
		if (self.RMSE == 0):
			print('calculate RMSE first')
		else:
			self.AIC = 2*(numParams - math.log(self.RMSE))

	def calculate_statistics(self, test_set, predicted_set):
		self.calculate_RMSE(test_set, predicted_set)
		self.calculate_F_Score(test_set, predicted_set)
		self.calculate_AIC()
		

dataSets = ['SYS1', 'SYS2', 'SYS3', 'CSR1', 'CSR2']
algoNames = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalities = ['l1', 'l2', 'elasticnet', 'none']

for i in dataSets:
	model = Model('model_data.xlsx', i)
	for j in algoNames:
		for l in penalities:
			max_iteration = 1000
			try:
				model.train_logistic_regression(l, max_iteration, j)
			except:
				print('Failed to fit %s using %s' % (l, j))
				continue

			while max_iteration <= 5000:
				model.train_logistic_regression(l, max_iteration, j)
				model.predict_logistic_regression('fail')
				model.graph_results(i, 'Logistic-Regression', l, max_iteration, j)
				max_iteration = max_iteration + 1000
