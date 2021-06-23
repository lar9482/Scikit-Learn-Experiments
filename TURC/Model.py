import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import os
import traceback

import graph
from graph import *

import DataSave
from DataSave import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

class Model:
	def __init__(self, dataSetName, sheetName, path, algoNames, testSize = 0.15):
		self.dataSet = pd.read_excel(dataSetName, sheetName)
		self.fail_indexes = np.array(self.dataSet['FN'])
		self.fail_times = np.array(self.dataSet['FT'])
		self.excelWriter = pd.ExcelWriter(os.path.join(path, sheetName) + '.xlsx')

		self.scale_data()

		self.index_train, self.index_test, self.fail_train, self.fail_test = train_test_split(self.fail_indexes, self.fail_times, test_size = testSize)
		self.index_predicted = np.zeros(len(self.index_test), dtype = int)
		self.fail_predicted = np.zeros(len(self.fail_test), dtype = int)

		self.reshape_data()

		self.Model = None

		self.RMSE = 0.00
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
		label_encoder = LabelEncoder()
		self.fail_indexes = label_encoder.fit_transform(self.fail_indexes)
		self.fail_times = label_encoder.fit_transform(self.fail_times)

	#In case the data comes in 1-D arrays, this method rearranges the data into 2-D arrays.
	#Scikit-learn only trains on 2-D arrays.
	def reshape_data(self):
		#index:X(input), fail:Y(output)
		self.index_train = self.index_train.reshape(-1, 1)
		self.index_test = self.index_test.reshape(-1, 1)
		self.fail_test = self.fail_test.reshape(-1, 1)

	def train_model(self, modelName, algoName, maxIterations = 1000, coFactors = None):
		model_Name = ''
		if (modelName == 'LR'):
			self.Model = LogisticRegression(penalty = coFactors, max_iter = maxIterations, solver = algoName)
			model_Name = 'Logistic-Regression '

		elif(modelName == 'SVR'):
			self.Model = SVR(kernel = algoName, max_iter = maxIterations)
			model_Name = 'Support-Vector-Machine '

		elif (modelName == 'DTR'):
			self.Model = DecisionTreeRegressor(criterion = algoName, max_depth = maxIterations)
			model_Name = 'Decision-Tree '

		elif (modelName == 'RFR'):
			self.Model = RandomForestRegressor(n_estimators = maxIterations, criterion = algoName, max_depth = maxIterations)
			model_Name = 'Random-Forest '

		elif (modelName == 'MLPR'):
			self.Model = MLPRegressor(hidden_layer_sizes = maxIterations, activation = coFactors, solver = algoName)
			model_Name = 'Mult-Layer-Perceptron '

		self.Model.fit(self.index_train, self.fail_train)
		return model_Name

	def predict_model(self, testSet):
		if (testSet == 'index'):	
			self.index_predicted = self.Model.predict(self.index_test)
			
		elif (testSet == 'fail'):
			self.fail_predicted = self.Model.predict(self.fail_test)

	def graph_results(self, modelName, names, path):
		try:
			Graph = graph(self.fail_test, self.fail_predicted, modelName, names, path)
			Graph.save_graph()
			Graph.clear_graph()
		except:
			traceback.print_exc()

	def calculate_RMSE(self, test_set, predicted_set):
		# print(test_set.reshape(1, -1)[0])
		# print(predicted_set)
		self.RMSE = mean_squared_error(test_set.reshape(1, -1)[0], predicted_set, squared = False)

	def calculate_AIC(self, numParams = 4):
		if (self.RMSE == 0):
			print('calculate RMSE first')
		else:
			self.AIC = 2*(numParams - math.log(self.RMSE))

	def calculate_statistics(self):
		self.calculate_RMSE(self.fail_test, self.fail_predicted)
		self.calculate_AIC()

	#This function creates a separate excel file that saves the test and predicted data,
	#into the same directory as self.excelWriter.
	#Please note that the directories are named based on the current algorithm, data set, and/or co-variant.
	def save_data(self, sheetName):
		df = pd.DataFrame({
			"FT": self.fail_test.reshape(1, -1)[0],
			"FP": self.fail_predicted
			})

		df.to_excel(self.excelWriter, sheet_name = sheetName, index = False)
		self.excelWriter.save()
	
	#These are helper functions that are used to create custom directories
	def construct_stat_table(self, max_iter):
		return [max_iter, self.RMSE, self.AIC]

	def construct_covariant_stat_table(self, max_iter, coFactor):
		return [coFactor, max_iter, self.RMSE, self.AIC]
	
	
	@staticmethod
	def update_stat_file(sheetName, statDictionary, excelWriter):
		print(statDictionary)
		df = pd.DataFrame(statDictionary)
		df.to_excel(excelWriter, sheet_name = sheetName, index = False)
		excelWriter.save()

	# #NOTE:
	#covariant factors(coFactors) can be metrics like 'penalities' etc.
	@staticmethod
	def runTestsWithCofactors(dataSets, modelName, algoNames, coFactors, excelWriter):
		for data_Sets in dataSets:
			statDictionary = {i : [' '] for i in algoNames}
			for algorithms in algoNames:
				for variant in coFactors:
					max_iteration = 1000
					model_Name = ''
					names = [data_Sets, algorithms, variant]
					directory = DataSave.createDirectoryPath(names)
					os.makedirs(directory)
					
					model = Model('model_data.xlsx', data_Sets, directory, algorithms)
					names.append(str(max_iteration))

					try:
						model_Name = model.train_model(modelName, algorithms, max_iteration, variant)
					except:
						print('Failed to fit %s using %s' % (variant, algorithms))
						statDictionary = statDictionary.pop(algorithm)
						continue
					
					while max_iteration <= 5000:
						convergeFlag = 'ConvergeSuccess'
						try:
							model_Name = model.train_model(modelName, algorithms, max_iteration, variant)
						except:
							convergeFlag = 'ConvergeFail'

						model.predict_model('fail')
						model.calculate_statistics()

						statsTable = model.construct_covariant_stat_table(variant, max_iteration)
						statsTable.append(convergeFlag)
						statsTable.append(' ')
						
						model.graph_results(modelName = model_Name, names = ' '.join([i for i in names]), path = directory)
						model.save_data(sheetName = ' '.join([i for i in names]))
						
						for i in statsTable:
							statDictionary[algorithms].append(i)

						max_iteration = max_iteration + 1000

						print('Processed' + ' '.join([i for i in names]))
						names = [data_Sets, algorithms, variant, str(max_iteration)]

			Model.update_stat_file(modelName + ' ' + data_Sets, statDictionary, excelWriter)

	@staticmethod
	def runTests(dataSets, modelName, algoNames, excelWriter):
		for data_Sets in dataSets:
			statDictionary = {i : [' '] for i in algoNames}
			for algorithms in algoNames:
					max_iteration = 1000
					model_Name = ''
					names = [data_Sets, algorithms]
					directory = DataSave.createDirectoryPath(names)
					os.makedirs(directory)
					
					model = Model('model_data.xlsx', data_Sets, directory, algorithms)
					names.append(str(max_iteration))

					try:
						model_Name = model.train_model(modelName, algorithms, max_iteration)
					except:
						print('Failed to fit %s' % (algorithms))
						#remove algorithm name from statDictionary soon.
						
						continue
					
					while max_iteration <= 5000:
						convergeFlag = 'ConvergeSuccess'
						try:
							model_Name = model.train_model(modelName, algorithms, max_iteration)
						except:
							convergeFlag = 'ConvergeFail'

						model.predict_model('fail')
						model.calculate_statistics()

						statsTable = model.construct_stat_table(max_iteration)
						statsTable.append(convergeFlag)
						statsTable.append(' ')
						
						model.graph_results(modelName = model_Name, names = ' '.join([i for i in names]), path = directory)
						model.save_data(sheetName = ' '.join([i for i in names]))
						
						for i in statsTable:
							statDictionary[algorithms].append(i)

						max_iteration = max_iteration + 1000

						print('Processed' + ' '.join([i for i in names]))
						names = [data_Sets, algorithms, str(max_iteration)]

			Model.update_stat_file(sheetName = modelName + ' ' + data_Sets, statDictionary = statDictionary, excelWriter = excelWriter)	


statWriter = pd.ExcelWriter(os.path.join(os.getcwd(), 'Stats') + '.xlsx')

# dataSets = ['SYS1']
# optimizeAlgorithms = ['newton-cg', 'lbfgs', 'sag', 'saga']
# penalities = ['l1', 'l2', 'elasticnet', 'none']
# modelName = 'LR'
# Model.runTestsWithCofactors(dataSets, modelName, solver, activation, statWriter)

dataSets = ['SYS1']
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
modelName = 'SVR'
Model.runTests(dataSets, modelName, kernels, statWriter)

# dataSets = ['SYS1']
# criterion = ['mse', 'friedman_mse', 'mae', 'poisson']
# modelName = 'DTR'
# Model.runTests(dataSets, modelName, criterion, statWriter)

# dataSets = ['SYS1']
# criterion = ['mse', 'mae']
# modelName = 'RFR'
# Model.runTests(dataSets, modelName, criterion, statWriter)

# dataSets = ['SYS1']
# activation = ['identity', 'logistic', 'tanh', 'relu']
# solver = ['lbfgs', 'adam']

# modelName = 'MLPR'
# Model.runTestsWithCofactors(dataSets, modelName, solver, activation, statWriter)
