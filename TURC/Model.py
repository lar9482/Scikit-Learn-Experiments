import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import os
import traceback
import threading

import DataSave
from DataSave import *

import graph
from graph import*

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from sklearn.exceptions import ConvergenceWarning
import warnings


class Model:
	def __init__(self, dataSetName, sheetName, algoNames, inputColumns = ['FN'], outputColumns = ['FT'], shuffled = True, testSize = 0.2):
		self.dataSet = pd.read_excel(dataSetName, sheetName)
		self.fail_indexes, self.fail_times = self._import_data(inputColumns, outputColumns)

		self._scale_data()

		self.index_train, self.index_test, self.fail_train, self.fail_test = train_test_split(self.fail_indexes, self.fail_times, test_size = testSize, shuffle = shuffled)
		self.index_predicted = np.zeros(len(self.index_test), dtype = int)
		self.fail_predicted = np.zeros(len(self.fail_test), dtype = int)

		self._reshape_data()

		self.Model = None

	#This function extracts columns from the dataset based on if it belongs in fail_indexes or fail_times
	def _import_data(self, inputColumns, outputColumns):
		fail_indexes = np.empty(shape = len(np.array(self.dataSet[inputColumns[0]])), dtype = int)
		fail_times = np.empty(shape = len(np.array(self.dataSet[inputColumns[0]])), dtype = int)

		for i in inputColumns:
			fail_indexes = np.vstack([fail_indexes, np.array(self.dataSet[i])])

		for i in outputColumns:
			fail_times = np.vstack([fail_times, np.array(self.dataSet[i])])

		#There are placeholder arrays in the first row for vstack to work(ehhhh....)
		#The next two lines deletes these placeholder arrays.
		fail_indexes = np.delete(fail_indexes, 0, 0)
		fail_times = np.delete(fail_times, 0, 0)

		#If there was only one column, the data is transformed into a 1-D array.
		if len(fail_indexes) == 1:
			fail_indexes = fail_indexes[0]
		if len(fail_times) == 1:
			fail_times = fail_times[0]

		return fail_indexes, fail_times

	#This function cleans up the data
	#Essentially, it scales the data to have zero mean and variance
	#(I.E, fail_index and fail_times are scaled to values between 0 and 1)
	def _scale_data(self):
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
	def _reshape_data(self):
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
			self.Model = MLPRegressor(activation = coFactors, solver = algoName, max_iter = maxIterations)
			model_Name = 'Mult-Layer-Perceptron '

		self.Model = self.Model.fit(self.index_train, self.fail_train)
		return model_Name

	def predict_model(self):
		self.index_predicted = self.Model.predict(self.index_test)
		self.fail_predicted = self.Model.predict(self.fail_test)


	def _convert_double_to_int(self, array):
		newArray = array
		for i in range(0, len(array)):
			newArray[i] = round(array[i])

		return newArray

	def calculate_statistics(self, numParams = 4):
		MSE = mean_squared_error(self.fail_test.reshape(1, -1)[0], self.fail_predicted, squared = True)
		RMSE = mean_squared_error(self.fail_test.reshape(1, -1)[0], self.fail_predicted, squared = False)
		AIC = 2*(numParams - math.log(RMSE))
		EVS = explained_variance_score(self.fail_test.reshape(1, -1)[0], self.fail_predicted)
		ME = max_error(self.fail_test.reshape(1, -1)[0], self.fail_predicted)
		MAE = mean_absolute_error(self.fail_test.reshape(1, -1)[0], self.fail_predicted)
		MedianSE = median_absolute_error(self.fail_test.reshape(1, -1)[0], self.fail_predicted)
		R2Score = r2_score(self.fail_test.reshape(1, -1)[0], self.fail_predicted)

		# print("Testing Datasets:")
		# print(self.fail_test.reshape(1, -1)[0])
		# print(self._convert_double_to_int(self.fail_predicted))

		train_acc_score = accuracy_score(self.fail_train, self._convert_double_to_int(self.Model.predict(self.fail_train.reshape(-1, 1))))
		test_acc_score = accuracy_score(self.fail_test.reshape(1, -1)[0], self._convert_double_to_int(self.fail_predicted))

		# print("Testing Accuracy:")
		# print(test_acc_score)

		return [RMSE, MSE, AIC, EVS, ME, MAE, MedianSE, R2Score, train_acc_score, test_acc_score]

	def graph_data(self, graphObject):
		#Graphing the predicted training data set.
		# print(self.index_train)
		# print(self.fail_train)
		graphObject.build_discrete_graph(self.Model.predict(self.index_train), self.Model.predict(self.fail_train.reshape(-1, 1)), 'Training_Set')

		#Graphing the predicted testing data set.
		print(self.index_test)
		print(self.fail_test)
		graphObject.build_discrete_graph(self.Model.predict(self.index_test), self.Model.predict(self.fail_test.reshape(-1, 1)), 'Testing_Set')
		print(self.Model.predict(self.index_test))
		print(self.Model.predict(self.fail_test.reshape(-1, 1)))
		#Graphing the original data set.
		graphObject.build_continuous_graph(self.fail_indexes, self.fail_times, 'Original_Set')
		graphObject.build_legend()
		graphObject.save_graph()

	def evaluate_graph(self, graphObject, modelName, algorithmName, covariant = None):
		if (modelName == 'SVR' and algorithmName == 'linear'):
			self.graph_data(graphObject)
		elif (modelName == 'DTR' and algorithmName == 'mse'):
			self.graph_data(graphObject)
		elif (modelName == 'RFR' and algorithmName == 'mse'):
			self.graph_data(graphObject)
		elif (modelName == 'LR' and algorithmName == 'newton-cg' and covariant == 'none'):
			self.graph_data(graphObject)
		elif (modelName == 'MLPR' and algorithmName == 'identity' and covariant == 'lbfgs'):
			self.graph_data(graphObject)


	@staticmethod
	def return_stat_names():
		return ['RMSE', 'MSE', 'AIC', 'EVS', 'ME', 'MAE', 'MedianSE', 'R2Score', 'train_acc_score', 'test_acc_score']

	@staticmethod
	def update_stat_file(sheetName, totalDictionary, excelWriter):
		df = pd.DataFrame(totalDictionary)
		df.to_excel(excelWriter, sheet_name = sheetName, index = False)
		excelWriter.save()
	
	@staticmethod
	def construct_covariant_stat_dict(dataDictionary, statNames, iter_diff, iter_bound):
		totalDictionary = {}
		index = 0
		totalDictionary['iteration'] = []
		iteration = iter_diff

		for i in range(0, len(statNames)):
			while iteration <= iter_bound:
				totalDictionary['iteration'].append(iteration)
				iteration = iteration + iter_diff

			totalDictionary['iteration'].append(' ')
			iteration = iter_diff
		totalDictionary['iteration'].append(' ')

		for key in dataDictionary: #range(0, len(dataDictionary)):
			variantDictionary = dataDictionary[key]

			for i in variantDictionary: 
				totalDictionary[key + ' ' + str(i)] = []
				variantList = variantDictionary[i]
				for j in range(0, len(variantList)):
					totalDictionary[key + ' ' + str(i)].append(variantList[j])

				totalDictionary[key + ' ' + str(i)].append(' ')
				index = index + 1


		return totalDictionary

	@staticmethod
	def construct_stat_dict(dataDictionary, statNames, iter_diff, iter_bound):
		totalDictionary = {}
		totalDictionary['iteration'] = []
		iteration = iter_diff

		for i in range(0, len(statNames)):
			while iteration <= iter_bound:
				totalDictionary['iteration'].append(iteration)
				iteration = iteration + iter_diff

			totalDictionary['iteration'].append(' ')
			iteration = iter_diff
		totalDictionary['iteration'].append(' ')

		for key in dataDictionary:
			totalDictionary[key] = []
			currentList = dataDictionary[key]
			for i in range(0, len(currentList)):
				for j in range(0, len(currentList[i])):
					totalDictionary[key].append(currentList[i][j])

			totalDictionary[key].append(' ')

		return totalDictionary


	# #NOTE:
	#covariant factors(coFactors) can be metrics like 'penalities' etc.
	@staticmethod
	def runTestsWithCofactors(dataSets, modelName, algoNames, coFactors, inputColumnName, outputColumnName, iter_diff, iter_bound, excelWriter, shuffled = True, testSize = 0.2):
		for data_Sets in dataSets:
			
			dataDictionary = {i : {} for i in algoNames}
			statNames = Model.return_stat_names()

			graphObject = graph(data_Sets + ' ' + modelName, os.getcwd())	
			for algorithms in algoNames:
				variantDictionary = {i : [] for i in coFactors}
				for variant in coFactors:
					iteration = iter_diff
					model_Name = ''
					paramNames = [data_Sets, algorithms, variant, str(iteration)]
					statDictionary = {i : [] for i in statNames}

					fitFlag = True
					model = Model('model_data.xlsx', data_Sets, algorithms, inputColumnName, outputColumnName, shuffled, testSize)
					try:
						model_Name = model.train_model(modelName, algorithms, iteration, variant)
					except Exception:
						fitFlag = False
						failIndex = iter_diff
						for i in statNames:
							while failIndex <= iter_bound:
								statDictionary[i].append('error')
								failIndex = failIndex + iter_diff
							failIndex = iter_diff
						print('failed to fit model')
					
					while (iteration <= iter_bound) and (fitFlag is not False):
						convergeFlag = 'ConvergeSuccess'
						
						try:
							model_Name = model.train_model(modelName, algorithms, iteration, variant)
						except ConvergenceWarning:
							convergeFlag = 'ConvergeFail'

						print(convergeFlag)
						model.predict_model()
						statsTable = model.calculate_statistics()
						model.evaluate_graph(graphObject, modelName, algorithms, variant)

						index = 0
						for key in statDictionary:
							statDictionary[key].append(statsTable[index])
							index = index + 1

						iteration = iteration + iter_diff
						print('Processed ' + ' '.join([i for i in paramNames]))
						paramNames = [data_Sets, algorithms, variant, str(iteration)]

					for key in statDictionary:
						currentStatElement = statDictionary[key]
						for j in range(0, len(currentStatElement)):
							variantDictionary[variant].append(currentStatElement[j])
						variantDictionary[variant].append(' ')

					#end(variant loop)

				dataDictionary[algorithms] = variantDictionary
				#end(algorithm loop)

			completeDictionary = Model.construct_covariant_stat_dict(dataDictionary, statNames, iter_diff, iter_bound)
			Model.update_stat_file(modelName + ' ' + data_Sets, completeDictionary, excelWriter)
			

	@staticmethod
	def runTests(dataSets, modelName, algoNames, inputColumnName, outputColumnName, iter_diff, iter_bound, excelWriter, shuffled = True, testSize = 0.2):
		for data_Sets in dataSets:
			dataDictionary = {i : [] for i in algoNames}
			statNames = Model.return_stat_names()
			graphObject = graph(data_Sets + ' ' + modelName, os.getcwd())

			for algorithms in algoNames:

					iteration = iter_diff
					model_Name = ''
					paramNames = [data_Sets, algorithms, str(iteration)]
					# directory = DataSave.createDirectoryPath(paramNames)
					# os.makedirs(directory)
					statDictionary = {i : [] for i in statNames}
					
					model = Model('model_data.xlsx', data_Sets, algorithms, inputColumnName, outputColumnName, shuffled, testSize)

					try:
						model_Name = model.train_model(modelName, algorithms, iteration) 
					except Exception:
						# statDictionary.pop(algorithms)
						# print('Failed to fit %s' % (algorithms))
						#remove algorithm name from statDictionary soon.
						continue
					
					while iteration <= iter_bound:
						convergeFlag = 'ConvergeSuccess'
						try:
							model_Name = model.train_model(modelName, algorithms, iteration)
						except ConvergenceWarning:
							converge = 'ConvergeFail'

						print(convergeFlag)

						model.predict_model()
						statTable = model.calculate_statistics()
						model.evaluate_graph(graphObject, modelName, algorithms)

						#add_to_statistics_dictionary
						index = 0
						for key in statDictionary:
							statDictionary[key].append(statTable[index])
							index = index + 1

						iteration = iteration + iter_diff
						print('Processed ' + ' '.join([i for i in paramNames]))
						paramNames = [data_Sets, algorithms, str(iteration)]

					for key in statDictionary:
						dataDictionary[algorithms].append(statDictionary[key])
						dataDictionary[algorithms].append(' ')
					#end(algorithm loop)

			completeDictionary = Model.construct_stat_dict(dataDictionary, statNames, iter_diff, iter_bound)	
			Model.update_stat_file(modelName + ' ' + data_Sets, completeDictionary, excelWriter)
			



statWriter = pd.ExcelWriter(os.path.join(os.getcwd(), 'Stats') + '.xlsx')

iter_diff = 5000
iter_bound = 5000

dataSets = ['SYS1', 'SYS2', 'SYS3']
inputColumnName = ['FN']
outputColumnName = ['FT']

shuffled = False
testSize = 0.2

# dataSets = ['J1', 'J2', 'J3', 'J4', 'J5']
# inputColumnName = ['T']
# outputColumnName = ['CFC']

# dataSets = ['DS1', 'DS2']
# inputColumnName = ['T', 'F']
# outputColumnName = ['FC', 'E']


# iteration_bound = 2000
# iteration_diff = 200


# try:
# 	threading.Thread(target = Model.runTestsWithCofactors, args = (dataSets, 'LR', ['newton-cg', 'lbfgs', 'sag', 'saga'], ['l1', 'l2', 'none'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTests, args = (dataSets, 'SVR', ['linear', 'poly', 'rbf', 'sigmoid'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTests, args = (dataSets, 'DTR', ['mse', 'friedman_mse', 'mae', 'poisson'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTests, args = (dataSets, 'RFR', ['mse', 'mae'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTestsWithCofactors, args = (dataSets, 'MLPR', ['lbfgs', 'adam'], ['identity', 'logistic', 'tanh', 'relu'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()

# except:
# 	traceback.print_exe()

#optimization algorithms
algorithms = ['newton-cg', 'lbfgs', 'sag', 'saga']
#penalities
covariants = ['l1', 'l2', 'none']

modelName = 'LR'
Model.runTestsWithCofactors(dataSets, modelName, algorithms, covariants, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)

#kernels
algorithms = ['linear', 'poly', 'rbf', 'sigmoid']
modelName = 'SVR'
Model.runTests(dataSets, modelName, algorithms, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)


#criterion
algorithms = ['mse', 'friedman_mse', 'mae', 'poisson']
modelName = 'DTR'
Model.runTests(dataSets, modelName, algorithms, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)


#criterion
algorithms = ['mse', 'mae']
modelName = 'RFR'
Model.runTests(dataSets, modelName, algorithms, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)


#activation
covariants = ['identity', 'logistic', 'tanh', 'relu']
#solver
algorithms = ['lbfgs', 'adam']

modelName = 'MLPR'
Model.runTestsWithCofactors(dataSets, modelName, algorithms, covariants, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)