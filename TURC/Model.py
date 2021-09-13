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
from sklearn.model_selection import GridSearchCV

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
	def __init__(self, dataSetName, sheetName, inputColumns = ['FT'], outputColumns = ['FN'], shuffled = False, testSize = 0.2):
		self.dataSet = pd.read_excel(dataSetName, sheetName)
		self.fail_times, self.fail_indexes = self._import_data(inputColumns, outputColumns)

		self.time_train, self.time_test, self.index_train, self.index_test = train_test_split(self.fail_times, self.fail_indexes, test_size = testSize, shuffle = shuffled)
		self.time_predicted = np.zeros(len(self.time_test), dtype = int)
		self.index_predicted = np.zeros(len(self.index_test), dtype = int)

		self._reshape_data()

		self.scaler_transform_Time, self.scaler_transform_Index = StandardScaler(), StandardScaler()
		self.scaler_transform_Time_Train, self.scaler_transform_Time_Test = StandardScaler(), StandardScaler()
		self.scaler_transform_Index_Train, self.scaler_transform_Index_Test = StandardScaler(), StandardScaler()

		self.Model = None

	#This function extracts columns from the dataset based on if it belongs in fail_indexes or fail_times
	def _import_data(self, inputColumns, outputColumns):
		fail_times = np.empty(shape = len(np.array(self.dataSet[inputColumns[0]])), dtype = int)
		fail_indexes = np.empty(shape = len(np.array(self.dataSet[inputColumns[0]])), dtype = int)

		for i in inputColumns:
			fail_times = np.vstack([fail_times, np.array(self.dataSet[i])])

		for i in outputColumns:
			fail_indexes = np.vstack([fail_indexes, np.array(self.dataSet[i])])

		#There are placeholder arrays in the first row for vstack to work(ehhhh....)
		#The next two lines deletes these placeholder arrays.
		fail_indexes = np.delete(fail_indexes, 0, 0)
		fail_times = np.delete(fail_times, 0, 0)

		#If there was only one column, the data is transformed into a 1-D array.
		
		if len(fail_indexes) == 1:
			fail_indexes = fail_indexes[0]
		if len(fail_times) == 1:
			fail_times = fail_times[0]

		return fail_times, fail_indexes

	#This function cleans up the data
	#Essentially, it scales the data to have zero mean and variance
	#(I.E, fail_index and fail_times are scaled to values between 0 and 1)
	def _scale_data(self):

		self.scaler_transform_Time = StandardScaler().fit(self.fail_times)
		self.scaler_transform_Index = StandardScaler().fit(self.fail_indexes)
		self.scaler_transform_Time_Train = StandardScaler().fit(self.time_train)
		self.scaler_transform_Time_Test = StandardScaler().fit(self.time_test)
		self.scaler_transform_Index_Train = StandardScaler().fit(self.index_train)
		self.scaler_transform_Index_Test = StandardScaler().fit(self.index_test)

		self.fail_indexes = self.scaler_transform_Time.transform(self.fail_indexes)
		self.fail_times = self.scaler_transform_Index.transform(self.fail_times)
		self.time_train = self.scaler_transform_Time_Train.transform(self.time_train)
		self.time_test = self.scaler_transform_Time_Test.transform(self.time_test)
		self.index_train = self.scaler_transform_Index_Test.transform(self.index_train)
		self.index_test = self.scaler_transform_Index_Test.transform(self.index_test)

		# #The fail_times data is encoded as floating points(not present in the SYS1 file)
		# #This encodes the fail_times data as ints
		# label_encoder = LabelEncoder()
		# self.fail_indexes = label_encoder.fit_transform(self.fail_indexes)
		# self.fail_times = label_encoder.fit_transform(self.fail_times)

	#In case the data comes in 1-D arrays, this method rearranges the data into 2-D arrays.
	#Scikit-learn only trains on 2-D arrays.
	def _reshape_data(self):
		#index:X(input), fail:Y(output)
		self.fail_times = self.fail_times.reshape(-1, 1)
		self.fail_indexes = self.fail_indexes.reshape(-1, 1)

		self.time_train = self.time_train.reshape(-1, 1)
		self.time_test = self.time_test.reshape(-1, 1)
		self.index_train = self.index_train.reshape(-1, 1)
		self.index_test = self.index_test.reshape(-1, 1)

	def brute_force_CV(self, modelName, params):
		print(self.time_train)
		print(self.index_train.ravel())

		if (modelName == 'Logistic-Regression'):
			self.Model = GridSearchCV(LogisticRegression(), params, scoring = 'r2', cv=20)

		elif(modelName == 'Support-Vector-Machine'):
			self.Model = GridSearchCV(SVR(), params, scoring = 'r2', cv=20)
		elif (modelName == 'Multi-Layer-Perceptron'):
			print('Neural Network')
			self.Model = GridSearchCV(MLPRegressor(shuffle = False), params, scoring = 'r2', cv=20)
		print('Test1')
		self.Model = self.Model.fit(self.fail_times, self.fail_indexes.ravel())
		print('Test2')

	def train_model(self, modelName, algoName, maxIterations = 1000, coFactors = None):
		model_Name = ''
		if (modelName == 'Logistic-Regression'):
			self.Model = LogisticRegression(penalty = coFactors, max_iter = maxIterations, solver = algoName)
			model_Name = 'Logistic-Regression '

		elif(modelName == 'Support-Vector-Machine'):
			self.Model = SVR(kernel = algoName, max_iter = maxIterations)
			model_Name = 'Support-Vector-Machine '

		elif (modelName == 'Decision-Tree'):
			self.Model = DecisionTreeRegressor(criterion = algoName, max_depth = maxIterations)
			model_Name = 'Decision-Tree '

		elif (modelName == 'Random-Forest'):
			self.Model = RandomForestRegressor(n_estimators = maxIterations, criterion = algoName, max_depth = maxIterations)
			model_Name = 'Random-Forest '

		elif (modelName == 'Multi-Layer-Perceptron'):
			nodes = 136
			self.Model = MLPRegressor(hidden_layer_sizes = (nodes, nodes), activation = coFactors, solver = algoName, max_iter = maxIterations, learning_rate_init= 0.5)
			model_Name = 'Multi-Layer-Perceptron '

		self.Model = self.Model.fit(self.time_train, self.index_train.ravel())
		return model_Name

	def predict_model(self):
		self.time_predicted = self.time_test
		self.index_predicted = self.Model.predict(self.time_test)
		

	def calculate_statistics(self, numParams = 4):
		#previously fail_test & fail_predicted
		MSE = mean_squared_error(self.index_test.reshape(1, -1)[0], self.index_predicted, squared = True)
		RMSE = mean_squared_error(self.index_test.reshape(1, -1)[0], self.index_predicted, squared = False)
		AIC = 2*(numParams - math.log(RMSE))
		EVS = explained_variance_score(self.index_test.reshape(1, -1)[0], self.index_predicted)
		ME = max_error(self.index_test.reshape(1, -1)[0], self.index_predicted)
		MAE = mean_absolute_error(self.index_test.reshape(1, -1)[0], self.index_predicted)
		MedianSE = median_absolute_error(self.index_test.reshape(1, -1)[0], self.index_predicted)
		R2Score = r2_score(self.index_test.reshape(1, -1)[0], self.index_predicted)

		# print("Testing Datasets:")
		# print(self.fail_test.reshape(1, -1)[0])
		# print(self._convert_double_to_int(self.fail_predicted))

		# print("Testing Accuracy:")
		# print(test_acc_score)

		return [RMSE, MSE, AIC, EVS, ME, MAE, MedianSE, R2Score]

	def graph_data(self, graphObject):
		print('Running graph_data')
		# graphObject.build_discrete_graph(self.scaler_transform_Time_Train.inverse_transform(self.time_train), self.scaler_transform_Index_Train.inverse_transform(self.Model.predict(self.time_train)), 'Training_Set')
		# graphObject.build_discrete_graph(self.scaler_transform_Time_Test.inverse_transform(self.time_predicted), self.scaler_transform_Index_Test.inverse_transform(self.index_predicted), 'Testing_Set')
		# graphObject.build_continuous_graph(self.scaler_transform_Time.inverse_transform(self.fail_times), self.scaler_transform_Index.inverse_transform(self.fail_indexes), 'Original_Set')

		graphObject.build_discrete_graph(self.time_train, self.Model.predict(self.time_train), 'Training_Set')
		graphObject.build_discrete_graph(self.time_predicted, self.index_predicted, 'Testing_Set')
		graphObject.build_continuous_graph(self.fail_times, self.fail_indexes, 'Original_Set')

		graphObject.build_legend()
		graphObject.save_graph()



	@staticmethod
	def return_stat_names():
		return ['RMSE', 'MSE', 'AIC', 'EVS', 'ME', 'MAE', 'MedianSE', 'R2Score']

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
			
			for algorithms in algoNames:
				variantDictionary = {i : [] for i in coFactors}
				for variant in coFactors:
					# graphObject = graph(modelName = data_Sets + ' ' + modelName + ': ' + algorithms + ' applied to ' +  variant, path = os.getcwd())
					graphObject = graph(modelName = data_Sets + ' ' + modelName, algorithm = '; ' + algorithms, variant = ' applied to ' + variant)
					iteration = iter_diff
					model_Name = ''
					paramNames = [data_Sets, algorithms, variant, str(iteration)]
					statDictionary = {i : [] for i in statNames}

					fitFlag = True
					

					model = Model('model_data.xlsx', data_Sets, inputColumnName, outputColumnName, shuffled, testSize)
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
						
						
						model.predict_model()
						statsTable = model.calculate_statistics()
						model.graph_data(graphObject)
						# model.evaluate_graph(graphObject, modelName, algorithms, variant)
						# model.graph_data(graphObject)

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

			for algorithms in algoNames:
					graphObject = graph(modelName = data_Sets + ' ' + modelName, algorithm = '; ' + algorithms)
					iteration = iter_diff
					model_Name = ''
					paramNames = [data_Sets, algorithms, str(iteration)]
					# directory = DataSave.createDirectoryPath(paramNames)
					# os.makedirs(directory)
					statDictionary = {i : [] for i in statNames}
					# graphObject = graph(data_Sets + ' ' + modelName + ': ' + algorithms, os.getcwd())
					

					model = Model('model_data.xlsx', data_Sets, inputColumnName, outputColumnName, shuffled, testSize)

					try:
						model_Name = model.train_model(modelName, algorithms, iteration) 
					except Exception:
						continue
					
					while iteration <= iter_bound:
						model.predict_model()
						statTable = model.calculate_statistics()
						model.graph_data(graphObject)

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
	
	@staticmethod
	def runCV(params, dataSets, modelName, inputColumnName, outputColumnName, excelWriter, shuffled = False, testSize = 0.2):
		for data_Sets in dataSets:
			model = Model('model_data.xlsx', data_Sets, inputColumnName, outputColumnName, shuffled, testSize)
			graphObject = graph(modelName = data_Sets + ' ' + modelName, algorithm = '')
			model.brute_force_CV(modelName, params)
			model.predict_model()
			model.calculate_statistics()
			model.graph_data(graphObject)

statWriter = pd.ExcelWriter(os.path.join(os.getcwd(), 'Stats') + '.xlsx')

iter_diff = 1000
iter_bound = 1000

dataSets = ['SYS1', 'SYS2', 'SYS3']
# dataSets = ['SYS2', 'SYS3']
inputColumnName = ['FT']
outputColumnName = ['FN']

shuffled = False
testSize = 0.2

# dataSets = ['J1', 'J2', 'J3', 'J4', 'J5']
# inputColumnName = ['T']
# outputColumnName = ['CFC']

# dataSets = ['DS1', 'DS2']
# inputColumnName = ['T', 'F']
# outputColumnName = ['FC', 'E']


# try:
# 	threading.Thread(target = Model.runTestsWithCofactors, args = (dataSets, 'LR', ['newton-cg', 'lbfgs', 'sag', 'saga'], ['l1', 'l2', 'none'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTests, args = (dataSets, 'SVR', ['linear', 'poly', 'rbf', 'sigmoid'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTests, args = (dataSets, 'DTR', ['mse', 'friedman_mse', 'mae', 'poisson'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTests, args = (dataSets, 'RFR', ['mse', 'mae'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()
# 	threading.Thread(target = Model.runTestsWithCofactors, args = (dataSets, 'MLPR', ['lbfgs', 'adam'], ['identity', 'logistic', 'tanh', 'relu'], inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)).start()

# except:
# 	traceback.print_exe()

# #optimization algorithms
# algorithms = ['newton-cg', 'lbfgs', 'sag', 'saga']
# #penalities
# covariants = ['l1', 'l2', 'none']


# modelName = 'Logistic-Regression'
# Model.runTestsWithCofactors(dataSets, modelName, algorithms, covariants, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)


# algorithms = ['linear', 'poly', 'rbf', 'sigmoid']
# params = {''}
# modelName = 'Support-Vector-Machine'
# Model.runTests(dataSets, modelName, algorithms, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)



# algorithms = ['mse', 'friedman_mse', 'mae', 'poisson']
# modelName = 'Decision-Tree'
# Model.runTests(dataSets, modelName, algorithms, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)


# algorithms = ['mse', 'mae']
# modelName = 'Random-Forest'
# Model.runTests(dataSets, modelName, algorithms, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)



algorithms = ['lbfgs', 'adam']
covariants = ['identity', 'logistic', 'tanh', 'relu']
hidden_layer_sizes = [100, 100]
alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
beta_1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
beta_2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
early_stopping = [False, True]
validation_fraction = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
params = {'hidden_layer_sizes': hidden_layer_sizes, 'activation': covariants, 'solver': algorithms, 'alpha' : alpha, 'beta_1' : beta_1, 'beta_2' : beta_2, 'early_stopping' : early_stopping, 'validation_fraction' : validation_fraction}

modelName = 'Multi-Layer-Perceptron'
Model.runTestsWithCofactors(dataSets, modelName, algorithms, covariants, inputColumnName, outputColumnName, iter_diff, iter_bound, statWriter, shuffled, testSize)
# Model.runCV(params = params, dataSets = dataSets, modelName = modelName, inputColumnName = inputColumnName, outputColumnName = outputColumnName, excelWriter = statWriter)