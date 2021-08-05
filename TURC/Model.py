import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import os
import traceback
import threading

# import graph
# from graph import *

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
	def __init__(self, dataSetName, sheetName, algoNames, shuffleData = True, testSize = 0.2):
		self.dataSet = pd.read_excel(dataSetName, sheetName)
		self.fail_indexes = np.array(self.dataSet['FN'])
		self.fail_times = np.array(self.dataSet['FT'])

		self._scale_data()

		self.index_train, self.index_test, self.fail_train, self.fail_test = train_test_split(self.fail_indexes, self.fail_times, test_size = testSize, shuffle = shuffleData)
		self.index_predicted = np.zeros(len(self.index_test), dtype = int)
		self.fail_predicted = np.zeros(len(self.fail_test), dtype = int)

		self._reshape_data()

		self.Model = None


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

	def predict_model(self, testSet):
		if (testSet == 'index'):	
			self.index_predicted = self.Model.predict(self.index_test)
			
		elif (testSet == 'fail'):
			self.fail_predicted = self.Model.predict(self.fail_test)

	# def graph_results(self, testData, predictedData, modelName, names, path):
	# 	try:
	# 		Graph = graph(testData, predictedData, modelName, names, path)
	# 		Graph.save_graph()
	# 	except:
	# 		traceback.print_exc()


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

		print("Testing Datasets:")
		print(self.fail_test.reshape(1, -1)[0])
		print(self._convert_double_to_int(self.fail_predicted))

		train_acc_score = accuracy_score(self.fail_train, self._convert_double_to_int(self.Model.predict(self.fail_train.reshape(-1, 1))))
		test_acc_score = accuracy_score(self.fail_test.reshape(1, -1)[0], self._convert_double_to_int(self.fail_predicted))

		print("Testing Accuracy:")
		print(test_acc_score)

		return [RMSE, MSE, AIC, EVS, ME, MAE, MedianSE, R2Score, train_acc_score, test_acc_score]

	@staticmethod
	def return_stat_names():
		return ['RMSE', 'MSE', 'AIC', 'EVS', 'ME', 'MAE', 'MedianSE', 'R2Score', 'train_acc_score', 'test_acc_score']

	@staticmethod
	def update_stat_file(sheetName, totalDictionary, excelWriter):
		df = pd.DataFrame(totalDictionary)
		df.to_excel(excelWriter, sheet_name = sheetName, index = False)
		excelWriter.save()
	
	@staticmethod
	def construct_covariant_stat_dict(dataDictionary, statNames, iter_bound):
		totalDictionary = {}
		index = 0
		totalDictionary['iteration'] = []
		iteration = 200

		for i in range(0, len(statNames)):
			while iteration <= iter_bound:
				totalDictionary['iteration'].append(iteration)
				iteration = iteration + 200

			totalDictionary['iteration'].append(' ')
			iteration = 200
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
	def construct_stat_dict(dataDictionary, statNames, iter_bound):
		totalDictionary = {}
		totalDictionary['iteration'] = []
		iteration = 200

		for i in range(0, len(statNames)):
			while iteration <= iter_bound:
				totalDictionary['iteration'].append(iteration)
				iteration = iteration + 200

			totalDictionary['iteration'].append(' ')
			iteration = 200
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
	def runTestsWithCofactors(dataSets, modelName, algoNames, coFactors, excelWriter, shuffleData = True):
		for data_Sets in dataSets:
			
			dataDictionary = {i : {} for i in algoNames}
			statNames = Model.return_stat_names()
			iter_bound = 2000
			for algorithms in algoNames:
				variantDictionary = {i : [] for i in coFactors}
				for variant in coFactors:
					iteration = 200
					model_Name = ''
					paramNames = [data_Sets, algorithms, variant, str(iteration)]
					statDictionary = {i : [] for i in statNames}

					fitFlag = True
					model = Model('model_data.xlsx', data_Sets, algorithms, shuffleData)
					try:
						model_Name = model.train_model(modelName, algorithms, iteration, variant)
					except Exception:
						fitFlag = False
						failIndex = 200
						for i in statNames:
							while failIndex <= iter_bound:
								statDictionary[i].append('error')
								failIndex = failIndex + 200
							failIndex = 200
						print('failed to fit model')
					
					while (iteration <= iter_bound) and (fitFlag is not False):
						convergeFlag = 'ConvergeSuccess'
						
						try:
							model_Name = model.train_model(modelName, algorithms, iteration, variant)
						except ConvergenceWarning:
							convergeFlag = 'ConvergeFail'

						print(convergeFlag)
						model.predict_model('fail')
						statsTable = model.calculate_statistics()

						index = 0
						for key in statDictionary:
							statDictionary[key].append(statsTable[index])
							index = index + 1

						iteration = iteration + 200
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

			completeDictionary = Model.construct_covariant_stat_dict(dataDictionary, statNames, iter_bound)
			Model.update_stat_file(modelName + ' ' + data_Sets, completeDictionary, excelWriter)

	@staticmethod
	def runTests(dataSets, modelName, algoNames, excelWriter, shuffleData = True):
		for data_Sets in dataSets:
			dataDictionary = {i : [] for i in algoNames}
			iter_bound = 2000
			statNames = Model.return_stat_names()
			for algorithms in algoNames:

					iteration = 200
					model_Name = ''
					paramNames = [data_Sets, algorithms, str(iteration)]
					# directory = DataSave.createDirectoryPath(paramNames)
					# os.makedirs(directory)
					statDictionary = {i : [] for i in statNames}
					
					model = Model('model_data.xlsx', data_Sets, algorithms, shuffleData)

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

						model.predict_model('fail')
						statTable = model.calculate_statistics()

						#add_to_statistics_dictionary
						index = 0
						for key in statDictionary:
							statDictionary[key].append(statTable[index])
							index = index + 1

						iteration = iteration + 200
						print('Processed ' + ' '.join([i for i in paramNames]))
						paramNames = [data_Sets, algorithms, str(iteration)]

					for key in statDictionary:
						dataDictionary[algorithms].append(statDictionary[key])
						dataDictionary[algorithms].append(' ')
					#end(algorithm loop)

			completeDictionary = Model.construct_stat_dict(dataDictionary, statNames, iter_bound)	
			Model.update_stat_file(modelName + ' ' + data_Sets, completeDictionary, excelWriter)



statWriter = pd.ExcelWriter(os.path.join(os.getcwd(), 'Stats') + '.xlsx')
dataSets = ['SYS1', 'SYS2', 'SYS3']
inputColumnName = 'FN'
outputColumnName = 'FT'

# dataSets = ['J1', 'J2', 'J3', 'J4', 'J5']
# inputColumnName = 'T'
# outputColumnName = 'CFC'

shuffleData = True


try:
	threading.Thread(target = Model.runTestsWithCofactors, args = (dataSets, 'LR', ['newton-cg', 'lbfgs', 'sag', 'saga'], ['l1', 'l2', 'none'], statWriter, shuffleData)).start()
	threading.Thread(target = Model.runTests, args = (dataSets, 'SVR', ['linear', 'poly', 'rbf', 'sigmoid'], statWriter, shuffleData)).start()
	threading.Thread(target = Model.runTests, args = (dataSets, 'DTR', ['mse', 'friedman_mse', 'mae', 'poisson'], statWriter, shuffleData)).start()
	threading.Thread(target = Model.runTests, args = (dataSets, 'RFR', ['mse', 'mae'], statWriter, shuffleData)).start()
	threading.Thread(target = Model.runTestsWithCofactors, args = (dataSets, 'MLPR', ['lbfgs', 'adam'], ['identity', 'logistic', 'tanh', 'relu'], statWriter, shuffleData)).start()

except:
	traceback.print_exe()

# #optimization algorithms
# algorithms = ['newton-cg', 'lbfgs', 'sag', 'saga']
# #penalities
# covariants = ['l1', 'l2', 'none']

# modelName = 'LR'
# Model.runTestsWithCofactors(dataSets, modelName, algorithms, covariants, statWriter, shuffleData)

# #kernels
# algorithms = ['linear', 'poly', 'rbf', 'sigmoid']
# modelName = 'SVR'
# Model.runTests(dataSets, modelName, algorithms, statWriter, shuffleData)


# #criterion
# algorithms = ['mse', 'friedman_mse', 'mae', 'poisson']
# modelName = 'DTR'
# Model.runTests(dataSets, modelName, algorithms, statWriter, shuffleData)


# #criterion
# algorithms = ['mse', 'mae']
# modelName = 'RFR'
# Model.runTests(dataSets, modelName, algorithms, statWriter, shuffleData)


# #activation
# covariants = ['identity', 'logistic', 'tanh', 'relu']
# #solver
# algorithms = ['lbfgs', 'adam']

# modelName = 'MLPR'
# Model.runTestsWithCofactors(dataSets, modelName, algorithms, covariants, statWriter, shuffleData)