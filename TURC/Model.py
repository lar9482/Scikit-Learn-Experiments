import pandas as pd
import numpy as np
import sklearn
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

class Model:
	def __init__(self, dataSetName, sheetName, testSize = 0.2):
		self.dataSet = pd.read_excel(dataSetName, sheetName)
		self.fail_indexes = np.array(self.dataSet['FN'])
		self.fail_times = np.array(self.dataSet['FT'])
		
		self.scale_data()

		self.index_train, self.index_test, self.fail_train, self.fail_test = train_test_split(self.fail_indexes, self.fail_times, test_size = testSize)

		self.reshape_data()

		self.Model = None

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
	def reshape_data(self):
		self.index_train = self.index_train.reshape(-1, 1)
		self.index_test = self.index_test.reshape(-1, 1)
		self.fail_train = self.fail_train.reshape(-1, 1)
		self.fail_test = self.fail_test.reshape(-1, 1)


	def train_logistic_regression(self, Penalty = 'l2', maxIterations = 100, Solver = 'lbfgs'):
		self.Model = LogisticRegression(penalty = Penalty, max_iter = maxIterations, solver = Solver)
		self.Model.fit(self.index_train, self.fail_train)

	def predict_logistic_regression(self, testSet):
		if (testSet == 'index'):
			#print(self.index_test)
			print('Predicted output on index_test:')
			print(self.Model.predict(self.index_test))

			print("Index Test RMSE:")
			print(self.calculate_RMSE(self.index_test, self.Model.predict(self.index_test)))

		elif (testSet == 'fail'):
			#print(self.fail_test)
			print('Predicted output on fail_test:')
			print(self.Model.predict(self.fail_test))

			print("Fail test RMSE:")
			print(self.calculate_RMSE(self.fail_test, self.Model.predict(self.fail_test)))

	def train_LRSD(self, maxIterations = 100):
		self.Model = SGDClassifier(loss = 'log', max_iter = maxIterations)
		self.Model.fit(self.index_train, self.fail_train)


	def predict_LRSD(self, testSet):
		if (testSet == 'index'):
			#print(self.index_test)
			print('Predicted output on index_test:')
			print(self.Model.predict(self.index_test))

			print(self.Model.decision_function(self.index_test))

		elif(testSet == 'fail'):
			#print(self.fail_test)
			print('Predicted output on fail_test:')
			print(self.Model.predict(self.fail_test))

			print(self.Model.decision_function(self.fail_test))


	def calculate_RMSE(self, test_set, predicted_set):
		value = 0
		for i in range(0, len(predicted_set)):
			temp = predicted_set[i] - test_set[i][0]
			value = value + (temp*temp)

		value = value/len(predicted_set)
		value = math.sqrt(value)

		return value

	def print_params(self):
		print(self.Model.get_params())


model = Model('model_data.xlsx', 'SYS1')

max_iteration = 1000
solver = 'saga'
penalty = 'l1'
model.train_logistic_regression(penalty, max_iteration, solver)
model.predict_logistic_regression('fail')

# max_iteration = 1000
# model.train_LRSD(max_iteration)
# model.predict_LRSD('fail')