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


def _import_data(inputColumns, outputColumns, dataSet):
	fail_indexes = np.empty(shape = len(np.array(dataSet[inputColumns[0]])), dtype = int)
	fail_times = np.empty(shape = len(np.array(dataSet[inputColumns[0]])), dtype = int)

	for i in inputColumns:
		fail_indexes = np.vstack([fail_indexes, np.array(dataSet[i])])

	for i in outputColumns:
		fail_times = np.vstack([fail_times, np.array(dataSet[i])])

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


sheetName = 'SYS3'
dataSetName = 'model_data.xlsx'
inputColumn = ['FT']
outputColumn = ['FN']

dataSet = pd.read_excel(dataSetName, sheetName)

fail_times, fail_indexes = _import_data(inputColumn, outputColumn, dataSet)

graphTitle = sheetName + ' Graph'
graphObject = graph(graphTitle, os.getcwd())
graphObject.build_discrete_graph(fail_times, fail_indexes, name = graphTitle + ' cumulative failure points')
graphObject.set_x_and_y_bound(fail_times[len(fail_times) - 1], len(fail_indexes))

graphObject.build_legend()
graphObject.save_graph()


