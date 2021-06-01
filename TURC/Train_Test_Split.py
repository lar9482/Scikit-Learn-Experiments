from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dataset = pd.read_excel('model_data.xlsx')

#Separating the data into two labels based on the name of the data
x = np.array(dataset.FT)
y = np.array(dataset.IF)

#80% of the data is used for training
#20% of the data is used for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
print(x_train.shape)
print(x_train)
x_train = x_train.reshape(-1, 1)
print(x_train.shape)
print(x_train)

print(x_train[0][0])

