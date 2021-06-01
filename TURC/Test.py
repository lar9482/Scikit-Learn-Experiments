import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_excel('model_data.xlsx')
fail_times = np.array(dataset['FT']) # reads in the 'FT' column
inter_fails = np.array(dataset['IF']) #reads in the 'IF' column


print(fail_times)
print(fail_times.reshape(1, -1))
# print(fail_times[1]) #prints the 2nd item in the 'FT' column
# print(dataset.iloc[0:3]) #prints the 1st three rows of the dataset

# #Graph example(Graphs the dataset)
x = [i for i in range(0, len(fail_times))]
y = [fail_times[i] for i in range(0, len(fail_times))]

plt.plot(x, y)
plt.xlabel('index')
plt.ylabel('fail_times')
plt.show()


