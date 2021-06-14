import os
import pandas as pd
import numpy as np


def createDirectoryPath(names):
	Dir = os.getcwd()
	for i in names:
		Dir = os.path.join(Dir, i)

	return Dir
