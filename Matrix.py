import numpy as np


class Matrix:
    def __init__(self, n, m):
        self._matrix = np.zeros((n, m))
    
    def set_value(self, i, j, value):
        self._matrix[i][j] = value
    
    def get_value(self, i, j):
        return self._matrix[i][j]
