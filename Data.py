
import matplotlib.pyplot as plt
import numpy as np
import torch

'''
Class: Data
input_variable() will return a two dimension data set, [sin(x), cos(x)]
get() will reture two torch tensor,x and y, where y = sin(x)**2 
matrix() will return a matrix with dimension [m ,m] for calculating the Loss
@Author: Naicheng Deng
'''

class Data():

    def __init__(self, points):
        self.points = points
        self.x = torch.unsqueeze(torch.linspace(0.00001, 2 * np.pi, self.points), dim=1)  # x data (tensor), shape=(100, 1)
        self.y = torch.square(torch.sin(self.x))  # noisy y data (tensor), shape=(100, 1)

    def input_variables(self):
        self.input_var = torch.zeros((self.points, 2))
        for i in range (self.points):
            self.input_var[i][0] = torch.cos(self.x[i])
            self.input_var[i][1] = torch.sin(self.x[i])
        return self.input_var

    def get(self):
        return self.x, self.y

    def matrix(self):
        self.matrics = torch.zeros([self.points, self.points])
        for i in range(self.points):
            for j in range(self.points):
                self.matrics[i][j] = torch.abs(torch.sin(self.x[i] - self.x[j]))
        return self.matrics

    def plot_data(self):
        plt.plot(self.x, self.y)
        plt.xlabel("$ \\theta $")
        plt.ylabel()
        plt.show()

