import matplotlib.pyplot as plt
import numpy as np
import torch

'''
Class: Data
input_variable() will return a two dimension data set, [sin(x), cos(x)]
get() will return two torch tensor,x and y, where y = sin(x)**2 
matrix() will return a matrix with dimension [m ,m] for calculating the Loss
@Author: Naicheng Deng
'''


class Data():

    def __init__(self, points, lower, upper):
        self.points = points
        self.upper = upper
        self.lower = lower
        self.x = np.linspace(self.lower, self.upper,self.points, endpoint=False)
        self.x = torch.from_numpy(self.x)
        self.x = torch.unsqueeze(self.x, dim=1)

        self.y = torch.square(torch.sin(self.x))
        self.matrics = torch.zeros([len(self.x), len(self.x)])
        self.input_var = torch.zeros((len(self.x), 2))

    def input_variables(self):
        for i in range(len(self.x)):
            self.input_var[i][0] = torch.cos(self.x[i])
            self.input_var[i][1] = torch.sin(self.x[i])
        return self.input_var

    def get(self):
        return self.x, self.y

    def matrix(self):
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                self.matrics[i][j] = torch.abs(torch.sin(self.x[i] - self.x[j]))
        return self.matrics

    def plot_data(self):
        plt.plot(self.x, self.y)
        plt.xlabel("$ \\theta $")
        plt.ylabel()
        plt.show()
