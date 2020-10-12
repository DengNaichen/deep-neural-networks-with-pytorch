import matplotlib.pyplot as plt
import torch
from Net import Net
from Data import Data
from Train import train_free_energy, fit, train_free_energy_two_inputs
import numpy as np
torch.manual_seed(3)

data_set = Data(512)
x, y = data_set.get()
matrix = data_set.matrix()
input_var = data_set.input_variables()
Layers = [2, 16, 16, 16, 16, 1]
net = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# print (input_var)

rho = 8
lambd = 6
net = train_free_energy_two_inputs(net, x,  input_var, rho, lambd, optimizer, matrix, epochs=5000, diagram=True)

## todo: 想想为什么y的范围还可以在0到2pi
#%%
y = net(input_var)
plt.plot(x, y)
plt.show()