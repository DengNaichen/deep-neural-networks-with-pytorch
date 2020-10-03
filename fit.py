#%%
import torch
from Net import Net
from Data import Data
from Train import fit
import matplotlib.pyplot as plt

data_set = Data(128)
x, y = data_set.get()
matrix = data_set.matrix()
Layers = [1, 16, 16, 16, 16, 1]
net = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# data_set.plot_data()
net = fit(x, y, optimizer, net, 500, diagram=True)  # todo: tell Yousef that cross entropy doesn't work

# print (net(x))
# plt.plot (x, net(x))