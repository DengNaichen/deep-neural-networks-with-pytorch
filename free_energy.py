import torch
from Net import Net
from Data import Data
from Train import train_free_energy, fit

data_set = Data(512)
x, y = data_set.get()
matrix = data_set.matrix()
Layers = [1, 16, 16, 16, 16, 1]
net = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# data_set.plot_data()
# net = fit(x, y, optimizer, net, 500, diagram=False)  # todo: tell Yousef that cross entropy doesn't work


rho = 8
lambd = 6
train_free_energy(net, x, rho, lambd, optimizer, matrix, epochs=50000, diagram=True)


