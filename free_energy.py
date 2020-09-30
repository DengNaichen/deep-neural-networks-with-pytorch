# %%
import torch
from Net import Net
from Data import Data
from Train import train_free_energy

data_set = Data(128)
x, y = data_set.get()
matrics = data_set.matrix()
Layers = [1, 16, 16, 16, 16, 1]
net = Net(Layers)
learning_rate = 0.07
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

rho = 5
lambd = 6
train_free_energy(net, x, rho, lambd, optimizer, matrics, epochs=5000)

# todo, figure out how to improve the accuracy
# todo, figure out how Pytorch get the gradient for each parameters?
# todo, which parts of my previous model are wrong?
