#%%
import torch
from Net import Net
from Data import Data
from Train import fit

#%%
data_set = Data(128)
x, y = data_set.get()
matrix = data_set.matrix()
Layers = [1, 16, 16, 16, 16, 1]
net = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

fit(x, y, optimizer, net, 500)
