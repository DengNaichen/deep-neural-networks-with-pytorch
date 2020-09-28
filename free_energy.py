# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
torch.manual_seed(2)    # reproducible
# %%
class Data():

    def __init__(self, points):
        self.points = points
        self.x = torch.unsqueeze(torch.linspace(0.001, 2 * np.pi, self.points), dim=1)  # x data (tensor), shape=(100, 1)
        self.y = torch.square(torch.sin(self.x))  # noisy y data (tensor), shape=(100, 1)
        self.input1 = torch.sin(self.x)
        self.input2 = torch.cos(self.y)

    def get(self):
        return self.input1, self.input2

    def matrix(self):
        self.matrics = torch.zeros ([self.points, self.points])
        for i in range (self.points):
            for j in range (self.points):
                self.matrics[i][j] = torch.abs(torch.sin(self.x[i] - self.x[j]))
        return self.matrics

    def plot_data(self):
        plt.plot(self.x, self.y)
        plt.xlabel("$ \\theta $")
        plt.ylabel()
        plt.show()


# %%
# define the class of my neural network
class Net(torch.nn.Module):

    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.relu(linear_transform(activation))
            else:
                activation = torch.sigmoid(linear_transform(activation))
        return activation


# %%
def my_loss(loss_type, yhat, y):
    L = len(yhat)
    if loss_type == "cross":
        # Loss = - torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))  # todo, figure the dimension
        Loss = - (1/L)*(torch.mm(y.T, torch.log(yhat)) + torch.mm((1-y).T, torch.log(1-yhat)))  # todo, 2pi is not needed any more?
    if loss_type == "mse":
        Loss = torch.mean((y - yhat) ** 2)

    return Loss

def free_energy(yhat, rho, lambd, matrix):
    L = len(yhat)
    # first_term = torch.mean(yhat * torch.log(rho*yhat))
    first_term = (1/L) * (torch.mm(yhat.T, torch.log(rho*yhat)))   
    # second_term = (1/L**2) * torch.mm(torch.mm(yhat.T, matrix), yhat)  #todo, check again
    third_term = lambd * torch.square(torch.mean(yhat) - 1) 
    return first_term +  third_term

#%%
def train(x, y, optimizer, Loss_function, epochs):
    plt.ion()
    Loss_points = []
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for i in range(epochs):

        yhat = net(x)  # input x and predict based on x
        loss = my_loss(Loss_function, yhat, y)
        a = loss.item()
        Loss_points.append(a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            ax1.cla()
            ax2.cla()
            ax1.scatter(x.data.numpy(), y.data.numpy(), marker=".")
            ax1.plot(x.data.numpy(), yhat.data.numpy(), 'r-', lw=1)
            ax1.set_xlabel (" \\theta")
            ax2.set_xlim([0, epochs])
            ax2.text(150, 0.5, 'Loss=%.5f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
            ax2.text(300, 0.5, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
            ax2.plot(range(epochs)[:i], Loss_points[:i])
            ax2.set_xlabel("epochs")
            ax2.set_ylabel("Loss")
            plt.pause(0.1)

    plt.ioff()
    plt.show()

#%%
def train_free_e (x, rho, lambd, optimizer, epochs, matrix):
    plt.ion()
    Loss_points = []
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    plt.subplots_adjust(left=0.15,bottom=0.1,top=0.9,right=0.95,hspace=0.6,wspace=0.25)

    for i in range(epochs):

        yhat = net(x) 
        loss = free_energy(yhat, rho, lambd, matrix)
        Loss_points.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            ax1.cla()
            ax2.cla()
            ax1.scatter(x.data.numpy(), yhat.data.numpy(), marker= ".")
            ax1.set_xlabel("$\\theta$")
            ax2.set_ylabel("$f$")
            ax1.set_title("Distribution Function")
            ax2.set_xlim([0, epochs])
            ax2.text(epochs/4, 1.1, 'Loss=%.7f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
            ax2.text(epochs/1.5, 1.1, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
            ax2.set_xlabel("epochs")
            ax2.set_ylabel("Loss(Free Energy)")
            ax2.set_title("Free Energy with $\\rho =$ {} and $\\lambda = $ {}".format(rho, lambd))
            ax2.plot(range(epochs)[:i], Loss_points[:i])
            plt.pause(0.1)

    plt.ioff()
    plt.show()
# %%
data_set = Data(128)
x, y = data_set.get()
matrix = data_set.matrix()
print (x.size())
print (y.size())
print (matrix.size())
Layers = [1, 16, 16, 16, 16, 1]
net = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#%%
rho = 5
lambd = 2
# train(x, y, optimizer, "cross", 500)
train_free_e (x, rho, lambd, optimizer, 10000, matrix)


#todo, figure out how to improve the accuracy
#todo, figure out how Pythorch get the gradient for each parameters?
#todo, which parts of my previous model are wrong?
