#%%
import torch
import numpy as np
from numpy import sin, pi

'''
Define my loss functions
cross_entropy_loss function 
mse_loss function
and the loss function form of free energy
yhat: the real output of NN
y: traget output of NN
x: input of NN
@Author: Naicheng Deng
'''


def cross_entropy_loss(yhat, y):
    L = len(yhat)
    loss = - (1 / L) * (torch.mm(y.T, torch.log(yhat)) # here we don't need 2pi, but we need to divide 2pi when calculate the integration
                        + torch.mm((1 - y).T, torch.log(1 - yhat)))
    return loss


def mse_loss(yhat, y):
    loss = torch.mean((y - yhat) ** 2)

    return loss


def free_energy(yhat, x, matrix, rho, lambd):
    L = len(yhat)
    first_term = (2 * pi / L) * (torch.mm(yhat.T, torch.log(rho * yhat)))
    second_term = (2 * pi**2 * rho / L ** 2) * torch.mm(torch.mm(yhat.T, matrix), yhat)  # todo
    third_term = lambd * torch.square(torch.trapz(yhat.T, x.T) - 1)  # the only reason we need x
    loss = first_term \
         + second_term \
         + third_term
    

    return loss


