import torch


'''
Define my own loss functions
cross_entropy_loss function 
mse_loss function
and the loss function form for free energy
yhat: the real output of NN
y: traget output of NN
x: input of NN
@Author: Naicheng Deng
'''

def cross_entropy_loss(yhat, y):
    L = len(yhat)
    Loss = - (1 / L) * (torch.mm(y.T, torch.log(yhat))   # todo, 2pi is not needed any more?
                        + torch.mm((1 - y).T, torch.log(1 - yhat)))
    return Loss

def mse_loss(yhat, y):
    Loss = torch.mean((y - yhat) ** 2)

    return Loss

def free_energy(yhat, matrix, rho, lambd):
    L = len(yhat)
    first_term = (1/L) * (torch.mm(yhat.T, torch.log(rho * yhat)))
    second_term = (rho/L**2) * torch.mm(torch.mm(yhat.T, matrix), yhat)
    # third_term = lambd * torch.square(torch.mean(yhat) - 1)  ## todo
    Loss = first_term + second_term + third_term

    return Loss


