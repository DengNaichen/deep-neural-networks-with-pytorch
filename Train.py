import matplotlib.pyplot as plt
from MyLossFunction import cross_entropy_loss, free_energy

def fit(x, y, optimizer, net, epochs, diagram = False):
    if diagram == True:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.4, wspace=0.25)

    loss_points = []    
    for i in range(epochs):
        yhat = net(x)
        loss = cross_entropy_loss(yhat, y)
        loss_points.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and diagram == True: 
            ax1.cla()
            ax2.cla()
            ax1.scatter(x.data.numpy(), y.data.numpy(), marker=".")
            ax1.plot(x.data.numpy(), yhat.data.numpy(), 'r-', lw=1)
            ax1.set_xlabel (" \\theta")
            ax2.set_xlim([0, epochs])
            ax2.text(150, 0.5, 'Loss=%.5f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
            ax2.text(300, 0.5, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
            ax2.plot(range(epochs)[:i], loss_points[:i])
            ax2.set_xlabel("epochs")
            ax2.set_ylabel("Loss")
            plt.pause(0.1)
    if diagram == True:
        plt.ioff()
        plt.show()
    return net


def train_free_energy(net, x, rho, lambd, optimizer, matrix, epochs, diagram= True):
    if diagram == True:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.4, wspace=0.25)
    loss_points = []

    for i in range(epochs):
        yhat = net(x)
        loss = free_energy(yhat, x, matrix, rho, lambd)
        loss_points.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0 and diagram == True:
            ax1.cla()
            ax2.cla()
            ax1.scatter(x.data.numpy(), yhat.data.numpy(), marker=".")
            ax1.set_xlabel("$\\theta$")
            ax2.set_ylabel("$f$")
            ax1.set_title("Distribution Function")
            ax2.set_xlim([0, epochs])
            ax2.text(epochs/4, 1.1, 'Loss=%.7f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
            ax2.text(epochs/1.5, 1.1, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
            ax2.set_xlabel("epochs")
            ax2.set_ylabel("Loss(Free Energy)")
            ax2.set_title("Free Energy with $\\rho =$ {} and $\\lambda = $ {}".format(rho, lambd))
            ax2.plot(range(epochs)[:i], loss_points[:i])
            plt.pause(0.1)
    if diagram == True:
        plt.ioff()
        plt.show()
    return net

# why we need tensor data: we only want to use the "tensor" data in tensor without gradient data,
# .data can do this work