import matplotlib.pyplot as plt
import numpy


def plot_set_up():
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plt.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.6, wspace=0.25)
    return ax1, ax2


def iteration_fit(ax1, ax2, x, y, yhat, epochs, Loss_points, i, loss):
    if i % 50 == 0:
        ax1.cla()
        ax2.cla()
        ax1.scatter(x.data.numpy(), y.data.numpy(), marker=".")
        ax1.plot(x.data.numpy(), yhat.data.numpy(), 'r-', lw=1)
        ax1.set_xlabel (" \\theta")
        ax2.set_xlim([0, epochs])
        ax2.text(epochs/4., 0.5, 'Loss=%.5f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
        ax2.text(epochs/1.5, 0.5, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
        ax2.plot(range(epochs)[:i], Loss_points[:i])
        ax2.set_xlabel("epochs")
        ax2.set_ylabel("Loss")
        plt.pause(0.1)
    else: pass

def iteration_fit(ax1, ax2, x, yhat, epochs, Loss_points, i, loss, rho, lambd):
    if i% 500 == 0:
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
    else: pass
    

def end():
    plt.ioff()
    plt.show()