from numpy import linspace,exp,vstack
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import uniform, seed

def main():
    seed(1234)
    x = uniform(-2,2,100)
    y = uniform(-2,2,100)
    data = vstack((x*exp(-x**2-y**2),0.5*x*exp(-x**2-y**2),0.2*x*exp(-x**2-y**2)))
    xi = linspace(min(x), max(x))
    yi = linspace(min(y), max(y))
    zi = []
    numframes = data.shape[0]
    for ii in range(numframes):
        zi.append(griddata((x, y), data[ii], (xi[None,:], yi[:,None]), method='cubic'))

    fig = plt.figure()
    im = plt.contour(xi, yi, zi[0], 15, linewidths=0.5, colors='k')
    ax = fig.gca()
    ani = animation.FuncAnimation(fig, update_contour_plot, frames=xrange(numframes), fargs=(zi, ax, fig, xi, yi), interval=1000)
    plt.colorbar(im)
    plt.show()
    return ani


def update_contour_plot(i, data,  ax, fig, xi, yi):
    ax.cla()
    im = ax.contour(xi, yi, data[i], 15, linewidths=0.5, colors='k')
    plt.title(str(i))
    return im,

main()