import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import axes3d  # implicitly required for projection='3d' call...
from matplotlib import cm
from itertools import cycle


def animate_1d(x, ys, animate_times, pause, labels=None):
    """
    Animates simple 1d line plots over time.
    :param x: The x coordinates of the plot
    :param ys: A list of a list of y coordinates to plot. Each contained list must be the same length and a 1d array.
    :param animate_times: A list of times corresponding to the time of each of the solutions in ys.
    :param pause: The time in ms to pause between to consecutive plots
    :return: None
    """
    fig = plt.figure()
    min_y = min(min(np.amin(vals.real) for vals in y) for y in ys)
    max_y = max(max(np.amax(vals.real) for vals in y) for y in ys)
    ax = plt.axes(xlim=(min(x), max(x)), ylim=(min_y, max_y))
    if labels is None:
        labels = []
    labels.extend(["Plot {}".format(i) for i in range(len(labels), len(ys))])
    lines = [ax.plot([], [], lw=2, color=color, label=label)[0]
             for _, color, label in zip(range(len(ys)),
                                        cycle(['r', 'b', 'g', 'k', 'm', 'c', 'y']),
                                        labels)]
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return (*lines), time_text

    # animation function.  This is called sequentially
    def animate(i):
        for line, y in zip(lines, ys):
            line.set_data(x, y[i].real)
        time_text.set_text("Solution at time=" + str(animate_times[i]))
        return (*lines), time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.

    # because else the object will not get creates and nothing will show! ...
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(animate_times), interval=pause, blit=True)
    plt.legend()
    plt.show()


def animate_2d(x1, x2, y, animate_times, pause):
    fig = plt.figure()
    ax = plt.axes()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    plt.pcolormesh(x1, x2, y[0], vmin=-1, vmax=1)  # fix color bar range over different figures
    plt.colorbar()

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        return time_text,

    # animation function.  This is called sequentially
    def animate(i):
        pdata = plt.pcolormesh(x1, x2, y[i], vmin=-1, vmax=1)  # fix color bar range over different figures
        time_text.set_text("Solution at time=" + str(animate_times[i]))
        return pdata, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.

    # because else the object will not get creates and nothing will show! ...
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(animate_times), interval=pause, blit=True)

    plt.show()


# better version may be possible using mayavi (see http://docs.enthought.com/mayavi/mayavi/mlab_animating.html)
def animate_2d_surface(x1, x2, ys, animate_times, pause):
    x1, x2 = np.meshgrid(x1, x2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # implicitly requires import! (from mpl_toolkits.mplot3d import axes3d)

    ax.plot_surface(x1, x2, ys[0].real, cmap=cm.PuBu, linewidth=0)
    ax.set_zlim(-2, 2)

    # animation function.  This is called sequentially
    def animate(i):
        ax.clear()
        surf = ax.plot_surface(x1, x2, ys[i].real, cmap=cm.PuBu, linewidth=0, alpha=1.)
        ax.set_zlim(-2, 2)

        time_text = ax.text(0.02, 0.95, 0.02, "Solution at time=" + str(animate_times[i]), transform=ax.transAxes)
        return time_text, surf

    # because else the object will not get creates and nothing will show! ...
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(animate_times), interval=pause, blit=False)  # blit=True will not work!

    plt.show()
