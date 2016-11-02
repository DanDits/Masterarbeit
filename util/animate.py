import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def animate_1d(x, ys, animate_times, pause):
    fig = plt.figure()
    ax = plt.axes(xlim=(min(x), max(x)), ylim=(min(np.amin(vals.real) for vals in ys),
                                               max(np.amax(vals.real) for vals in ys)))
    line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(x, ys[i].real)
        time_text.set_text("Solution at time=" + str(animate_times[i]))
        return line, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.

    # because else the object will not get creates and nothing will show! ...
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(animate_times), interval=pause, blit=True)
    plt.show()


def animate_2d(x1, x2, ys, animate_times, pause):
    fig = plt.figure()
    ax = plt.axes()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    plt.pcolormesh(x1, x2, ys[0], vmin=-1, vmax=1)  # fix color bar range over different figures
    plt.colorbar()

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        return time_text,

    # animation function.  This is called sequentially
    def animate(i):
        pdata = plt.pcolormesh(x1, x2, ys[i], vmin=-1, vmax=1)  # fix color bar range over different figures
        time_text.set_text("Solution at time=" + str(animate_times[i]))
        return pdata, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.

    # because else the object will not get creates and nothing will show! ...
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(animate_times), interval=pause, blit=True)

    plt.show()


# TODO teste mal mayavi.soureforge.net
def animate_2d_surface(x1, x2, ys, animate_times, pause):
    x1, x2 = np.meshgrid(x1, x2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x1, x2, ys[0].real, cmap=cm.coolwarm)
    ax.set_zlim(-2, 2)

    # animation function.  This is called sequentially
    def animate(i):
        ax.clear()
        surf = ax.plot_surface(x1, x2, ys[i].real, cmap=cm.coolwarm)
        ax.set_zlim(-2, 2)

        time_text = ax.text(0.02, 0.95, 0.02, '', transform=ax.transAxes)
        time_text.set_text("Solution at time=" + str(animate_times[i]))
        return time_text, surf

    # because else the object will not get creates and nothing will show! ...
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(animate_times), interval=pause, blit=False)  # blit=True will not work!

    plt.show()
