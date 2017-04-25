import matplotlib.pyplot as plt
import numpy as np

import demo.chapter1.splitting as sp

trial = sp.trial_frog
dim = 2
time = 0.5
grid_size_N = 64


def set_font_size(axes, size):
    items = [axes.title, axes.xaxis.label, axes.yaxis.label]
    if hasattr(axes, "zaxis"):
        items.append(axes.zaxis.label)
    for item in items:
        item.set_fontsize(size)

if dim == 1:
    title = ""
    if trial == sp.trial_frog:
        title = "$u(t,x)=\sin(2t)\exp(-\cos(x))$ für $t={}$".format(time)
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='$x$', ylabel='$u(t,x)$', title=title)
    x = np.linspace(-np.pi, np.pi, num=grid_size_N)
    x_mesh = np.meshgrid(x)
    y = trial.reference(x_mesh, time)
    plt.plot(x, y)
    plt.ylim((0, 2.5))
    set_font_size(ax, 24)
    plt.show()
elif dim == 2:
    #from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    x = np.linspace(-np.pi, np.pi, num=grid_size_N)
    xs_mesh = np.meshgrid(x, x)
    xs_mesh_sparse = np.meshgrid(x, x, sparse=True)
    y = trial.reference(xs_mesh, time)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if trial == sp.trial_frog:
        plt.title("$u(t,x)=\sin(2t)\exp(-\cos(x_1+x_2))$ für $t={}$".format(time))
    ax.plot_surface(*xs_mesh, y, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$u(t,x)$')
    set_font_size(ax, 22)
    plt.show()
