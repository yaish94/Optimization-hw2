import numpy as np
import matplotlib.pyplot as plt


def plot_outlines(f, sol, start_point, history, header):
    delta = 0.0025
    x = np.arange(sol[0] - start_point[0], sol[0] + start_point[0], delta)
    # x = np.arange(-100.0, 100.0, delta)
    y = np.arange(sol[1] - start_point[1], sol[1] + start_point[1], delta)
    # y = np.arange(-100.0, 100.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            vec = np.array([x[i], y[j]])
            f_x, df_x = f(vec)
            Z[i][j] = f_x

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z.T)

    for k in range(1, len(history)):
        ax.annotate('', xy=history[k], xytext=history[k - 1],
                    arrowprops={'arrowstyle': 'simple', 'color': 'blue', 'lw': 2},
                    va='center', ha='center')

    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(header)

    plt.show()

    plot_iteration_value(f, history, header)


def plot_iteration_value(f, history, header):
    values = [f(history[i])[0] for i in range(len(history))]

    plt.plot(values)
    plt.xlabel('iteration')
    plt.ylabel('objective function value')
    plt.title(header)
    plt.show()


def plot_feasible_region_lp(history):
    # plot the feasible region
    d = np.linspace(-2, 4, 300)
    x, y = np.meshgrid(d, d)
    plt.imshow(((y >= -x + 1) & (y <= 1) & (y >= 0) & (x <= 2)).astype(int),
               extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3)

    # plot the lines defining the constraints
    x = np.linspace(0, 4, 2000)
    # y >= -x + 1
    y1 = (-1 * x) + 1
    # y <= 1
    y2 = (x * 0) + 1
    # y >= 0
    y3 = (x * 0) + 0

    history_x = [v[0] for v in history]
    history_y = [v[1] for v in history]
    plt.plot(history_x, history_y, linewidth=1, marker='>', color="k", label="algorithm path")

    plt.plot(x, y1, label=r'$y \geq -x+1$')
    plt.plot(x, y2, label=r'$y \leq 1$')
    plt.plot(x, y3, label=r'$y \geq 0$')
    plt.axvline(2, color='red', label=r'$x \leq 2$')  # x <= 2

    plt.xlim(0, 3)
    plt.ylim(-1, 2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()


def plot_feasible_region_qp(history):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(range(2), range(2))
    c1 = ax.plot_surface(X, Y, 0 * Y, alpha=0.4, label=r'$z \geq 0$')
    c1._facecolors2d = c1._facecolor3d
    c1._edgecolors2d = c1._edgecolor3d

    X, Z = np.meshgrid(range(2), range(2))
    c2 = ax.plot_surface(X, 0 * X, Z, alpha=0.4, label=r'$y \geq 0$')
    c2._facecolors2d = c2._facecolor3d
    c2._edgecolors2d = c2._edgecolor3d

    Z, Y = np.meshgrid(range(2), range(2))
    c3 = ax.plot_surface(0 * Y, Y, Z, alpha=0.4, label=r'$x \geq 0$')
    c3._facecolors2d = c3._facecolor3d
    c3._edgecolors2d = c3._edgecolor3d

    c4 = ax.plot_surface(X, Y, 1 - X - Y, color="k", alpha=0.4, label=r'$x+y+z = 1$')
    ax.contour3D(X, Y, 1 - X - Y, 30, cmap='binary', label=r'$x+y+z = 1$', alpha=0.5)
    c4._facecolors2d = c4._facecolor3d
    c4._edgecolors2d = c4._edgecolor3d

    history_x = [v[0] for v in history]
    history_y = [v[1] for v in history]
    history_z = [v[2] for v in history]

    ax.plot(history_x, history_y, history_z ,linewidth=2, marker='>', color="k", label="algorithm path")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)

    plt.show()
