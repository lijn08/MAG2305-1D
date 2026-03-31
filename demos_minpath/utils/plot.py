# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def create_canvas():
    parameters = {
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
    }
    plt.rcParams.update(parameters)

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 100)
    ax = [fig.add_subplot(gs[:, :55]), fig.add_subplot(gs[:, 65:])]

    return fig, ax


def path_nodes(ax, sample, title):
    ax.cla()
    x = np.arange(len(sample.model)) * sample.cellsize

    curve_max = 11
    plot_stride = sample.path_nodes // curve_max + 1
    for i in range(0, sample.path_nodes, plot_stride):
        y = sample.Nodes[i].Theta
        rgb = 1.0 / sample.path_nodes * i
        ax.plot(x, np.cos(y), color=(rgb, 0.2, 0.2), label="node " + str(i))
    ax.set_xlabel("$x$ [nm]", fontsize=15)
    ax.set_ylabel(r"cos $\theta$", fontsize=15)
    ax.legend(loc="right")
    ax.set_title(title)

    return None


def energy(ax, sample, Hext):
    ax.cla()

    y = sample.GetEnergy(Hext=Hext)
    Edw = 4.0 * np.sqrt(sample.Nodes[0].Ax[-1] * sample.Nodes[0].Ku[-1])

    ax.plot(y / Edw, marker="o", ls="--")
    ax.set_xlabel("Path node", fontsize=15)
    ax.set_ylabel(r"$\mathscr{E}$ / $\mathscr{E}_{DW}$", fontsize=15)
    ax.set_title("Energies along path")

    return None
