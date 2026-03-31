# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def create_canvas():
    parameters = {
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 15,
        "font.size": 13,
        "figure.dpi": 150,
    }
    plt.rcParams.update(parameters)

    fig, ax = plt.subplots(figsize=(8, 5))

    return fig, ax


def spin_curve(ax, sample, stable, saddle, Hext, Temp, title):
    # Some features for the sample
    dhard = np.sqrt(sample.Ax[-1] / sample.Ku[-1]) * 1.0e7
    Hkhard = 2.0 * sample.Ku[-1] / sample.Ms[-1]
    Hc = Hkhard
    Ehard = 4.0 * np.sqrt(sample.Ax[-1] * sample.Ku[-1])

    ax.cla()
    ax.plot(stable[0], np.cos(sample.Theta), label="current", color="black")
    ax.plot(
        stable[0], np.cos(stable[1]), label="stable", ls="--", color="orange", alpha=0.7
    )
    ax.plot(
        saddle[0], np.cos(saddle[1]), label="saddle", ls="--", color="green", alpha=0.7
    )

    xlim0, xlim1 = 1.0 * saddle[0][0] - 10, 1.0 * saddle[0][-1] + 10
    ylim0, ylim1 = -1.1, 1.1
    ax.fill(
        [xlim0, xlim0, xlim1, xlim1],
        [ylim0, ylim1, ylim1, ylim0],
        color="grey",
        alpha=0.2,
    )
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(ylim0, ylim1)

    ax.legend(loc=[0.7, 0.6])
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"cos $\theta$")
    ax.set_title(title)

    xtxt = 0.7 * (xlim1 - 0.1 * (xlim1 - xlim0))
    ax.text(x=xtxt, y=-0.38, s=r"$T$" + " = {:.1f} K".format(Temp))
    ax.text(x=xtxt, y=-0.50, s=r"$\delta_{hard}$" + " = {:.2f} nm".format(dhard))
    ax.text(x=xtxt, y=-0.62, s=r"$E_{hard}$" + " = {:.2f} erg/cm$^2$".format(Ehard))
    ax.text(x=xtxt, y=-0.74, s=r"$H_{kh}$" + " = {:.1f} Oe".format(Hkhard))
    ax.text(x=xtxt, y=-0.86, s=r"$H_{c}$ " + " = {:.1f} Oe".format(Hc))
    ax.text(x=xtxt, y=-0.98, s=r"-$H_{ext}$" + " = {:.1f} Oe".format(-Hext))

    return None
