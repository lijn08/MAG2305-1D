# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def create_canvas(axip=None):
    parameters = {
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 15,
        "font.size": 12,
        "figure.dpi": 150,
    }
    plt.rcParams.update(parameters)

    fig, ax = plt.subplots(figsize=(8, 5))

    if axip is not None:
        axi = fig.add_axes([axip, 0.19, 0.2, 0.2])
        axi.tick_params(axis="both", labelsize=7)
        return fig, ax, axi

    else:
        return fig, ax


def spin_curve(ax, sample, stable, saddle, Hext, Hc, Temp, title, txtp=0.7):
    # Some features for the sample
    dhard = np.sqrt(sample.Ax[-1] / sample.Ku[-1]) * 1.0e7
    Hkhard = 2.0 * sample.Ku[-1] / sample.Ms[-1]
    Ehard = 4.0 * np.sqrt(sample.Ax[-1] * sample.Ku[-1])

    ax.cla()
    ax.plot(stable[0], np.cos(sample.Theta), label="current", color="black")
    if stable is not None:
        ax.plot(
            stable[0],
            np.cos(stable[1]),
            label="stable",
            ls="--",
            color="orange",
            alpha=0.7,
        )
    if saddle is not None:
        ax.plot(
            saddle[0],
            np.cos(saddle[1]),
            label="saddle",
            ls="--",
            color="green",
            alpha=0.7,
        )

    xlim0, xlim1 = 1.0 * stable[0][0] - 10, 1.0 * stable[0][-1] + 10
    ylim0, ylim1 = -1.1, 1.1
    ax.fill(
        [0.0, 0.0, xlim1, xlim1],
        [ylim0, ylim1, ylim1, ylim0],
        color="grey",
        alpha=0.2,
    )
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(ylim0, ylim1)

    ax.text(
        x=0.8 * xlim0,
        y=0.1,
        s="soft\nphase",
        color="grey",
        style="italic",
        weight="bold",
        fontsize=12,
        alpha=0.7,
    )
    ax.text(
        x=0.15 * xlim1,
        y=0.1,
        s="hard\nphase",
        color="grey",
        style="italic",
        weight="bold",
        fontsize=12,
    )

    ax.legend(loc=[0.7, 0.6])
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"cos $\theta$")
    ax.set_title(title)

    xtxt = txtp * (xlim1 - 0.1 * (xlim1 - xlim0))
    ax.text(x=xtxt, y=-0.38, s=r"$T$" + " = {:.1f} K".format(Temp))
    ax.text(x=xtxt, y=-0.50, s=r"$\delta_{hard}$" + " = {:.2f} nm".format(dhard))
    ax.text(x=xtxt, y=-0.62, s=r"$E_{hard}$" + " = {:.2f} erg/cm$^2$".format(Ehard))
    ax.text(x=xtxt, y=-0.74, s=r"$H_{kh}$" + " = {:.1f} Oe".format(Hkhard))
    ax.text(x=xtxt, y=-0.86, s=r"$H_{c}$ " + " = {:.1f} Oe".format(Hc))
    ax.text(x=xtxt, y=-0.98, s=r"-$H_{ext}$" + " = {:.1f} Oe".format(-Hext))

    return None


def energy_curve(axi, energy, Estable):
    axi.cla()
    axi.plot(energy[0], energy[1], lw=1.2, color="black", alpha=0.7)
    axi.plot(
        [energy[0][0], energy[0][-1]],
        [Estable, Estable],
        lw=1.2,
        ls="--",
        color="orange",
        alpha=0.7,
    )
    axi.set_ylabel("Energy [erg/cm$^2$]", fontsize=9)

    return None
