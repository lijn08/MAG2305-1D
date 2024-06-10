# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:00:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jiangnan@kust.edu.cn
-------------------

Thermally assisted reversal against a soft/hard interface

"""
import os, sys
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import core.MAG2305_1D as MAG2305
import core.SoftHard_module as SH
import numpy as np
import matplotlib.pyplot as plt
import argparse, shutil

####################
# alternative args #
####################
def get_args_in():
    parser = argparse.ArgumentParser(description='code parameters')

    parser.add_argument('--cell',    type=float,   default=1.0,      help='cell size [nm], default 1.0')
    parser.add_argument('--tsoft',   type=float,   default=50,       help='soft layer thickness [nm], default 50')
    parser.add_argument('--Msoft',   type=float,   default=1000.,    help='soft layer saturation [emu/cc], default 1000')
    parser.add_argument('--Asoft',   type=float,   default=1.0e-6,   help='soft layer exchange stiffness [erg/cm], default 1.0e-6')
    parser.add_argument('--Ksoft',   type=float,   default=0.0,      help='soft layer anisotropy energy [erg/cc], default 0.0')
    parser.add_argument('--thard',   type=float,   default=100,      help='hard layer thickness [nm], default 100')
    parser.add_argument('--Mhard',   type=float,   default=1000.,    help='hard layer saturation [emu/cc], default 1000')
    parser.add_argument('--Ahard',   type=float,   default=1.0e-6,   help='hard layer exchange stiffness [erg/cm], default 1.0e-6')
    parser.add_argument('--Khard',   type=float,   default=1.0e6,    help='hard layer anisotropy energy [erg/cc], default 1.0e6')

    parser.add_argument('--Hext',    type=float,   default=-500,     help='external field [Oe], default -500')
    parser.add_argument('--dtime',   type=float,   default=1.0e-12,  help='time integral step [s], default 1.0e-12')
    parser.add_argument('--errlim',  type=float,   default=1.0e-6,   help='error limit to stop iteration, default 1.0e-6')
    parser.add_argument('--Temp',    type=float,   default=300,      help='temperature [K], default 300')
    parser.add_argument('--rseed',   type=int,     default=None,     help='random seed for thermal fluctuation, default None')

    parser.add_argument('--niters',  type=int,     default=100000,   help='iteration steps for spin update, default 100000')
    parser.add_argument('--nplot',   type=int,     default=200,      help='iteration stride for plot animation, default 200')
    parser.add_argument('--save',    type=bool,    default=False,    help='save animation as figures, default False')

    return parser.parse_args()


def main(args, path):
#################
# figure canvas #
#################
    parameters = {'axes.labelsize' : 17,
                  'axes.titlesize' : 17,
                  'xtick.labelsize': 12,
                  'ytick.labelsize': 12,
                  'legend.fontsize': 15,
                  'figure.dpi'     : 150}
    plt.rcParams.update(parameters)

    fig, ax =plt.subplots(figsize=(8,5))


########################
# mmSample preparation #
########################
    # some features for the sample
    eAM = args.Msoft * args.Asoft / args.Mhard / args.Ahard
    dhard = np.sqrt(args.Ahard / args.Khard) * 1.0e7
    Hkhard = 2.0 * args.Khard / args.Mhard
    Ehard = 4.0 * np.sqrt(args.Ahard * args.Khard)
    tstar = args.tsoft / dhard * np.sqrt(args.Msoft*args.Ahard/args.Mhard/args.Asoft)

    statec = SH.hc_theta_fntsoft(eAM = eAM, tstar = tstar)
    hc = statec[0]
    Hc = hc * Hkhard

    # create standard states (y0: stable, y1: saddle)
    matters = ((args.Msoft, args.Asoft, args.Ksoft),
               (args.Mhard, args.Ahard, args.Khard))

    y0, y1 = SH.domain_wall_fntsoft(matters=matters, 
                                    tsoft=args.tsoft, thard=args.thard,
                                    cell_size=args.cell, Hext=args.Hext,
                                    statec = statec)

    # create mm sample
    model = np.zeros_like(y0[0])
    model[y0[0] < 0] = 1
    model[y0[0] >=0] = 2

    sample = MAG2305.mmSample(model=model, matters=matters, 
                              cell_size=args.cell)

    # initialize spin in stable state
    spinin = y0[1]

    sample.SpinInit(Spin_in = spinin)

###############
# spin update #
###############
    for n in range(args.niters):
        error = sample.SpinDescent(Hext = args.Hext, dtime = args.dtime, T = args.Temp)
        if error < args.errlim:
            break

        # plot animation
        if n % args.nplot == 0 or n == args.niters-1:
            print("error : {}".format(error))
            theta = sample.Theta

            # plot major
            ax.cla()
            ax.plot(y0[0], np.cos(theta), label="current")
            ax.plot(y0[0], np.cos(y0[1]), label="stable", ls = "--", alpha=0.8)
            ax.plot(y1[0], np.cos(y1[1]), label="saddle", ls = "--",
                    color="grey", alpha=0.7)

            xlim0, xlim1 = 1.1 * y0[0][0], 1.1*y0[0][-1]
            ylim0, ylim1 = -1.1, 1.1
            ax.fill([0.0, 0.0, xlim1, xlim1], 
                    [ylim0, ylim1, ylim1, ylim0], 
                    color="grey", alpha=0.2)
            ax.set_xlim(xlim0, xlim1)
            ax.set_ylim(ylim0, ylim1)

            ax.legend(loc=[0.65, 0.65])
            ax.set_xlabel(r"$x$ [nm]")
            ax.set_ylabel(r"cos $\theta$")
            ax.set_title("time: {:.2f} [ns]  (iters {:d})".format(n*args.dtime/1.0e-9, n))

            ax.text(x=0.80*xlim0, y=0.1, s="soft\nphase",
                    color="grey", style="italic", weight="bold", fontsize=12, alpha=0.7)
            ax.text(x=0.15*xlim1, y=0.1, s="hard\nphase",
                    color="grey", style="italic", weight="bold", fontsize=12)

            xtxt = 0.7 * (xlim1 - 0.2 * (xlim1-xlim0))
            ax.text(x=xtxt, y=-0.3, s=r"$T$" + " = {:.1f} K".format(args.Temp))
            ax.text(x=xtxt, y=-0.4, s=r"$\epsilon_{AM}$" + " = {:.2f}".format(eAM))
            ax.text(x=xtxt, y=-0.5, s=r"$\delta_{hard}$" + " = {:.2f} nm".format(dhard))
            ax.text(x=xtxt, y=-0.6, s=r"$E_{hard}$"+ " = {:.2f} erg/cm$^2$".format(Ehard))
            ax.text(x=xtxt, y=-0.7, s=r"$H_{kh}$" + " = {:.1f} Oe".format(Hkhard))
            ax.text(x=xtxt, y=-0.8, s=r"$H_{c}$ " + " = {:.1f} Oe".format(Hc))
            ax.text(x=xtxt, y=-0.9, s=r"-$H_{ext}$"+ " = {:.1f} Oe".format(-args.Hext))


            plt.pause(0.05)

            if args.save:
                plt.savefig(path + "DW_config_" + str(n).zfill(6))

    plt.close()

    return None


##########################


if __name__ == "__main__":

    args = get_args_in()

    path = "./data/"
    if args.save:
        if os.path.exists(path):
            shutil.rmtree(path)

        os.mkdir(path)

    if args.rseed is not None:
        np.random.seed(args.rseed)

    main(args=args, path=path)
