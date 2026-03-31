# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:00:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

Thermally assisted reversal against a soft/hard interface

"""

import os, sys

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import core.MAG2305_1Dlayers as MAG
import core.SoftHard_module as SH
from utils import plot, info
import numpy as np
import matplotlib.pyplot as plt
import argparse, shutil


####################
# alternative args #
####################
def get_args_in():
    parser = argparse.ArgumentParser(description="code parameters")

    parser.add_argument(
        "--cell", type=float, default=1.0, help="cell size [nm], default 1.0"
    )
    parser.add_argument(
        "--tsoft", type=float, default=50, help="soft layer thickness [nm], default 50"
    )
    parser.add_argument(
        "--Msoft",
        type=float,
        default=1000.0,
        help="soft layer saturation [emu/cc], default 1000",
    )
    parser.add_argument(
        "--Asoft",
        type=float,
        default=1.0e-6,
        help="soft layer exchange stiffness [erg/cm], default 1.0e-6",
    )
    parser.add_argument(
        "--Ksoft",
        type=float,
        default=0.0,
        help="soft layer anisotropy energy [erg/cc], default 0.0",
    )
    parser.add_argument(
        "--thard",
        type=float,
        default=100,
        help="hard layer thickness [nm], default 100",
    )
    parser.add_argument(
        "--Mhard",
        type=float,
        default=1000.0,
        help="hard layer saturation [emu/cc], default 1000",
    )
    parser.add_argument(
        "--Ahard",
        type=float,
        default=1.0e-6,
        help="hard layer exchange stiffness [erg/cm], default 1.0e-6",
    )
    parser.add_argument(
        "--Khard",
        type=float,
        default=1.0e6,
        help="hard layer anisotropy energy [erg/cc], default 1.0e6",
    )

    parser.add_argument(
        "--Hext", type=float, default=-500, help="external field [Oe], default -500"
    )
    parser.add_argument(
        "--dtime",
        type=float,
        default=1.0e-12,
        help="time integral step [s], default 1.0e-12",
    )
    parser.add_argument(
        "--errlim",
        type=float,
        default=1.0e-6,
        help="error limit to stop iteration, default 1.0e-6",
    )
    parser.add_argument(
        "--Temp", type=float, default=300, help="temperature [K], default 300"
    )
    parser.add_argument(
        "--rseed",
        type=int,
        default=None,
        help="random seed for thermal fluctuation, default None",
    )

    parser.add_argument(
        "--niters",
        type=int,
        default=100000,
        help="iteration steps for spin update, default 100000",
    )
    parser.add_argument(
        "--nplot",
        type=int,
        default=200,
        help="iteration stride for plot animation, default 200",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="save animation as figures, default False",
    )

    return parser, parser.parse_args()


def main(args, path):
    fig, ax = plot.create_canvas()

    # some features for the sample
    eAM = args.Msoft * args.Asoft / args.Mhard / args.Ahard
    dhard = np.sqrt(args.Ahard / args.Khard) * 1.0e7
    Hkhard = 2.0 * args.Khard / args.Mhard
    Ehard = 4.0 * np.sqrt(args.Ahard * args.Khard)
    tstar = (
        args.tsoft / dhard * np.sqrt(args.Msoft * args.Ahard / args.Mhard / args.Asoft)
    )

    statec = SH.hc_theta_fntsoft(eAM=eAM, tstar=tstar)
    hc = statec[0]
    Hc = hc * Hkhard

    # Create standard states (y0: stable, y1: saddle)
    matters = (
        MAG.Matter(Ms=args.Msoft, Ax=args.Asoft, Ku=args.Ksoft),
        MAG.Matter(Ms=args.Mhard, Ax=args.Ahard, Ku=args.Khard),
    )

    y0, y1 = SH.domain_wall_fntsoft(
        matters=matters,
        tsoft=args.tsoft,
        thard=args.thard,
        cell_size=args.cell,
        Hext=args.Hext,
        statec=statec,
    )

    # Create mm sample
    model = np.zeros_like(y0[0])
    model[y0[0] < 0] = 1
    model[y0[0] >= 0] = 2

    sample = MAG.mmSample(model=model, matters=matters, cell_size=args.cell)

    # Initialize spin in stable state
    spinin = y0[1]

    sample.SpinInit(spin_in=spinin)

    # Update spin configuration
    for n in range(args.niters):
        error = sample.SpinDescent(Hext=args.Hext, dtime=args.dtime, T=args.Temp)
        if error < args.errlim:
            break

        # plot animation
        if n % args.nplot == 0 or n == args.niters - 1:
            print("error : {}".format(error))

            # plot major
            title = "time: {:.2f} [ns]  (iters {:d})".format(n * args.dtime / 1.0e-9, n)
            plot.spin_curve(
                ax,
                sample,
                stable=y0,
                saddle=y1,
                Hext=args.Hext,
                Hc=Hc,
                Temp=args.Temp,
                title=title,
                txtp=0.6,
            )

            plt.pause(0.05)

            if args.save:
                plt.savefig(path + "DW_config_" + str(n).zfill(6))

    plt.close()

    return None


##########################


if __name__ == "__main__":
    parser, args = get_args_in()

    path = "./data/"
    if args.save:
        # delete path if existed
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        # write down args
        info.write_down_args(path=path, parser=parser, args=args)

    if args.rseed is not None:
        np.random.seed(args.rseed)

    with MAG.print_log():
        main(args=args, path=path)
