# -*- coding: utf-8 -*-
"""
Created on Wen Aug 14 16:00:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

Verify the motion of DW inside a magnetic bar

"""

import os, sys

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import core.MAG2305_1Dlayers as MAG
import core.MagBar_module as MB
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
        "--thard",
        type=float,
        default=200,
        help="hard layer thickness [nm], default 200",
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
        "--Hext", type=float, default=-900, help="external field [Oe], default -200"
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
        "--Temp", type=float, default=0, help="temperature [K], default 0"
    )
    parser.add_argument(
        "--rseed",
        type=int,
        default=None,
        help="random seed for thermal fluctuation, default None",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=1,
        help="shift direction of saddle state, [1] rightward, [-1] leftward, default 1",
    )
    parser.add_argument(
        "--nshift", type=int, default=1, help="steps for saddle state shift, default 1"
    )

    parser.add_argument(
        "--niters",
        type=int,
        default=50000,
        help="iteration steps for spin update, default 50000",
    )
    parser.add_argument(
        "--nplot",
        type=int,
        default=100,
        help="iteration stride for plot animation, default 100",
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

    # Create standard states (y0: stable, y1: saddle)
    matters = MAG.Matter(Ms=args.Mhard, Ax=args.Ahard, Ku=args.Khard)
    y0, y1 = MB.domain_wall_inf(
        matters=matters, tbar=args.thard, cell_size=args.cell, Hext=args.Hext
    )

    # Create mm sample
    model = np.ones_like(y0[0])
    sample = MAG.mmSample(model=model, matters=matters, cell_size=args.cell)

    # Initialize spin close to saddle state
    spinin = y1[1]
    for _ in range(args.nshift):
        spinin = MAG.numpy_roll(spinin, shift=args.shift, pbc=0)

    sample.SpinInit(spin_in=spinin)

    # Update spin configuration
    for n in range(args.niters):
        error = sample.SpinDescent(Hext=args.Hext, dtime=args.dtime)
        if error < args.errlim:
            break

        # plot animation
        if n % args.nplot == 0 or n == args.niters - 1:
            print("error : {}".format(error))

            title = "time: {:.2f} [ns]  (iters {:d})".format(n * args.dtime / 1.0e-9, n)
            plot.spin_curve(
                ax,
                sample,
                stable=y0,
                saddle=y1,
                Hext=args.Hext,
                Temp=args.Temp,
                title=title,
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
