# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 14:23:00 2025

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

Minimal Energy Path for Soft/Hard Bilayers

"""

import os, sys

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import core.MAG2305_1Dlayers as MAG
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
        "--nodes", type=int, default=21, help="number of path nodes, default 21"
    )
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
        "--Hext", type=float, default=0, help="external field [Oe], default 0"
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
        "--niters",
        type=int,
        default=1000000,
        help="iteration steps for spin update, default 1000000",
    )
    parser.add_argument(
        "--nplot",
        type=int,
        default=500,
        help="iteration stride for plot animation, default 500",
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

    # model configuration
    model = np.empty(int(args.tsoft / args.cell) + int(args.thard / args.cell))
    model[: int(args.tsoft / args.cell)] = 1  # matter 1
    model[int(args.tsoft / args.cell) :] = 2  # matter 2

    # magnetic parameter configuration
    matters = (
        MAG.Matter(Ms=args.Msoft, Ax=args.Asoft, Ku=args.Ksoft),
        MAG.Matter(Ms=args.Mhard, Ax=args.Ahard, Ku=args.Khard),
    )

    # Make a MinPath model called 'sample'
    sample = MAG.MinPath(
        cell_size=args.cell, model=model, matters=matters, path_nodes=args.nodes
    )

    # Initialize spin state
    spin0 = np.zeros_like(model) + 0.01  # slightly tiltled from 0 to avoid singularity
    spin1 = np.zeros_like(model) + np.pi
    sample.SpinInit(spin_begin=spin0, spin_end=spin1, method="linear")

    # Find stable state
    for n in range(args.niters):
        error = sample.NodesMove(Hext=args.Hext, dtime=args.dtime)
        print("error : {}".format(error))

        if n % args.nplot == 0:
            plot.path_nodes(ax[0], sample, title="Iteration " + str(n))
            plot.energy(ax[1], sample, Hext=args.Hext)
            plt.subplots_adjust(left=0.1, right=0.98)
            plt.pause(0.05)
            if args.save:
                plt.savefig(path + "path_" + str(n).zfill(6))

        if error < args.errlim:
            break

    plt.close()


##########################
##########################

if __name__ == "__main__":
    parser, args = get_args_in()

    path = "./data_softhard/"
    if args.save:
        # delete path if existed
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        # write down args
        info.write_down_args(path=path, parser=parser, args=args)

    with MAG.print_log():
        main(args=args, path=path)
