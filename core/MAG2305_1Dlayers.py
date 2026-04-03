# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:13:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

MAG2305-1D : An FDM simulator specialized in 1D micromagnetics

-------------------

"""

import sys
from datetime import datetime
from contextlib import contextmanager


__version__ = "1Dlayers_2026.04.01"


def print_version():
    print("************************************")
    print("MAG2305 version: {:s}".format(__version__))
    print("************************************\n")


@contextmanager
def print_log(logfile="./mag2305.log"):
    # wrapper to print log file
    original_stdout = sys.stdout
    sys.stdout = open(logfile, "w")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")
    print_version()
    try:
        yield
    finally:
        print("\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        sys.stdout.close()
        sys.stdout = original_stdout

    return None


print_version()
# For users
from .Matter import Matter
from .mmSample import mmSample, numpy_roll
from .MinPath import MinPath
from .Solver import Solver, BatchSolver, StableSolver
