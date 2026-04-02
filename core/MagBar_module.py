# -*- coding: utf-8 -*-
"""
Created on Wen Aug 14 14:42:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

MagBar_module : Functions for single phase 1D model

-------------------

"""

import numpy as np
from .Matter import Matter


# =============================================================================
# Functions for soft-hard dual layer model; both infinitely thick
# =============================================================================


def theta_inf(h: float = 0.0):
    """
    Spin angle theta0 for magnetic bar with infinite length

    Arguments
    ---------
    h   : Float
          Normalized external field, h = Hext / Hkhard

    Returns
    -------
    th0st: Float
           Spin tilting angle theta0 especially for stable state
    th0sd: Float
           Spin tilting angle theta0 especially for saddle state
    """

    h = np.abs(h)
    hc = 1.0

    if h > hc:
        print("[Warning] External field larger than coercivity:")
        print("              h = {:.4f}, hc = {:.4f}".format(h, hc))
        print("          No stable/saddle DW could be found!\n")
        th0st = None
        th0sd = None

    else:
        th0st = 0.0
        th0sd = np.arccos(2 * h - 1)

    return th0st, th0sd


def domain_wall_inf(
    matters: tuple[Matter] | Matter = (Matter(),),
    tbar: float = 1.0,
    cell_size: float = 1.0,
    Hext: float = 0.0,
):
    """
    To generate DW configuration for magnetic bar with infinite length

    Arguments
    ---------
    matters: (Matter, ) or Matter
             Magnetic properties of matters
    tbar   : Float
             Length of the magnetic bar [unit nm]
    cell_size : Float
                Size (or length) per cell along variation axis [unit nm]

    Returns
    -------
    [x, theta_st] : Numpy Float(...)
                    Spin configuration for the stable state
    [x, theta_sd] : Numpy Float(...)
                    Spin configuration for the saddle state
    """
    if type(matters) is tuple:
        matters = matters[0]
    Mhard, Ahard, Khard = matters.Ms, matters.Ax, matters.Ku
    DWhard = np.sqrt(Ahard / Khard) * 1.0e7  # unit: nm
    Hkhard = 2.0 * Khard / Mhard

    Hext = np.abs(Hext)
    h = Hext / Hkhard

    cell_count = int(tbar / cell_size)

    x = (
        np.linspace(
            start=0.0, stop=cell_count * cell_size, num=cell_count, endpoint=False
        )
        + 0.5 * cell_size
    )

    theta = [np.zeros_like(x), np.zeros_like(x)]
    th0_list = theta_inf(h=h)

    for i, th0 in enumerate(th0_list):
        if th0 is not None:
            beta = np.sqrt(h / (1.0 - h))
            u = np.arccosh(1.0 / np.tan(th0 / 2) / beta) + x / DWhard * np.sqrt(1.0 - h)

            theta[i] = 2.0 * np.arctan(1.0 / np.cosh(u) / beta)

        else:
            theta[i] = None

    return [x, theta[0]], [x, theta[1]]
