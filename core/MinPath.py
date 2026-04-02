# -*- coding: utf-8 -*-
"""
Created on Tue Jul 08 12:00:00 2025

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

[Class] MinPath : Path with minimum energy,
                  consisting of a series of the same mmSamples

-------------------

"""

import numpy as np
from numpy.typing import ArrayLike
import sys

from .Constants import gamma0
from .Matter import Matter
from .mmSample import mmSample
from .Wrappers import silent_print


# =============================================================================
# Define Class: MinPath here
# =============================================================================


class MinPath:
    "To Find the Minimum Energy Path"

    # =========================================================================
    # PART I - Initialize MinPath
    # =========================================================================

    def __init__(
        self,
        cell_size: float = 1.0,
        model: tuple | list | ArrayLike = (1,),
        matters: tuple | Matter = (Matter(),),
        path_nodes: int = 20,
    ):
        """
        Arguments
        ---------
        path_nodes : Int >= 3
                     Number of nodes along minimal energy path
        cell_size  : Float
                     Size (or length) per cell along variable axis [unit nm]
                     # Default = 1
                     # Recorded as self.cellsize
        model : Int(self.cellcount)
                Input data of model configuration, defining matter id for each cell
                # Default = (1,)
                # Recorded as self.model
        matters : Matter(self.Nmats)
                  Magnetic properties of matters
                  # Default = (Matter(default),)

        Parameters
        ----------
        self.Nmats : Int
                     Number of matters
        self.cellcount : Int
                         Micromagnetic cell count in the sample
        """
        print("Initialize a Minimal Energy Path model ...\n")

        # path_nodes
        if path_nodes >= 3:
            self.path_nodes = int(path_nodes)
        else:
            print(
                "  [Input Error] Value of path_nodes=[Input] must be an integer >= 3!"
            )
            sys.exit(0)

        # Nodes
        with silent_print():
            self.Nodes = [
                mmSample(
                    cell_size=cell_size,
                    model=model,
                    matters=matters,
                )
                for _ in range(self.path_nodes)
            ]

        print(
            "... Created a Minimal Energy Path model with [{:d}] nodes.\n".format(
                self.path_nodes
            )
        )

        # Basic infomation
        self.model = self.Nodes[0].model
        self.Nmats = self.Nodes[0].Nmats
        self.cellsize = self.Nodes[0].cellsize
        self.cellcount = self.Nodes[0].cellcount

        # Sample attritutes
        self.Energy = np.empty(self.path_nodes)

        return None

    def SpinInit(self, spin_begin, spin_end, method="linear"):
        """
        Initialize Spin state from input

        Arguments
        ---------
        spin_begin: Float(cell_count)
                    Input spin state for the first node of path
        spin_end  : Float(cell_count)
                    Input spin state for the last node of path
        method    : String
                    Method to initialize the rest of path nodes
        """
        print("Initialize spin state ...\n")

        spin_begin = np.array(spin_begin, dtype=float)
        spin_end = np.array(spin_end, dtype=float)

        # Initialize spins for the first and last path nodes
        with silent_print():
            self.Nodes[0].SpinInit(spin_in=spin_begin)
            self.Nodes[-1].SpinInit(spin_in=spin_end)

        # Initialize spins for all the rest of path nodes
        if method == "linear":
            mid_state = np.array(
                [
                    spin_end[0]
                    + i * (spin_begin[-1] - spin_end[0]) / (len(spin_end) - 1)
                    for i in range(len(spin_end))
                ]
            )

            # left half
            if self.path_nodes % 2 == 1:
                nodes_num1 = self.path_nodes // 2 + 1
                stride = (mid_state - spin_begin) / (nodes_num1 - 1)
            else:
                nodes_num1 = self.path_nodes // 2
                stride = (mid_state - spin_begin) / (2 * nodes_num1 - 1) * 2
            with silent_print():
                [
                    self.Nodes[i].SpinInit(spin_in=spin_begin + i * stride)
                    for i in range(1, nodes_num1)
                ]

            # right half
            if self.path_nodes % 2 == 1:
                nodes_num2 = self.path_nodes - nodes_num1 + 1
                stride = (spin_end - mid_state) / (nodes_num2 - 1)
            else:
                nodes_num2 = self.path_nodes - nodes_num1
                stride = (spin_end - mid_state) / (2 * nodes_num2 - 1) * 2
            with silent_print():
                [
                    self.Nodes[-1 - i].SpinInit(spin_in=spin_end - i * stride)
                    for i in range(1, nodes_num2)
                ]

            print("... Spin state initialized with method = [{}].\n".format(method))

        else:
            print("[Input Error] Unknown input: method=[Input] should be [linear]!")

        return None

    # =========================================================================
    # PART II - Update path nodes configuration
    # =========================================================================

    def NodesMove(
        self, Hext: float = 0.0, dtime: float = 1.0e-14, damping: float = 0.1
    ):
        """
        To update Nodes states based on nudged-elastic-band algorithm

        Arguments
        ---------
        Hext  : Float
                Applied external field
        dtime : Float
                Time step for Spin update
                # default = 1.0e-14
        damping : Float
                  Damping constant

        Returns
        -------
        error : Float
                Maximal Spin change among all path nodes
        """
        # Energy and energy difference
        nodes_energy = np.array(
            [self.Nodes[i].GetEnergy(Hext=Hext) for i in range(self.path_nodes)]
        )
        energy_diff = np.diff(nodes_energy, n=1) > 0

        # Update nodes states
        error_list = []
        for n in range(1, self.path_nodes - 1):
            theta_diff_p = self.Nodes[n + 1].Theta - self.Nodes[n].Theta
            theta_diff_n = self.Nodes[n - 1].Theta - self.Nodes[n].Theta
            # Get path tangent
            if energy_diff[n - 1] and energy_diff[n]:
                theta_diff = theta_diff_p

            elif not energy_diff[n - 1] and not energy_diff[n]:
                theta_diff = theta_diff_n

            else:
                theta_diff = self.Nodes[n + 1].Theta - self.Nodes[n - 1].Theta

            # Path tangent
            tangent = theta_diff / np.linalg.norm(theta_diff)
            force = (
                0
                * (np.linalg.norm(theta_diff_p) - np.linalg.norm(theta_diff_n))
                * tangent
            )

            # Spin change
            GSpin = self.Nodes[n]._GetGSpin_For_SpinDescent(Hext=Hext)
            DSpin = (
                gamma0.value
                * dtime
                * damping
                * (force + GSpin - (GSpin @ tangent) * tangent)
            )

            error_list.append(np.abs(DSpin).max())

            self.Nodes[n].Theta += DSpin

        return float(max(error_list))

    def GetEnergy(self, Hext: float = 0.0):
        """
        To get the energies along the path

        Arguments
        ---------
        Hext  : Float
                Applied external field

        Returns
        -------
        self.Energy.copy : Float
                           Energy of the sample
        """
        self.Energy = np.array(
            [self.Nodes[i].GetEnergy(Hext=Hext) for i in range(self.path_nodes)]
        )

        return np.copy(self.Energy)
