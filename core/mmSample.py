# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:13:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

[Class] mmSample : Micromagnetic sample to be calculated

-------------------

"""

import numpy as np
from numpy.typing import ArrayLike
import sys, os

from .Constants import gamma0, kBoltz
from .Matter import Matter


# =============================================================================
# Define local functions here
# =============================================================================


def numpy_roll(arr, shift, pbc):
    """
    Re-defined numpy.roll(), including pbc judgement

    Arguments
    ---------
    arr      : Numpy Float(...)
               Array to be rolled
    shift    : Int
               Roll with how many steps
    pbc      : Int or Bool
               Periodic condition for rolling; 1: pbc, 0: non-pbc

    Returns
    -------
    arr_roll : Numpy Float(...)
               arr after rolling
    """
    arr_roll = np.roll(arr, shift=shift)

    if not pbc:
        if shift == 1:
            arr_roll[0] = arr[0]
        elif shift == -1:
            arr_roll[-1] = arr[-1]

    return arr_roll


def check_matters(matters):
    """
    Check the format of input [matters] for mmSample

    Returns
    -------
    standard_matters (check passed) | None (check failed)
    """
    if not isinstance(matters, tuple):
        matters = (matters,)

    for n, item in enumerate(matters):
        if isinstance(item, Matter):
            print("  Format check passed!")
            return matters
        else:
            print("  [Input Error] Elements of [matters] should be Matter!")
            print("                Please verify the element of inndex {}.".format(n))
            return None


def optimize_dtime(H):
    """
    Find optimized time step for spin evolution
    """
    return float("{:.2e}".format(5.0e-7 / H))


# =============================================================================
# Define Class: mmSample here
# =============================================================================


class mmSample:
    "Define a micromagnetic sample"

    # =========================================================================
    # PART I - Initialize mmSample
    # =========================================================================

    def __init__(
        self,
        cell_size: float = 1.0,
        model: tuple | list | ArrayLike = (1,),
        matters: tuple = (Matter(),),
        start_position: float = 0.0,
    ):
        """
        Arguments
        ---------
        cell_size : Float
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
        start_position : Float
                         Starting position [unit nm] of the sample on the variable axis
                         # Default = 0.0
                         # Recorded as self.startps

        Parameters
        ----------
        self.Nmats : Int
                     Number of matters
        self.Theta : Float(self.cellcount)
                     Spin tilting angle for each cell
        self.cellcount : Int
                         Micromagnetic cell count in the sample
        """
        print("Initialize an mm-1D sample:\n")

        # Basic inputs
        self.cellsize = float(cell_size)
        self.startps = float(start_position)

        # model
        try:
            self.model = np.array(model, dtype=int)
        except ValueError:
            print("  [Input Error] Please verify the format of model=[input] !")
            sys.exit(0)

        self.cellcount = len(model)
        self.Nmats = self.model.max()
        print("  # Cells  : {:d}".format(self.cellcount))
        print("  # Matters: {:d}".format(self.Nmats))

        # matters
        matters = check_matters(matters)
        if type(matters) is None:
            sys.exit(0)
        else:
            enough_matters = len(matters) >= self.Nmats

        if enough_matters:
            self.Ms = np.zeros(self.Nmats)
            self.Ax = np.zeros(self.Nmats)
            self.Ku = np.zeros(self.Nmats)
            self.Ku_angle = np.zeros(self.Nmats)
            for i in range(self.Nmats):
                self.Ms[i], self.Ax[i], self.Ku[i], self.Ku_angle[i] = (
                    matters[i].Ms,
                    matters[i].Ax,
                    matters[i].Ku,
                    matters[i].Ku_angle,
                )

        else:
            print(
                "  [Input Error] Length of matters=[Input] less than matters count in [model]!"
            )
            print(
                "                Length of matters=[Input] are expected to be {:d}!".format(
                    self.Nmats
                )
            )
            sys.exit(0)

        print(
            "  Ms check : 1st {:8.3f}, last {:8.3f}   [emu/cc]".format(
                self.Ms[0], self.Ms[-1]
            )
        )
        print(
            "  Ax check : 1st {:.2e}, last {:.2e}   [erg/cm]".format(
                self.Ax[0], self.Ax[-1]
            )
        )
        print(
            "  Ku check : 1st {:.2e}, last {:.2e}   [erg/cc]\n".format(
                self.Ku[0], self.Ku[-1]
            )
        )

        # Magnetization, anisotropy, and exchange matrix
        self._MakeConstantMatrix()

        # Sample configuration features: spin (theta), and energy
        self.Theta = np.zeros(self.cellcount)
        self.Energy = 0.0

        return None

    def _MakeConstantMatrix(self):
        """
        To create constant matrix for further calculations
        { Called in self.__init__() }

        Parameters
        ----------
        self.Hk0  : Float(self.cellcount)
                    1st order uniaxial anisotropy field constant [unit Oe^1/2]
        self.Hx0  : Float(self.cellcount)
                    Heisenberg exchange field constant [unit Oe]
        self.Ht0  : Float(self.cellcount)
                    Brown's thermal fluctuation field constant [unit Oe]
                    # Ht0 = SQRT(2.0 * kB / Ms / cell_volume)
                    # Ht = 3D_normal_dis * Ht0 * SQRT(damping * T / gamma / dtime)
        self.Msmx : Float(self.cellcount)
                    Ms for each cell [unit emu/cc]
        self.Kumx : Float(self.cellcount)
                    Ku for each cell [unit erg/cc]
        self.Axmx : Float(4, self.cellcount)
                    Mean exchange energy constant for each cell
                    # A = A0 * (A1 * Theta_right - A2 * Theta_left + A3 * Theta)
        """

        # critical length, unit [nm]
        critsize = np.zeros(self.cellcount)
        for i in range(self.Nmats):
            size_Bloch = (
                np.sqrt(self.Ax[i] / self.Ku[i]) if self.Ku[i] != 0.0 else np.inf
            )
            size_Neel = (
                np.sqrt(self.Ax[i] / self.Ms[i] ** 2 / 2 / np.pi)
                if self.Ms[i] != 0.0
                else np.inf
            )
            critsize[self.model == i + 1] = min(size_Bloch, size_Neel) * 1.0e7

        # Magnetization, anisotropy, and exchange matrix
        Msmx = np.zeros(self.cellcount)
        Kumx = np.zeros(self.cellcount)
        Hk0 = np.zeros(self.cellcount)
        Ht0 = np.zeros(self.cellcount)
        for i in range(self.Nmats):
            Msmx[self.model == i + 1] = self.Ms[i]
            Kumx[self.model == i + 1] = self.Ku[i]
            Hk0[self.model == i + 1] = (
                2.0 * self.Ku[i] / self.Ms[i] if self.Ms[i] != 0.0 else 0.0
            )
            Ht0[self.model == i + 1] = (
                np.sqrt(
                    2.0
                    * kBoltz.value
                    / self.Ms[i]
                    / self.cellsize
                    / critsize[i] ** 2
                    * 1.0e21
                )
                if self.Ms[i] != 0.0
                else 0.0
            )

        Hx0 = np.zeros((2, self.cellcount))
        Axmx = np.zeros((4, self.cellcount))
        Ax = np.zeros_like(Msmx)
        for i in range(self.Nmats):
            Ax[self.model == i + 1] = self.Ax[i] if self.Ms[i] != 0.0 else 0.0
        Axmx[0, ...] = 1.0e14 * Ax / self.cellsize**2

        Ax_nb = numpy_roll(Ax, shift=+1, pbc=0)
        np.divide(
            4.0 * 1.0e14 * Ax * Ax_nb,
            Msmx * (Ax + Ax_nb) * self.cellsize**2,
            where=(Msmx != 0),
            out=Hx0[0],
        )
        Axmx[1, ...] = Ax_nb / (Ax + Ax_nb)

        Ax_nb = numpy_roll(Ax, shift=-1, pbc=0)
        np.divide(
            4.0 * 1.0e14 * Ax * Ax_nb,
            Msmx * (Ax + Ax_nb) * self.cellsize**2,
            where=(Msmx != 0),
            out=Hx0[1],
        )
        Axmx[2, ...] = Ax_nb / (Ax + Ax_nb)

        Axmx[3, ...] = Ax * (
            1.0 / (numpy_roll(Ax, shift=+1, pbc=0) + Ax)
            - 1.0 / (numpy_roll(Ax, shift=-1, pbc=0) + Ax)
        )

        # Simple statistic
        self.Msavg = Msmx.sum() / self.cellcount
        self.Hmax = Hk0.max() + 2.0 * Hx0.max()

        print("  Average magnetization  : {:9.2f}  [emu/cc]".format(self.Msavg))
        print("  Maximal anisotropy Hk  : {:9.3e}  [Oe]    ".format(Hk0.max()))
        print("  Maximal Heisenberg Hx  : {:9.3e}  [Oe]    ".format(Hx0.max()))
        print("  Maximal effective Heff : {:9.3e}  [Oe]\n  ".format(self.Hmax))
        print(
            "  Est thermal Ht at 300K : {:9.3e}  [Oe]\n  ".format(
                Ht0.max() * np.sqrt(300 / 1.0e-5)
            )
        )

        # Numpy arrays for the sample
        self.Msmx = np.array(Msmx)
        self.Kumx = np.array(Kumx)
        self.Axmx = np.array(Axmx)
        self.Hk0 = np.array(Hk0)
        self.Hx0 = np.array(Hx0)
        self.Ht0 = np.array(Ht0)

        return None

    def SpinInit(self, spin_in):
        """
        Initialize Spin state from input

        Arguments
        ---------
        spin_in : Float(self.cellcount)
                  Input Spin state

        Returns
        -------
        passed  : Bool(True or False)

        Parameters
        ----------
        self.Theta : Float(self.cellcount)
                     Spin tilting angle for each cell
        """
        print("Initialize spin state ...\n")

        spin_in = np.array(spin_in, dtype=float)

        if len(spin_in) != self.cellcount:
            print(
                "  [Input error] Length of spin_in=[Input] mismatched! Should be {}\n".format(
                    self.cellcount
                )
            )

            passed = False

        else:
            self.Theta = np.array(spin_in)
            print("... Spin state initialized according to input.\n")

            if (self.model <= 0).sum() > 0:
                self.Theta[self.model <= 0] = np.nan

            passed = True

        return passed

    # =========================================================================
    # PART II - Energy calculation
    # =========================================================================

    def GetEnergy(self, Hext: float = 0.0):
        """
        To get the energy of the sample

        Arguments
        ---------
        Hext  : Float
                Applied external field

        Parameters
        ----------
        self.Theta : Float(self.cellcount)
                     Spin tilting angle for each cell
        self.Msmx : Float(self.cellcount)
                    Ms for each cell [unit emu/cc]
        self.Kumx : Float(self.cellcount)
                    Ku for each cell [unit erg/cc]
        self.Axmx : Float(4, self.cellcount)
                    Mean exchange energy constant for each cell
                    # A = A0 * (A1 * Theta_right - A2 * Theta_left + A3 * Theta)

        Returns
        -------
        self.Energy.copy : Float
                           Energy of the sample
        """

        self.Energy = (
            (
                self.Axmx[0]
                * (
                    self.Axmx[1] * numpy_roll(self.Theta, shift=+1, pbc=0)
                    - self.Axmx[2] * numpy_roll(self.Theta, shift=-1, pbc=0)
                    + self.Axmx[3] * self.Theta
                )
                ** 2
                + self.Kumx * np.sin(self.Theta) ** 2
                - Hext * self.Msmx * np.cos(self.Theta)
            ).sum()
            * self.cellsize
            * 1.0e-7
        )

        return self.Energy.copy()

    def GetEnergy_detailed(self, Hext: float = 0.0):
        """
        To get the energy of the sample

        Arguments
        ---------
        Hext  : Float
                Applied external field

        Parameters
        ----------
        self.Theta : Float(self.cellcount)
                     Spin tilting angle for each cell
        self.Msmx : Float(self.cellcount)
                    Ms for each cell [unit emu/cc]
        self.Kumx : Float(self.cellcount)
                    Ku for each cell [unit erg/cc]
        self.Axmx : Float(4, self.cellcount)
                    Mean exchange energy constant for each cell
                    # A = A0 * (A1 * Theta_right - A2 * Theta_left + A3 * Theta)

        Returns
        -------
        self.Energy.copy : Float
                           Energy of the sample
        """

        Ex = (
            (
                self.Axmx[0]
                * (
                    self.Axmx[1] * numpy_roll(self.Theta, shift=+1, pbc=0)
                    - self.Axmx[2] * numpy_roll(self.Theta, shift=-1, pbc=0)
                    + self.Axmx[3] * self.Theta
                )
                ** 2
            ).sum()
            * self.cellsize
            * 1.0e-7
        )

        Ek = (self.Kumx * np.sin(self.Theta) ** 2).sum() * self.cellsize * 1.0e-7

        Ez = (-Hext * self.Msmx * np.cos(self.Theta)).sum() * self.cellsize * 1.0e-7

        return Ex, Ek, Ez

    # =========================================================================
    # PART III - Update spin configuration
    # =========================================================================

    def _GetGSpin_For_SpinDescent(self, Hext: float = 0.0):
        """
        To get energy descent direction without thermal fluctuation

        Arguments
        ---------
        Hext  : Float
                Applied external field

        Returns
        -------
        GSpin      : Float(self.cellcount)
                     GSpin = - delta_E / delta_M
        """
        # Energy descent direction
        GSpin = (
            self.Hx0[0] * (numpy_roll(self.Theta, shift=+1, pbc=0) - self.Theta)
            + self.Hx0[1] * (numpy_roll(self.Theta, shift=-1, pbc=0) - self.Theta)
            - self.Hk0 * np.sin(self.Theta) * np.cos(self.Theta)
            - Hext * np.sin(self.Theta)
        )

        return GSpin

    def SpinDescent(
        self,
        Hext: float = 0.0,
        T: float = 0.0,
        dtime: float = 1.0e-14,
        damping: float = 0.1,
    ):
        """
        To update Spin state based on energy descent direction

        Arguments
        ---------
        Hext  : Float
                Applied external field
        T     : Float
                System temperature, unit [K]
        dtime : Float
                Time step for Spin update
                # default = 1.0e-14
        damping : Float
                  Damping constant
                  # default = 0.1

        Parameters
        ----------
        GSpin      : Float(self.cellcount)
                     GSpin = - delta_E / delta_M
        DSpin      : Float(self.cellcount)
                     Delta Spin; DSpin = dtime * gamma * damping * GSpin
        self.Theta : Float(self.cellcount)
                     Spin tilting angle for each cell

        Returns
        -------
        error : Float
                Maximal Spin change among all cells ( |DSpin|.max )

        """
        # Energy descent direction
        # GSpin = self.Hx0[0] * ( numpy_roll(self.Theta, shift=+1, pbc=0)
        #                       - self.Theta )                            \
        #       + self.Hx0[1] * ( numpy_roll(self.Theta, shift=-1, pbc=0)
        #                       - self.Theta )                            \
        #       - self.Hk0 * np.sin(self.Theta) * np.cos(self.Theta)      \
        #                                                                 \
        #       - Hext * np.sin(self.Theta)
        GSpin = self._GetGSpin_For_SpinDescent(Hext=Hext)

        # Thermal agitation
        if T > 0:
            # random direction
            phi = np.random.rand(self.cellcount) * 2 * np.pi
            costh = np.random.rand(self.cellcount) * 2 - 1.0
            theta = np.arccos(costh)
            randy = np.sin(theta) * np.cos(phi)
            randz = np.sin(theta) * np.sin(phi)

            # random scale follows standard normal distribution
            randscl = np.random.standard_normal(self.cellcount)
            randscl[randscl > 5] = 5
            randscl[randscl < -5] = -5
            randscl *= self.Ht0 * np.sqrt(damping * T / dtime / gamma0.value)

            # Brown's thermal fluctuation
            GSpin += randscl * (randy * np.cos(self.Theta) - randz * np.sin(self.Theta))

        # Spin change
        DSpin = gamma0.value * dtime * damping * GSpin
        error = np.abs(DSpin).max()

        self.Theta += DSpin

        return float(error)

    # =========================================================================
    # PART IV - Integrated functions
    # =========================================================================

    def SpinBatchEvolution(
        self,
        Hext: float = 0.0,
        T: float = 0.0,
        dtime: float = 1.0e-14,
        damping: float = 0.1,
        error_limit: float = 1.0e-6,
        num_iters: int = 10000,
        save_spin: bool = False,
        save_stride: int = 1000,
        save_path: str = "./output/",
        plot_func=None,
        plot_func_args: dict | None = None,
        plot_stride: int = 100,
        print_stride: int = 1000,
    ):
        """
        To update Spin state based on energy descent direction

        Arguments
        ---------
        Hext  : Float
                Applied external field
        T     : Float
                System temperature, unit [K]
        dtime : Float
                Time step for Spin update
                # default = 1.0e-14
        damping : Float
                  Damping constant
                  # default = 0.1
        error_limit : Float
                      Lower limit of spin change error
                      # default = 1.0e-6
        num_iters : Int
                    Maximal number of iteration
                    # default = 10,000
        save_spin : True or False
                    Save intermediate spin state (Theta) or not
                    # default = False
        save_stride : Int
                      Iteration stride to save intermediate spin state
                      # Saved data named as spin_xxxxx.npy
                      # default = 1,000
        save_path : String
                    Path to save spin data
                    # default = "./output/"
        plot_func : Callable function
                    Function for plot
                    # default = None
        plot_func_args : Dict
                         Args for plot_func_args
                         # Must be formatted as Dict
        plot_stride : Int
                      Iteration stride to call plot_func
                      # default = 100
        print_stride: Int
                      Iteration stride to print error information
                      # default = 1,000

        Returns
        -------
        end_errlim : Bool
                     True: End at error limit
                     False: End because of run out loops

        """
        num_iters = int(num_iters)
        digit = len(str(num_iters)) + 1
        print_stride = plot_stride if plot_func is not None else print_stride
        print("Starting batch evolution of spin ...\n")

        # Make path if saving spin
        if save_spin:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        # Iterations
        print("Hext = {:.4e} Oe\nDtime = {:.4e} s".format(Hext, dtime))
        for n in range(num_iters):
            nout = n + 1
            error = self.SpinDescent(Hext=Hext, T=T, dtime=dtime, damping=damping)

            if n % print_stride == 0:
                print("Iters {:d} , Error: {:e}".format(nout, error))

            if (plot_func is not None) and (n % plot_stride == 0):
                try:
                    plot_func(**plot_func_args)
                except:
                    print(
                        "  [Error] met at plot_func or plot_func_args. Plot action is skipped!"
                    )

            # Save spin or not
            if save_spin and n % save_stride == 0:
                np.save(
                    file=save_path + "spin_" + str(nout).zfill(digit), arr=self.Theta
                )

            # End loop
            if error < error_limit:
                print("Iters {:d} , Error: {:e}".format(nout, error))
                print("\n... Evolution ended at error limit {}.\n".format(error_limit))
                if save_spin:
                    np.save(
                        file=save_path + "spin_" + str(nout).zfill(digit),
                        arr=self.Theta,
                    )

                return True

        # Final action if running out loops
        print("Iters {:d} , Error: {:e}".format(nout, error))
        print("\n... Evolution ended at iteration limit.\n")
        if save_spin:
            np.save(file="spin_" + str(num_iters).zfill(digit), arr=self.Theta)

        return False

    def GetStableState(
        self,
        Hext: float = 0.0,
        error_limit: float = 1.0e-6,
        num_iters: int = 10000000,
        save_spin: bool = False,
        save_stride: int = 1000,
        save_path: str = "./output/",
        plot_func=None,
        plot_func_args: dict | None = None,
        plot_stride: int = 100,
        print_stride: int = 10000,
    ):
        """
        To update Spin state based on energy descent direction

        Arguments
        ---------
        Hext  : Float
                Applied external field
        error_limit : Float
                      Lower limit of spin change error
                      # default = 1.0e-6
        num_iters : Int
                    Maximal number of iteration
                    # default = 10,000,000
        save_spin : True or False
                    Save intermediate spin state (Theta) or not
                    # default = False
        save_stride : Int
                      Iteration stride to save intermediate spin state
                      # Saved data named as spin_xxxxx.npy
                      # default = 1,000
        save_path : String
                    Path to save spin data
                    # default = "./output/"
        plot_func : Callable function
                    Function for plot
                    # default = None
        plot_func_args : Dict
                         Args for plot_func_args
                         # Must be formatted as Dict
        plot_stride : Int
                      Iteration stride to call plot_func
                      # default = 100
        print_stride: Int
                      Iteration stride to print error information
                      # default = 10,000

        Returns
        -------
        end_errlim : Bool
                     True: End at error limit
                     False: End because of run out loops

        """
        dtime = optimize_dtime(self.Hmax)

        # Call SpinBatchEvolution() at T = 0K
        end_errlim = self.SpinBatchEvolution(
            Hext=Hext,
            T=0,
            dtime=dtime,
            damping=0.1,
            error_limit=error_limit,
            num_iters=num_iters,
            save_spin=save_spin,
            save_stride=save_stride,
            save_path=save_path,
            plot_func=plot_func,
            plot_func_args=plot_func_args,
            plot_stride=plot_stride,
            print_stride=print_stride,
        )

        return end_errlim
