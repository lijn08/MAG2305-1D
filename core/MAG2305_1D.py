# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:13:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
-------------------

MAG2305-1D : An FDM simulator for one-dimensional micromagnetics

Developer lib version : numpy >= 1.21.5

-------------------

"""


__version__ = '1Dmm_2024.04.22'
print('MAG2305 version: {:s}\n'.format(__version__))


import numpy as np
import sys


# =============================================================================
# Define constants here
# =============================================================================

class Const():
    " Define constants in this class "

    def __init__(self, value, unit):

        self.__value = value
        self.__unit  = unit

    @property
    def value(self):
        return self.__value

    @property
    def unit(self):
        return self.__unit

"""
gamma0: Gyromagnetic ratio of spin
        [Lande g factor] * [electron charge] / [electron mass] / [light speed]
"""
gamma0 = Const(1.75882e7, '[Oe s]^-1')

"""
kBoltz: Boltzmann constant
"""
kBoltz = Const(1.38065e-16, '[erg K^-1]')


# =============================================================================
# Define general functions here
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


# =============================================================================
# Define mmModel here
# =============================================================================

class mmSample():
    " Define a micromagnetic sample "


    # =========================================================================
    # PART I - Initialize mmModel
    # =========================================================================


    def __init__(self, cell_count :int   = 1,
                       cell_size  :float = 1.,
                       model :tuple[int] = None,
                       Ms :tuple[float]  = (1.,),
                       Ax :tuple[float]  = (1.0e-6,),
                       Ku :tuple[float]  = (0.0e0,),
                       matters :tuple[float] = None,
                       start_position :float = 0.):
        """
        Arguments
        ---------
        cell_count : Int
                     Micromagnetic cell count in the sample
                     # Default = 1
                     # Recorded as self.cellcount
        cell_size  : Float
                     Size (or length) per cell along variation axis [unit nm]
                     # Default = 1
                     # Recorded as self.cellsize
        Ms    : Float(Nmats)
                Saturation magnetization [unit emu/cc] for each matter
                # Default = 1
                # Recorded as self.Ms
        Ax    : Float(Nmats)
                Heisenberg exchange stiffness constant [unit erg/cm] for each matter
                # Default = 1.0e-6
                # Recorded as self.Ax
        Ku    : Float(Nmats)
                1st order uniaxial anisotropy energy density [unit erg/cc] for each matter
                # Default = 1
                # [In the 1D case], easy axis asigned perpendicular to the variation axis
                # Recorded as self.Ku
        start_position : Float
                         Starting position [unit nm] of the sample on the variation axis
                         # Default=0
                         # Recorded as self.startps
        model : Int(cell_count)
                Input data of model configuration, defining matter id for each cell
                # Default = None
                # If [model] not None, input [cell_count] will be ignored
                # Recorded as self.model
        matters : Float(Nmats,3)
                  Magnetic properties of matters
                  # Format : Ms[1], Ax[1], Ku[1]
                             Ms[2], Ax[2], Ku[2]
                             Ms[3], Ax[3], Ku[3]
                             ...
                   [Units] : emu/cc, erg/cm, erg/cc
                  # Default = None
                  # If [matters] not None, inputs [Ms], [Ax], and [Ku] will be ignored

        Parameters
        ----------
        self.Nmats: Int
                    Number of matters
        self.Theta: Float(self.cellcount)
                    Spin tilting angle for each cell
        """
        print("\nInitialize an mm-1D sample:\n")

        # Basic inputs
        self.cellsize = float(cell_size)
        self.startps  = float(start_position)

        # model
        if model is None:
            self.cellcount = int(cell_count)
            self.model = np.ones(self.cellcount, dtype=int)
            self.Nmats = 1
        else:
            self.cellcount = len(model)
            self.model = np.array(model, dtype=int)
            self.Nmats = self.model.max()
        print("  # Cells  : {:d}".format(self.cellcount))
        print("  # Matters: {:d}".format(self.Nmats))

        # matters
        if matters is None:
            if type(Ms) == int or type(Ms) == float:
                self.Ms = np.full(self.Nmats, Ms, dtype=float)
            elif len(Ms) == self.Nmats:
                self.Ms = np.array(Ms, dtype=float)
            else:
                print("  [Input Error] Length of Ms=[Input] does not match with matters count in [model]!")
                print("                Length of Ms=[Input] is expected to be {:d}!".format(self.Nmats))
                sys.exit(0)

            if type(Ax) == int or type(Ax) == float:
                self.Ax = np.full(self.Nmats, Ax, dtype=float)
            elif len(Ax) == self.Nmats:
                self.Ax = np.array(Ax, dtype=float)
            else:
                print("  [Input Error] Length of Ax=[Input] does not match with matters count in [model]!")
                print("                Length of Ax=[Input] is expected to be {:d}!".format(self.Nmats))
                sys.exit(0)

            if type(Ku) == int or type(Ku) == float:
                self.Ku = np.full(self.Nmats, Ku, dtype=float)
            elif len(Ku) == self.Nmats:
                self.Ku = np.array(Ku, dtype=float)
            else:
                print("  [Input Error] Length of Ku=[Input] does not match with matters count in [model]!")
                print("                Length of Ku=[Input] is expected to be {:d}!".format(self.Nmats))
                sys.exit(0)

        else:
            if len(matters) >= self.Nmats:
                self.Ms = np.zeros(self.Nmats)
                self.Ax = np.zeros(self.Nmats)
                self.Ku = np.zeros(self.Nmats)
                matters = np.array(matters, dtype=float)
                for i in range(self.Nmats):
                    self.Ms[i], self.Ax[i], self.Ku[i] = matters[i]

            else:
                print("  [Input Error] Lines of matters=[Input] less than matters count in [model]!")
                print("                Lines of matters=[Input] are expected to be {:d}!".format(self.Nmats))
                sys.exit(0)

        print("  Ms check : 1st {:8.3f}, last {:8.3f}   [emu/cc]"
              .format(self.Ms[0], self.Ms[-1]))
        print("  Ax check : 1st {:.2e}, last {:.2e}   [erg/cm]"
              .format(self.Ax[0], self.Ax[-1]))
        print("  Ku check : 1st {:.2e}, last {:.2e}   [erg/cc]\n"
              .format(self.Ku[0], self.Ku[-1]))

        # Magnetization, anisotropy, and exchange matrix
        self.MakeConstantMatrix()

        # Sample configuration features: spin (theta), and energy
        self.Theta = np.zeros( self.cellcount )
        self.Energy = 0.0

        return None


    def MakeConstantMatrix(self):
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
            size_Bloch = np.sqrt(self.Ax[i] / self.Ku[i]) \
                         if self.Ku[i] != 0.0 else np.inf
            size_Neel  = np.sqrt(self.Ax[i] / self.Ms[i]**2 /2/np.pi) \
                         if self.Ms[i] != 0.0 else np.inf
            critsize[ self.model == i+1 ] = min(size_Bloch, size_Neel) * 1.0e7

        # Magnetization, anisotropy, and exchange matrix
        Msmx = np.zeros(self.cellcount)
        Kumx = np.zeros(self.cellcount)
        Hk0  = np.zeros(self.cellcount)
        Ht0  = np.zeros(self.cellcount)
        for i in range(self.Nmats):
            Msmx[ self.model == i+1 ] = self.Ms[i]
            Kumx[ self.model == i+1 ] = self.Ku[i]
            Hk0[ self.model == i+1 ] = 2.0 * self.Ku[i] / self.Ms[i] \
                                       if self.Ms[i] != 0.0 else 0.0
            Ht0[ self.model == i+1 ] = np.sqrt(2.0 * kBoltz.value / self.Ms[i]
                                               / self.cellsize / critsize[i]**2 
                                               * 1.0e21) \
                                       if self.Ms[i] != 0.0 else 0.0

        Hx0  = np.zeros( (2, self.cellcount) )
        Axmx = np.zeros( (4, self.cellcount) )
        Ax = np.zeros_like(Msmx)
        for i in range(self.Nmats):
            Ax[ self.model == i+1 ] = self.Ax[i] if self.Ms[i] !=0.0 else 0.0
        Axmx[0,...] = 1.0e14 * Ax / self.cellsize**2

        Ax_nb = numpy_roll(Ax, shift=+1, pbc=0)
        Hx0[0,...] = np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                                Msmx * (Ax + Ax_nb) * self.cellsize**2, 
                                where= (Msmx!=0) )
        Axmx[1,...] = Ax_nb / (Ax + Ax_nb)

        Ax_nb = numpy_roll(Ax, shift=-1, pbc=0)
        Hx0[1,...] = np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                                Msmx * (Ax + Ax_nb) * self.cellsize**2, 
                                where= (Msmx!=0) )
        Axmx[2,...] = Ax_nb / (Ax + Ax_nb)

        Axmx[3,...] = Ax * ( 1.0 / (numpy_roll(Ax, shift=+1, pbc=0) 
                                    + Ax) 
                           - 1.0 / (numpy_roll(Ax, shift=-1, pbc=0)
                                    + Ax) )

        # Simple statistic
        self.Msavg = Msmx.sum() / self.cellcount
        self.Hmax  = Hk0.max() + 2.0 * Hx0.max()

        print("  Average magnetization  : {:9.2f}  [emu/cc]".format(self.Msavg))
        print("  Maximal anisotropy Hk  : {:9.3e}  [Oe]    ".format(Hk0.max()))
        print("  Maximal Heisenberg Hx  : {:9.3e}  [Oe]    ".format(Hx0.max()))
        print("  Maximal effective Heff : {:9.3e}  [Oe]\n  ".format(self.Hmax))
        print("  Est thermal Ht at 300K : {:9.3e}  [Oe]\n  ".format(Ht0.max() 
                                                                    *np.sqrt(300/1.0e-5)))

        # Numpy arrays for the sample
        self.Msmx = np.array(Msmx)
        self.Kumx = np.array(Kumx)
        self.Axmx = np.array(Axmx)
        self.Hk0  = np.array(Hk0)
        self.Hx0  = np.array(Hx0)
        self.Ht0  = np.array(Ht0)

        return None


    def SpinInit(self, Spin_in):
        """
        Initialize Spin state from input

        Arguments
        ---------
        Spin_in : Float(self.cellcount)
                  Input Spin state

        Returns
        -------
        passed  : Bool(True or False)

        Parameters
        ----------
        self.Theta : Float(self.cellcount)
                     Spin tilting angle for each cell
        """

        Spin_in = np.array(Spin_in, dtype=float)

        if len(Spin_in) != self.cellcount:
            print('[Input error] Length of Spin_in=[Input] mismatched! Should be {}\n'
                  .format( self.cellcount ) )

            passed = False

        else:
            self.Theta = np.array(Spin_in)
            print('Spin state initialized according to input.\n')

            if (self.model <= 0).sum() > 0:
                self.Theta[self.model<=0] = np.nan

            passed = True

        return passed


    # =========================================================================
    # PART II - Energy calculation
    # =========================================================================


    def GetEnergy(self, Hext :float = 0.):
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

        self.Energy = ( self.Axmx[0] *
                        ( self.Axmx[1] * numpy_roll(self.Theta, 
                                                    shift=+1, pbc=0)
                        - self.Axmx[2] * numpy_roll(self.Theta,
                                                    shift=-1, pbc=0)
                        + self.Axmx[3] * self.Theta )**2

                    + self.Kumx * np.sin(self.Theta)**2

                    - Hext * self.Msmx * np.cos(self.Theta) ).sum() \
                                                                    \
                    * self.cellsize * 1.0e-7

        return self.Energy.copy()


    def GetEnergy_check(self, Hext :float = 0.):
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

        Ex = ( self.Axmx[0] *
                ( self.Axmx[1] * numpy_roll(self.Theta, 
                                            shift=+1, pbc=0)
                - self.Axmx[2] * numpy_roll(self.Theta,
                                            shift=-1, pbc=0)
                + self.Axmx[3] * self.Theta )**2).sum() * self.cellsize * 1.0e-7

        Ek = (self.Kumx * np.sin(self.Theta)**2).sum() * self.cellsize * 1.0e-7

        Ez = (- Hext * self.Msmx * np.cos(self.Theta)).sum() * self.cellsize * 1.0e-7

        return Ex, Ek, Ez


    # =========================================================================
    # PART III - Update spin configuration
    # =========================================================================


    def SpinDescent(self, Hext    :float = 0.,
                          T       :float = 0.,
                          dtime   :float = 1.0e-14,
                          damping :float = 0.1):
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
                # default = 1.0e-13
        damping : Float
                  Damping constant

        Returns
        -------
        error : Float
                Maximal Spin change among all cells ( |DSpin|.max )

        Parameters
        ----------
        GSpin      : Float(self.cellcount)
                     GSpin = - delta_E / delta_M
        DSpin      : Float(self.cellcount)
                     Delta Spin; DSpin = dtime * gamma * damping * GSpin
        self.Theta : Float(self.cellcount)
                     Spin tilting angle for each cell
        """
        # Energy descent direction
        GSpin = self.Hx0[0] * ( numpy_roll(self.Theta, shift=+1, pbc=0) 
                              - self.Theta )                            \
              + self.Hx0[1] * ( numpy_roll(self.Theta, shift=-1, pbc=0) 
                              - self.Theta )                            \
              - self.Hk0 * np.sin(self.Theta) * np.cos(self.Theta)      \
                                                                        \
              - Hext * np.sin(self.Theta)

        # Thermal agitation
        if T > 0:
            # random direction
            phi   = np.random.rand( self.cellcount ) * 2 * np.pi
            costh = np.random.rand( self.cellcount ) * 2 - 1.0
            theta = np.arccos(costh)
            randy = np.sin(theta) * np.cos(phi)
            randz = np.sin(theta) * np.sin(phi)

            # random scale follows standard normal distribution
            randscl = np.random.standard_normal( self.cellcount )
            randscl[randscl > 5] = 5
            randscl[randscl < -5] = -5
            randscl *= self.Ht0 * np.sqrt(damping * T / dtime / gamma0.value )

            # Brown's thermal fluctuation
            GSpin += randscl * ( randy * np.cos(self.Theta)
                               - randz * np.sin(self.Theta) )

        # Spin change
        DSpin = gamma0.value * dtime * damping * GSpin
        error = np.abs(DSpin).max()

        self.Theta += DSpin

        return float(error)
