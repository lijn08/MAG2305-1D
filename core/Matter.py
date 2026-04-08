# -*- coding: utf-8 -*-
"""
Created on Wen Apr 01 22:00:00 2026

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

[Class] Matter : Define a matter and its magnetic properties

-------------------

"""

import numpy as np
from .Wrappers_Funcs import check_float, check_int, check_bool, check_str


# =============================================================================
# Define Class: BuiltinFerroMatter here
# =============================================================================


class BuiltinFerroMatter:
    "Define a builtin ferromatter and its (fixed) magnetic properties"

    def __init__(
        self,
        Ms: float = 1000.0,
        Ax: float = 1.0e-6,
        Ku: float = 0.0e0,
        Ku_angle: float = 0.0,
    ):
        """
        Arguments
        ---------
        Ms : Float
             Saturation magnetization [unit emu/cc] for each matter
             # Default = 1000.0
        Ax : Float
             Heisenberg exchange stiffness constant [unit erg/cm] for each matter
             # Default = 1.0e-6
        Ku : Float
             1st order uniaxial anisotropy energy density [unit erg/cc] for each matter
             # Default = 0.0
             # [In the 1D case], easy axis asigned perpendicular to the variatble axis
        Ku_angle : Float
                   Tilting angle of easy axis for Ku
                   # Default = 0.0
        """
        self._Ms = check_float(Ms, "Ms")
        self._Ax = check_float(Ax, "Ax")
        self._Ku = check_float(Ku, "Ku")
        self._Ku_angle = check_float(Ku_angle, "Ku_angle")

        return None

    def list(self):
        "List all attributes"
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("_"):
                print(f"{attr}: {getattr(self, attr)}")

        return None

    def copy(self):
        "Copy a new [changeable] instance"
        new_instance = FerroMatter.__new__(FerroMatter)
        new_instance.__dict__.update(self.__dict__)

        return new_instance

    @property
    def Ms(self):
        return self._Ms

    @property
    def Ax(self):
        return self._Ax

    @property
    def Ku(self):
        return self._Ku

    @property
    def Ku_angle(self):
        return self._Ku_angle


"""
Nd2Fe14B
"""
Nd2Fe14B = BuiltinFerroMatter(Ms=1281, Ax=0.8e-6, Ku=4.36e7)


# =============================================================================
# Define Class: FerroMatter (Matter) here
# =============================================================================


class FerroMatter(BuiltinFerroMatter):
    "Define a ferromatter and its magnetic properties"

    @property
    def Ms(self):
        return self._Ms

    @Ms.setter
    def Ms(self, value):
        self._Ms = check_float(value, "Ms")

    @property
    def Ax(self):
        return self._Ax

    @Ax.setter
    def Ax(self, value):
        self._Ax = check_float(value, "Ax")

    @property
    def Ku(self):
        return self._Ku

    @Ku.setter
    def Ku(self, value):
        self._Ku = check_float(value, "Ku")

    @property
    def Ku_angle(self):
        return self._Ku_angle

    @Ku_angle.setter
    def Ku_angle(self, value):
        self._Ku_angle = check_float(value, "Ku_angle")


class Matter(FerroMatter):
    "Define a matter(=ferromatter) and its magnetic properties"
