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


# =============================================================================
# Define Class: Matter here
# =============================================================================

Known_matters = (
    dict(name="Non-Magnetic", Ms=0.0, Ax=np.nan, Ku=np.nan),
    dict(name="NdFeB", Ms=1281, Ax=0.8e-6, Ku=4.36e7),
)


class Matter:
    "Define a matter and its magnetic properties"

    def __init__(
        self,
        form: str = "Unknown",
        Ms: float = 1000.0,
        Ax: float = 1.0e-6,
        Ku: float = 0.0e0,
        Ku_angle: float = 0.0,
    ):
        """
        Arguments
        ---------
        form: String
              Form of the matter
              # Default = "Unknown"
              # Recorded as self.form
        Ms  : Float
              Saturation magnetization [unit emu/cc] for each matter
              # Default = 1000.0
              # Recorded as self.Ms
        Ax  : Float
              Heisenberg exchange stiffness constant [unit erg/cm] for each matter
              # Default = 1.0e-6
              # Recorded as self.Ax
        Ku  : Float
              1st order uniaxial anisotropy energy density [unit erg/cc] for each matter
              # Default = 0.0
              # [In the 1D case], easy axis asigned perpendicular to the variatble axis
              # Recorded as self.Ku
        Ku_angle : Float
                   Tilting angle of easy axis for Ku
                   # Default = 0.0
                   # Recorded as self.Ku_angle
        """
        # Unknown matters
        self._form = str(form)
        self._Ms = float(Ms)
        self._Ax = float(Ax)
        self._Ku = float(Ku)

        # Known matters
        for item in Known_matters:
            if item["name"].casefold() == form.casefold():
                self._form = item["name"]
                self._Ms = item["Ms"]
                self._Ax = item["Ax"]
                self._Ku = item["Ku"]

        self._Ku_angle = float(Ku_angle)

        return None

    @property
    def form(self):
        return self._form

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
