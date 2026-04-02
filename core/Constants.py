# -*- coding: utf-8 -*-
"""
Created on Wen Apr 01 22:00:00 2026

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

[Class] Const : Define useful constants

-------------------

"""


# =============================================================================
# Define constants here
# =============================================================================


class Const:
    "Define constants in this class"

    def __init__(self, value, unit):
        self._value = value
        self._unit = unit

    @property
    def value(self):
        return self._value

    @property
    def unit(self):
        return self._unit


"""
gamma0: Gyromagnetic ratio of spin
        [Lande g factor] * [electron charge] / [electron mass] / [light speed]
"""
gamma0 = Const(1.75882e7, "[Oe s]^-1")

"""
kBoltz: Boltzmann constant
"""
kBoltz = Const(1.38065e-16, "[erg K^-1]")
