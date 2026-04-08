# -*- coding: utf-8 -*-
"""
Created on Wen Apr 01 22:00:00 2026

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

Wrappers and common Functions for MAG2305-1D

-------------------

"""

import sys, os
from contextlib import contextmanager


# =============================================================================
# Define wrappers here
# =============================================================================


@contextmanager
def silent_print():
    # wrapper to block print
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    return None


# =============================================================================
# Define functions here
# =============================================================================


def check_float(value, name):
    "Input should be Float"
    error_msg = "Unknown input for [solver.{}]! Try float number.".format(name)

    if isinstance(value, int) or isinstance(value, float):
        output = float(value)
    else:
        raise TypeError(error_msg)

    return output


def check_int(value, name):
    "Input should be Integral"
    error_msg = "Unknown input for [solver.{}]! Try integral number.".format(name)

    if isinstance(value, int):
        output = value
    else:
        raise TypeError(error_msg)

    return output


def check_bool(value, name):
    "Input should be Boolean"
    error_msg = "Unknown input for [solver.{}]! Try boolean number.".format(name)

    if isinstance(value, bool):
        output = value
    else:
        raise TypeError(error_msg)

    return output


def check_str(value, name):
    "Input should be String"
    error_msg = "Unknown input for [solver.{}]! Try string.".format(name)

    if isinstance(value, str):
        output = value
    else:
        raise TypeError(error_msg)

    return output
