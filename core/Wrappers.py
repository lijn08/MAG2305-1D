# -*- coding: utf-8 -*-
"""
Created on Wen Apr 01 22:00:00 2026

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

Wrappers for MAG2305-1D

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
