# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 16:30:00 2026

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
        li-jiangnan@kust.edu.cn
-------------------

[Class] Solver : Define solvers for mmSample spin evolution, used in
                 SpinBatchEvolution(), GetStableState()

-------------------

"""
# =============================================================================
# Define local functions here
# =============================================================================


def check_dtime(value):
    "solver.dtime should be Float or String 'auto'"
    error_msg = "Unknown input for [solver.dtime]! Try float number or 'auto'."

    if isinstance(value, str):
        if value.casefold() == "auto":
            output = value
        else:
            raise TypeError(error_msg)
    elif isinstance(value, float):
        output = float(value)
    else:
        raise TypeError(error_msg)

    return output


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


# =============================================================================
# Define Class: BuiltinSolver here
# =============================================================================


class BuiltinSolver:
    "Define a builin solver and its (fixed) parameters"

    def __init__(
        self,
        dtime: float | str = 1.0e-14,
        damping: float = 0.1,
        error_limit: float = 1.0e-6,
        num_iters: int = 100000,
        save_spin: bool = False,
        save_path: str = "./output/",
        save_stride: int = 1000,
        print_stride: int = 1000,
    ):
        """
        Arguments
        ---------
        dtime   : Float or String
                  (Psudo) time step for Spin update
                  # default = 1.0e-14
                  # dtime = "auto" allowed in GetStableState()
        damping : Float
                  Damping constant
                  # default = 0.1
        num_iters  : Int
                     Maximal number of iteration
                     # default = 100,000
        error_limit: Float
                     Spin change lower limit. Iteration ended if spin change < error_limit
                     # default = 1.0e-6
        save_spin  : True or False
                     Save intermediate spin state (Theta) or not
                     # default = False
        save_path  : String
                     Path to save spin data
                     # default = "./output/"
        save_stride: Int
                     Iteration stride to save intermediate spin state
                     # Saved data named as spin_xxxxx.npy
                     # default = 1,000
        print_stride: Int
                      Iteration stride to print error information
                      # default = 1,000
        """
        self._dtime = check_dtime(dtime)
        self._damping = check_float(damping, "damping")
        self._error_limit = check_float(error_limit, "error_limit")
        self._num_iters = check_int(num_iters, "num_iters")
        self._save_spin = check_bool(save_spin, "save_spin")
        self._save_path = check_str(save_path, "save_path")
        self._save_stride = check_int(save_stride, "save_stride")
        self._print_stride = check_int(print_stride, "print_stride")

        return None

    def copy(self):
        "Copy a new [changeable] instance"
        new_instance = Solver.__new__(Solver)
        new_instance.__dict__.update(self.__dict__)

        return new_instance

    @property
    def dtime(self):
        return self._dtime

    @property
    def damping(self):
        return self._damping

    @property
    def error_limit(self):
        return self._error_limit

    @property
    def num_iters(self):
        return self._num_iters

    @property
    def save_spin(self):
        return self._save_spin

    @property
    def save_path(self):
        return self._save_path

    @property
    def save_stride(self):
        return self._save_stride

    @property
    def print_stride(self):
        return self._print_stride


"""
Default solver for SpinBatchEvolution()
"""
BatchSolver = BuiltinSolver(
    dtime=1.0e-14,
    damping=0.1,
    error_limit=1.0e-6,
    num_iters=10000,
    save_spin=False,
    save_path="./output/",
    save_stride=1000,
    print_stride=1000,
)

"""
Default solver for GetStableState()
"""
StableSolver = BuiltinSolver(
    dtime="auto",
    damping=0.1,
    error_limit=1.0e-6,
    num_iters=10000000,
    save_spin=False,
    save_path="./output/",
    save_stride=1000,
    print_stride=10000,
)


# =============================================================================
# Define Class: Solver here
# =============================================================================


class Solver(BuiltinSolver):
    "Define a changeable solver and its parameters"

    def copy(self):
        # Create a new instance as a copy
        new_instance = Solver.__new__(Solver)
        new_instance.__dict__.update(self.__dict__)

        return new_instance

    @property
    def dtime(self):
        return self._dtime

    @dtime.setter
    def dtime(self, value):
        self._dtime = check_dtime(value)

    @property
    def damping(self):
        return self._damping

    @damping.setter
    def damping(self, value):
        self._damping = check_float(value, "damping")

    @property
    def error_limit(self):
        return self._error_limit

    @error_limit.setter
    def error_limit(self, value):
        self._error_limit = check_float(value, "error_limit")

    @property
    def num_iters(self):
        return self._num_iters

    @num_iters.setter
    def num_iters(self, value):
        self._num_iters = check_int(value, "num_iters")

    @property
    def save_spin(self):
        return self._save_spin

    @save_spin.setter
    def save_spin(self, value):
        self._save_spin = check_bool(value, "save_spin")

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        self._save_path = check_str(value, "save_path")

    @property
    def save_stride(self):
        return self._save_stride

    @save_stride.setter
    def save_stride(self, value):
        self._save_stride = check_int(value, "save_stride")

    @property
    def print_stride(self):
        return self._print_stride

    @print_stride.setter
    def print_stride(self, value):
        self._print_stride = check_int(value, "print_stride")
