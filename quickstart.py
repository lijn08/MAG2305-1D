# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:37:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
-------------------

Quickstart for MAG2305-1D

"""

# Import core module
import core.MAG2305_1D as MAG

# Prepare model configuration
import numpy as np
model = np.empty(200)
model[:100] = 1    # cells   0 - 99 : matter 1
model[100:] = 2    # cells 100 - 199: matter 2


# Prepare matters configuration
# Format : [ [Ms[1], Ax[1], Ku[1]],
#            [Ms[2], Ax[2], Ku[2]],
#            [Ms[3], Ax[3], Ku[3]],
#            ... ]
#
# Ms [emu/cc] : saturation magnetization
# Ax [erg/cm] : exchange stiffness constant
# Ku [erg/cc] : uniaxial anisotropy energy density
#
# Note that the 'matters' start from id 1 !!!
#
# matter 1 : Ms=1000 emu/cc, Ax=1.0e-6 erg/cm, Ku=0.0 erg/cc
# matter 2 : Ms=1000 emu/cc, Ax=1.0e-6 erg/cm, Ku=1.0e6 erg/cc
matters = np.array([ [1000, 1.0e-6, 0.0], 
                     [1000, 1.0e-6, 1.0e6] ])


# Make an mmSmaple called 'sample0'
cell_size = 1.0
sample0 = MAG.mmSample(cell_size=cell_size, model=model, matters=matters)

# Initialize spin state with all theta angle ~ 0.0
spin0 = np.zeros_like(model) + 0.01  # slightly tiltled from 0 to avoid singularity
sample0.SpinInit(Spin_in = spin0)


# Simulate spin evolution
# -- External field, Hext = -450 [Oe]
# -- Psudo time step, dtime = 1.0e-12 [s]
# The maximal spin change is returned as 'error'
error = sample0.SpinDescent(Hext=-450, dtime=1.0e-12)


# Find stable state when Hext = -450 Oe
# -- Stop iteration when error <= 1.0e-7
# -- Stop if loops > 100000 (to avoid endless loop)
errlim  = 1.0e-7
itermax = 100000
for n in range(itermax):
    error = sample0.SpinDescent(Hext=-450, dtime=1.0e-12)
    print("error : {}".format(error))
    if error < errlim:
        break


import matplotlib.pyplot as plt
x = np.arange(len(model)) * cell_size
y = np.cos(sample0.Theta)
plt.plot(x, y)
plt.xlabel("position [nm]", fontsize=15)
plt.ylabel(r"cos $\theta$", fontsize=15)
plt.show()
