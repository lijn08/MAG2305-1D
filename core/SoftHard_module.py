# -*- coding: utf-8 -*-
"""
Created on Tue May 07 09:23:00 2024

-------------------
Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn
-------------------

SoftHard_module : Including functions for soft-hard dual phase model

Developer lib version : numpy >= 1.21.5
                        scipy >= 1.12.0

-------------------

"""


import numpy as np
from scipy.optimize import root
from scipy.special import ellipkinc  # 1st kind integral
from scipy.special import ellipeinc  # 2nd kind integral
from scipy.special import ellipj     # Jacobian function
import sys, time


# =============================================================================
# Functions for soft-hard dual phase model; both infinitely thick
# =============================================================================

def hc_theta_inf(eAM :float = 1., 
                   h :float = 0.):
    """
    Coercivity & interface spin angle theta0 for soft-hard dual phase model
    assuming that soft and hard phases are inifitely thick

    Arguments
    ---------
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    h   : Float
          Normalized external field, h = Hext / Hkhard

    Returns
    -------
    hc   : Float
           Normalized coercivity, hc = Hc / Hkhard
    th0c : Float
           Spin tilting angle theta0 at interface, especially for switching state
    th0st: Float
           Spin tilting angle theta0 at interface, especially for stable state
    th0sd: Float
           Spin tilting angle theta0 at interface, especially for saddle state
    """

    h = np.abs(h)
    hc = 1.0 / (1.0 + np.sqrt(eAM))**2
    th0c = np.arccos( (1.0 - np.sqrt(eAM)) / (1.0 + np.sqrt(eAM)) )

    if h > hc:
        print("[Warning] External field larger than coercivity:")
        print("              h = {:.4f}, hc = {:.4f}".format(h, hc))
        print("          No stable/saddle DW could be found!\n")
        th0st = None
        th0sd = None

    else:
        u = np.sqrt( h**2 * (1.0 - eAM)**2 - 2.0 * h * (1.0 + eAM) + 1.0 )
        th0st = np.arccos( h * (1.0 - eAM) + u)
        th0sd = np.arccos( h * (1.0 - eAM) - u)

    return hc, th0c, th0st, th0sd


def energy_inf( eM :float = 1.,
               eAM :float = 1., 
                 h :float = 0.,
                ts :float = 1.,
                th :float = 1.):
    """
    Energy for soft-hard dual phase model
    assuming that soft and hard phases are inifitely thick

    Arguments
    ---------
    eM  : Float
          Magnetization ratio, eM = Msoft / Mhard
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    h   : Float
          Normalized external field, h = Hext / Hkhard
    ts  : Float
          Normalized thickness of the soft phase
          # ts = tsoft / dwh
    th  : Float
          Normalized thickness of the hard phase
          # th = thard / dwh

    Returns
    -------
    Ec  : Float
          Normalized energy for switching state
    Est : Float
          Normalized energy for stable state
    Esd : Float
          Normalized energy for saddle state
          # Norm(E) = E / Ehard = E / SQRT(Ahard*Khard)

    Parameters
    ----------
    dwh : Float
          Critical thickness of the hard phase [unit nm]
          # dwh = SQRT(Ahard / Khard)
    """

    h = np.abs(h)

    Elist = [None, None, None]
    th0_list = hc_theta_inf(eAM = eAM, h = h)[1:]

    for i, th0 in enumerate(th0_list):

        Es = 2.0 * np.sqrt(h * eAM) * (1.0 - np.sin(th0/2)) \
           - 0.5 * h * eM * ts

        Eh = ( h * np.arccosh(np.cos(th0/2)/np.sqrt(h)) 
             - np.cos(th0/2) * np.sqrt(np.cos(th0/2)**2 - h) 
             - h * np.arccosh(1/np.sqrt(h)) + np.sqrt(1-h) ) \
           + 0.5 * h * th

        Elist[i] = Es + Eh

    return tuple(Elist)


def energy_barrier_inf(eAM :float = 1., 
                         h :float = 0.):
    """
    Energy barrier for soft-hard dual phase model
    assuming that soft and hard phases are inifitely thick

    Arguments
    ---------
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    h   : Float
          Normalized external field, h = Hext / Hkhard

    Returns
    -------
    Eb : Float
         Normalized energy barrier
         # Norm(E) = E / Ehard = E / SQRT(Ahard*Khard)

    Parameters
    ----------
    dwh : Float
          Critical thickness of the hard phase [unit nm]
          # dwh = SQRT(Ahard / Khard)
    """

    h = np.abs(h)

    E = [None, None]
    th0_list = hc_theta_inf(eAM = eAM, h = h)[2:]

    for i, th0 in enumerate(th0_list):

        E[i] = - 2.0 * np.sqrt(h * eAM) * np.sin(th0/2)   \
               + h * np.arccosh(np.cos(th0/2)/np.sqrt(h)) \
               - np.cos(th0/2) * np.sqrt(np.cos(th0/2)**2 - h) 

    Eb = E[1] - E[0]

    return Eb


def domain_wall_inf(matters, tsoft :float = 1.,
                             thard :float = 1., 
                             cell_size :float = 1.,
                             Hext  :float = 0.):
    """
    To generate DW configuration for soft-hard dual phase model
    assuming that soft and hard phases are inifitely thick

    Arguments
    ---------
    matters: Float(Nmats,3)
             Magnetic properties of matters
             # Format : Ms[1], Ax[1], Ku[1]
                        Ms[2], Ax[2], Ku[2]
                        ...
              [Units] : emu/cc, erg/cm, erg/cc
    tsoft  : Float
             Thickness of the soft phase [unit nm]
    thard  : Float
             Thickness of the hard phase [unit nm]
    cell_size : Float
                Size (or length) per cell along variation axis [unit nm]

    Returns
    -------
    [x, theta_st] : Numpy Float(...)
                    Spin configuration for the stable state
    [x, theta_sd] : Numpy Float(...)
                    Spin configuration for the saddle state
    """

    Msoft, Asoft, Ksoft = matters[0]
    Mhard, Ahard, Khard = matters[1]
    eAM = Msoft*Asoft/Mhard/Ahard
    DWhard = np.sqrt(Ahard/Khard) *1.0e7  # unit: nm
    Hkhard = 2.0 * Khard / Mhard

    Hext = np.abs(Hext)
    h = Hext / Hkhard

    cell_soft = int(tsoft / cell_size)
    cell_hard = int(thard / cell_size)
    cell_count = cell_soft + cell_hard

    x = np.linspace(start=0., stop=cell_count*cell_size, 
                    num=cell_count, endpoint=False) \
      + (0.5 - cell_soft) * cell_size

    theta = [np.zeros_like(x), np.zeros_like(x)]
    th0_list = hc_theta_inf(eAM = eAM, h = h)[2:]

    for i, th0 in enumerate(th0_list):

        if th0 is not None:

            u = np.log( (1.0 + np.sin(th0/2)) / np.cos(th0/2) ) \
              - x[x<0] / DWhard * np.sqrt(h * Msoft*Ahard/Mhard/Asoft)

            theta[i][x<0] = 2.0 * np.arcsin( np.tanh(u) )

            beta = np.sqrt(h / (1.0 - h))
            u = np.arccosh( 1.0 /np.tan(th0/2)/beta ) \
              + x[x>=0] / DWhard * np.sqrt(1.0 - h)

            theta[i][x>=0] = 2.0 * np.arctan(1.0 / np.cosh(u) / beta)

        else:

            theta[i] = None

    return [x,theta[0]], [x,theta[1]]


# =============================================================================
# Functions for soft-hard dual phase model; hard infinite but soft finite
# =============================================================================

def cal_thickS(h, th0, ths, phi=0.0):
    """
    Calculate the normalized thickness of soft phase, 
    given external field h and angles theta0 and thetaS

    Arguments
    ---------
    h   : Float
          Normalized external field, h = Hext / Hkhard
    th0 : Float
          Spin tilting angle theta0 at interface
    ths : Float
          Spin tilting angle thetaS at the edge

    Returns
    -------
    tstar : Float
            Fully normalized thickness of soft phase
            # tstar = soft_thickness / dwh * sqrt(Msoft*Ahard/Mhard/Asoft)
            # dwh = SQRT(Ahard / Khard)
    """

    t0 = np.arcsin( np.sin((th0 + phi)/2) / np.sin((ths + phi)/2) )

    m = np.sin((ths + phi)/2)**2

    tstar = ( ellipkinc(np.pi/2, m) - ellipkinc(t0, m) ) / np.sqrt(h)

    return tstar


def functions_for_thetaS(x, h, th0, tstar):
    """
    Functions to determine angle thetaS from theta0
    { For scipy.optimize.root to solve x }

    Arguments
    ---------
    x = sin(ths/2)
    th0 : Float
          Spin tilting angle theta0 at interface
    ths : Float
          Spin tilting angle thetaS at the edge
    h   : Float
          Normalized external field, h = Hext / Hkhard
    tstar : Float
            Fully normalized thickness of soft phase
            # tstar = soft_thickness / dwh * sqrt(Msoft*Ahard/Mhard/Asoft)
            # dwh = SQRT(Ahard / Khard)

    Returns
    -------
    out : Functions list
    """

    t0 = np.arcsin( np.sin(th0/2) / x[0] )

    m = x[0]**2

    out = [ tstar - ( ellipkinc(np.pi/2, m) - ellipkinc(t0, m) ) / np.sqrt(h) ]

    return out


def functions_for_theta(x, h, eAM, tstar):
    """
    Functions to determine angles theta0 and thetaS from soft thickness
    { For scipy.optimize.root to solve x }

    Arguments
    ---------
    x = [sin(th0/2), sin(ths/2)]
    th0 : Float
          Spin tilting angle theta0 at interface
    ths : Float
          Spin tilting angle thetaS at the edge
    h   : Float
          Normalized external field, h = Hext / Hkhard
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    tstar : Float
            Fully normalized thickness of soft phase
            # tstar = soft_thickness / dwh * sqrt(Msoft*Ahard/Mhard/Asoft)
            # dwh = SQRT(Ahard / Khard)

    Returns
    -------
    out : Functions list
    """

    out = [ 2.0 * x[0]**2 * (1.0 - x[0]**2)  +  h * (1.0 - 2.0 * x[0]**2) * (1.0 - eAM) - h * ( 1.0 - eAM * (1.0 - 2.0 * x[1]**2) ) ]

    t0 = np.arcsin( x[0] / x[1] )

    m = x[1]**2

    out.append(
                tstar - ( ellipkinc(np.pi/2, m) - ellipkinc(t0, m) ) / np.sqrt(h)
              )

    return out


def functions_for_hc(x, ths, eAM, phi=0.0):
    """
    Functions to determine coercivity, 
    including state equation, switching equation, and equation for Dths0
    { For scipy.optimize.root to solve x }

    Arguments
    ---------
    x = [hc, th0, Dths0]
    hc  : Float
          Normalized coercivity, h = Hc / Hkhard
    th0 : Float
          Spin tilting angle theta0 at interface
    ths : Float
          Spin tilting angle thetaS at the edge
    Dths0 : Float
            Derivative d(thetaS) / d(theta0)
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard

    Returns
    -------
    out : Functions list
    """

    out = [ 0.5 * np.sin(x[1])**2 + x[0] * np.cos(x[1] + phi) * (1.0 - eAM) - x[0] * ( np.cos(phi) - eAM * np.cos(ths + phi) ) ]

    out.append(
                 np.sin(x[1]) * np.cos(x[1]) - x[0] * np.sin(x[1] + phi) * (1.0 - eAM) - x[0] * eAM * np.sin(ths + phi) * x[2]
              )

    t0 = np.arcsin( np.sin((x[1] + phi)/2) / np.sin((ths + phi)/2) )

    m = np.sin((ths + phi)/2)**2

    out.append(
                 x[2] + 0.5 * np.cos((x[1] + phi)/2) / np.sqrt( np.sin((ths + phi)/2)**2 - np.sin((x[1] + phi)/2)**2 )  / (

                      - 0.5 * np.sin((x[1] + phi)/2) / np.sqrt( np.sin((ths + phi)/2)**2 - np.sin((x[1] + phi)/2)**2 ) / np.tan((ths + phi)/2)

                      - 0.5 * np.cos((x[1] + phi)/2) * (   (ellipeinc(np.pi/2, m) - ellipeinc(t0, m)) / np.sin((ths + phi)/2) / np.cos((ths + phi)/2)

                                                         - (ellipkinc(np.pi/2, m) - ellipkinc(t0, m)) / np.sin((ths + phi)/2) * np.cos((ths + phi)/2)

                                                         + np.tan((x[1] + phi)/2) * np.cos(t0) / np.cos((ths + phi)/2)   )  )
                )

    return out


def functions_thinnuc(x, eAM, tstar):
    """
    Functions to determine coercivity, 
    including state equation, switching equation, and equation for Dths0
    { For scipy.optimize.root to solve x }

    Arguments
    ---------
    x = [hc, th02ths]
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    tstar : Float
            Fully normalized thickness of soft phase
            # tstar = soft_thickness / dwh * sqrt(Msoft*Ahard/Mhard/Asoft)
            # dwh = SQRT(Ahard / Khard)
    th02ths : Float
              theta0 / thetaS

    Returns
    -------
    out : Functions list
    """

    out = [ x[1] - np.sin( 0.5*np.pi - tstar * np.sqrt(x[0]) ) ]

    out.append( x[1]**2 * ( 1.0/x[0] + eAM - 1 ) - eAM )

    return out


def function_for_state(x, h, ths, eAM, phi=0.0):
    """
    Function to determine stable state,
    { For scipy.optimize.root to solve x }

    Arguments
    ---------
    x = [th0]
    h   : Float
          Normalized external field, h = Hext / Hkhard
    th0 : Float
          Spin tilting angle theta0 at interface
    ths : Float
          Spin tilting angle thetaS at the edge
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard

    Returns
    -------
    out : Functions list
    """

    out = [ 0.5 * np.sin(x[0])**2 + h * np.cos(x[0] + phi) * (1.0 - eAM) - h * ( np.cos(phi) - eAM * np.cos(ths + phi) ) ]

    return out


def hc_theta_fntsoft(eAM :float = 1., 
                   tstar :float = 1.0,
                     err :float = 1.0e-6):
    """
    Coercivity & soft spin angles theta0, thetaS for soft-hard dual phase model
    assuming that hard phase is inifite but soft is finite

    Arguments
    ---------
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    h   : Float
          Normalized external field, h = Hext / Hkhard
    tstar : Float
            Fully normalized thickness of soft phase
            # tstar = soft_thickness / dwh * sqrt(Msoft*Ahard/Mhard/Asoft)
            # dwh = SQRT(Ahard / Khard)

    Returns
    -------
    hc   : Float
           Normalized coercivity, hc = Hc / Hkhard
    th0c : Float
           Spin tilting angle theta0 at interface, especially for switching state
    thsc : Float
           Spin tilting angle thetaS at end point, especially for switching state
    """

    # to solve hc using scipy.optimize.root
    ini_x = [ 1.0 / (1.0 + np.sqrt(eAM))**2,
              np.arccos((1.0 - np.sqrt(eAM)/(1.0 + np.sqrt(eAM)))),
              0.0 ]

    print("\nTry local scanning on thetaS to find |coercivity| ...\n")
    nuc_switch = False
    step = 1; sign = -1; ths_try = np.pi; nloop = 0
    while nloop < 10000:
        nloop += 1

        ths_try += step * sign / 180
        cal = root( functions_for_hc, ini_x, args = (ths_try, eAM), method = 'lm',
                    options={'xtol': 1.0e-10, 'maxiter': 100000, 'factor': 0.1} )

        if cal.success & (not np.isnan(cal.fun.sum())):
            hc_try, th0_try, Dth0s_try = cal.x[0], cal.x[1], cal.x[2]
            tstar_try = cal_thickS(h=hc_try, th0=th0_try, ths=ths_try)

            if abs(tstar_try - tstar) < max(err * tstar, err):
                break
            else:
                ini_x = [hc_try, th0_try, Dth0s_try]

            if (tstar_try - tstar) * sign > 0:
                sign = sign * -1
                step = 0.1 * step

        else:
            print("cal.status: {} found in iteration process".format(cal.status))
            if cal.status >= 4:
                nuc_switch = True
                break
            else:
                sys.exit("\n[ERROR] Failed to find hc through proper ini_x!\n")

    if nloop == 10000:
        print("\n[ERROR] Failed to find hc through 10000 iterations!")
        print("        Final thetaS for trial: {}\n".format(ths_try))
        sys.exit(0)

    if nuc_switch:
        print("\nTry nucleation process to find |coercivity| ...\n")
        ini_nuc = [hc_try, th0_try/ths_try]
        cal_nuc = root( functions_thinnuc, ini_nuc, args = (eAM, tstar), method = 'lm',
                        options={'xtol': 1.0e-10, 'maxiter': 100000, 'factor': 0.1} )

        if cal_nuc.success:
            hc_try, th02ths = cal_nuc.x[0], cal_nuc.x[1]
            th0_try = 0.0001
            ths_try = th0_try / th02ths
            print("... Success of action:\n")

        else:
            print("cal.status: ", cal.status)
            sys.exit("\n[ERROR] Failed to find hc at nucleation process!\n")

    hc, th0c, thsc = hc_try, th0_try, ths_try
    print("    hc: {:.5f}, theta0: {:.5f}, thetaS: {:.5f}, tstar {:.5f}\n"
          .format(hc, th0c, thsc, cal_thickS(h=hc, th0=th0c, ths=thsc)))

    return hc, th0c, thsc, nuc_switch


def state_thetas_fntsoft(eAM :float = 1., 
                           h :float = 0.,
                       tstar :float = 1.0,
                         err :float = 1.0e-6,
                      statec :tuple = None):
    """
    Coercivity & soft spin angles theta0, thetaS for soft-hard dual phase model
    assuming that hard phase is inifite but soft is finite

    Arguments
    ---------
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    h   : Float
          Normalized external field, h = Hext / Hkhard
    tstar : Float
            Fully normalized thickness of soft phase
            # tstar = soft_thickness / dwh * sqrt(Msoft*Ahard/Mhard/Asoft)
            # dwh = SQRT(Ahard / Khard)
    statec: (float, float, float, Boolean)
            State of switching as input, statec = (hc, th0c, thsc, nuc_switch)
            # If statec == None, func: softhard_hc_theta_fntsoft() will be called

    Returns
    -------
    hc   : Float
           Normalized coercivity, hc = Hc / Hkhard
    th0c : Float
           Spin tilting angle theta0 at interface, especially for switching state
    thsc : Float
           Spin tilting angle thetaS at end point, especially for switching state
    th0st: Float
           Spin tilting angle theta0 at interface, especially for stable state
    thssd: Float
           Spin tilting angle theta0 at end point, especially for stable state
    th0sd: Float
           Spin tilting angle theta0 at interface, especially for saddle state
    thssd: Float
           Spin tilting angle theta0 at end point, especially for saddle state
    """

    if statec is None:
        hc, th0c, thsc, nuc_switch = hc_theta_fntsoft(eAM = eAM,
                                                    tstar = tstar,
                                                      err = err)
    else:
        hc, th0c, thsc, nuc_switch = statec

    h = np.abs(h)
    if h >= hc:
        print("\n[Warning] External field reaches switching condition:")
        print("          h {:.5f}  >=  hc {:.5f}".format(h, hc))
        print("          No stable/saddle DW could be found!\n")
        th0st, thsst, th0sd, thssd = None, None, None, None
        return hc, th0c, thsc, th0st, thsst, th0sd, thssd

    if h == 0:
        print("\n[Warning] External field equals zero (Hext = 0)")
        print("          Trival states defined for stable/saddle!\n")
        th0st, thsst, th0sd, thssd = 0.0, 0.0, np.pi, np.pi
        return hc, th0c, thsc, th0st, thsst, th0sd, thssd

    # to find stable state
    print("\nTry local scanning on theta0 to find |stable| state ...\n")
    step = 1; sign = 1; th0_try = 0.0; nloop = 0
    while nloop < 10000:
        nloop += 1

        th0_try += step * sign / 180
        if th0_try < 0:
            print("... State found to be zero!\n")
            th0st, thsst, tstarst = 0.0, 0.0, tstar
            break

        u = (1.0 - (1.0 - eAM)*np.cos(th0_try)
                 - np.sin(th0_try)**2/h/2) / eAM

        if -1 <= u <= 1:
            ths_try = np.arccos(u)
            if th0_try >= ths_try:
                tstar_try = 0.0
            else:
                tstar_try = cal_thickS(h=h, th0=th0_try, ths=ths_try)
        else:
            ths_try = np.pi
            tstar_try = np.inf

        if abs(tstar_try - tstar) < max(err * tstar, err):
            th0st, thsst, tstarst = th0_try, ths_try, tstar_try
            break

        if (tstar_try - tstar) * sign > 0:
            sign = sign * -1
            step = 0.1 * step

    if nloop == 10000:
        print("\n[ERROR] Failed to find |stable| state through 10000 iterations!")
        print("        Final theta0 for trial: {}\n".format(th0_try))
        sys.exit(0)

    print("... |Stable| state configuration:\n")
    print("    theta0: {:.5f}, thetaS: {:.5f}, tstar: {:.5f}\n"
          .format(th0st, thsst, tstarst))

    # to find saddle state
    print("\nTry local scanning on theta0 to find |saddle| state ...\n")
    step = 1; sign = 1; th0_try = th0st; nloop = 0
    while nloop < 10000:
        nloop += 1

        th0_try += step * sign / 180
        if th0_try < 0:
            print("... State found to be zero!\n")
            th0sd, thssd, tstarsd = 0.0, 0.0, tstar
            break

        u = (1.0 - (1.0 - eAM)*np.cos(th0_try)
                 - np.sin(th0_try)**2/h/2) / eAM

        if -1 <= u <= 1:
            ths_try = np.arccos(u)
            if th0_try >= ths_try:
                tstar_try = 0.0
            else:
                tstar_try = cal_thickS(h=h, th0=th0_try, ths=ths_try)
        else:
            ths_try = np.pi
            tstar_try = np.inf

        if abs(tstar_try - tstar) < max(err * tstar, err):
            th0sd, thssd, tstarsd = th0_try, ths_try, tstar_try
            break

        if (tstar_try - tstar) * sign < 0:
            sign = sign * -1
            step = 0.1 * step

    if nloop == 10000:
        print("\n[ERROR] Failed to find |saddle| state through 10000 iterations!")
        print("        Final theta0 for trial: {}\n".format(th0_try))
        sys.exit(0)

    print("... |Saddle| state configuration:\n")
    print("    theta0: {:.5f}, thetaS: {:.5f}, tstar: {:.5f}\n"
          .format(th0sd, thssd, tstarsd))

    return hc, th0c, thsc, th0st, thsst, th0sd, thssd


def energy_fntsoft( eM :float = 1.,
                   eAM :float = 1., 
                     h :float = 0.,
                    ts :float = 1.,
                    th :float = 1.,
                statec :tuple = None):
    """
    Energy for soft-hard dual phase model
    assuming that hard phase is inifite but soft is finite

    Arguments
    ---------
    eM  : Float
          Magnetization ratio, eM = Msoft / Mhard
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    h   : Float
          Normalized external field, h = Hext / Hkhard
    ts  : Float
          Normalized thickness of the soft phase
          # ts = tsoft / dwh
    th  : Float
          Normalized thickness of the hard phase
          # th = thard / dwh

    Returns
    -------
    Ec  : Float
          Normalized energy for switching state
    Est : Float
          Normalized energy for stable state
    Esd : Float
          Normalized energy for saddle state
          # Norm(E) = E / Ehard = E / SQRT(Ahard*Khard)

    Parameters
    ----------
    dwh : Float
          Critical thickness of the hard phase [unit nm]
          # dwh = SQRT(Ahard / Khard)
    """

    h = np.abs(h)
    tstar = ts * np.sqrt(eM**2 / eAM)

    Elist = [None, None, None]
    th_list = state_thetas_fntsoft(eAM = eAM, 
                                     h = h, 
                                 tstar = tstar,
                                statec = statec)[1:]

    for i in range(3):
        th0 = th_list[2*i]
        ths = th_list[2*i+1]
        if th0 == 0.0 and ths == 0.0:
            t0 = np.pi/2
        else:
            t0 = np.arcsin( np.sin(th0/2) / np.sin(ths/2) )
        m = np.sin(ths/2)**2

        Es = - 2.0 * np.sqrt(h * eAM) * ( ellipeinc(t0,m) - ellipeinc(np.pi/2,m) 
                                        - np.cos(ths/2)**2 * ellipkinc(t0,m) 
                                        + np.cos(ths/2)**2 * ellipkinc(np.pi/2,m) ) \
             + 0.5 * h * np.sqrt(eAM) * tstar * np.cos(ths)

        Eh =   ( h * np.arccosh(np.cos(th0/2)/np.sqrt(h)) 
               - np.cos(th0/2) * np.sqrt(np.cos(th0/2)**2 - h) 
               - h * np.arccosh(1/np.sqrt(h)) + np.sqrt(1-h) ) \
             + 0.5 * h * th

        Elist[i] = Es + Eh

    return tuple(Elist)


def energy_barrier_fntsoft(eAM :float = 1., 
                             h :float = 0.,
                         tstar :float = 1.,
                        statec :tuple = None):
    """
    Energy barrier for soft-hard dual phase model
    assuming that hard phase is inifite but soft is finite

    Arguments
    ---------
    eAM : Float
          Magnetic propety ratio, eAM = Msoft * Asoft / Mhard / Ahard
    h   : Float
          Normalized external field, h = Hext / Hkhard
    tstar : Float
            Fully normalized thickness of soft phase
            # tstar = soft_thickness / dwh * sqrt(Msoft*Ahard/Mhard/Asoft)
            # dwh = SQRT(Ahard / Khard)

    Returns
    -------
    Eb : Float
         Normalized energy barrier
         # Norm(E) = E / Ehard = E / SQRT(Ahard*Khard)

    Parameters
    ----------
    dwh : Float
          Critical thickness of the hard phase [unit nm]
          # dwh = SQRT(Ahard / Khard)
    """

    h = np.abs(h)

    E = [None, None]
    th_list = state_thetas_fntsoft(eAM = eAM, 
                                     h = h, 
                                 tstar = tstar,
                                statec = statec)[3:]

    for i in range(2):
        th0 = th_list[2*i]
        ths = th_list[2*i+1]
        if th0 == 0.0 and ths == 0.0:
            t0 = np.pi/2
        else:
            t0 = np.arcsin( np.sin(th0/2) / np.sin(ths/2) )
        m = np.sin(ths/2)**2

        E[i] = - 2.0 * np.sqrt(h * eAM) * ( ellipeinc(t0,m) - ellipeinc(np.pi/2,m)
                                          - np.cos(ths/2)**2 * ellipkinc(t0,m)
                                          + np.cos(ths/2)**2 * ellipkinc(np.pi/2,m) ) \
               + 0.5 * h * np.sqrt(eAM) * tstar * np.cos(ths)  \
               + h * np.arccosh(np.cos(th0/2)/np.sqrt(h))      \
               - np.cos(th0/2) * np.sqrt(np.cos(th0/2)**2 - h) 

    Eb = E[1] - E[0]

    return Eb


def domain_wall_fntsoft(matters, tsoft :float = 1.,
                                 thard :float = 1., 
                             cell_size :float = 1.,
                                 Hext  :float = 0.,
                                statec :tuple = None):
    """
    To generate DW configuration for soft-hard dual phase model
    assuming that hard phase is inifite but soft is finite

    Arguments
    ---------
    matters: Float(Nmats,3)
             Magnetic properties of matters
             # Format : Ms[1], Ax[1], Ku[1]
                        Ms[2], Ax[2], Ku[2]
                        ...
              [Units] : emu/cc, erg/cm, erg/cc
    tsoft  : Float
             Thickness of the soft phase [unit nm]
    thard  : Float
             Thickness of the hard phase [unit nm]
    cell_size : Float
                Size (or length) per cell along variation axis [unit nm]

    Returns
    -------
    (x, theta_st) : Numpy Float(...)
                    Spin configuration for the stable state
    """

    Msoft, Asoft, Ksoft = matters[0]
    Mhard, Ahard, Khard = matters[1]
    eAM = Msoft*Asoft/Mhard/Ahard
    eMA_ = Msoft*Ahard/Mhard/Asoft
    DWhard = np.sqrt(Ahard/Khard) *1.0e7  # unit: nm
    Hkhard = 2.0 * Khard / Mhard

    h = np.abs(Hext) / Hkhard
    tstar = tsoft / DWhard * np.sqrt(eMA_)

    cell_soft = int(tsoft / cell_size)
    cell_hard = int(thard / cell_size)
    cell_count = cell_soft + cell_hard

    x = np.linspace(start=0., stop=cell_count*cell_size, 
                    num=cell_count, endpoint=False) \
      + (0.5 - cell_soft) * cell_size

    theta = [np.zeros_like(x), np.zeros_like(x)]

    th_list = state_thetas_fntsoft(eAM = eAM, 
                                     h = h, 
                                 tstar = tstar, 
                                statec = statec)[3:]

    for i in range(2):
        th0 = th_list[2*i]
        ths = th_list[2*i+1]

        if (th0 is None) or (ths is None):
            theta[i] = None

        else:
            if th0 == 0.0 and ths == 0.0:
                theta[i][x>=0] = 0.0
            else:
                t0 = np.arcsin( np.sin(th0/2) / np.sin(ths/2) )

                m = np.sin(ths/2)**2

                u = ellipkinc(t0,m) - x[x<0] / DWhard * np.sqrt(h * eMA_)

                theta[i][x<0] = 2.0 * np.arcsin( np.sin(ths/2) * ellipj(u,m)[0] )

                beta = np.sqrt(h / (1.0 - h))
                u = np.arccosh( 1.0 /np.tan(th0/2)/beta ) \
                  + x[x>=0] / DWhard * np.sqrt(1.0 - h)

                theta[i][x>=0] = 2.0 * np.arctan(1.0 / np.cosh(u) / beta)

    return [x,theta[0]], [x,theta[1]]
