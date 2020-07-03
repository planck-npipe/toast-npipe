#cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=True, cdivision=True

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

# This package implements the COBE Zodiacal emission model
# according to Ken Ganga

import healpy as hp
import numpy as np
import toast.qarray as qa
import toast.timing as timing

from ..preproc_modules.signal_estimation import SignalEstimator
from ..reproc_modules import destripe_tools

cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack
cimport cython
# from cython.parallel import parallel, prange


np.import_array()

from libc.math cimport sin, cos, sqrt, fabs, exp, pow, M_PI, atan2

# light speed [m/s]
cdef double c = 299792458.0
# Planck constant [J/Hz]
cdef double h = 6.62607004e-34
# Boltzmann's constant [J/K]
cdef double k = 1.38064852e-23
cdef double degree = 0.017453292519943295

xaxis, yaxis, zaxis = np.eye(3)


cdef class Cloud:

    cdef double emissivity
    cdef double X0
    cdef double Y0
    cdef double Z0
    cdef double Incl
    cdef double Omega
    cdef double n0
    cdef double alpha
    cdef double beta
    cdef double gamma
    cdef double mu
    cdef double sinOmega
    cdef double cosOmega
    cdef double sinIncl
    cdef double cosIncl
    cdef double muInv

    def __cinit__(
            self, double emissivity=1, double X0=0.011887801,
            double Y0=0.0054765065, double Z0=-0.0021530908,
            double Incl=2.0335188, double Omega=77.657956,
            double n0=1.1344374e-7, double alpha=1.3370697,
            double beta=4.1415004, double gamma=0.94206179,
            double mu=0.18873176):

        self.emissivity = emissivity
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0
        self.Incl = Incl
        self.Omega = Omega
        self.n0 = n0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu

        # derive frequently used quantities

        self.sinOmega = sin(Omega * degree)
        self.cosOmega = cos(Omega * degree)
        self.sinIncl = sin(Incl * degree)
        self.cosIncl = cos(Incl * degree)
        self.muInv = 1 / mu


cdef class Band:

    cdef double emissivity
    cdef double X0
    cdef double Y0
    cdef double Z0
    cdef double N0
    cdef double Dz
    cdef double Dr
    cdef double R0
    cdef double Vi
    cdef double Vr
    cdef double Pi
    cdef double Pr
    cdef double Omega
    cdef double Incl
    cdef double sinOmega
    cdef double cosOmega
    cdef double sinIncl
    cdef double cosIncl
    cdef double DzRInv
    cdef double DrInv
    cdef double ViInv

    # Default values correspond to Band1

    def __cinit__(
            self, double emissivity=1, double N0=5.5890290e-10,
            double Dz=8.7850534, double Dr=1.5, double R0=3.0, double Vi=0.1,
            double Vr=0.05, double Pi=4, double Pr=1, double P2r=4,
            double Omega=80, double Incl=0.56438265, double X0=0, double Y0=0,
            double Z0=0):

        self.emissivity = emissivity
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0
        self.N0 = N0
        self.Dz = Dz
        self.Dr = Dr
        self.R0 = R0
        self.Vi = Vi
        self.Vr = Vr
        self.Pi = Pi
        self.Pr = Pr
        self.Omega = Omega
        self.Incl = Incl

        # derive frequently used quantities

        self.sinOmega = sin(Omega * degree)
        self.cosOmega = cos(Omega * degree)
        self.sinIncl = sin(Incl * degree)
        self.cosIncl = cos(Incl * degree)
        self.DzRInv = 1 / (Dz * degree)
        self.DrInv = 1 / Dr
        self.ViInv = 1 / Vi


cdef class Ring:

    cdef double emissivity
    cdef double X0
    cdef double Y0
    cdef double Z0
    cdef double Omega
    cdef double Incl
    cdef double nsr
    cdef double Rsr
    cdef double sigmaRsr
    cdef double sigmaZsr
    cdef double sinOmega
    cdef double cosOmega
    cdef double sinIncl
    cdef double cosIncl
    cdef double sigmaRsr2Inv
    cdef double sigmaZsrInv

    def __cinit__(
            self, double emissivity=1, double Omega=22.278980,
            double Incl=0.48707166, double nsr=1.8260528e-08,
            double Rsr=1.0281924, double sigmaRsr=0.025000000,
            double sigmaZsr=0.054068037, double X0=0, double Y0=0, double Z0=0):

        self.emissivity = emissivity
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0
        self.Omega = Omega
        self.Incl = Incl
        self.nsr = nsr
        self.Rsr = Rsr
        self.sigmaRsr = sigmaRsr
        self.sigmaZsr = sigmaZsr

        # derive frequently used quantities

        self.sinOmega = sin(Omega * degree)
        self.cosOmega = cos(Omega * degree)
        self.sinIncl = sin(Incl * degree)
        self.cosIncl = cos(Incl * degree)
        self.sigmaRsr2Inv = 1 / (sigmaRsr * sigmaRsr)
        self.sigmaZsrInv = 1 / sigmaZsr


cdef class Blob:

    cdef double emissivity
    cdef double X0
    cdef double Y0
    cdef double Z0
    cdef double Omega
    cdef double Incl
    cdef double ntb
    cdef double Rtb
    cdef double sigmaRtb
    cdef double sigmaZtb
    cdef double thetaTB
    cdef double sigmaThetaTB
    cdef double sinOmega
    cdef double cosOmega
    cdef double sinIncl
    cdef double cosIncl
    cdef double thetaTBR
    cdef double sigmaRtbInv
    cdef double sigmaZtbInv
    cdef double sigmaThetaTBRinv

    def __cinit__(
            self, double emissivity=1, double Omega=22.278980,
            double Incl=0.48707166, double ntb=2.0094267e-08,
            double Rtb=1.0579183, double sigmaRtb=0.10287315,
            double sigmaZtb=0.091442964, double thetaTB=-10.0,
            double sigmaThetaTB=12.115211, double X0=0,
            double Y0=0, double Z0=0):

        self.emissivity = emissivity
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0
        self.Omega = Omega
        self.Incl = Incl
        self.ntb = ntb
        self.Rtb = Rtb
        self.sigmaRtb = sigmaRtb
        self.sigmaZtb = sigmaZtb
        # thetaTB is in degrees
        self.thetaTB = thetaTB
        # sigmaThetaTB is in degrees
        self.sigmaThetaTB = sigmaThetaTB

        # derive frequently used quantities

        self.sinOmega = sin(Omega * degree)
        self.cosOmega = cos(Omega * degree)
        self.sinIncl = sin(Incl * degree)
        self.cosIncl = cos(Incl * degree)
        # Translate to radians
        self.thetaTBR = thetaTB * degree
        self.sigmaRtbInv = 1 / sigmaRtb
        self.sigmaZtbInv = 1 / sigmaZtb
        # Translate to radians
        self.sigmaThetaTBRinv = 1 / (sigmaThetaTB * degree)


# Pre-fitted emissivities for Planck frequencies ASSUME TOI IS IN MJy/Sr

fitted_emissivities = {
    # cloud, band1, band2, band3, ring, blob
    100: (.012, 1.02, .08,  .72, .0, .0),
    143: (.022, 1.23, .15, 1.16, .0, .0),
    217: (.051, 1.30, .15, 1.27, .0, .0),
    353: (.106, 1.58, .39, 1.88, .0, .0),
    545: (.167, 1.74, .54, 2.54, .0, .0),
    857: (.256, 2.06, .85, 3.37, .0, .0)
}


cdef int get_band_emission(double[:] x, double[:] y, double[:] z,
                           Band band, double[:] result) nogil:
    cdef int n = x.shape[0], i
    cdef double X0 = band.X0
    cdef double Y0 = band.Y0
    cdef double Z0 = band.Z0
    cdef double N0 = band.N0
    cdef double R0 = band.R0
    cdef double Pr = band.Pr
    cdef double Pi = band.Pi
    cdef double sinOmega = band.sinOmega
    cdef double cosOmega = band.cosOmega
    cdef double sinIncl = band.sinIncl
    cdef double cosIncl = band.cosIncl
    cdef double DzRInv = band.DzRInv
    cdef double ViInv = band.ViInv
    cdef double DrInv = band.DrInv
    cdef double emissivity = band.emissivity

    cdef double R, Rinv, Z, zeta, ZDz, ViTerm, WtTerm, xprime, yprime, zprime
    cdef double ZDz2, ZDz4, ZDz6

    if band.X0 == 0 and band.Y0 == 0 and band.Z0 == 0 \
       and band.Pr == 1 and band.Pi == 4:

        # This version is optimized for Planck

        # Distance from the center of the cloud

        # for i in prange(n, schedule='static', chunksize=10):
        for i in range(n):
            xprime = x[i]
            yprime = y[i]
            zprime = z[i]
            R = sqrt(xprime*xprime + yprime*yprime + zprime*zprime)
            Rinv = 1 / R
            Z = (xprime*sinOmega - yprime*cosOmega) * sinIncl + zprime*cosIncl
            zeta = fabs(Z) * Rinv
            ZDz = zeta * DzRInv
            ZDz2 = ZDz * ZDz
            ZDz4 = ZDz2 * ZDz2
            ZDz6 = ZDz4 * ZDz2
            ViTerm = 1.0 + ZDz4 * ViInv
            WtTerm = 1 - exp(-pow(R*DrInv, 20))

            result[i] = N0 * exp(-ZDz6) * ViTerm * WtTerm \
                * R0 * Rinv * emissivity
    else:
        # Shift to center-of-the-cloud coordinates
        for i in range(n):
            xprime = x[i] - X0
            yprime = y[i] - Y0
            zprime = z[i] - Z0

            # Distance from the center of the cloud

            R = sqrt(xprime*xprime + yprime*yprime + zprime*zprime)
            Rinv = 1. / R
            Z = (xprime*sinOmega - yprime*cosOmega) * sinIncl \
                + zprime*cosIncl
            zeta = fabs(Z) * Rinv
            ZDz = zeta * band.DzRInv
            ViTerm = 1.0 + pow(ZDz, Pi) * ViInv
            WtTerm = 1 - exp(-pow(R*DrInv, 20))

            result[i] = N0 * exp(-pow(ZDz, 6)) * ViTerm * WtTerm \
                * pow(R0*Rinv, Pr) * emissivity

    return 0

cdef int get_blob_emission(double[:] x, double[:] y, double[:] z,
                           double xEarth, double yEarth, Blob blob,
                           double[:] result) nogil:
    """
    The trailing blob part of the ring emission.
    """

    cdef double atanEarth = atan2(yEarth, xEarth)

    cdef int n = x.shape[0], i
    cdef double X0 = blob.X0
    cdef double Y0 = blob.Y0
    cdef double Z0 = blob.Z0
    cdef double sinOmega = blob.sinOmega
    cdef double cosOmega = blob.cosOmega
    cdef double sinIncl = blob.sinIncl
    cdef double cosIncl = blob.cosIncl
    cdef double thetaTBR = blob.thetaTBR
    cdef double ntb = blob.ntb
    cdef double Rtb = blob.Rtb
    cdef double sigmaRtbInv = blob.sigmaRtbInv
    cdef double sigmaZtbInv = blob.sigmaZtbInv
    cdef double sigmaThetaTBRinv = blob.sigmaThetaTBRinv
    cdef double emissivity = blob.emissivity

    cdef double R, Z, xprime, yprime, zprime, dlon

    # for i in prange(n, schedule='static', chunksize=10):
    for i in range(n):
        xprime = x[i] - X0
        yprime = y[i] - Y0
        zprime = z[i] - Z0

        R = sqrt(xprime*xprime + yprime*yprime + zprime*zprime)
        Z = (xprime*sinOmega - yprime*cosOmega)*sinIncl + zprime*cosIncl
        # Relative heliocentric longitude between (x,y,z) and the blob
        dlon = atan2(y[i], x[i]) - (atanEarth + thetaTBR)

        while dlon < -M_PI:
            dlon = dlon + 2*M_PI
        while dlon > M_PI:
            dlon = dlon - 2*M_PI

        result[i] = ntb * exp(
            -0.5*pow((R - Rtb) * sigmaRtbInv, 2)
            - fabs(Z) * sigmaZtbInv
            - 0.5*pow(dlon * sigmaThetaTBRinv, 2)) * emissivity

    return 0


cdef int get_cloud_emission(double[:] x, double[:] y, double[:] z,
                            Cloud cloud, double[:] result) nogil:
    """
    /*
    * Inputs:
    *  s: The position at which to evaluate the function, in AU.
    *  X0, Y0, Z0: The position of the center of the cloud, in AU.
    *  Incl,Omega: The cloud inclination and ascending node, in degrees.
    *  n0        : density. Seems to have an extra (AU^-1) units,
    *              to account for the units.
    *  alpha,beta,gamma,mu: scalars.
    */
    """

    cdef int n = x.shape[0], i
    cdef double X0 = cloud.X0
    cdef double Y0 = cloud.Y0
    cdef double Z0 = cloud.Z0
    cdef double sinOmega = cloud.sinOmega
    cdef double cosOmega = cloud.cosOmega
    cdef double sinIncl = cloud.sinIncl
    cdef double cosIncl = cloud.cosIncl
    cdef double mu = cloud.mu
    cdef double muInv = cloud.muInv
    cdef double n0 = cloud.n0
    cdef double alpha = cloud.alpha
    cdef double beta = cloud.beta
    cdef double gamma = cloud.gamma
    cdef double emissivity = cloud.emissivity

    cdef double R, Rinv, Z, xprime, yprime, zprime, zeta, g

    # for i in prange(n, schedule='static', chunksize=10):
    for i in range(n):
        xprime = x[i] - X0
        yprime = y[i] - Y0
        zprime = z[i] - Z0
        R = sqrt(xprime*xprime + yprime*yprime + zprime*zprime)
        Rinv = 1 / R
        Z = (xprime*sinOmega - yprime*cosOmega)*sinIncl + zprime*cosIncl

        zeta = fabs(Z) * Rinv
        if zeta < mu:
            g = 0.5 * zeta * zeta * muInv
        else:
            g = zeta - 0.5 * mu

        result[i] = n0 * pow(R, -alpha) * exp(-beta*pow(g, gamma)) * emissivity

    return 0


cdef int get_ring_emission(double[:] x, double[:] y, double[:] z,
                           Ring ring, double[:] result) nogil:
    """
    This is half of the ring emission in Kelsall et al.
    The trailing blob is handled separately.
    """

    cdef int n = x.shape[0], i
    cdef double X0 = ring.X0
    cdef double Y0 = ring.Y0
    cdef double Z0 = ring.Z0
    cdef double sinOmega = ring.sinOmega
    cdef double cosOmega = ring.cosOmega
    cdef double sinIncl = ring.sinIncl
    cdef double cosIncl = ring.cosIncl
    cdef double nsr = ring.nsr
    cdef double Rsr = ring.Rsr
    cdef double sigmaRsr2Inv = ring.sigmaRsr2Inv
    cdef double sigmaZsrInv = ring.sigmaZsrInv
    cdef double emissivity = ring.emissivity

    cdef double xprime, yprime, zprime, R, Z

    # for i in prange(n, schedule='static', chunksize=10):
    for i in range(n):
        xprime = x[i] - X0
        yprime = y[i] - Y0
        zprime = z[i] - Z0
        R = sqrt(xprime*xprime + yprime*yprime + zprime*zprime)
        Z = (xprime*sinOmega - yprime*cosOmega)*sinIncl + zprime*cosIncl

        result[i] = nsr * exp(-0.5 * pow(R-Rsr, 2) * sigmaRsr2Inv
                              - fabs(Z)*sigmaZsrInv) * emissivity

    return 0


cdef ZodiValue(double nu, double elon, double elat, double xObs, double yObs,
               double zObs, Cloud cloud, Band band1, Band band2, Band band3,
               Ring ring, Blob blob, double T0=286.0, double delta=0.46686260,
               int nsteps=201, double Rmax=5.2, bint total=0):

    """
    /* ZodiValue
    *
    * Inputs:
    *  nu    : Frequency (in Hz) at which to do the calculations.
    *  elon  : Ecliptic longitude (in Radians) of the direction of observation
    *  elat  : Ecliptic latitude  (in Radians) of the direction of observation
    *  xObs  : Ecliptic x-coordinate (in AU) of the observer's location
    *  yObs  : Ecliptic y-coordinate (in AU) of the observer's location
    *  zObs  : Ecliptic z-coordinate (in AU) of the observer's location
    *  T0    : Temperature (in K) of the interplanetary dust at 1 AU
    *  delta : T(R)=T0*R**(-delta)
    *  cloud : Structure defining parameters of cloud
    *  band1 : Structure defining parameters of band1
    *  band2 : Structure defining parameters of band2
    *  band3 : Structure defining parameters of band3
    *  ring : Structure defining parameters of the circumsolar ring.
    *  blob : Structure defining parameters of the trailing blob
    *  nsteps: The number of integration steps to use (COBE used 201).
        Must be odd!
    *  Rmax  : Maximum radius (in AU) out to which we should integrate.
    *  total(False): If true, collapse the component emissions into
        a single number
    */
    """

    # Approximate Earth direction (distance is irrelevant) for the
    # trailing blob by the observatory position (relative to Sun).

    cdef double xEarth = xObs
    cdef double yEarth = yObs

    # Convert the lon./lat. to a position on the sphere at the outer limit

    # new calculation from Ken
    cdef double rSquared = xObs*xObs + yObs*yObs + zObs*zObs
    cdef double coselat = cos(elat)
    cdef double uvx = coselat * cos(elon)
    cdef double uvy = coselat * sin(elon)
    cdef double uvz = sin(elat)
    rCosTheta = xObs*uvx + yObs*uvy + zObs*uvz
    cdef double rho = sqrt(Rmax*Rmax - rSquared + rCosTheta*rCosTheta) \
        - rCosTheta
    # Find the point at the outer edge of the cloud along our line of sight
    cdef double x1 = xObs + rho*uvx
    cdef double y1 = yObs + rho*uvy
    cdef double z1 = zObs + rho*uvz

    # old calculation for line-of-sight
    # x1 = Rmax * np.cos(elat) * np.cos(elon)
    # y1 = Rmax * np.cos(elat) * np.sin(elon)
    # z1 = Rmax * np.sin(elat)

    cdef double dx = (x1-xObs) / (nsteps-1.0)
    cdef double dy = (y1-yObs) / (nsteps-1.0)
    cdef double dz = (z1-zObs) / (nsteps-1.0)

    cdef double const1 = 2 * h * nu * nu * nu / (c * c)
    cdef double const2 = h * nu / k
    cdef double T0inv = 1. / T0
    cdef double delta2 = delta / 2

    cdef double[:] x = np.zeros(nsteps, dtype=np.float64)
    cdef double[:] y = np.zeros(nsteps, dtype=np.float64)
    cdef double[:] z = np.zeros(nsteps, dtype=np.float64)
    cdef double[:] bb = np.zeros(nsteps, dtype=np.float64)

    cdef int i
    cdef double s, Tinv

    for i in range(nsteps):
        x[i] = xObs + i*dx
        y[i] = yObs + i*dy
        z[i] = zObs + i*dz
        # Blackbody emission
        s = x[i]*x[i] + y[i]*y[i] + z[i]*z[i]
        Tinv = pow(s, delta2) * T0inv
        bb[i] = const1 / (exp(const2*Tinv) - 1)

    cdef int nintegrand
    if total:
        nintegrand = 1
    else:
        nintegrand = 0
        if cloud.emissivity != 0:
            nintegrand += 1
        if band1.emissivity != 0:
            nintegrand += 1
        if band2.emissivity != 0:
            nintegrand += 1
        if band3.emissivity != 0:
            nintegrand += 1
        if ring.emissivity != 0:
            nintegrand += 1
        if blob.emissivity != 0:
            nintegrand += 1

    cdef double[:, :] integrands = np.zeros([nintegrand, nsteps],
                                            dtype=np.float64)
    cdef double[:] integrand = np.zeros(nsteps, dtype=np.float64)

    cdef int row = 0, col
    # use cython memoryview to speed up indexing
    cdef double[:] target

    if total:
        target = integrand
    else:
        target = integrands[row]

    if cloud.emissivity != 0:
        get_cloud_emission(x, y, z, cloud, target)
        if total:
            for col in range(nsteps):
                integrands[0, col] += target[col]
        else:
            row += 1
            target = integrands[row]

    if band1.emissivity != 0:
        get_band_emission(x, y, z, band1, target)
        if total:
            for col in range(nsteps):
                integrands[0, col] += target[col]
        else:
            row += 1
            target = integrands[row]

    if band2.emissivity != 0:
        get_band_emission(x, y, z, band2, target)
        if total:
            for col in range(nsteps):
                target[col] += integrand[col]
        else:
            row += 1
            target = integrands[row]

    if band3.emissivity != 0:
        get_band_emission(x, y, z, band3, target)
        if total:
            for col in range(nsteps):
                integrands[0, col] += target[col]
        else:
            row += 1
            target = integrands[row]

    if ring.emissivity != 0:
        get_ring_emission(x, y, z, ring, target)
        if total:
            for col in range(nsteps):
                integrands[0, col] += target[col]
        else:
            row += 1
            target = integrands[row]

    if blob.emissivity != 0:
        get_blob_emission(x, y, z, xEarth, yEarth, blob, target)
        if total:
            for col in range(nsteps):
                integrands[0, col] += target[col]

    for row in range(nintegrand):
        target = integrands[row]
        for col in range(nsteps):
            target[col] *= bb[col]

    cdef np.ndarray[double] integrals = np.zeros(nintegrand, dtype=np.float64)
    cdef double[:] integrals_view = integrals
    cdef double integral
    cdef double norm = 2 * sqrt(dx*dx + dy*dy + dz*dz) * 0.333333333333 * 1e20

    for row in range(nintegrand):
        target = integrands[row]
        integral = target[0] * 0.5
        for col in range(1, nsteps-1, 2):
            integral += 2 * target[col]
            integral += target[col+1]
        integral += target[nsteps-1] * 0.5
        integral *= norm
        integrals_view[row] = integral

    return integrals


class Zodier():

    def __init__(
            self, freq, nbin=1000, use_2015_emissivities=False,
            coord='G', coord_pos='E', bufsize=100000, emissivities=None):
        """
        Instantiate the zodier object
        Arguments:
        freq -- Observation frequency in GHz.
        nbin -- Number of phase bins to use when the phase is supplied.
        use_2015_emissivities(False) -- Sets the emissivity values to
            2015 best fits. Otherwise, "1.0" is used for all components.
        coord(G) -- Input pointing coordinate system
        coord_pos(E) -- Coordinate system for observatory position (in AU)
        emissivities -- Allow user to set the emissivities and disable
            Zodi components (especially the trailing blob) via zero emissivity
        """

        self.freq = freq

        self.nbin = nbin
        self.estim = SignalEstimator(nbin=self.nbin)

        if use_2015_emissivities:
            if freq in fitted_emissivities:
                ems = fitted_emissivities[freq]
            else:
                raise Exception('There are no hard-coded emissivities for {}GHz'
                                ''.format(freq))
        else:
            if emissivities is None:
                ems = np.ones(6)
            else:
                ems = emissivities

        self.cloud = Cloud(emissivity=ems[0])
        self.band1 = Band(emissivity=ems[1], N0=5.5890290e-10, Dz=8.7850534,
                          Dr=1.5, Vi=0.1, Vr=0.05, P2r=4.0, Omega=80.0,
                          Incl=0.56438265)
        self.band2 = Band(emissivity=ems[2], N0=1.9877609e-09, Dz=1.9917032,
                          Dr=0.94121881, Vi=0.89999998, Vr=.15,  P2r=4.0,
                          Omega=30.347476, Incl=1.2)
        self.band3 = Band(emissivity=ems[3], N0=1.4369827e-10, Dz=15.0,
                          Dr=1.5, Vi=0.05, Vr=-1.0, P2r=-1.0, Omega=80.0,
                          Incl=0.8)
        self.ring = Ring(emissivity=ems[4])
        self.blob = Blob(emissivity=ems[5])

        self.nzodi = np.sum(ems != 0)

        self.coord = coord
        self.coord_pos = coord_pos
        self.bufsize = bufsize

        # Zodiacal emission is estimated in ecliptic coordinates.
        # Prepare to rotate direction and position.

        if self.coord != 'E':
            self.rotmatrix_dir, do_conv, normcoord \
                = hp.rotator.get_coordconv_matrix([self.coord, 'E'])
            self.rotquat_dir = qa.from_rotmat(self.rotmatrix_dir)
        else:
            self.rotmatrix_dir = None
            self.rotquat_dir = None

        if self.coord_pos != 'E':
            self.rotmatrix_pos, do_conv, normcoord \
                = hp.rotator.get_coordconv_matrix([self.coord_pos, 'E'])
            self.rotquat_pos = qa.from_rotmat(self.rotmatrix_pos)
        else:
            self.rotmatrix_pos = None
            self.rotquat_pos = None

    def zodi(self, quats, position, phase=None, total=False, pntflag=None,
             nbin=3000):
        """
        Evaluate the Zodiacal emission based on orientation and position
        (in AU) of the observatory.
        Inputs:
        quats -- Detector orientation
        position -- observatory position
        phase(None) -- If the phases are supplied, the code will estimate
            the emission as a function of phase and interpolate. [RADIANS]
        total(False) -- if False, return an array of timeline vectors,
            one for each zodi component with a nonzero emissivity.
            Otherwise sum the timelines.
        pntflag(None) -- Stable pointing flags to identify the science scan
        """

        cdef int n = len(quats)

        zodi = np.zeros([self.nzodi, n], dtype=np.float64)

        if self.rotquat_pos is not None:
            position = qa.rotate(self.rotquat_pos, position.T).T

        dir = qa.rotate(quats, zaxis)

        if self.rotquat_dir is not None:
            dir = qa.rotate(self.rotquat_dir, dir)

        dir = np.atleast_2d(dir)

        elon, elat = hp.vec2dir(dir.T, lonlat=True)

        if phase is not None:
            # estimate the zodi on a grid of phases and interpolate
            if pntflag is None or np.all(pntflag != 0):
                x = phase.astype(np.float64).copy()
                y = np.atleast_1d(elon).copy()
                z = np.atleast_1d(elat).copy()
            else:
                x = phase[pntflag == 0].astype(np.float64).copy()
                y = np.atleast_1d(elon)[pntflag == 0]
                z = np.atleast_1d(elat)[pntflag == 0]
            ii = np.argsort(x)
            x = x[ii].astype(np.float64)
            y = y[ii].astype(np.float64)
            z = z[ii].astype(np.float64)
            wbin = 2*np.pi / nbin
            bins = (x / wbin).astype(np.int32)

            if np.any(bins < 0) or np.any(bins >= nbin):
                good = np.logical_and(bins >= 0, bins < nbin)
                x = x[good].copy()
                y = y[good].copy()
                z = z[good].copy()

            hitmap = np.zeros(nbin, dtype=np.int32)
            phasemap = np.zeros(nbin, dtype=np.float64)
            lonmap = np.zeros(nbin, dtype=np.float64)
            latmap = np.zeros(nbin, dtype=np.float64)

            destripe_tools.fast_hit_binning(bins, hitmap)
            destripe_tools.fast_binning(x, bins, phasemap)
            destripe_tools.fast_binning(y, bins, lonmap)
            destripe_tools.fast_binning(z, bins, latmap)

            good = hitmap != 0
            ngood = np.sum(good)
            hitmap = hitmap[good]
            phasemap = phasemap[good] / hitmap
            lonmap = lonmap[good] / hitmap * degree
            latmap = latmap[good] / hitmap * degree

            xpos, ypos, zpos = np.mean(position, axis=0)

            zodimaps = np.zeros([ngood, self.nzodi])

            for ibin, (binlon, binlat) in enumerate(zip(lonmap, latmap)):
                zodimaps[ibin] = ZodiValue(
                    self.freq*1e9, binlon, binlat, xpos, ypos, zpos,
                    self.cloud, self.band1, self.band2, self.band3,
                    self.ring, self.blob)

            zodimaps = zodimaps.T.copy()

            # Extend the phase and the signal to interpolate across
            # zero phase
            phasemap = np.hstack([phasemap[nbin-1]-2*np.pi,
                                  phasemap,
                                  phasemap[0]+2*np.pi])
            for i, zodimap in enumerate(zodimaps):
                zodimap = np.hstack([zodimap[nbin-1], zodimap, zodimap[0]])
                zodi[i] = np.interp(phase, phasemap, zodimap)
        else:
            istart = 0
            while istart < n:
                istop = min(istart+self.bufsize, n)
                for i in range(istart, istop):
                    try:
                        zodi[:, i] = ZodiValue(
                            self.freq*1e9, elon[i]*degree, elat[i]*degree,
                            position[i, 0], position[i, 1], position[i, 2],
                            self.cloud, self.band1, self.band2, self.band3,
                            self.ring, self.blob)
                    except Exception as e:
                        raise Exception(
                            'ZodiValue failed with "{}". istart = {}, '
                            'istop = {}, i = {}, shape(elon) = {}, '
                            'shape(elat) = {}, shape(position) = {}'.format(
                                e, istart, istop, i, np.shape(elon),
                                np.shape(elat), np.shape(position)))
                istart = istop

        if total:
            zodi = np.sum(zodi, axis=1)

        return zodi
