# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os.path

from scipy.constants import degree, arcmin

import healpy as hp
import numpy as np

import toast.qarray as qa
import toast.timing as timing

# Returned offsets will be in Dxx coordinates and arc minutes

xaxis, yaxis, zaxis = np.eye(3)

__path__ = os.path.dirname(__file__)
EPHEMERIS_PATH = 'ephemeris'


class PlanetFlagger():

    def __init__(self):

        self.mars = Pnt2Planeter('mars')
        self.jupiter = Pnt2Planeter('jupiter')
        self.saturn = Pnt2Planeter('saturn')
        self.uranus = Pnt2Planeter('uranus')
        self.neptune = Pnt2Planeter('neptune')

        self.planets = [self.mars, self.jupiter, self.saturn, self.uranus,
                        self.neptune]

    def flag(self, theta, phi, timestamps, radius):
        """
        Flag all planets out to radius [arcmin]
        """
        flags = np.zeros(len(theta), dtype=bool)
        cosmin = np.cos(radius * arcmin)
        for planet in self.planets:
            cosdist = planet.cosdist(theta, phi, timestamps)
            flags[cosdist > cosmin] = True
        return flags


class Pnt2Planeter():

    def __init__(self, target):
        """
        Initialize an object to translate detector pointing to
        planet-centric coordinates. Parameters:
        target -- Name of the target supplied
        """
        self.target = target
        fn = os.path.join(__path__, EPHEMERIS_PATH,
                          'positions_{}_planck.txt'.format(target.upper()))
        if not os.path.isfile(fn):
            raise Exception('pnt2planeter: No ephemeris file for {}: {}'
                            ''.format(target, fn))
        self.path = fn
        self.time, self.glon, self.glat = np.genfromtxt(
            self.path, usecols=(0, 1, 2), unpack=True)  # position is in degrees

        # The quats describe rotation of Z-axis to detector pointing.
        # We want the rotation from X-axis, so add a rotation from
        # X -> Z. We also rotate about the X-axis to have the scan
        # direction vertical instead of horizontal

        phi = 0
        theta = -np.pi / 2
        psi = np.pi / 2
        zquat = np.zeros(4)

        # XYX rotation
        zquat[3] = np.cos(.5 * theta) * np.cos(.5 * (phi + psi))
        zquat[0] = np.cos(.5 * theta) * np.sin(.5 * (phi + psi))
        zquat[1] = np.sin(.5 * theta) * np.cos(.5 * (phi - psi))
        zquat[2] = np.sin(.5 * theta) * np.sin(.5 * (phi - psi))

        self.zquat = zquat
        return

    def cosdist(self, theta, phi, timestamps):
        """
        Return the cosine of the angular distance between the target
        and pointing direction (theta, phi)
        Inputs are in radians.
        """

        if theta.ptp() > np.pi or phi.ptp() > 4 * np.pi:
            print('WARNING: theta and/or phi have large scatter. '
                  'They are expected to be in radians.')

        vec = hp.dir2vec(theta, phi)
        cosdist = self.cosdist_vec(vec, timestamps)

        return cosdist

    def cosdist_vec(self, vec, timestamps, full_output=False):
        """
        Return the cosine of the angular distance between the target
        and pointing direction
        """
        if timestamps[0] > self.time[-1] or timestamps[-1] < self.time[0]:
            raise Exception(
                'There is no overlap in the stored and provided time stamps.')

        tol = 3600.
        ind = np.logical_and(self.time >= timestamps[0] - tol,
                             self.time <= timestamps[-1] + tol)

        planettime = self.time[ind]
        planetglon_temp = self.glon[ind]
        planetglat_temp = self.glat[ind]

        planetglon = np.interp(timestamps, planettime, planetglon_temp)
        planetglat = np.interp(timestamps, planettime, planetglat_temp)

        planetvec = hp.dir2vec(planetglon, planetglat, lonlat=True)
        cosdist = np.sum(vec * planetvec, axis=0)

        if full_output:
            return cosdist, planetvec
        else:
            return cosdist

    def translate(self, quats, timestamps, psi_pol=None):
        """
        Translate the input quaternions into planet-centric coordinates
        and convert the offsets to arc minutes.  The quaternions ARE
        EXPECTED to be in galactic coordinates although adding a
        rotation would be straightforward.

        The output coordinate system is Pxx, unless either
        a) quats do not include the psi_pol rotation or
        b) psi_pol is provided in radians. translate() will then remove
            the psi_pol rotation from the quaternions
        """
        if timestamps[0] > self.time[-1] or timestamps[-1] < self.time[0]:
            raise Exception(
                'There is no overlap in the stored and provided time stamps.')

        if psi_pol is not None:
            pol_quat = qa.rotation(zaxis, -psi_pol)
            zquat = qa.mult(pol_quat, self.zquat)
        else:
            zquat = self.zquat

        my_quats = qa.mult(quats, zquat)  # From X-axis to detector position

        # Transform the planet positions into quaternions

        tol = 3600.

        ind = np.logical_and(self.time >= timestamps[0] - tol,
                             self.time <= timestamps[-1] + tol)
        nind = np.sum(ind)

        planettime = self.time[ind]
        planetglon = self.glon[ind]
        planetglat = self.glat[ind]
        planetquat = np.zeros([nind, 4])

        # ZYZ rotation to put the X-axis to the planet position

        phi = planetglon * degree
        theta = -planetglat * degree
        psi = 0

        planetquat[:, 3] = np.cos(.5 * theta) * np.cos(.5 * (phi + psi))
        planetquat[:, 0] = -np.sin(.5 * theta) * np.sin(.5 * (phi - psi))
        planetquat[:, 1] = np.sin(.5 * theta) * np.cos(.5 * (phi - psi))
        planetquat[:, 2] = np.cos(.5 * theta) * np.sin(.5 * (phi + psi))

        targetquats = qa.slerp(timestamps, planettime, planetquat)

        planetvec = qa.rotate(targetquats, xaxis)

        # Rotate the planet into Dxx frame (detector on X-axis)

        planetvec = qa.rotate(qa.inv(my_quats), planetvec)

        # The detector position relative to the planet is the inverse
        # of the planet coordinates in Dxx

        lon, lat = hp.vec2dir(planetvec.T, lonlat=True)

        az, el = -np.array([lon, lat]) * 60  # To arc minutes
        return az, el
