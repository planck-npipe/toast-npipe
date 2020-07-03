# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import csv
import os
from toast_planck.utilities import to_radiometer

from scipy.constants import c, h, k

import healpy as hp
import numpy as np
import toast.qarray as qa
import toast.timing as timing

__path__ = os.path.dirname(__file__)
PARAM_PATH = os.path.join(__path__,
                          'lfi_fsl_data/DX12_dBdTcmb_release_S_param.csv')
PARAM_PATH_NPIPE = os.path.join(__path__, 'lfi_fsl_data/npipe_s_factors.csv')

XAXIS, YAXIS, ZAXIS = np.eye(3, dtype=np.float64)

SPINANGLE = np.radians(85)
SPINROT = qa.rotation(YAXIS, np.pi / 2 - SPINANGLE)
# Inverse light speed in km / s (the assumed unit for velocity)
CINV = 1e3 / c


class Dipoler():
    """
    Dipoler objects return the orbital and solar system dipole as seen
    by a specified detector.
    """

    def __init__(self, solsys_speed=370.082, solsys_glon=264.00,
                 solsys_glat=48.24, TCMB=2.72548, coord='G', coord_vel='E',
                 mode='QUATERNION', bufsize=1000000,
                 full4pi=False, comm=None, RIMO=None,
                 relativistic_addition=True, freq=0,
                 symmetrize_4pi=False):
        """
        Instantiate the dipoler object
        Arguments:
        solsys_speed(370.082) -- Solar system speed wrt. CMB rest
            frame in km/s. Default is Planck 2015 best fit value
        solsys_glon(264.00) -- Solar system velocity direction longitude
            in degrees
        solsys_glat(48.24) -- Solar system velocity direction latitude
            in degrees
        TCMB(2.72548) -- CMB monopole temperature
        coord(G) -- Input pointing coordinate system
        coord_vel(E) -- Coordinate system for orbital velocity
        mode(QUATERNION) -- Input pointing format, presently only
            QUATERNION is supported
        relativistic_addition (bool): Add the the velocities relativistically
        symmetrize_4pi (bool): Use average 4pi parameters in each horn
        """
        self.comm = comm
        if comm is not None:
            self.rank = comm.rank
        else:
            self.rank = 0
        self.coord = coord
        self.coord_vel = coord_vel
        self.bufsize = bufsize
        self.rimo = RIMO
        self.tcmb = TCMB
        self.freq = freq
        self._set_q()
        self.symmetrize_4pi = symmetrize_4pi
        self.full4pi = full4pi
        if full4pi == 'npipe':
            self.load_4pi_params(npipe=True)
        elif full4pi:
            self.load_4pi_params()
        self.mode = mode
        if self.mode != 'QUATERNION':
            raise Exception('Dipoler: Unknown pointing format: {}'.format(
                self.mode))
        self.solsys_speed = solsys_speed
        self.solsys_glon = solsys_glon
        self.solsys_glat = solsys_glat
        self.baryvel = np.zeros(3)
        dipole_proj = self.solsys_speed \
            * np.sin(np.radians(90 - self.solsys_glat))
        self.baryvel[0] = dipole_proj * np.cos(np.radians(self.solsys_glon))
        self.baryvel[1] = dipole_proj * np.sin(np.radians(self.solsys_glon))
        self.baryvel[2] = self.solsys_speed \
            * np.cos(np.radians(90 - self.solsys_glat))
        self.relativistic_addition = relativistic_addition
        if self.coord == 'G':
            self.rotmatrix = None
            self.rotquat = None
        else:
            # Rotate the solar system velocity to pointing coordinate system
            self.rotmatrix = hp.rotator.get_coordconv_matrix(
                ['G', self.coord])[0]
            self.rotquat = qa.from_rotmat(self.rotmatrix)
            self.baryvel = hp.rotator.rotateVector(self.rotmatrix, self.baryvel)

        if self.coord != self.coord_vel:
            self.rotmatrix_vel = hp.rotator.get_coordconv_matrix(
                    [self.coord_vel, self.coord])[0]
            self.rotquat_vel = qa.from_rotmat(self.rotmatrix_vel)

        self._last_det = None
        self._last_params = None
        return

    def _set_q(self):
        """ Set the relativistic frequency factor for
        frequency-dependent dipole.
        """
        x = h * self.freq * 1e9 / (k * self.tcmb)
        self.x = x
        if x != 0:
            self.q = (x / 2) * (np.exp(x) + 1) / (np.exp(x) - 1)
        else:
            self.q = 1
        return

    def load_4pi_params(self, npipe=False):
        """ Load the dipole/far side lobe convolution parameters.
        """

        if npipe:
            fname = PARAM_PATH_NPIPE
        else:
            fname = PARAM_PATH
        if not os.path.isfile(fname):
            raise RuntimeError(
                'dipoler: no FSL parameter file found at {}'.format(fname))

        if self.rank == 0:
            self.fsl_params = {}

            with open(fname, 'r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                keys = None
                for row in csvreader:
                    if keys is None:
                        keys = row[2:]
                        continue
                    freq = int(row[2]) // 1000
                    if freq < 100:
                        det = 'LFI' + row[0] + row[1]
                    else:
                        det = '{}-{}{}'.format(freq, row[0], row[1].lower())
                    detdict = {}
                    for key, value in zip(keys, row[2:]):
                        detdict[key] = np.float(value)
                    self.fsl_params[det] = detdict
            if self.symmetrize_4pi:
                # Average the 4pi parameters in each horn
                for det in self.fsl_params.keys():
                    if det[-1] in 'aM':
                        pairdet = det.replace('a', 'b').replace('M', 'S')
                        dict1 = self.fsl_params[det]
                        dict2 = self.fsl_params[pairdet]
                        for key in keys:
                            avg = .5 * (dict1[key] + dict2[key])
                            dict1[key] = avg
                            dict2[key] = avg
        else:
            self.fsl_params = None

        if self.comm is not None:
            self.fsl_params = self.comm.bcast(self.fsl_params, root=0)

    def _get_total_velocity(self, ind, nind, velocity, orbital_only):
        if velocity is None:
            proper = np.tile(self.baryvel, (nind, 1))
            tvel = None
        else:
            tvel = np.atleast_2d(velocity)[ind].copy()
            if self.coord_vel != self.coord:
                tvel = np.atleast_2d(qa.rotate(self.rotquat_vel, tvel))
            if self.relativistic_addition:
                tvel_parallel = np.sum(
                    tvel * self.baryvel, 1).reshape([-1, 1]) \
                    * np.tile(self.baryvel / self.solsys_speed ** 2, (nind, 1))
                tvel_perp = tvel - tvel_parallel
                vdot = 1 / (1 + (self.baryvel * CINV ** 2) * tvel)
                invgamma = np.sqrt(1 - (self.solsys_speed * CINV) ** 2)
                tvel_parallel = vdot * (tvel_parallel + self.baryvel)
                tvel_perp = vdot * tvel_perp * invgamma
                proper = tvel_parallel + tvel_perp
            else:
                proper = tvel + self.baryvel
            if orbital_only:
                proper -= self.baryvel
        return proper, tvel

    def _get_4pi_dipole(self, detector, proper, quats, ind, dipole):
        # Rotate velocity into the detector frame.  This must be the same
        # frame (Pxx or Dxx) the 4pi coefficients were computed in.
        # Pure qa.inv(quats) rotates into the Dxx frame
        # Adding psi_uv rotates into Pxx
        if self.full4pi == 'npipe':
            # NPIPE factors are computed in Dxx
            psi_uv = np.radians(self.rimo[detector].psi_uv)
            psi_pol = np.radians(self.rimo[detector].psi_pol)
            polrot = qa.rotation(ZAXIS, -(psi_uv + psi_pol))
            vel = qa.rotate(
                qa.inv(qa.mult(np.atleast_2d(quats)[ind], polrot)),
                proper * CINV)
        else:
            # LFI factors are in Pxx
            psi_pol = np.radians(self.rimo[detector].psi_pol)
            polrot = qa.rotation(ZAXIS, -psi_pol)
            vel = qa.rotate(
                qa.inv(qa.mult(np.atleast_2d(quats)[ind], polrot)),
                proper * CINV)
        dipole_amplitude = self.get_fourpi_prod(vel, detector, 0)
        # relativistic corrections for the quadrupole
        vel2 = vel.T.copy()
        for i in range(3):
            dipole_amplitude += self.q * vel2[i] \
                * self.get_fourpi_prod(vel, detector, i + 1)
        dipole_amplitude *= self.tcmb
        if self.full4pi == 'npipe':
            # Apply beam efficiency correction so the template
            # reflects unit response to a dipole signal
            dipole_amplitude /= self._last_params[4]
        dipole[ind] = dipole_amplitude
        return

    def _get_pencil_dipole(self, proper, quats, ind, dipole):
        speed = np.sqrt(np.sum(proper ** 2, axis=1))
        invspeed = 1 / speed
        proper_dir = np.tile(invspeed, (3, 1)).T * proper
        beta = speed * CINV
        det_dir = qa.rotate(np.atleast_2d(quats)[ind], ZAXIS)
        # Relativistic calculation up to second order WITH
        # quadrupole correction
        # See Eq. (2.5) in arXiv:1504.02076v2
        # We omit the beta**2 / 2 offset term
        z = np.sum(proper_dir * det_dir, axis=1)
        betaz = beta * z
        dipole[ind] = self.tcmb * (betaz * (1 + betaz * self.q))
        return det_dir

    def _get_fgdipole(self, tvel, quats, det_dir, ind, fg, fgdipole):
        # Foregrounds are only modulated by the orbital motion
        speed = np.sqrt(np.sum(tvel ** 2, axis=1))
        if np.any(speed == 0):
            raise Exception('Zero speed in dipole calculation')
        invspeed = 1 / speed
        proper_dir = np.tile(invspeed, (3, 1)).T * tvel
        beta = speed * CINV
        if det_dir is None:
            det_dir = qa.rotate(np.atleast_2d(quats)[ind], ZAXIS)
        num = 1 - beta * np.sum(proper_dir * det_dir, axis=1)
        invgamma = np.sqrt(1 - beta ** 2)
        fgdipole[ind] = fg[ind] * (1 / num * invgamma - 1)
        return

    def dipole(self, quats, velocity=None, fg=None, det=None,
               orbital_only=False):
        """ Evaluate the CMB dipole.

        Evaluate the CMB dipole (in K_CMB) according to the solar system
        motion and the optional orbital velocity information (in km/s).
        if a foreground map and orbital velocity are provided, will also
        return a foreground Doppler effect.
        """
        if velocity is None and fg is not None:
            raise RuntimeError(
                'Cannot evaluate foreground dipole without velocity')

        detector = det
        if det is not None and det[-1] in '01' and det[-2] != '-':
            detector = to_radiometer(det)

        nsamp = len(np.atleast_2d(quats))
        dipole = np.zeros(nsamp)
        if fg is not None:
            fgdipole = np.zeros(nsamp)

        istart = 0
        while istart < nsamp:
            istop = min(istart + self.bufsize, nsamp)
            ind = slice(istart, istop)
            nind = istop - istart
            istart = istop
            proper, tvel = self._get_total_velocity(ind, nind, velocity,
                                                    orbital_only)
            if self.full4pi:
                self._get_4pi_dipole(detector, proper, quats, ind, dipole)
                det_dir = None
            else:
                det_dir = self._get_pencil_dipole(proper, quats, ind, dipole)
            if fg is not None:
                self._get_fgdipole(tvel, quats, det_dir, ind, fg, fgdipole)

        if len(np.shape(quats)) == 1:
            dipole = dipole[0]
            if fg is not None:
                fg = fg[0]

        if fg is None:
            return dipole
        else:
            return dipole, fgdipole

    def get_fourpi_prod(self, vel, det, kind):
        if self._last_det != det:
            params = []
            for comps in [['S100', 'S010', 'S001'],  #   x,  y,  z
                          ['S200', 'S110', 'S101'],  #  xx, xy, xz
                          ['S110', 'S020', 'S011'],  #  xy, yy, yz
                          ['S101', 'S011', 'S002']]:  # xz, yz, zz
                params.append(
                    np.array([self.fsl_params[det][comp] for comp in comps]))
            # Append the main beam efficiency
            params.append(self.fsl_params[det]['Int001'])
            self._last_det = det
            self._last_params = params
        fsl_params = self._last_params[kind]
        # result = qa.arraylist_dot(vel, fsl_params).flatten()
        result = np.sum(vel * fsl_params, 1)
        return result


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.style.use('classic')

    nside = 256
    npix = 12 * nside ** 2

    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix)

    thetaquat = qa.rotation(YAXIS, theta)
    phiquat = qa.rotation(ZAXIS, phi)
    quat = qa.mult(phiquat, thetaquat)

    dipoler = Dipoler(freq=0)
    dipo = dipoler.dipole(quat) * 1e3
    hp.mollview(dipo, title='Solar system dipole, freq=0', unit='mK')
    plt.gca().graticule(30)

    plt.figure(figsize=[20, 6])
    plt.suptitle('Doppler quadrupole')
    for ifreq, freq in enumerate([0, 30, 44, 70, 100, 143, 217, 353, 545, 857]):
        dipoler = Dipoler(freq=freq)
        dipo = dipoler.dipole(quat) * 1e6
        quad = hp.remove_dipole(dipo)
        hp.mollview(quad, title='{}GHz, q={:.3f}, P-to-P={:.1f}uK'.format(
                    freq, dipoler.q, np.ptp(quad)), sub=[2, 5, 1 + ifreq],
                    unit=r'$\mu$K', min=-2, max=4)
        plt.gca().graticule(30)

    plt.show()
