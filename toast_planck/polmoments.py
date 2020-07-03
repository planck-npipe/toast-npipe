# Copyright (c) 2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np

import toast
from toast.mpi import MPI
from toast import qarray

from .reproc_modules.destripe_tools import fast_binning, fast_hit_binning
from .utilities import to_radiometer, to_diodes


class OpPolMomentsPlanck(toast.Operator):
    """
    Operator that builds polarization angle moment maps for QuickPol.

    Args:
        nside (int): NSIDE resolution for Healpix maps.
        pixels (str): write pixels to the cache with name <pixels>_<detector>.
            If the named cache objects do not exist, then they are created.
        weights (str): write pixel weights to the cache with name
            <weights>_<detector>.  If the named cache objects do not exist,
            then they are created.
        single_precision (bool): Store the quaternions, weights and
            pixel numbers in 32bit cache vectors
        smax (int): highest moment to map
        prefix (str): filename prefix (suffix will be '_DETECTOR.fits')
    """

    def __init__(
            self, nside=1024, RIMO=None, margin=0,
            single_precision=False,
            keep_vel=True, keep_pos=True, keep_phase=True, keep_quats=True,
            smax=6, prefix='polmoments', nest=False):
        self._nside = nside
        self._npix = 12 * nside ** 2
        self._margin = margin
        self._single_precision = single_precision
        self._keep_vel = keep_vel
        self._keep_pos = keep_pos
        self._keep_phase = keep_phase
        self._keep_quats = keep_quats
        self._prefix = prefix
        self._smax = smax
        self._nest = nest

        if RIMO is None:
            raise ValueError('You must specify which RIMO to use')

        # The Reduced Instrument Model contains the necessary detector
        # parameters
        self.RIMO = RIMO

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    @property
    def rimo(self):
        return self.RIMO

    @property
    def nside(self):
        return self._nside

    def exec(self, data):

        comm = data.comm.comm_world

        xaxis, _, zaxis = np.eye(3, dtype=np.float64)
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

        # Read velocity

        for obs in data.obs:
            tod = obs['tod']
            vel = tod.local_velocity(margin=self._margin)
            if self._single_precision:
                vel = vel.astype(np.float32)
                vel = tod.cache.put(tod.VELOCITY_NAME, vel, replace=True)
            del vel

        for obs in data.obs:
            tod = obs['tod']
            tod.purge_eff_cache()

        # Read position

        if self._keep_pos:
            for obs in data.obs:
                tod = obs['tod']
                try:  # LFI pointing files are missing the position
                    pos = tod.local_position_position(margin=self._margin)
                except Exception:
                    pos = None
                if pos is not None and self._single_precision:
                    pos = pos.astype(np.float32)
                    tod.cache.put(tod.POSITION_NAME, pos, replace=True)
                del pos
            for obs in data.obs:
                tod = obs['tod']
                tod.purge_eff_cache()

        # Read phase

        if self._keep_phase:
            for obs in data.obs:
                tod = obs['tod']
                phase = tod.local_phase(margin=self._margin)
                if self._single_precision:
                    phase = phase.astype(np.float32)
                    tod.cache.put(tod.PHASE_NAME, phase, replace=True)
                del phase
            for obs in data.obs:
                tod = obs['tod']
                tod.purge_eff_cache()

        # Generate attitude

        first_obs = True
        for obs in data.obs:
            tod = obs['tod']
            nsamp = tod.local_samples[1]
            commonflags = tod.local_common_flags(margin=self._margin)
            satquats = None
            for detector in tod.local_dets:
                if detector[-1] in '01' and detector[-2] != '-':
                    # Single diode, share pointing with the other diode
                    # in the same radiometer arm
                    if detector[-1] == '1':
                        # We may not need to process this diode
                        detector2 = detector[:-1] + '0'
                        if detector2 in tod.local_dets:
                            continue
                    det = to_radiometer(detector)
                    diodes = to_diodes(det)
                else:
                    det = detector
                    diodes = []
                # psidet = np.radians(
                #        self.RIMO[det].psi_uv + self.RIMO[det].psi_pol)

                vel = tod.local_velocity()
                if len(vel) != nsamp + 2 * self._margin:
                    raise Exception('Cached velocities do not include margins.')
                if satquats is None:
                    detquats, satquats = tod.read_pntg(
                        detector=detector, margin=self._margin, deaberrate=True,
                        velocity=vel, full_output=True)
                else:
                    detquats = tod.read_pntg(
                        detector=detector, margin=self._margin, deaberrate=True,
                        velocity=vel, satquats=satquats)
                del vel
                if len(detquats) != nsamp + 2 * self._margin:
                    raise Exception('Cached quats do not include margins.')

                theta, phi, psi = qarray.to_angles(detquats)

                flags = tod.local_flags(det, margin=self._margin)
                if len(flags) != nsamp + 2 * self._margin:
                    raise Exception('Cached flags do not include margins.')
                totflags = flags != 0
                totflags[commonflags != 0] = True

                if self._single_precision:
                    cachename = '{}_{}'.format(tod.POINTING_NAME, det)
                    tod.cache.put(cachename, detquats.astype(np.float32),
                                  replace=True)
                    for diode in diodes:
                        alias = '{}_{}'.format(tod.POINTING_NAME, diode)
                        tod.cache.add_alias(alias, cachename)

                pixels = hp.ang2pix(self._nside, theta, phi, nest=self._nest)
                pixels = pixels.astype(np.int32)
                pixels[totflags] = -1

                # Save hits

                hmap = np.zeros(self._npix, dtype=np.int32)
                fast_hit_binning(pixels, hmap)
                comm.Allreduce(MPI.IN_PLACE, hmap, op=MPI.SUM)

                if comm.rank == 0:
                    fname = self._prefix + '_{}_hits.fits'.format(det)
                    if os.path.isfile(fname) and not first_obs:
                        print('Loading existing hit map from {} to co-add'
                              ''.format(fname), flush=True)
                        old_hmap = hp.read_map(fname, nest=self._nest,
                                               dtype=np.int32)
                        hmap += old_hmap
                    hp.write_map(fname, hmap, dtype=np.int32,
                                 column_names=['hits'], coord='G',
                                 nest=self._nest, overwrite=True)
                    print('Saved hitmap in {}'.format(fname), flush=True)
                del hmap

                # From Pxx to Dxx
                # psi -= psidet

                colnames = []
                coldata = []
                for s in range(1, self._smax + 1):
                    for func in ['COS', 'SIN']:
                        if func == 'COS':
                            toi = np.cos(s * psi)
                        else:
                            toi = np.sin(s * psi)
                        mmap = np.zeros(self._npix)
                        fast_binning(toi, pixels, mmap)
                        comm.Allreduce(MPI.IN_PLACE, mmap, op=MPI.SUM)
                        if comm.rank == 0:
                            colnames.append('SUM_{}_{}_PSI'.format(func, s))
                            coldata.append(mmap.astype(np.float32))
                        del mmap

                if comm.rank == 0:
                    fname = self._prefix + '_{}.fits'.format(det)
                    if os.path.isfile(fname) and not first_obs:
                        print('Loading existing moment maps from {} to co-add'
                              ''.format(fname), flush=True)
                        old_coldata = hp.read_map(fname, None, nest=self._nest)
                        coldata += old_coldata
                    hp.write_map(fname, coldata, dtype=np.float32,
                                 column_names=colnames, coord='G',
                                 nest=self._nest, overwrite=True)
                    print('Saved moment maps in {}'.format(fname), flush=True)
                del colnames
                del coldata

            first_obs = False

        for obs in data.obs:
            tod = obs['tod']
            tod.purge_eff_cache()

        if not self._keep_vel:
            for obs in data.obs:
                tod = obs['tod']
                tod.cache.destroy(tod.VELOCITY_NAME)

        return
