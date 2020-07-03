# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

import toast
from toast_planck.preproc_modules import Dipoler

import numpy as np

from .utilities import DEFAULT_PARAMETERS


# import warnings
# warnings.filterwarnings('error')
class OpDipolePlanck(toast.Operator):
    """
    Operator for simulating dipoles

    Args:
        freq (int):  Observing frequency in GHz
        solsys_speed(float) -- Solar system velocity wrt. CMB rest
             frame in km/s. Default is Planck 2015 best fit value
        solsys_glon(float) -- Solar system velocity direction longitude
             in degrees
        solsys_glat(float) -- Solar system velocity direction latitude
             in degrees
        output (str):  if None, write TOD, otherwise the name to use in
            the cache.
        margin (int):  Extra processing margin to include in the output
             dipole TOD.
        add_to_existing (boolean): if True, add the dipole to the output
              cache if False, create new cache containing dipole
             (default is False)
        keep_quats (boolean): If false, quaternions are cleared from the
             cache or never put there.
    """

    def __init__(
            self, freq, solsys_speed=None, solsys_glon=None,
            solsys_glat=None,
            output='dipole', margin=0, mode='total', add_to_existing=False,
            keep_quats=True, npipe_mode=False, lfi_mode=True):

        self._freq = freq
        self._solsys_speed = (solsys_speed if solsys_speed else
                              DEFAULT_PARAMETERS["solsys_speed"])
        self._solsys_glon = (solsys_glon if solsys_glon else
                             DEFAULT_PARAMETERS["solsys_glon"])
        self._solsys_glat = (solsys_glat if solsys_glat else
                             DEFAULT_PARAMETERS["solsys_glat"])
        self._npipe_mode = npipe_mode
        self._lfi_mode = lfi_mode and np.int(freq) < 100
        self._output = output
        self._margin = margin
        self._mode = mode.upper()
        self._add_to_existing = add_to_existing
        self._keep_quats = keep_quats
        super().__init__()

    # @profile
    def exec(self, data):
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world

        margin = self._margin

        if self._npipe_mode:
            dipoler = Dipoler(
                full4pi='npipe', solsys_speed=self._solsys_speed,
                solsys_glon=self._solsys_glon, solsys_glat=self._solsys_glat,
                comm=cworld, RIMO=data.obs[0]['tod'].RIMO)
        elif self._lfi_mode:
            dipoler = Dipoler(
                full4pi=True, solsys_speed=self._solsys_speed,
                solsys_glon=self._solsys_glon, solsys_glat=self._solsys_glat,
                comm=cworld, RIMO=data.obs[0]['tod'].RIMO)
        else:
            dipoler = Dipoler(
                freq=self._freq, solsys_speed=self._solsys_speed,
                solsys_glon=self._solsys_glon, solsys_glat=self._solsys_glat)

        for obs in data.obs:
            tod = obs['tod']

            velocity = tod.local_velocity(margin=margin)

            for det in tod.local_dets:
                quat = tod.local_pointing(det, margin=margin)

                if self._mode == 'TOTAL' or self._mode == 'ORBITAL':
                    dipo_total = dipoler.dipole(quat, velocity=velocity,
                                                det=det)
                if self._mode == 'SOLSYS' or self._mode == 'ORBITAL':
                    dipo_solsys = dipoler.dipole(quat, det=det)

                del quat

                if self._mode == 'TOTAL':
                    dipo = dipo_total
                elif self._mode == 'SOLSYS':
                    dipo = dipo_solsys
                elif self._mode == 'ORBITAL':
                    dipo = dipo_total - dipo_solsys
                else:
                    raise Exception(
                        'Unknown dipole mode: {}'.format(self._mode))

                cachename = '{}_{}'.format(self._output, det)
                if not self._add_to_existing:
                    tod.cache.put(cachename, dipo, replace=True)
                else:
                    tod.cache.reference(cachename)[:] += dipo

                if not self._keep_quats:
                    tod.cache.clear('{}_.*'.format(tod.POINTING_NAME))

        return
