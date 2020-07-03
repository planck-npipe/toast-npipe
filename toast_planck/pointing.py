# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np
import toast
import toast.qarray as qa
import toast.tod as tt

from .utilities import to_radiometer, to_diodes


class OpPointingPlanck(toast.Operator):
    """
    Operator which generates healpix pointing

    Args:
        nside (int): NSIDE resolution for Healpix maps.
        pixels (str): write pixels to the cache with name <pixels>_<detector>.
            If the named cache objects do not exist, then they are created.
        weights (str): write pixel weights to the cache with name
            <weights>_<detector>.  If the named cache objects do not exist,
            then they are created.
        single_precision (bool): Store the quaternions, weights and
            pixel numbers in 32bit cache vectors
    """

    def __init__(
        self,
        nside=1024,
        mode="I",
        RIMO=None,
        margin=0,
        apply_flags=True,
        single_precision=False,
        keep_vel=True,
        keep_pos=True,
        keep_phase=True,
        keep_quats=True,
    ):
        self._nside = nside
        self._mode = mode
        self._margin = margin
        self._apply_flags = apply_flags
        self._single_precision = single_precision
        self._keep_vel = keep_vel
        self._keep_pos = keep_pos
        self._keep_phase = keep_phase
        self._keep_quats = keep_quats

        if RIMO is None:
            raise ValueError("You must specify which RIMO to use")

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
        xaxis, _, zaxis = np.eye(3, dtype=np.float64)
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

        # Read velocity

        for obs in data.obs:
            tod = obs["tod"]
            vel = tod.local_velocity(margin=self._margin)
            if self._single_precision:
                vel = vel.astype(np.float32)
                vel = tod.cache.put(tod.VELOCITY_NAME, vel, replace=True)
            del vel

        for obs in data.obs:
            tod = obs["tod"]
            tod.purge_eff_cache()

        # Read position

        if self._keep_pos:
            for obs in data.obs:
                tod = obs["tod"]
                try:  # LFI pointing files are missing the position
                    pos = tod.local_position_position(margin=self._margin)
                except Exception:
                    pos = None
                if pos is not None and self._single_precision:
                    pos = pos.astype(np.float32)
                    tod.cache.put(tod.POSITION_NAME, pos, replace=True)
                del pos
            for obs in data.obs:
                tod = obs["tod"]
                tod.purge_eff_cache()

        # Read phase

        if self._keep_phase:
            for obs in data.obs:
                tod = obs["tod"]
                phase = tod.local_phase(margin=self._margin)
                if self._single_precision:
                    phase = phase.astype(np.float32)
                    tod.cache.put(tod.PHASE_NAME, phase, replace=True)
                del phase
            for obs in data.obs:
                tod = obs["tod"]
                tod.purge_eff_cache()

        # Generate attitude

        for obs in data.obs:
            tod = obs["tod"]
            nsamp = tod.local_samples[1]
            commonflags = tod.local_common_flags(margin=self._margin)
            satquats = None
            for detector in tod.local_dets:
                if detector[-1] in "01" and detector[-2] != "-":
                    # Single diode, share pointing with the other diode
                    # in the same radiometer arm
                    if detector[-1] == "1":
                        # We may not need to process this diode
                        detector2 = detector[:-1] + "0"
                        if detector2 in tod.local_dets:
                            continue
                    det = to_radiometer(detector)
                    diodes = to_diodes(det)
                else:
                    det = detector
                    diodes = []

                vel = tod.local_velocity()
                if len(vel) != nsamp + 2 * self._margin:
                    raise Exception("Cached velocities do not include margins.")
                if satquats is None:
                    pdata, satquats = tod.read_pntg(
                        detector=detector,
                        margin=self._margin,
                        deaberrate=True,
                        velocity=vel,
                        full_output=True,
                    )
                else:
                    pdata = tod.read_pntg(
                        detector=detector,
                        margin=self._margin,
                        deaberrate=True,
                        velocity=vel,
                        satquats=satquats,
                    )
                del vel
                if len(pdata) != nsamp + 2 * self._margin:
                    raise Exception("Cached quats do not include margins.")

                if self._apply_flags:
                    flags = tod.local_flags(det, margin=self._margin)
                    if len(flags) != nsamp + 2 * self._margin:
                        raise Exception("Cached flags do not include margins.")
                    totflags = flags != 0
                    totflags[commonflags != 0] = True

                if self._single_precision:
                    cachename = "{}_{}".format(tod.POINTING_NAME, det)
                    tod.cache.put(cachename, pdata.astype(np.float32), replace=True)
                    for diode in diodes:
                        alias = "{}_{}".format(tod.POINTING_NAME, diode)
                        tod.cache.add_alias(alias, cachename)

                if self._apply_flags:
                    pdata[totflags, :] = nullquat

                theta, phi, psi = qa.to_angles(pdata)

                pixels = hp.ang2pix(self._nside, theta, phi, nest=True)
                if self._apply_flags:
                    pixels[totflags] = -1

                epsilon = self.RIMO[det].epsilon
                eta = (1 - epsilon) / (1 + epsilon)

                weights = None
                if self._mode == "I":
                    weights = np.ones([nsamp + 2 * self._margin, 1], dtype=np.float64)
                elif self._mode == "IQU":
                    Ival = np.ones(nsamp + 2 * self._margin)
                    Qval = eta * np.cos(2 * psi)
                    Uval = eta * np.sin(2 * psi)

                    weights = np.column_stack((Ival, Qval, Uval))
                else:
                    raise RuntimeError("invalid mode for Planck Pointing")

                pixelsname = "{}_{}".format(tod.PIXEL_NAME, det)
                if self._single_precision:
                    tod.cache.put(pixelsname, pixels.astype(np.int32), replace=True)
                else:
                    tod.cache.put(pixelsname, pixels.astype(np.int64), replace=True)
                for diode in diodes:
                    alias = "{}_{}".format(tod.PIXEL_NAME, diode)
                    tod.cache.add_alias(alias, pixelsname)

                weightsname = "{}_{}".format(tod.WEIGHT_NAME, det)
                if self._single_precision:
                    tod.cache.put(weightsname, weights.astype(np.float32), replace=True)
                else:
                    tod.cache.put(weightsname, weights.astype(np.float64), replace=True)
                for diode in diodes:
                    alias = "{}_{}".format(tod.WEIGHT_NAME, diode)
                    tod.cache.add_alias(alias, weightsname)

        for obs in data.obs:
            tod = obs["tod"]
            tod.purge_eff_cache()

        if not self._keep_vel:
            for obs in data.obs:
                tod = obs["tod"]
                tod.cache.destroy(tod.VELOCITY_NAME)

        return
