# Copyright (c) 2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from toast_planck.preproc_modules.filters import flagged_running_average

from scipy.signal.signaltools import fftconvolve

import numpy as np
import toast.timing as timing


class GlitchRemoverLFI():

    def __init__(self):
        """
        Instantiate a glitch remover object. Parameters:
        """
        pass

    def remove(self, signal, flags, signal_estimate=None, ssoflags=None):

        if len(np.shape(signal)) == 2:
            # Only use the LOAD signals for glitch detection
            flags1 = self.flag_glitches(signal[:, 1].ravel(), flags)
            flags2 = self.flag_glitches(signal[:, 3].ravel(), flags)
            flags_out = flags1 + flags2
        else:
            flags_in = flags.copy()
            if ssoflags is not None:
                flags_in |= ssoflags
            flags_out = self.flag_glitches(signal - signal_estimate, flags_in)
            flags[flags_out] = True

        return signal, flags

    def flag_glitches(self, signal, flags):

        flags_out = np.zeros_like(flags)

        # Search for outliers in raw samples
        good = flags == 0
        for _ in range(10):
            med = np.median(signal[good])
            dist = np.median((signal[good] - med) ** 2) ** .5
            bad = np.abs(signal - med) > 10 * dist
            bad[np.logical_not(good)] = False
            nbad = np.sum(bad)
            if nbad == 0:
                break
            good[bad] = False
        bad = np.logical_and(flags == 0, np.logical_not(good))
        nbad = np.sum(bad)
        if nbad != 0:
            # extend the flags
            bad = fftconvolve(bad, np.ones(1000), mode='same') >= 1
            flags_out[bad] = True

        # Search for outliers in a median-smoothed signal
        good = flags == 0
        wkernel = 1000
        for _ in range(10):
            fsignal = flagged_running_average(
                signal - med, np.logical_not(good), wkernel) + med
            med = np.median(fsignal[good])
            dist = np.median((fsignal[good] - med) ** 2) ** .5
            bad = np.abs(fsignal - med) > 10 * dist
            bad[np.logical_not(good)] = False
            nbad = np.sum(bad)
            if nbad == 0:
                break
            good[bad] = False
        bad = np.logical_and(flags == 0, np.logical_not(good))
        nbad = np.sum(bad)
        if nbad != 0:
            # extend the flags
            bad = fftconvolve(bad, np.ones(wkernel), mode='same') >= 1
            flags_out[bad] = True
        return flags_out
