# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from toast._libtoast import filter_polynomial as polyfilter

from ..utilities import PLANCK_DETINDX
from .signal_estimation import SignalEstimator


class GapFiller():

    def __init__(self, order=5, nbin=10000):
        """
        Instantiate a gap filler object. Parameters:
        order -- order of a trending polynomial
        nbin -- number of phase bins in the signal estimate.
        """
        self.order = order
        self.nbin = nbin
        self.estim = SignalEstimator(self.nbin)

    def fill(self, signal, phase, flag, det, ring_number,
             dark=False, signal_estimate=None, pntflag=None):
        """
        Perform gap filling on given signal. Inputs:
        signal -- demodulated and gap-filled signal to be corrected
        phase -- spin phase in RADIANS
        flag -- extra processing flags not present in signal.mask
        det -- detector name (used for RNG)
        ring_number -- ring index (used for RNG)
        dark -- enable dark bolometer mode (disable signal estimation)
        signal_estimate(None) -- estimate of the total sky emission in
            the same units as signal
        """
        np.random.seed(
            123456 + 100000 * (PLANCK_DETINDX[det] + 1) + ring_number)

        cleaned_signal = signal.copy()
        good = (flag == 0)
        if pntflag is not None:
            good[pntflag] = False

        if not dark:
            # subtract signal
            if signal_estimate is not None:
                unrolled_signal = (
                    signal_estimate +
                    np.median(cleaned_signal[good] - signal_estimate[good]))
            else:
                try:
                    self.estim.fit(phase[good], cleaned_signal[good])
                except Exception as e:
                    raise Exception('gap_filler: Signal estimation '
                                    'failed: {}'.format(e))

                unrolled_signal = self.estim.eval(phase)

            cleaned_signal -= unrolled_signal

        n = cleaned_signal.size
        cleaned_signal_copy = cleaned_signal.copy()

        # fit a polynomial to the signal

        polyfilter(
            self.order,
            np.logical_not(good).astype(np.uint8),
            [cleaned_signal],
            np.array([0]),
            np.array([n]),
        )
        trend = cleaned_signal_copy - cleaned_signal
        del cleaned_signal_copy

        # get clean data rms

        rms = np.std(cleaned_signal[good])

        # Fill the gaps with signal estimate, white noise and polynomial trend.

        corrected_signal = signal.copy()

        bad = (flag != 0)
        nbad = np.sum(bad)

        if not dark:
            corrected_signal[bad] = (
                np.random.randn(nbad) * rms +
                trend[bad] + unrolled_signal[bad])
        else:
            corrected_signal[bad] = np.random.randn(nbad) * rms + trend[bad]

        return corrected_signal
