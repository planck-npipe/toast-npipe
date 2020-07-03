# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.signal import fftconvolve
import toast.timing as timing


def flagged_running_average(signal, flag, wkernel, return_flags=False,
                            downsample=False):
    """
    Compute a running average considering only the unflagged samples.
    Args:
        signal (float)
        flag (bool)
        wkernel (int):  Running average width
        return_flags (bool):  If true, also return flags which are
            a subset of the input flags.
        downsample (bool):  If True, return a downsampled version of the
            filtered timestream

    """
    if len(signal) != len(flag):
        raise Exception('Signal and flag lengths do not match.')

    bad = flag != 0
    masked_signal = signal.copy()
    masked_signal[bad] = 0

    good = np.ones(len(signal), dtype=np.float64)
    good[bad] = 0

    kernel = np.ones(wkernel, dtype=np.float64)

    filtered_signal = fftconvolve(masked_signal, kernel, mode='same')
    filtered_hits = fftconvolve(good, kernel, mode='same')

    hit = filtered_hits > 0.1
    nothit = np.logical_not(hit)

    filtered_signal[hit] /= filtered_hits[hit]
    filtered_signal[nothit] = 0

    if return_flags or downsample:
        filtered_flags = np.zeros_like(flag)
        filtered_flags[nothit] = True

    if downsample:
        good = filtered_flags == 0
        if return_flags:
            filtered_flags[good][::wkernel]
        filtered_signal[good][::wkernel]

    if return_flags:
        return filtered_signal, filtered_flags
    else:
        return filtered_signal
