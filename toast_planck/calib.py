# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

from toast_planck.utilities import read_gains

import toast

import toast.timing as timing
import toast.tod as tt


class OpCalibPlanck(toast.Operator):
    """
    Operator for calibrating cached timestreams

    Args:
        signal_in (str):  Cache key for the signal to calibrate
        signal_out (str):  Cache key for storing calibrated signal
        file_gain (str):  Planck gain file to read gains from.
        decalibrate (bool):  If True, decalibrate instead of calibrating
    """

    def __init__(self, signal_in=None, signal_out='tod',
                 file_gain=None, decalibrate=False):
        self._signal_in = signal_in
        self._signal_out = signal_out
        self._file_gain = file_gain
        self._decalibrate = decalibrate
        super().__init__()

    # @profile
    def exec(self, data):
        detgains = {}

        for obs in data.obs:
            tod = obs['tod']
            times = tod.local_timestamps()

            for det in tod.local_dets:
                if det not in detgains:
                    gains = read_gains(self._file_gain, det)
                    if self._decalibrate:
                        good = gains[1] != 0
                        gains[1][good] = 1 / gains[1][good]
                    detgains[det] = gains
                else:
                    gains = detgains[det]

                signal = tod.local_signal(det, name=self._signal_in)
                signal = tt.calibrate(times, signal, *gains)

                cachename_out = '{}_{}'.format(self._signal_out, det)
                tod.cache.put(cachename_out, signal, replace=True)
        return
