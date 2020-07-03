# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

from toast_planck.preproc_modules import RingMasker

import toast

import toast.timing as timing


class OpBadIntervals(toast.Operator):
    """
    Operator for loading timestreams from disk

    Args:
    """

    def __init__(self, path, margin=0,
                 effdir=None, file_pattern=None,
                 commonflagmask=1, detflagmask=1):
        self._path = path
        self._ringmasker = RingMasker(path)
        self._margin = margin
        self._effdir = effdir
        self._file_pattern = file_pattern
        self._commonflagmask = commonflagmask
        self._detflagmask = detflagmask
        super().__init__()

    # @profile
    def exec(self, data):
        for obs in data.obs:
            tod = obs['tod']
            timestamps = tod.local_timestamps()
            commonflags = tod.local_common_flags()

            interval_flags = self._ringmasker.get_mask(timestamps, 'ALL')
            commonflags[interval_flags] |= self._commonflagmask

            for det in tod.local_dets:
                flags = tod.local_flags(det)
                interval_flags = self._ringmasker.get_mask(timestamps, det)
                flags[interval_flags] |= self._detflagmask

            tod.purge_eff_cache()
        return
