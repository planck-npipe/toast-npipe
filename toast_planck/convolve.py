# Copyright (c) 2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

import numpy as np

import toast
import toast.timing as timing

from toast_planck.preproc_modules import TauDeconvolver
from .utilities import det2fsample


# import warnings
# warnings.filterwarnings('error')
class OpConvolvePlanck(toast.Operator):
    """
    Operator for applying extra filtering to the data

    Args:
        filterfile(str):  Path to the filter file.  Special tag
            DETECTOR will be replaced with the detector name
        filterlen(int): length of the filter to apply
        fsample(float):  Sampling frequency
    """

    def __init__(self, filterfile, filterlen=2 ** 20, normalize=True,
                 extend_flags=True):
        self.filterfile = filterfile
        self.filterlen = filterlen
        self.normalize = normalize
        self.extend_flags = extend_flags
        super().__init__()

    # @profile
    def exec(self, data):

        comm = data.comm.comm_group
        convolvers = {}

        for obs in data.obs:
            tod = obs['tod']
            if 'intervals' not in obs:
                raise RuntimeError(
                    'observation must specify intervals')
            intervals = tod.local_intervals(obs['intervals'])
            local_starts = [ival.first for ival in intervals]
            local_stops = [ival.last + 1 for ival in intervals]
            for det in tod.local_dets:
                fsample = det2fsample(det)
                if det not in convolvers:
                    filterfile = self.filterfile.replace('DETECTOR', det)
                    if filterfile == self.filterfile:
                        key = 'generic'
                    else:
                        key = det
                    if key not in convolvers:
                        convolvers[key] = TauDeconvolver(
                            det, None, filterlen=self.filterlen,
                            filterfile=filterfile, fsample=fsample, comm=comm,
                            normalize_filter=self.normalize)
                    if key != det:
                        convolvers[det] = convolvers[key]
                convolver = convolvers[det]
                # The convolver will extend flagging to all samples that have
                # at least 10 % contribution from compromised samples
                signal = tod.local_signal(det)
                if self.extend_flags:
                    flags = tod.local_flags(det)
                else:
                    # Just a dummy vector
                    flags = np.zeros(signal.size, dtype=np.bool)
                for (istart, istop) in zip(local_starts, local_stops):
                    ind = slice(istart, istop)
                    signal[ind], flags[ind] = convolver.deconvolve(signal[ind],
                                                                   flags[ind])

        del convolvers
        return
