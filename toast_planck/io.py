# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

import toast

import toast.timing as timing


class OpInputPlanck(toast.Operator):
    """
    Operator for loading timestreams from disk

    Args:
    """

    def __init__(self, signal_name='signal', flags_name='flags',
                 timestamps_name='timestamps',
                 commonflags_name='common_flags', margin=0,
                 effdir=None, file_pattern=None):

        self._signal_name = signal_name
        self._flags_name = flags_name
        self._timestamps_name = timestamps_name
        self._commonflags_name = commonflags_name
        self._margin = margin
        self._effdir = effdir
        self._file_pattern = file_pattern
        super().__init__()

    # @profile
    def exec(self, data):
        for obs in data.obs:
            tod = obs['tod']

            if self._timestamps_name is not None \
               or self._commonflags_name is not None:
                timestamps, commonflags = tod.read_times(
                    margin=self._margin, and_flags=True)
                if self._timestamps_name is not None:
                    tod.cache.put(self._timestamps_name, timestamps,
                                  replace=True)
                if self._commonflags_name is not None:
                    tod.cache.put(self._commonflags_name, commonflags,
                                  replace=True)

            and_flags = self._flags_name is not None

            if self._signal_name is not None or self._flags_name is not None:
                for det in tod.local_dets:

                    result = tod.read(
                        detector=det, margin=self._margin, and_flags=and_flags,
                        effdir=self._effdir, file_pattern=self._file_pattern)

                    if and_flags:
                        signal, flags = result
                    else:
                        signal = result

                    if self._signal_name is not None:
                        cachename = '{}_{}'.format(self._signal_name, det)
                        tod.cache.put(cachename, signal, replace=True)
                    if self._flags_name is not None:
                        cachename = '{}_{}'.format(self._flags_name, det)
                        tod.cache.put(cachename, flags, replace=True)

        tod.purge_eff_cache()
        return


class OpOutputPlanck(toast.Operator):
    """
    Operator for writing timestreams to disk

    Args:
    """

    def __init__(self, signal_name='signal', flags_name='flags',
                 commonflags_name='common_flags',
                 effdir_out=None, effdir_out_diode0=None,
                 effdir_out_diode1=None, margin=0):
        self._signal_name = signal_name
        self._flags_name = flags_name
        self._commonflags_name = commonflags_name
        self._effdir_out = effdir_out
        self._effdir_out_diode0 = effdir_out_diode0
        self._effdir_out_diode1 = effdir_out_diode1
        self._margin = margin
        super().__init__()

    # @profile
    def exec(self, data):

        for obs in data.obs:
            tod = obs['tod']
            nsamp = tod.local_samples[1]

            if self._commonflags_name is not None:
                commonflags = tod.cache.reference(self._commonflags_name)
                for effdir_out in [self._effdir_out, self._effdir_out_diode0,
                                   self._effdir_out_diode1]:
                    if effdir_out is not None:
                        tod.write_common_flags(flags=commonflags,
                                               effdir_out=effdir_out)

            for det in tod.local_dets:
                if self._signal_name is not None:
                    cachename = "{}_{}".format(self._signal_name, det)
                    signal = tod.cache.reference(cachename)
                else:
                    signal = None

                if self._flags_name is not None:
                    cachename = "{}_{}".format(self._flags_name, det)
                    flags = tod.cache.reference(cachename)
                else:
                    flags = None

                if det[-1] in '01':
                    # single diode
                    if det[-1] == '0':
                        effdir_out = self._effdir_out_diode0
                    else:
                        effdir_out = self._effdir_out_diode1
                else:
                    effdir_out = self._effdir_out

                if self._margin == 0:
                    ind = slice(0, nsamp)
                else:
                    ind = slice(self._margin, nsamp - self._margin)

                if signal is not None and flags is not None:
                    tod.write_tod_and_flags(detector=det, data=signal[ind],
                                            flags=flags[ind],
                                            effdir_out=effdir_out)
                else:
                    if signal is not None:
                        tod.write(detector=det, data=signal[ind],
                                  effdir_out=effdir_out)
                    if flags is not None:
                        tod.write_det_flags(detector=det, flags=flags[ind],
                                            effdir_out=effdir_out)
        return
