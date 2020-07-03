# Copyright (c) 2016-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
from toast_planck.reproc_modules import destripe_tools

import numpy as np
import toast.timing as timing

try:
    import h5py
except ImportError as e:
    h5py = None


class OpRingMaker():

    def __init__(self, nside, nside_in, signal=None, detmask=1, commonmask=3,
                 fileroot=None, out=None):

        if nside > nside_in:
            raise Exception('OpRingMaker: input nside must not be smaller '
                            'than the binning resolution.')
        if fileroot and nside > 4096:
            raise ValueError('HDF5 file dtype only supports NSIDE up to 4096')
        self._nside = nside
        self._nside_in = nside_in
        self._signal = signal
        self._npix = 12 * nside ** 2
        self._dgrade = (nside_in // nside) ** 2
        self._detmask = detmask
        self._commonmask = commonmask
        if h5py is None and fileroot is not None:
            raise RuntimeError('Cannot write ringsets without h5py module.')
        self._fileroot = fileroot
        self._out = out

    def exec(self, data):
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world

        rank = cworld.Get_rank()

        ringsets = {}

        for obs in data.obs:
            tod = obs['tod']

            timestamps = tod.local_times()
            commonflags = tod.local_common_flags()

            if commonflags is not None:
                commonflags = (commonflags & self._commonmask) != 0

            intervals = tod.local_intervals(obs['intervals'])
            local_starts = [ival.first for ival in intervals]
            local_stops = [ival.last + 1 for ival in intervals]

            ring_offset = tod.globalfirst_ring
            for interval in intervals:
                if interval.last < tod.local_samples[0]:
                    ring_offset += 1

            # Map of pixel numbers
            pixmap = np.arange(self._npix, dtype=np.int32)

            for det in tod.local_dets:

                if det not in ringsets:
                    ringsets[det] = {}

                pixels_in = tod.local_pixels(det)
                signal = tod.local_signal(det, name=self._signal)
                flags = (tod.local_flags(det) & self._detmask) != 0
                flags[commonflags] = True

                ring_number = ring_offset - 1

                for ring_start, ring_stop in zip(local_starts, local_stops):
                    ring_number += 1
                    ind = slice(ring_start, ring_stop)

                    flg = flags[ind]
                    good = flg == 0

                    if np.sum(good) == 0:
                        continue

                    tme = timestamps[ind][good].copy()
                    pix_in = pixels_in[ind][good].copy()
                    pix = pix_in // self._dgrade
                    sig = signal[ind][good].copy()

                    # Check to ensure the same pixel isn't repeated
                    # too many times.

                    first = 0
                    while first < len(pix):
                        last = first + 1
                        while last < len(pix) and pix[first] == pix[last]:
                            last += 1
                        if last - first > 100:
                            raise RuntimeError(
                                'There are {} consecutive hits to {} on ring '
                                '{} between {} and {}'.format(
                                    last - first, pix[first], ring_number,
                                    tme[first], tme[last - 1]))
                        first = last

                    hitmap = np.zeros(self._npix, dtype=np.int32)
                    sigmap = np.zeros(self._npix, dtype=np.float64)

                    destripe_tools.fast_hit_binning(pix.astype(np.int32),
                                                    hitmap)
                    destripe_tools.fast_binning(sig.astype(np.float64),
                                                pix.astype(np.int32), sigmap)
                    hit = hitmap != 0
                    sigmap[hit] /= hitmap[hit]

                    ringsets[det][ring_number] = (
                        tme[0], tme[-1], pixmap[hit].copy(), sigmap[hit].copy(),
                        hitmap[hit].copy())

        if self._fileroot is not None:
            for det in sorted(ringsets.keys()):
                fn = '{}_{}_{}.h5'.format(self._fileroot, det, self._nside)
                fn = os.path.join(self._out, fn)

                dtypes = np.dtype([
                    # pids are 40K, uint16 65535
                    ('pid', np.uint16),
                    # uint32 goes up to 4294967295, this is good until 4096
                    ('pix', np.uint32),
                    # uint16 65535, hits in 40 min even at 200Hz,
                    # just 8 pixels are enough to make this ok
                    ('hits', np.uint16),
                    ('c', np.float64)])
                f = None
                ind = None
                for i, (key, col) in enumerate(zip(
                        ['pid', 'pix', 'hits', 'c'], [-1, 2, 4, 3])):
                    sendlist = []
                    dtype = dtypes[i]

                    for ring_number in sorted(ringsets[det].keys()):
                        if key == 'pid':
                            nsamp_ring = len(ringsets[det][ring_number][2])
                            sendlist.append(np.ones(nsamp_ring, dtype=dtype)
                                            * ring_number)
                        else:
                            sendlist.append(
                                ringsets[det][ring_number][col].astype(dtype))

                    if len(sendlist) == 0:
                        my_vec = np.array([], dtype=dtype)
                    else:
                        my_vec = np.hstack(sendlist)

                    vec = None
                    vec = cworld.gather(my_vec, root=0)

                    if rank == 0:
                        vec = np.hstack(vec)
                        if ind is None:
                            # Order the ringset by PID, use mergesort
                            # because quicksort is not stable
                            ind = np.argsort(vec, kind='mergesort')
                        vec = vec[ind]
                        if f is None:
                            f = h5py.File(fn, 'w')
                            f.create_dataset('data', shape=(len(vec),),
                                             dtype=dtypes)
                        f['data'][key] = vec.copy()

                if rank == 0:
                    f.close()
                    print('Ringset written into {}.'.format(fn), flush=True)
        return
