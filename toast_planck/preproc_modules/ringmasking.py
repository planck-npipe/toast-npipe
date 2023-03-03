# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


class RingMasker():

    def __init__(self, path, bolo=None):
        """
        Instantiate the bad ring object
        path -- path to the bad ring file
        bolo(None) -- bolometer to consider, default is to store all
        """
        self.path = path
        self.bolo = bolo

        self.bad_rings = {}
        try:
            f = open(self.path, 'r')
            for iline, line in enumerate(f):
                if line.strip().startswith('#'):
                    continue
                try:
                    bolo, tstart, tstop = line.split()
                except Exception as e:
                    print('Line {} failed to parse: {} : {}'.format(
                        iline, line, e))
                    continue
                if bolo != 'ALL':
                    if self.bolo is not None \
                       and self.bolo.upper() != bolo.upper():
                        continue
                if bolo not in self.bad_rings:
                    self.bad_rings[bolo] = []
                self.bad_rings[bolo].append(
                    np.array([tstart, tstop], dtype=np.float64))
            f.close()
        except Exception as e:
            raise Exception('Failed to read bad rings from {} : {}'.format(
                path, e))
        return

    def get_mask(self, time, det):
        """
        Compare the supplied time stamps to the list of bad intervals
        and produce a vector of flags.

        """
        out = np.zeros(len(time), dtype=bool)

        if det not in self.bad_rings and 'ALL' not in self.bad_rings:
            return out

        tfirst = time[0]
        tlast = time[-1]

        for key in [det, 'ALL']:
            if key in self.bad_rings:
                for interval in self.bad_rings[key]:
                    tstart, tstop = interval
                    if tfirst <= tstop and tlast >= tstart:
                        # Found overlapping period, mask the relevant samples
                        overlap = np.logical_and(time >= tstart, time <= tstop)
                        out[overlap] = True
        return out
