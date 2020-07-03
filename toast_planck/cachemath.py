# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

import toast

import numpy as np
import toast.timing as timing


class OpCacheMath(toast.Operator):
    """
    Operator for writing timestreams to disk

    Args:
        in1 (str):  Input1. Cache key or scalar
        in2 (str):  Input2. Cache key or scalar
        out (str):  Output cache key.
        add (bool):  Add in1 and in2
        subtract (bool):  Add in1 and in2
        multiply (bool):  Multiply in1 and in2
        divide (bool):  Divide in1 and in2
    """

    def __init__(self, in1=None, in2=None, out='tod',
                 add=False, subtract=False, multiply=False, divide=False):
        if in1 is None:
            raise RuntimeError('OpCacheMath requires in1 to be set')
        self._in1 = in1
        if in2 is None:
            raise RuntimeError('OpCacheMath requires in2 to be set')
        self._in2 = in2
        if out is None:
            raise RuntimeError('OpCacheMath requires out to be set')
        self._out = out
        if add + subtract + multiply + divide != 1:
            raise RuntimeError('OpCacheMath requires exactly one operation to '
                               'be chosen.')
        self._add = add
        self._subtract = subtract
        self._multiply = multiply
        self._divide = divide
        super().__init__()

    # @profile
    def exec(self, data):
        for obs in data.obs:
            tod = obs['tod']
            for det in tod.local_dets:
                try:
                    cachename = "{}_{}".format(self._in1, det)
                    in1 = tod.cache.reference(cachename)
                except Exception:
                    in1 = np.float(self._in1)
                try:
                    cachename = "{}_{}".format(self._in2, det)
                    in2 = tod.cache.reference(cachename)
                except Exception:
                    in2 = np.float(self._in2)

                if self._add:
                    out = in1 + in2
                elif self._subtract:
                    out = in1 - in2
                elif self._multiply:
                    out = in1 * in2
                elif self._divide:
                    out = in1 / in2

                del in1
                del in2

                cachename = "{}_{}".format(self._out, det)
                tod.cache.put(cachename, out, replace=True)
        return
