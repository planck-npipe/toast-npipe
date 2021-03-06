# Copyright (c) 2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import namedtuple

Ring = namedtuple(
    'Ring', ['pixels', 'hits', 'signal', 'quat', 'weights', 'phase', 'nbytes',
             'phaseorder', 'mask'])

globals()[Ring.__name__] = Ring
