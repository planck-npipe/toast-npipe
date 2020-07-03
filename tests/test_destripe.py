# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from toast_planck.reproc_modules.destriping import Destriper

from toast.mpi import MPI
from toast.tests.mpi import MPITestCase

import numpy as np


class DestriperTest(MPITestCase):

    def setUp(self):
        self.disable = True
        if self.disable:
            return
        self.npix = 100
        self.destriper = Destriper(self.npix, MPI.COMM_WORLD, cglimit=1e-10,
                                   itermax=100)
        self.verbose = False

    def test_destripe(self):
        if self.disable:
            return

        ninterval = 10
        leninterval = 1000
        pixels = []
        toi = []
        flags = []
        for i in range(ninterval):
            pixels.append(np.arange(leninterval, dtype=np.int32) % self.npix)
            toi.append(np.ones(leninterval, dtype=np.float64) * i
                       + np.random.randn(leninterval))
            flags.append(np.arange(leninterval, dtype=np.int) % 10 < 5)

        print('RMS before destriping = {}'.format(np.std(np.hstack(toi))))

        self.destriper.destripe(toi, flags, pixels, verbose=self.verbose,
                                in_place=False)

        print('RMS after destriping 1/2 = {}'.format(np.std(np.hstack(toi))))

        self.destriper.destripe(toi, flags, pixels, verbose=False,
                                in_place=True)

        print('RMS after destriping 2/2 = {}'.format(np.std(np.hstack(toi))))

        return

    def test_destripe_with_templates(self):
        if self.disable:
            return

        ninterval = 10
        leninterval = 1000
        pixels = []
        toi = []
        flags = []
        templates = []
        for i in range(ninterval):
            pixels.append(np.arange(leninterval, dtype=np.int32) % self.npix)
            toi.append(np.ones(leninterval, dtype=np.float64) * i
                       + np.random.randn(leninterval))
            atemplate = np.random.randn(leninterval)
            btemplate = np.random.randn(leninterval)
            toi[-1] += atemplate + btemplate
            templates.append([atemplate, btemplate])
            flags.append(np.arange(leninterval, dtype=np.int) % 10 < 5)

        print('RMS before destriping = {}'.format(np.std(np.hstack(toi))))

        self.destriper.destripe(toi, flags, pixels, verbose=self.verbose,
                                in_place=False, templates=templates)

        print('RMS after destriping 1/2 = {}'.format(np.std(np.hstack(toi))))

        self.destriper.destripe(toi, flags, pixels, verbose=self.verbose,
                                in_place=True, templates=templates)

        print('RMS after destriping 2/2 = {}'.format(np.std(np.hstack(toi))))

        return
