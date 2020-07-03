# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from toast_planck.preproc_modules import lfi_adc

from toast.mpi import MPI
from toast.tests.mpi import MPITestCase

import numpy as np


class LFIADCTest(MPITestCase):

    def setUp(self):
        # Note: self.comm is set by the test infrastructure
        self.radiometer = 'LFI28M'
        self.corrector = lfi_adc.LFINLCorrector(self.radiometer,
                                                comm=MPI.COMM_WORLD)

    """
    def test_props(self):
        print('Running LFIADC properties test. diodes = {}\n freq = {}\n'
              ' diodefiles = {}'.format(
                self.corrector.diodes, self.corrector.freq,
                self.corrector.diodefiles))
    """

    def test_correct(self):
        # print('Running LFIADC correct test')
        n = 1000
        timestamps = np.linspace(1628849580000000, 1628849580001000, n)
        signal = np.zeros([n, 4])
        signal[:, 0:2] = np.random.randn(2 * n).reshape([n, 2]) * .001 + 1.12
        signal[:, 2:4] = np.random.randn(2 * n).reshape([n, 2]) * .001 + 1.40

        # print('Input values: ', signal[:10])

        signal = self.corrector.correct(signal, timestamps)

        # print('Output values: ', signal[:10])
