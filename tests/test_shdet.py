# Copyright (c) 2015-2016 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from toast_planck import shdet

from toast.mpi import MPI
from toast.tests.mpi import MPITestCase

import numpy as np


class SHDetTest(MPITestCase):

    def setUp(self):
        # Note: self.comm is set by the test infrastructure
        self.shdet = shdet.SHDet()
        self.n = self.shdet.get_n()
        self.nadc = self.shdet.get_nadc()
        self.nparam = self.shdet.get_nparam()
        self.sentinel = self.shdet.get_sentinel()
        self.nparam2 = self.shdet.get_nparam2()

    """
    def test_props(self):

        # self.assertEqual(self.eff.detectors, self.dets)

        print('Running SHDet properties test. n = {}, nadc {}, nparam = {}, '
              'sentinel = {}, mx_blmtr_mdl = {}'.format(
                  self.n, self.nadc, self.nparam, self.sentinel, self.nparam2))
    """

    def test_simulate(self):

        n_simulate = 10000

        # print('Running SHDet simulate test ( n={} )'.format(n_simulate))

        # only test 1e5 points just so this executes relatively quickly
        sig = np.random.randn(n_simulate) * 0

        # change some of the SHDet parameters

        # Setting WN level at the raw sample level in V
        # (NS_DSN/sqrt(40)*V_ADC_RNG/2^N_BIT
        # (PS: if it is zero, no noise is generated)
        fct_dsn2rwV = 1 / np.sqrt(40.) * 10.2 / 65536

        # DSN wn level needs to be passed
        self.shdet.set_parameter('noise_dsn', 100 * fct_dsn2rwV)

        # Some seed (not an issue as float, gets promoted to
        # unsigned long inside shdet)
        self.shdet.set_parameter('seed', 1)

        # ADC off/on (0/1)
        self.shdet.set_parameter('adc_on', 1)

        # print('Input values: ', sig[:10])

        sig = self.shdet.simulate(sig)

        # print('Output values: ', sig[:10])
        # print('Last value: ', sig[-1])
