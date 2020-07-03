# Copyright (c) 2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from toast_planck import OpSignalSim

from toast.mpi import MPI
from toast.tests.mpi import MPITestCase

import healpy as hp
import numpy as np
import numpy.testing as nt

import toast.qarray as qarray


class PlanckSignalTest(MPITestCase):

    def setUp(self):
        # Create and a_lm expansion
        ell, _, cls = hp.sphtfunc.load_sample_spectra()
        self.lmax = len(ell) - 1
        alms = hp.synalm(cls, lmax=self.lmax, new=True)
        self.fname_alm = 'test/alm.fits'
        hp.write_alm(self.fname_alm, alms, overwrite=True)
        self.fwhm = 0
        self.nside = 16
        self.pol = True
        self.comm = MPI.COMM_SELF
        self.refmap = hp.alm2map(alms[0], self.nside, verbose=False)
        self.freq = 0

    def test_sample(self):
        signalsim = OpSignalSim(
            self.fname_alm, self.fwhm, self.freq, pol=self.pol,
            comm=self.comm, nside=self.nside)

        npix = 12 * self.nside ** 2
        pix = np.arange(npix)
        theta, phi = hp.pix2ang(self.nside, pix)
        quat = qarray.from_angles(theta, phi, theta * 0, IAU=False)
        sigmap = signalsim._sample_maps(quat)
        nt.assert_almost_equal(sigmap, self.refmap, decimal=4)
        return
