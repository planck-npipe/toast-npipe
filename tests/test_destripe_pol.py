# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from toast_planck.reproc_modules.destriping import FancyDestriperPol

from toast.mpi import MPI
from toast.tests.mpi import MPITestCase

import healpy as hp
import numpy as np


class DestriperPolTest(MPITestCase):

    def setUp(self):
        self.disable = True
        if self.disable:
            return
        self.nside = 16
        self.npix = 12 * self.nside ** 2
        self.nnz = 3
        self.destriper = FancyDestriperPol(
            self.npix, self.nnz, MPI.COMM_WORLD,
            do_offset=True, do_gain=True, do_pol_eff=True, do_pol_angle=True,
            ndegrade=4, fsample=1.0, lowpassfreq=.1, dir_out='test')
        self.verbose = False

    def test_destripe(self):
        if self.disable:
            return

        ntask = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank

        nsamp_tot = 100000
        nsamp_proc = nsamp_tot // ntask + 1
        my_first = rank * nsamp_proc
        my_last = my_first + nsamp_proc
        if my_last > nsamp_tot:
            my_last = nsamp_tot
        my_nsamp = my_last - my_first

        np.random.seed(12345)

        sky = np.array(hp.synfast(
            np.ones([4, 4 * self.nside]), self.nside, fwhm=10 * np.pi / 180,
            pol=True, new=True))
        sky[1:] *= 0.1  # Suppress polarization wrt temperature
        # hp.write_map('input_sky.fits', sky)
        # sky = np.zeros([self.npix, self.nnz], dtype=np.float64)
        # sky[:, 0] = np.arange(self.npix)
        # sky[:, 1] = np.cos(np.arange(self.npix))
        # sky[:, 2] = np.sin(np.arange(self.npix))
        # sky = np.array(hp.reorder(sky.T, r2n=True )).T
        sky = np.array(hp.reorder(sky, r2n=True)).T
        # import pylab
        # hp.mollview(sky[:,0], nest=True)
        # pylab.savefig('map.png')

        template_sky = np.array(hp.synfast(
            np.ones([4, 4 * self.nside]), self.nside, fwhm=20 * np.pi / 180,
            pol=True, new=True))
        template_sky[1:] *= 0.1
        # hp.write_map('template_sky.fits', template_sky)
        # template_sky = np.arange(self.npix, dtype=np.float) % 10
        # template_sky = np.sin(np.arange(self.npix, dtype=np.float)
        #                      / self.npix*np.pi)
        # template_sky -= np.mean(template_sky)
        # template_sky = np.array(hp.reorder(template_sky.T, r2n=True)).T
        template_sky = np.array(hp.reorder(template_sky, r2n=True)).T
        # if rank == 0:
        #    hp.mollview(template_sky[:,0], nest=True)
        #    pylab.savefig('templatemap.png')

        # Four detectors

        sigma1 = 1.
        sigma2 = 1.
        sigma3 = 1.
        sigma4 = 1.

        clean_toi = []
        dirty_toi = []

        t = np.arange(my_first, my_last)
        pixels = t % self.npix
        pixels[np.logical_and(t < nsamp_tot // 2, pixels < 8)] = 8
        pixels1 = pixels
        psi1 = t / self.npix * np.pi / 10
        weights1 = np.vstack(
            [np.ones(my_nsamp), np.cos(2 * psi1), np.sin(2 * psi1)]).T
        psi1scan = psi1 + 1 * np.pi / 180
        weights1scan = np.vstack(
            [np.ones(my_nsamp),
             0.9 * np.cos(2 * psi1scan),
             0.9 * np.sin(2 * psi1scan)]).T
        signal1 = np.zeros(my_nsamp)
        template1 = np.zeros(my_nsamp)
        for i, p in enumerate(pixels1):
            signal1[i] = np.sum(sky[p] * weights1scan[i])
            # signal1[i] = np.sum(sky[p] * weights1[i])
            template1[i] = np.sum(template_sky[p] * weights1[i])
        templates1 = [template1]
        flag1 = np.zeros(my_nsamp, dtype=bool)
        flag1[np.logical_and(t >= nsamp_tot // 4,
                             t <= 3 * nsamp_tot // 4)] = True
        signal1 += np.random.randn(nsamp_tot)[my_first:my_last] * sigma1

        clean_toi.append(signal1.copy())

        signal1 *= 1.01
        signal1 += templates1[0] * .1
        signal1 += 1

        dirty_toi.append(signal1.copy())

        pixels2 = pixels
        psi2 = t / self.npix * np.pi / 10 + np.pi / 2
        weights2 = np.vstack(
            [np.ones(my_nsamp), np.cos(2 * psi2), np.sin(2 * psi2)]).T
        signal2 = np.zeros(my_nsamp)
        template2 = np.zeros(my_nsamp)
        for i, p in enumerate(pixels2):
            signal2[i] = np.sum(sky[p] * weights2[i])
            template2[i] = np.sum(template_sky[p] * weights2[i])
        templates2 = [template2]
        flag2 = np.zeros(my_nsamp, dtype=bool)
        signal2 += np.random.randn(nsamp_tot)[my_first:my_last] * sigma2

        clean_toi.append(signal2.copy())

        signal2 *= .99
        signal2 -= templates2[0] * .1
        signal2 -= 1

        dirty_toi.append(signal2)

        pixels3 = pixels
        psi3 = t / self.npix * np.pi / 10 + np.pi / 4
        weights3 = np.vstack(
            [np.ones(my_nsamp), np.cos(2 * psi3), np.sin(2 * psi3)]).T
        signal3 = np.zeros(my_nsamp)
        template3 = np.zeros(my_nsamp)
        for i, p in enumerate(pixels3):
            signal3[i] = np.sum(sky[p] * weights3[i])
            template3[i] = np.sum(template_sky[p] * weights3[i])
        templates3 = [template3]
        flag3 = np.zeros(my_nsamp, dtype=bool)
        signal3 += np.random.randn(nsamp_tot)[my_first:my_last] * sigma3

        clean_toi.append(signal3.copy())

        # signal3 -= templates3[0]*2
        # signal3 -= 2

        dirty_toi.append(signal3)

        pixels4 = pixels
        psi4 = t / self.npix * np.pi / 10 + np.pi / 4 + np.pi / 2
        weights4 = np.vstack(
            [np.ones(my_nsamp), np.cos(2 * psi4), np.sin(2 * psi4)]).T
        signal4 = np.zeros(my_nsamp)
        template4 = np.zeros(my_nsamp)
        for i, p in enumerate(pixels4):
            signal4[i] = np.sum(sky[p] * weights4[i])
            template4[i] = np.sum(template_sky[p] * weights4[i])
        templates4 = [template4]
        flag4 = np.zeros(my_nsamp, dtype=bool)
        signal4 += np.random.randn(nsamp_tot)[my_first:my_last] * sigma4

        clean_toi.append(signal4.copy())
        dirty_toi.append(signal4)

        clean_toi = np.array(clean_toi)
        dirty_toi = np.array(dirty_toi)
        flags = np.vstack([flag1, flag2, flag3, flag4])
        pixels = np.vstack([pixels1, pixels2, pixels3, pixels4])
        weights = [weights1, weights2, weights3, weights4]
        templates = [templates1, templates2, templates3, templates4]

        resid = MPI.COMM_WORLD.allgather(
            (dirty_toi - clean_toi)[np.logical_not(flags)].ravel())
        if rank == 0:
            print('RMS before destriping = {}'.format(np.std(np.hstack(resid))))

        dirty_toi[1][10] = 1e4  # Add outlier

        self.destriper.flag_outliers(
            dirty_toi, flags, pixels, weights,
            verbose=self.verbose, save_maps=False)

        destriped_toi, _, _ = self.destriper.destripe(
            dirty_toi, flags, pixels, weights, templates,
            verbose=self.verbose, in_place=True, return_baselines=True,
            siter='_poltest')

        resid = MPI.COMM_WORLD.allgather(
            (destriped_toi - clean_toi)[np.logical_not(flags)].ravel())
        if rank == 0:
            print('RMS after destriping 1/2 = {}'.format(
                np.std(np.hstack(resid))))

        destriped_toi, _, _ = self.destriper.destripe(
            dirty_toi, flags, pixels, weights, templates,
            verbose=self.verbose, in_place=True, return_baselines=True)

        resid = MPI.COMM_WORLD.allgather(
            (destriped_toi - clean_toi)[np.logical_not(flags)].ravel())
        if rank == 0:
            print('RMS after destriping 2/2 = {}'.format(
                np.std(np.hstack(resid))))

        destriped_toi, _, _ = self.destriper.destripe(
            dirty_toi, flags, pixels, weights, templates,
            verbose=self.verbose, in_place=True, return_baselines=True)

        resid = MPI.COMM_WORLD.allgather(
            (destriped_toi - clean_toi)[np.logical_not(flags)].ravel())
        if rank == 0:
            print('RMS after destriping 3/2 = {}'.format(
                np.std(np.hstack(resid))))

        return

    def test_destripe_single(self):
        if self.disable:
            return

        ntask = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank

        nsamp_tot = 100000
        nsamp_proc = nsamp_tot // ntask + 1
        my_first = rank * nsamp_proc
        my_last = my_first + nsamp_proc
        if my_last > nsamp_tot:
            my_last = nsamp_tot
        my_nsamp = my_last - my_first

        sky = np.zeros([self.npix, self.nnz], dtype=np.float64)
        sky[:, 0] = np.arange(self.npix)
        sky[:, 1] = np.cos(np.arange(self.npix))
        sky[:, 2] = np.sin(np.arange(self.npix))

        # One detector

        sigma1 = 10.

        np.random.seed(12345)

        clean_toi = []
        dirty_toi = []

        t = np.arange(my_first, my_last)
        pixels = t % self.npix
        # pixels[np.logical_and(t<nsamp_tot//2, pixels<8)] = 8
        pixels1 = pixels
        psi1 = t / self.npix * np.pi / 10
        weights1 = np.vstack(
            [np.ones(my_nsamp), np.cos(2 * psi1), np.sin(2 * psi1)]).T
        signal1 = np.zeros(my_nsamp)
        template1 = t + np.sin(t / 1000) * 100
        for i, p in enumerate(pixels1):
            signal1[i] = np.sum(sky[p] * weights1[i])
        templates1 = [template1]
        flag1 = np.zeros(my_nsamp, dtype=bool)
        flag1[np.logical_and(t >= nsamp_tot // 4,
                             t <= 3 * nsamp_tot // 4)] = True
        signal1 += np.random.randn(nsamp_tot)[my_first:my_last] * sigma1

        clean_toi.append(signal1.copy())

        signal1 *= 1.01
        signal1 += 1
        signal1 += templates1[0] * 1e4

        dirty_toi.append(signal1.copy())

        clean_toi = np.array(clean_toi)
        dirty_toi = np.array(dirty_toi)
        flags = np.vstack([flag1])
        pixels = np.vstack([pixels1])
        weights = [weights1]
        templates = [templates1]

        resid = MPI.COMM_WORLD.allgather(
            (dirty_toi - clean_toi)[np.logical_not(flags)].ravel())
        if rank == 0:
            print('RMS before destriping = {}'.format(np.std(np.hstack(resid))))

        destriper = FancyDestriperPol(self.npix, self.nnz, MPI.COMM_WORLD,
                                      do_offset=True, do_gain=True, ndegrade=1,
                                      dir_out='test')
        _, amp, cov = destriper.destripe(
            dirty_toi, flags, pixels, weights, templates,
            verbose=self.verbose, in_place=False, return_baselines=True,
            siter='_singlepoltest')

        if rank == 0:
            print('template amplitude = {} +- {}'.format(
                amp[0, 0], np.sqrt(cov[0, 0]), flush=True))

        if np.abs((amp[0, 0] - 1e4) / np.sqrt(cov[0, 0])) > 3 and \
           np.abs((amp[0, 0] - 1e4) / 1e4) > 1e-4:
            raise Exception(
                'Failed to fit the template: {} +- {} != {}'.format(
                    amp[0, 0], np.sqrt(cov[0, 0]), 1e4))

        return

    def test_destripe_single_t_only(self):
        if self.disable:
            return

        ntask = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank

        nsamp_tot = 100000
        nsamp_proc = nsamp_tot // ntask + 1
        my_first = rank * nsamp_proc
        my_last = my_first + nsamp_proc
        if my_last > nsamp_tot:
            my_last = nsamp_tot
        my_nsamp = my_last - my_first

        sky = np.arange(self.npix)

        # One detector

        sigma1 = 10.

        np.random.seed(12345)

        clean_toi = []
        dirty_toi = []

        t = np.arange(my_first, my_last)
        pixels = t % self.npix
        # pixels[np.logical_and(t<nsamp_tot//2, pixels<8)] = 8
        pixels1 = pixels
        # psi1 = t / self.npix * np.pi / 10
        signal1 = np.zeros(my_nsamp)
        template1 = t + np.sin(t / 1000) * 100
        for i, p in enumerate(pixels1):
            signal1[i] = sky[p]
        templates1 = [template1]
        flag1 = np.zeros(my_nsamp, dtype=bool)
        flag1[np.logical_and(t >= nsamp_tot // 4,
                             t <= 3 * nsamp_tot // 4)] = True
        signal1 += np.random.randn(nsamp_tot)[my_first:my_last] * sigma1

        clean_toi.append(signal1.copy())

        signal1 *= 1.01
        signal1 += 1
        signal1 += templates1[0] * 1e4

        dirty_toi.append(signal1.copy())

        clean_toi = np.array(clean_toi)
        dirty_toi = np.array(dirty_toi)
        flags = np.vstack([flag1])
        pixels = np.vstack([pixels1])
        weights = None
        templates = [templates1]

        resid = MPI.COMM_WORLD.allgather(
            (dirty_toi - clean_toi)[np.logical_not(flags)].ravel())
        if rank == 0:
            print('RMS before destriping = {}'.format(
                np.std(np.hstack(resid))))

        destriper = FancyDestriperPol(self.npix, 1, MPI.COMM_WORLD,
                                      do_offset=True, do_gain=True, ndegrade=1,
                                      dir_out='test')
        _, amp, cov = destriper.destripe(
            dirty_toi, flags, pixels, weights, templates,
            verbose=self.verbose, in_place=False, return_baselines=True,
            siter='_temptest')

        if rank == 0:
            print('template amplitude = {} +- {}'.format(
                amp[0, 0], np.sqrt(cov[0, 0]), flush=True))

        if np.abs((amp[0, 0] - 1e4) / np.sqrt(cov[0, 0])) > 3 and \
           np.abs((amp[0, 0] - 1e4) / 1e4) > 1e-4:
            raise Exception(
                'Failed to fit the template: {} +- {} != {}'.format(
                    amp[0, 0], np.sqrt(cov[0, 0]), 1e4))

        return
