# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# A simple, single detector destriper that solves and subtracts an offset for
# each pointing period

# import pyximport; pyximport.install()

import os
import pickle
from toast_planck.preproc_modules.filters import flagged_running_average
from toast_planck.reproc_modules import destripe_tools

from scipy.constants import degree
from scipy.signal import fftconvolve
from toast.mpi import MPI

import healpy as hp
import numpy as np
import toast.cache as tc


class Destriper():

    def __init__(self, npix, mpicomm, cglimit=1e-12, itermax=1000):

        self.npix = npix
        self.comm = mpicomm
        self.cglimit = cglimit
        self.itermax = itermax

        self.rank = self.comm.Get_rank()
        self.ntask = self.comm.Get_size()

        self.hitinv = None

        # Profiling

        self.reset_timers()

    def reset_timers(self):

        self.time_subtract_baselines = 0.
        self.time_mpi_dot_product = 0.
        self.time_toi2base = 0.
        self.time_base2toi = 0.
        self.time_apply_Z = 0.
        self.time_map2toi = 0.
        self.time_toi2map = 0.
        self.time_destripe = 0.
        self.time_iterate = 0.
        self.time_rhs = 0.
        self.time_lhs = 0.
        self.time_precond_init = 0.
        self.time_precond = 0.
        self.time_cache_pointing = 0.
        self.time_mpi = 0.

    def cache_pointing(self, flag, pixels, reset=False):
        """
        Store pointing in flat internal arrays
        """

        if self.hitinv is not None and not reset:
            return

        t1 = MPI.Wtime()

        if len(pixels) == 0:
            self.flag = []
            self.good = []
            self.pixels = []
            self.hitinv = []
        else:
            try:
                self.flag = np.hstack(flag)
            except Exception as e:
                raise Exception('Failed to hstack this flag list: '
                                '{}, {}'.format(flag, e))
            self.good = np.logical_not(self.flag)

            self.pixels = np.hstack(pixels)[self.good].copy()

            hitmap = np.zeros(self.npix, dtype=np.int32)
            my_hitmap = np.zeros(self.npix, dtype=np.int32)

            destripe_tools.fast_hit_binning(self.pixels, my_hitmap)

            t2 = MPI.Wtime()
            self.comm.Allreduce(my_hitmap, hitmap, op=MPI.SUM)
            self.time_mpi += MPI.Wtime() - t2

            self.hitinv = np.zeros(self.npix, dtype=np.float64)
            hitpix = hitmap != 0
            self.hitinv[hitpix] = 1. / hitmap[hitpix]

        self.time_cache_pointing += MPI.Wtime() - t1

    def destripe(
            self, toi, flag, pixels, verbose=False, return_baselines=False,
            in_place=False, templates=None):

        """Destripe the TOI that is provided as a list of TOI vectors:
        one offset per vector.
        """

        # Always fit for an offset

        self.nbaseline = len(toi)
        self.ntemplate = 1

        # Optionally allow for additional baseline templates
        # Require the same number of templates for every segment

        if templates is not None:
            self.ntemplate += len(templates[0])

        self.reset_timers()

        # Will only cache once and reuse between calls

        self.cache_pointing(flag, pixels)

        # Evaluate the Right Hand Side of the destriping equation

        t0 = MPI.Wtime()

        t1 = MPI.Wtime()

        rhs = self.toi2base(self.apply_Z(toi, flag, pixels),
                            flag, templates=templates)

        self.time_rhs += MPI.Wtime() - t1

        # Define the linear operator that applies the sparse matrix
        # on the Left Hand Side

        def lhs(baselines, templates):

            t1 = MPI.Wtime()

            b = self.toi2base(
                self.apply_Z(
                    self.base2toi(
                        baselines, toi, templates=templates),
                    flag, pixels),
                flag, templates=templates)

            self.time_lhs += MPI.Wtime() - t1

            return b

        # Generate the preconditioner from F^T F, the primary component of LHS

        t1 = MPI.Wtime()

        nhitinv = np.zeros([self.nbaseline, self.ntemplate])
        for ibase, flagvec in enumerate(flag):
            nn = np.sum(np.logical_not(flagvec))
            if nn == 0:
                continue
            nn = 1. / nn
            nhitinv[ibase, 0] = nn
            if templates is not None:
                for itemp, temp in enumerate(templates[ibase]):
                    nn = destripe_tools.fast_masked_sum(
                        temp * temp, flagvec.astype(np.int32))
                    if nn != 0:
                        nhitinv[ibase, itemp + 1] = 1 / nn

        self.time_precond_init += MPI.Wtime() - t1

        def precond(baselines):

            t1 = MPI.Wtime()

            if len(nhitinv) > 0:
                b = nhitinv * baselines
            else:
                b = []

            self.time_precond += MPI.Wtime() - t1

            return b

        # CG-iterate to solve lhs(baselines) = rhs

        t2 = MPI.Wtime()

        x0 = np.zeros([self.nbaseline, self.ntemplate])
        r0 = rhs - lhs(x0, templates)  # r stands for the residual
        z0 = precond(r0)
        x = x0.copy()
        r = r0.copy()
        p = z0.copy()
        iiter = 0
        rz0 = self.mpi_dot_product(r0, z0)
        rz = rz0
        if self.rank == 0:
            print('Iteration {:03} : residual = {}'.format(iiter, rz0))
        while iiter < self.itermax:
            iiter += 1
            if np.isnan(rz):
                raise Exception(
                    'ERROR: rz is NaN on iteration {}'.format(iiter))
            Ap = lhs(p, templates)
            denom = self.mpi_dot_product(p, Ap)
            if denom == 0:
                raise Exception(
                    'ERROR: denom is zero on iteration {}'.format(iiter))
            alpha = rz / denom
            x += alpha * p
            r -= alpha * Ap
            z = precond(r)
            new_rz = self.mpi_dot_product(r, z)
            if self.rank == 0:
                print('Iteration {:03} : relative residual = {}'.format(
                    iiter, new_rz / rz0))
            if rz / rz0 < self.cglimit:
                break
            beta = new_rz / rz
            p = z + beta * p
            rz = new_rz

        self.time_iterate += MPI.Wtime() - t2

        clean_toi = self.subtract_baselines(toi, x, in_place=in_place,
                                            templates=templates)

        self.time_destripe += MPI.Wtime() - t0

        if verbose:
            self.report_timing()

        if return_baselines:
            return clean_toi, x
        else:
            return clean_toi

    def subtract_baselines(self, toi, baselines, in_place=False,
                           templates=None):

        t1 = MPI.Wtime()
        if in_place:
            clean_toi = toi
        else:
            clean_toi = []
            for toivec, baseline in zip(toi, baselines):
                clean_toi.append(toivec.copy())

        for ibase, (toivec, baseline) in enumerate(zip(clean_toi, baselines)):
            toivec -= baseline[0]
            if templates is not None:
                for itemp, temp in enumerate(templates[ibase]):
                    toivec -= baseline[itemp + 1] * temp

        self.time_subtract_baselines += MPI.Wtime() - t1

        return clean_toi

    def mpi_dot_product(self, x, y):

        t1 = MPI.Wtime()
        my_sqsum = np.sum(x * y)
        t2 = MPI.Wtime()
        sqsum = self.comm.allreduce(my_sqsum, op=MPI.SUM)
        self.time_mpi += MPI.Wtime() - t2

        self.time_mpi_dot_product += MPI.Wtime() - t1

        return sqsum

    def toi2base(self, toi, flag, templates=None):

        """ Bin TOI onto baselines """

        t1 = MPI.Wtime()
        baselines = np.zeros([self.nbaseline, self.ntemplate])
        for ibase, (toivec, flagvec) in enumerate(zip(toi, flag)):
            if np.all(flagvec):
                baselines[ibase] = 0
            else:
                baselines[ibase][0] = destripe_tools.fast_masked_sum(
                    toivec, flagvec.astype(np.int32))
                if templates is not None:
                    for itemp, temp in enumerate(templates[ibase]):
                        baselines[ibase][itemp + 1] = \
                            destripe_tools.fast_masked_sum(
                            toivec * temp, flagvec.astype(np.int32))

        self.time_toi2base += MPI.Wtime() - t1

        return baselines

    def base2toi(self, baselines, toi, templates):

        """ Scan baselines to TOI """

        t1 = MPI.Wtime()
        basetoi = []
        for ibase, (baseline, toivec) in enumerate(zip(baselines, toi)):
            basetoi.append(np.ones(len(toivec)) * baseline[0])
            if templates is not None:
                for itemp, temp in enumerate(templates[ibase]):
                    basetoi[-1] += temp * baseline[itemp + 1]
        self.time_base2toi += MPI.Wtime() - t1

        return basetoi

    def apply_Z(self, toi, flag, pixels):

        """ Apply Z = I - P ( P^T P )^-1 P^T to the TOI """

        t1 = MPI.Wtime()
        zmap = self.toi2map(toi, flag, pixels)
        ztoi = self.map2toi(zmap, pixels)
        cleaned_toi = []
        for toivec, ztoivec in zip(toi, ztoi):
            clean_toi = toivec - ztoivec
            cleaned_toi.append(clean_toi.copy())
        self.time_apply_Z += MPI.Wtime() - t1

        return cleaned_toi

    def map2toi(self, sigmap, pixels):

        """ Scan TOI from the map based on a list of pixel vectors """

        t1 = MPI.Wtime()
        toi = []
        for pixvec in pixels:
            toivec = np.zeros(len(pixvec), dtype=np.float64)
            destripe_tools.fast_scanning(toivec, pixvec, sigmap)
            toi.append(toivec.copy())
        self.time_map2toi += MPI.Wtime() - t1

        return toi

    def toi2map(self, toi, flag, pixels):

        """ Bin the TOI based on the pointing in pixels. """

        t1 = MPI.Wtime()
        sigmap = np.zeros(self.npix)
        if len(toi) > 0:
            mytoi = np.hstack(toi)[self.good].copy()
            destripe_tools.fast_binning(
                mytoi.astype(np.float64), self.pixels, sigmap)
            sigmap *= self.hitinv
        self.time_toi2map += MPI.Wtime() - t1

        t1 = MPI.Wtime()
        self.comm.Allreduce(MPI.IN_PLACE, sigmap, op=MPI.SUM)
        self.time_mpi += MPI.Wtime() - t1

        return sigmap

    def report_timing(self):

        def report(name, t):
            ttot = np.array(self.comm.gather(t, root=0))
            if self.rank == 0:
                print('{} time mean {:7.2f} s min {:7.2f} s, '
                      'max {:7.2f}s'.format(name, np.mean(ttot),
                                            np.amin(ttot), np.amax(ttot)))

        report('Total destriping .......', self.time_destripe)
        report('  - cache pointing .... ', self.time_cache_pointing)
        report('  - init preconditioner ', self.time_precond_init)
        report('  - evaluate RHS .......', self.time_rhs)
        report('  - CG iterate .........', self.time_iterate)
        report('    - precond ..........', self.time_precond)
        report('    - LHS ..............', self.time_lhs)
        report('      - toi2base .......', self.time_toi2base)
        report('      - apply Z ........', self.time_apply_Z)
        report('        - toi2map ......', self.time_toi2map)
        report('        - map2toi ......', self.time_map2toi)
        report('      - base2toi .......', self.time_base2toi)
        report('    - MPI dot product ..', self.time_mpi_dot_product)
        report('  - subtract baselines .', self.time_subtract_baselines)
        report('  - MPI .............. .', self.time_mpi)


class FancyDestriper(Destriper):

    # Now permanently fixed:
    #   only_templates = True
    #   templates_are_destriped = True

    def destripe(self, toi, flag, pixels, templates, IV_templates=None,
                 verbose=False, return_baselines=False,
                 in_place=False, positive_templates=None, cglimit=None,
                 return_ring_by_ring=False):
        """
        Destripe with one offset per ring and additional full mission templates
        """

        if len(toi) == 0:
            raise Exception('ERROR: the destriper cannot presently handle '
                            'empty lists of TOI')

        self.reset_timers()

        self.cache_pointing(flag, pixels)

        # Evaluate the Right Hand Side of the destriping equation

        t0 = MPI.Wtime()
        t1 = MPI.Wtime()

        clean_toi = self.apply_Z(toi, flag, pixels)

        self.time_rhs += MPI.Wtime() - t1

        # Generate the preconditioner from F^T F

        t1 = MPI.Wtime()

        # The regular baseline block of the matrix is diagonal

        nhitinv = []
        for flagvec in flag:
            nn = np.sum(np.logical_not(flagvec))
            if nn != 0:
                nn = 1. / nn
            nhitinv.append(nn)
        nhitinv = np.array(nhitinv)

        ntemplate = len(templates)
        XZ = np.zeros([ntemplate, ntemplate])
        ZZ = np.zeros([ntemplate, ntemplate])
        Zy = np.zeros(ntemplate)

        # While we are at it, let's collect the fit coefficients ring-by-ring
        # for diagnostic purposes

        try:
            nring = len(templates[0])
        except Exception:
            nring = 0
        ringlens = []
        istart = 0
        for iring in range(nring):
            ringlen = len(templates[0][iring])
            istop = istart + ringlen
            ngood = np.sum(self.good[istart:istop])
            ringlens.append(ngood)
            istart = istop
        # The extra template is the offset
        ring_XZ = np.zeros([nring, ntemplate + 1, ntemplate + 1])
        # The extra template is the offset
        ring_ZZ = np.zeros([nring, ntemplate + 1, ntemplate + 1])
        ring_Zy = np.zeros([nring, ntemplate + 1])
        clean_toi = np.hstack(clean_toi)[self.good].copy()

        clean_template_tois = []
        clean_IV_template_tois = []
        if IV_templates is None:
            for i, template in enumerate(templates):
                # This definition yields the posterior covariance
                # instead of the prior
                itoi = np.hstack(self.apply_Z(
                    template, flag, pixels))[self.good].copy()
                clean_template_tois.append(itoi)
                clean_IV_template_tois.append(itoi)
        else:
            # process also the instrumental variable templates
            for i, (template, IV_template) in enumerate(
                    zip(templates, IV_templates)):
                itoi = np.hstack(self.apply_Z(
                    template, flag, pixels))[self.good].copy()
                clean_template_tois.append(itoi)
                if template is IV_template:
                    clean_IV_template_tois.append(itoi)
                else:
                    IV_itoi = np.hstack(self.apply_Z(
                        IV_template, flag, pixels))[self.good].copy()
                    clean_IV_template_tois.append(IV_itoi)

        for i, _ in enumerate(templates):
            itoi = clean_template_tois[i]
            IV_itoi = clean_IV_template_tois[i]
            dp_Zy = destripe_tools.fast_dot_product(
                IV_itoi.astype(np.float64), clean_toi.astype(np.float64))

            Zy[i] = dp_Zy

            for j, _ in enumerate(templates[:i + 1]):
                # The second application of Z is only redundant when
                # considering the full mission.
                jtoi = clean_template_tois[j]
                IV_jtoi = clean_IV_template_tois[j]

                dp_XZ = destripe_tools.fast_dot_product(
                    itoi.astype(np.float64), IV_jtoi.astype(np.float64))
                dp_ZZ = destripe_tools.fast_dot_product(
                    IV_itoi.astype(np.float64), IV_jtoi.astype(np.float64))
                XZ[i, j] += dp_XZ
                ZZ[i, j] += dp_ZZ
                if i != j:
                    XZ[j, i] += dp_XZ
                    ZZ[j, i] += dp_ZZ

                if return_ring_by_ring:
                    # ring-by-ring diagnostics:
                    istart = 0
                    for iring, ringlen in enumerate(ringlens):
                        istop = istart + ringlen
                        ind = slice(istart, istart + ringlen)
                        dp_XZ = destripe_tools.fast_dot_product(
                            itoi[ind].astype(np.float64),
                            IV_jtoi[ind].astype(np.float64))
                        dp_ZZ = destripe_tools.fast_dot_product(
                            IV_itoi[ind].astype(np.float64),
                            IV_jtoi[ind].astype(np.float64))
                        ring_XZ[iring, i, j] = dp_XZ
                        ring_ZZ[iring, i, j] = dp_ZZ
                        if i != j:
                            ring_XZ[iring, j, i] = dp_XZ
                            ring_ZZ[iring, j, i] = dp_ZZ
                        if i == j:
                            # Add the offset template dot products
                            offset = np.ones(ringlen)
                            if len(offset) != len(jtoi[ind]):
                                raise Exception(
                                    'Lengths do not match: ''iring = {}, '
                                    'nring = {}, ringlen={}, istart={}, '
                                    'istop={}'.format(
                                        iring, nring, ringlen, istart, istop))
                            dp = destripe_tools.fast_dot_product(
                                offset, IV_jtoi[ind].astype(np.float64))
                            ring_XZ[iring, ntemplate, j] = dp
                            ring_XZ[iring, j, ntemplate] = dp
                            ring_ZZ[iring, ntemplate, j] = dp
                            ring_ZZ[iring, j, ntemplate] = dp
                            # Use the j-template without Z-application to match
                            # full mission processing.
                            ring_Zy[
                                iring, i] = destripe_tools.fast_dot_product(
                                    IV_itoi[ind].astype(np.float64),
                                    clean_toi[ind].astype(np.float64))
                            if i == ntemplate - 1:
                                ring_XZ[iring, ntemplate,
                                        ntemplate] = np.dot(offset, offset)
                                ring_ZZ[iring, ntemplate,
                                        ntemplate] = np.dot(offset, offset)
                                ring_Zy[iring, ntemplate] = destripe_tools.fast_dot_product(
                                        offset,
                                        clean_toi[ind].astype(np.float64))
                        istart += ringlen

        del clean_template_tois

        t2 = MPI.Wtime()
        self.comm.Allreduce(MPI.IN_PLACE, XZ, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, ZZ, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, Zy, op=MPI.SUM)
        self.time_mpi += MPI.Wtime() - t2

        # Solve for the least squares template amplitudes.
        # This is a really small problem so no reason
        # to try to solve in parallel

        try:
            ZZinv = np.linalg.inv(ZZ)
        except Exception:
            raise Exception(
                'ERROR: failed to invert the {} instrument template '
                'covariance matrix'.format(ntemplate))

        invcov = np.dot(XZ, np.dot(ZZinv, XZ.T))

        try:
            cov = np.linalg.inv(invcov)
        except Exception:
            raise Exception('ERROR: failed to invert the {} template '
                            'covariance matrix'.format(ntemplate))

        proj = np.dot(XZ, np.dot(ZZinv, Zy))

        self.time_precond_init += MPI.Wtime() - t1

        x = np.dot(cov, proj)

        self.time_iterate += MPI.Wtime() - t2

        clean_toi = self.subtract_baselines(
            toi, x, templates, in_place=in_place)

        self.time_destripe += MPI.Wtime() - t0

        # Measure the RMS of the baseline-subtracted TOI by subtracting the
        # signal estimate (includes any possible overall offset). This produces
        # a slight overestimate of the noise RMS as the binned and scanned
        # signal estimate is also noisy.

        noise_toi = np.hstack(self.apply_Z(
            clean_toi, flag, pixels))[self.good].copy()
        my_n = len(noise_toi)
        my_sqsum = np.sum(noise_toi ** 2)
        n_tot = self.comm.allreduce(my_n, op=MPI.SUM)
        sqsum_tot = self.comm.allreduce(my_sqsum, op=MPI.SUM)
        rms = np.sqrt(sqsum_tot / n_tot)

        # report

        if verbose:
            if self.rank == 0:
                print('residual TOI RMS = ', rms)
                print('TOI template amplitudes: ', x)
            self.report_timing()

        if return_baselines:
            amplitudes = x
            if return_ring_by_ring:
                return (clean_toi, amplitudes, cov * rms,
                        ring_XZ, ring_ZZ, ring_Zy)
            else:
                return clean_toi, amplitudes, cov * rms
        else:
            return clean_toi

    def subtract_baselines(self, toi, amplitudes, templates, in_place=False):

        t1 = MPI.Wtime()

        if in_place:
            clean_toi = toi
        else:
            clean_toi = []
            for toivec in toi:
                clean_toi.append(toivec.copy())

        # templates

        if len(templates) > 0:
            for amplitude, template in zip(amplitudes, templates):
                for toivec, templatevec in zip(clean_toi, template):
                    toivec -= amplitude * templatevec

        self.time_subtract_baselines += MPI.Wtime() - t1

        return clean_toi

    def toi2base(self, toi, flag, templates):

        """ Bin TOI onto baselines, F^T y. """

        t1 = MPI.Wtime()

        if len(templates) > 0:
            my_amplitudes = np.zeros(len(templates))

            mytoi = np.hstack(toi)[self.good].copy()

            for iamp, template in enumerate(templates):
                my_amplitudes[iamp] += destripe_tools.fast_dot_product(
                    mytoi.astype(np.float64),
                    np.hstack(template)[self.good].astype(np.float64))

            # These amplitudes will be identical across processes
            amplitudes = np.zeros(len(templates))

            t2 = MPI.Wtime()
            self.comm.Allreduce(my_amplitudes, amplitudes, op=MPI.SUM)
            self.time_mpi += MPI.Wtime() - t2
        else:
            amplitudes = np.array([])

        self.time_toi2base += MPI.Wtime() - t1

        return amplitudes

    def base2toi(self, baselines, toi, amplitudes, templates):

        """ Scan baselines to TOI """

        t1 = MPI.Wtime()

        # templates

        for itemplate, (amplitude, template) in enumerate(
                zip(amplitudes, templates)):
            for templatevec in template:
                toivec += amplitude * templatevec

        self.time_base2toi += MPI.Wtime() - t1

        return basetoi


class FancyDestriperPol():

    """
    Polarized, multi-detector destriper that only fits full length templates.
    """

    def __init__(self, npix, nnz, mpicomm, do_offset=False, do_gain=False,
                 do_pol_eff=False, do_pol_angle=False,
                 threshold=1e-2, ndegrade=1, fsample=1.0, lowpassfreq=1.0,
                 fwhm=10 * degree, lmax=128, dir_out='./', precond=True):

        self.ndegrade = int(ndegrade)
        if self.ndegrade < 1:
            raise Exception('ERROR: ndegrade cannot be smaller than one: {}'
                            ''.format(ndegrade))
        self.npix = npix // ndegrade
        self.nside = hp.npix2nside(self.npix)
        self.nnz = nnz
        self.threshold = threshold
        self.comm = mpicomm
        # We support two templates that are constructed on the fly
        self.do_offset = do_offset
        if nnz == 2 and do_gain:
            raise RuntimeError('Cannot construct a gain template when nnz = 2')
        self.do_gain = do_gain
        if nnz == 2 and do_pol_eff:
            raise RuntimeError('Not set up to measure pol.eff. when nnz = 2')
        self.do_pol_eff = do_pol_eff and (nnz != 1)
        if nnz == 2 and do_pol_angle:
            raise RuntimeError('Not set up to measure pol.angle. when nnz = 2')
        self.do_pol_angle = do_pol_angle and (nnz != 1)
        self.fsample = fsample
        self.lowpassfreq = min(lowpassfreq, fsample)
        if self.lowpassfreq is None or self.lowpassfreq == 0:
            self.naverage = 1
        else:
            self.naverage = int(self.fsample / self.lowpassfreq)
        self.fwhm = fwhm
        self.lmax = min(lmax, 3 * self.nside)
        self.dir = dir_out
        self.precond = precond

        self.rank = self.comm.Get_rank()
        self.ntask = self.comm.Get_size()

        if self.rank == 0 and not os.path.isdir(self.dir):
            os.makedirs(self.dir)

        self.cc = None
        self.rcond = None
        self.my_npix = None
        self.local2global = None
        self.global2local = None
        self.pixels = None
        self.my_own = None
        self.commonpix = None
        self.counts = None
        self.displs = None
        self.bufsize = None

        self.cache = tc.Cache()

        # Profiling

        self.reset_timers()

    def reset_timers(self):

        self.time_subtract_baselines = 0.
        self.time_apply_Z = 0.
        self.time_map2toi = 0.
        self.time_toi2map = 0.
        self.time_destripe = 0.
        self.time_cache_pointing = 0.
        self.time_mpi = 0.
        self.time_accumulate_cc = 0.
        self.time_invert_cc = 0.
        self.time_cc_multiply = 0.
        self.time_clean_templates = 0.
        self.time_build_cov = 0.
        self.time_invert_cov = 0.
        self.time_solve_amplitudes = 0.
        self.time_rms = 0.
        self.time_alltoallv = 0.
        self.time_init_alltoallv = 0.
        self.time_collect_map = 0.
        self.time_lowpass = 0.
        self.time_write_map = 0.

    def flag_outliers(self, toi, flag, pixels, weights, verbose=False,
                      save_maps=False, siter='', flag_out=None,
                      threshold=10.):
        """Subtract binned signal and find outliers from the residual TOD
        """

        self.verbose = verbose

        self.reset_timers()

        if len(toi) == 0:
            raise Exception(
                'ERROR: the destriper cannot handle empty lists of TOI')

        self.ndet = len(toi)
        self.nsamp = len(toi[0])

        if weights is None:
            weights = np.ones([self.ndet, self.nsamp, 1], dtype=np.float32)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print('Initializing alltoallv', flush=True)
            self.comm.Barrier()

        self.initialize_alltoallv(pixels, flag)

        # Evaluate the Right Hand Side of the destriping equation

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print('Computing the pixel covariance matrices', flush=True)
            self.comm.Barrier()

        self.get_cc(flag, pixels, weights)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print('Binning full initial map', flush=True)
            self.comm.Barrier()

        full_initial_map = self.toi2map(toi, flag, pixels, weights)
        nnan = np.sum(np.isnan(full_initial_map))
        if nnan != 0:
            print('{:4} : WARNING: accumulated initial_map contains {} NaNs'
                  ''.format(self.rank, nnan))
            full_initial_map[np.isnan(full_initial_map)] = 0

        if self.verbose and self.rank == 0:
            print('CC-multiplying map', flush=True)
        self.comm.Barrier()
        full_initial_map = self.cc_multiply(full_initial_map)

        if save_maps:
            if self.verbose and self.rank == 0:
                print('Collecting map', flush=True)
            self.comm.Barrier()
            global_map = self.collect_map(full_initial_map).T.copy()

        if self.verbose:
            if self.verbose and self.rank == 0:
                print('Computing full initial RMS', flush=True)
            self.comm.Barrier()

            rms = self.get_rms(toi, flag, pixels, weights)

            self.comm.Barrier()
            if self.rank == 0:
                print('RMS computed', flush=True)

            if self.rank == 0:
                print('Input TOI RMS = ', rms, flush=True)
                if save_maps:
                    print('Initial map RMS:', flush=True)
                    for m in global_map:
                        print('{}'.format(np.nanstd(m.ravel())))
                    print('', flush=True)
                    t2 = MPI.Wtime()
                    fn = os.path.join(self.dir,
                                      'full_initial_map{}.fits'.format(siter))
                    hp.write_map(fn, global_map, nest=True)
                    print('Saved initial map to {}'.format(fn), flush=True)
                    self.time_write_map += MPI.Wtime() - t2

        if save_maps:
            del global_map

        if self.verbose:
            if self.rank == 0:
                print('Cleaning TOI', flush=True)
            self.comm.Barrier()

        clean_toi = self.apply_Z(toi, flag, pixels, weights, hstack=False)

        # The clean_toi is downsampled.  We find which steps have outlier
        # variances and flag the corresponding full rate flags

        # steplen = int(60 * self.fsample / self.naverage)
        steplen = int(300 * self.fsample / self.naverage)

        for idet in range(self.ndet):
            nbad = np.sum(flag[idet] != 0)
            ntotal = flag[idet].size
            nbad = self.comm.allreduce(nbad, op=MPI.SUM)
            ntotal = self.comm.allreduce(ntotal, op=MPI.SUM)
            if self.rank == 0:
                print('Detector {} has {} / {} = {:.3f} % flagged.'.format(
                    idet, nbad, ntotal, nbad * 100. / ntotal))
            my_hitsum = []
            my_stepsum = []
            my_ind = []
            istart = 0
            nn = clean_toi[0].size
            while istart < nn:
                istop = istart + steplen
                ii = slice(istart, istop)
                good = clean_toi[idet][ii] != 0
                nhit = np.sum(good)
                my_hitsum.append(nhit)
                my_stepsum.append(np.sum(clean_toi[idet][ii][good]))
                my_ind.append((istart, istop))
                istart = istop
            my_hitsum = np.array(my_hitsum)
            my_stepsum = np.array(my_stepsum)
            my_rank = np.ones(my_hitsum.size, dtype=int) * self.rank
            all_hitsum = np.hstack(self.comm.allgather(my_hitsum))
            all_stepsum = np.hstack(self.comm.allgather(my_stepsum))
            all_rank = np.hstack(self.comm.allgather(my_rank))
            good = all_hitsum > 0
            all_var = np.zeros_like(all_stepsum)
            all_var[good] = all_stepsum[good] ** 2 / all_hitsum[good]
            wmean = min(10001, len(all_var) // 10)
            werr = min(100001, len(all_var) // 10)
            for _ in range(10):
                # Smooth mean
                var_mean = flagged_running_average(
                    all_var, np.logical_not(good), wmean)
                # Smooth error estimate
                var_err = np.sqrt(flagged_running_average(
                    (all_var - var_mean) ** 2, np.logical_not(good), werr))
                # Detect outliers
                ngood1 = np.sum(good)
                good[np.abs(all_var - var_mean) > var_err * threshold] = False
                ngood2 = np.sum(good)
                if ngood1 == ngood2:
                    break
            # Flag the odd intervals that passed the test but are embedded
            # in a suspicious region.
            bad = np.logical_not(good)
            bad[all_hitsum == 0] = False
            kernel = np.ones(5) / 5
            rbad = fftconvolve(bad, kernel, mode='same')
            good[rbad > 0.2] = False
            # Flag lonely steps
            bad = np.logical_not(good)
            kernel = np.ones(10) / 10
            rbad = fftconvolve(bad, kernel, mode='same')
            good[rbad > 0.8] = False
            nbad = np.sum(np.logical_and(np.logical_not(good), all_hitsum > 0))
            if self.rank == 0:
                fn = os.path.join(self.dir,
                                  'step_var{}_{}.pck'.format(siter, idet))
                with open(fn, 'wb') as f:
                    pickle.dump([all_var, all_hitsum], f, protocol=2)
                print('step variances stored in {}'.format(fn))
                print('Flagging {} / {} outlier steps for detector {}'.format(
                    nbad, all_var.size, idet), flush=True)

            mine = all_rank == self.rank
            my_good = good[mine]
            bad = np.argwhere(np.logical_not(my_good))
            if len(bad) > 0:
                for ibad in bad.ravel():
                    if my_hitsum[ibad] == 0:
                        continue
                    ii = my_ind[ibad]
                    istart = ii[0] * self.naverage
                    istop = ii[1] * self.naverage
                    if flag[idet].dtype == np.bool:
                        flag[idet][istart:istop] = True
                    else:
                        flag[idet][istart:istop] |= np.uint8(255)
                    if flag_out is not None:
                        if flag_out[idet].dtype == np.bool:
                            flag_out[idet][istart:istop] = True
                        else:
                            flag_out[idet][istart:istop] |= np.uint8(255)

            nbad = np.sum(flag[idet] != 0)
            ntotal = flag[idet].size
            nbad = self.comm.allreduce(nbad, op=MPI.SUM)
            ntotal = self.comm.allreduce(ntotal, op=MPI.SUM)
            if self.rank == 0:
                print('Detector {} now has {} / {} = {:.3f} % flagged.'.format(
                    idet, nbad, ntotal, nbad * 100. / ntotal))

        del self.cc
        del self.rcond
        del self.pixels
        self.cache.clear()
        self.cc = None
        self.rcond = None
        self.my_npix = None
        self.local2global = None
        self.global2local = None
        self.pixels = None
        self.my_own = None
        self.commonpix = None
        self.counts = None
        self.displs = None
        self.bufsize = None

        return

    def destripe(
            self, toi, flag, pixels, weights, templates, IV_templates=None,
            verbose=False, return_baselines=False, in_place=False, siter='',
            save_maps=False):

        """Fix the provided multi-detector polarized TOI with the given
           templates and (optionally) provided instrumental variables.
        """

        self.verbose = verbose

        self.reset_timers()

        t0 = MPI.Wtime()

        if len(toi) == 0:
            raise Exception(
                'ERROR: the destriper cannot handle empty lists of TOI')

        self.ndet = len(toi)
        self.nsamp = len(toi[0])
        self.ntemplate = len(templates[0])
        do_offset = self.do_offset and (self.ndet > 1)
        do_gain = self.do_gain and (self.ndet > 1)
        do_pol_eff = self.do_pol_eff and (self.ndet > 1)
        do_pol_angle = self.do_pol_angle and (self.ndet > 1)
        if do_offset:
            self.ntemplate += 1
        if do_gain:
            self.ntemplate += 1
        if do_pol_eff:
            self.ntemplate += 1
        if do_pol_angle:
            self.ntemplate += 1

        if self.ntemplate == 0:
            if self.rank == 0:
                print('No templates to fit. No destriping done.', flush=True)
            if return_baselines:
                return toi, np.zeros(0), np.zeros([0, 0])
            else:
                return toi

        if weights is None:
            weights = np.ones([self.ndet, self.nsamp, 1], dtype=np.float32)

        nsamptot = self.comm.allreduce(self.nsamp, op=MPI.SUM)
        if self.verbose and self.rank == 0:
            print('ndet = {}, nsamptot = {}, ntemplate = {}'.format(
                self.ndet, nsamptot, self.ntemplate), flush=True)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print('Initializing alltoallv', flush=True)
            self.comm.Barrier()

        self.initialize_alltoallv(pixels, flag)

        # Evaluate the Right Hand Side of the destriping equation

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print('Computing the pixel covariance matrices', flush=True)
            self.comm.Barrier()

        self.get_cc(flag, pixels, weights)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print('Binning initial map', flush=True)
            self.comm.Barrier()

        initial_map = self.toi2map(toi, flag, pixels, weights)
        nnan = np.sum(np.isnan(initial_map))
        if nnan != 0:
            print('{:4} : WARNING: accumulated initial_map contains {} NaNs'
                  ''.format(self.rank, nnan))
            initial_map[np.isnan(initial_map)] = 0
        initial_map = self.cc_multiply(initial_map)

        if save_maps:
            global_map = self.collect_map(initial_map).T.copy()

        if self.verbose:
            if self.rank == 0:
                print('Computing initial RMS', flush=True)
            self.comm.Barrier()

            rms = self.get_rms(toi, flag, pixels, weights)
            if self.rank == 0:
                print('Input TOI RMS = ', rms)
                if save_maps:
                    print('Initial map RMS:')
                    for m in global_map:
                        print('{}'.format(np.nanstd(m.ravel())))
                    print('', flush=True)
                    t2 = MPI.Wtime()
                    fn = os.path.join(self.dir,
                                      'initial_map{}.fits'.format(siter))
                    try:
                        hp.write_map(fn, global_map, nest=True)
                    except Exception:
                        hp.write_map(fn, global_map, nest=True, overwrite=True)
                    print('Saved initial map to {}'.format(fn), flush=True)
                    self.time_write_map += MPI.Wtime() - t2

        """
        if do_gain or do_pol_eff or do_pol_angle:
            global_map = hp.reorder(global_map, n2r=True)
            global_map = hp.smoothing(
                global_map, fwhm=self.fwhm, lmax=self.lmax, iter=0,
                self.verbose=True)
            global_map = np.array(hp.reorder(global_map, r2n=True)).T
            smoothed_map = np.zeros([self.my_npix, self.nnz], dtype=np.float64)
            for my_pix, pix in enumerate(self.local2global):
                smoothed_map[my_pix] = global_map[pix]
            global_map = self.collect_map(smoothed_map).T.copy()
            fn = os.path.join(self.dir, 'smoothed_map{}.fits'.format(siter))
            hp.write_map(fn, global_map, nest=True)
            print('Saved smoothed map to {}'.format(fn), flush=True)
        """

        if save_maps:
            del global_map

        if self.verbose:
            if self.rank == 0:
                print('Cleaning TOI', flush=True)
            self.comm.Barrier()

        t1 = MPI.Wtime()

        clean_toi = self.apply_Z(toi, flag, pixels, weights)

        # In the following, "X" is the matrix of time domain templates
        # and "Z" is the matrix of matching instrument variables

        if self.verbose:
            if self.rank == 0:
                print('Cleaning template TOIs', flush=True)
            self.comm.Barrier()

        # We have to fix the first detector amplitudes to avoid degeneracies in
        # a multi detector run

        if self.ndet == 1 or self.nnz == 2:
            ioff = 0
            ioff_det = 0
        else:
            ioff = self.ntemplate
            ioff_det = 1

        ZX = np.zeros(
            [self.ndet - ioff_det, self.ntemplate, self.ndet - ioff_det,
             self.ntemplate], dtype=np.float64)
        Zy = np.zeros([self.ndet - ioff_det, self.ntemplate], dtype=np.float64)

        def check_size(itoi, idet, name):
            if itoi.size != clean_toi.size:
                raise Exception(
                    'ERROR: det = {}: template {} is different size than '
                    'clean_toi: {} != {}'.format(
                        idet, name, itoi.size, clean_toi.size))

        clean_template_tois = []
        clean_IV_template_tois = []
        for idet in range(ioff_det, self.ndet):
            clean_template_tois_det = []
            clean_IV_template_tois_det = []
            if do_offset:
                itoi = self.apply_Z(np.ones(self.nsamp), flag, pixels,
                                    weights, det=idet).copy()
                check_size(itoi, idet, 'offset')
                cachename = 'clean_template_offset_{}'.format(idet)
                if len(itoi) > 0:
                    itoi = self.cache.put(cachename, itoi, replace=True)
                clean_template_tois_det.append(itoi)
                clean_IV_template_tois_det.append(itoi)
                del itoi
            if do_gain:
                itoi = self.apply_Z(
                    self.map2toi(initial_map, flag, pixels, weights, det=idet),
                    flag, pixels, weights, det=idet).copy()
                check_size(itoi, idet, 'gain')
                cachename = 'clean_template_gain_{}'.format(idet)
                if len(itoi) > 0:
                    itoi = self.cache.put(cachename, itoi, replace=True)
                clean_template_tois_det.append(itoi)
                # The instrumental variable template comes from the smoothed map
                clean_IV_template_tois_det.append(itoi)  # No IV
                del itoi
                """
                itoi_IV = self.apply_Z(
                    self.map2toi(smoothed_map, flag, pixels, weights, det=idet),
                    flag, pixels, weights, det=idet).copy()
                cachename = 'clean_IV_template_gain_{}'.format(idet)
                itoi_IV = self.cache.put(cachename, itoi_IV, replace=True)
                clean_IV_template_tois_det.append(itoi_IV)
                """
            if do_pol_eff:
                itoi = self.apply_Z(
                    self.map2toi(initial_map, flag, pixels, weights, det=idet,
                                 pol_eff=True),
                    flag, pixels, weights, det=idet).copy()
                check_size(itoi, idet, 'pol_eff')
                cachename = 'clean_template_pol_eff_{}'.format(idet)
                if len(itoi) > 0:
                    itoi = self.cache.put(cachename, itoi, replace=True)
                clean_template_tois_det.append(itoi)
                # The instrumental variable template comes from the smoothed map
                clean_IV_template_tois_det.append(itoi)  # No IV
                del itoi
                """
                itoi_IV = self.apply_Z(
                    self.map2toi(smoothed_map, flag, pixels, weights, det=idet,
                                 pol_eff=True),
                    flag, pixels, weights, det=idet).copy()
                cachename = 'clean_IV_template_pol_eff_{}'.format(idet)
                itoi_IV = self.cache.put(cachename, itoi_IV, replace=True)
                clean_IV_template_tois_det.append(itoi_IV)
                """
            if do_pol_angle:
                itoi = self.apply_Z(
                    self.map2toi(initial_map, flag, pixels, weights, det=idet,
                                 pol_angle=True),
                    flag, pixels, weights, det=idet).copy()
                check_size(itoi, idet, 'pol_angle')
                cachename = 'clean_template_pol_ang_{}'.format(idet)
                if len(itoi) > 0:
                    itoi = self.cache.put(cachename, itoi, replace=True)
                clean_template_tois_det.append(itoi)
                # The instrumental variable template comes from the smoothed map
                clean_IV_template_tois_det.append(itoi)  # No IV
                del itoi
            if IV_templates is None:
                for i, template in enumerate(templates[idet]):
                    itoi = self.apply_Z(template, flag, pixels, weights,
                                        det=idet).copy()
                    check_size(itoi, idet, '# {}'.format(i))
                    cachename = 'clean_template_{}_{}'.format(i, idet)
                    if len(itoi) > 0:
                        itoi = self.cache.put(cachename, itoi, replace=True)
                    clean_template_tois_det.append(itoi)
                    clean_IV_template_tois_det.append(itoi)
                    del itoi
            else:
                # process also the instrumental variable templates
                for i, (template, IV_template) in enumerate(zip(
                        templates[idet], IV_templates[idet])):
                    itoi = self.apply_Z(template, flag, pixels, weights,
                                        det=idet).copy()
                    check_size(itoi, idet, '# {}'.format(i))
                    cachename = 'clean_template_{}_{}'.format(i, idet)
                    if len(itoi) > 0:
                        itoi = self.cache.put(cachename, itoi, replace=True)
                    clean_template_tois_det.append(itoi)
                    if template is IV_template:
                        clean_IV_template_tois_det.append(itoi)
                    else:
                        IV_itoi = self.apply_Z(
                            IV_template, flag, pixels, weights,
                            det=idet).copy()
                        check_size(IV_itoi, idet, '# {} IV'.format(i))
                        cachename = 'clean_IV_template_{}_{}'.format(i, idet)
                        if len(IV_itoi) > 0:
                            IV_itoi = self.cache.put(cachename, IV_itoi,
                                                     replace=True)
                        clean_IV_template_tois_det.append(IV_itoi)
                        del IV_itoi
                    del itoi
            clean_template_tois.append(clean_template_tois_det)
            clean_IV_template_tois.append(clean_IV_template_tois_det)
            del clean_template_tois_det
            del clean_IV_template_tois_det

        self.time_clean_templates += MPI.Wtime() - t1

        if self.verbose:
            if self.rank == 0:
                print('Multiplying TOI', flush=True)
            self.comm.Barrier()

        t1 = MPI.Wtime()

        for idet in range(self.ndet - ioff_det):
            for i in range(self.ntemplate):
                row = idet * self.ntemplate + i
                IV_itoi = clean_IV_template_tois[idet][i]
                dp_Zy = np.dot(IV_itoi, clean_toi)
                Zy[idet, i] = dp_Zy
                for jdet in range(self.ndet - ioff_det):
                    for j in range(self.ntemplate):
                        col = jdet * self.ntemplate + j
                        if col < row:
                            continue
                        jtoi = clean_template_tois[jdet][j]
                        dp_ZX = np.dot(IV_itoi, jtoi)
                        del jtoi

                        ZX[idet, i, jdet, j] += dp_ZX
                        if row != col:
                            ZX[jdet, j, idet, i] += dp_ZX
                del IV_itoi

        self.time_build_cov += MPI.Wtime() - t1

        if self.verbose:
            if self.rank == 0:
                print('Collecting dot products', flush=True)
            self.comm.Barrier()

        t1 = MPI.Wtime()
        self.comm.Allreduce(MPI.IN_PLACE, ZX, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, Zy, op=MPI.SUM)
        self.time_mpi += MPI.Wtime() - t1

        # Solve for the least squares template amplitudes.  This is really
        # a small problem.  No reason to try to solve in parallel.

        nn = (self.ndet - ioff_det) * self.ntemplate
        ZX = ZX.reshape([nn, nn])
        Zy = Zy.reshape([nn])

        if np.all(ZX == 0):
            if return_baselines:
                return (toi, np.zeros([self.ndet, self.ntemplate]),
                        np.zeros([nn, nn]))
            else:
                return toi

        t1 = MPI.Wtime()
        if self.rank == 0:
            try:
                ZXinv = None
                if self.precond:
                    # Different templates may have very different
                    # dynamical ranges.  This will result in poor
                    # condition numbers.  We can scale the
                    # templates to improve the condition numbers
                    precond = np.diag(1 / np.diag(ZX))
                    pZX = np.dot(precond, ZX)
                    evals = np.linalg.eigvals(pZX)
                    bad_evals = evals < np.amax(evals) * 1e-10
                    nbad = np.sum(bad_evals)
                    print('Preconditioned matrix has {} singular eigenmodes.\n'
                          'All eigenvalues: {}'.format(nbad, evals))
                    if nbad == 0:
                        ZXinv = np.linalg.inv(pZX)
                        Zy = np.dot(precond, Zy)
                if ZXinv is None:
                    # No preconditioning or the preconditioning left the
                    # matrix numerically singular.  ZX is symmetric.
                    evals, evecs = np.linalg.eigh(ZX)
                    if self.rank == 0:
                        fn_out = os.path.join(self.dir,
                                              'eigen{}.pck'.format(siter))
                        with open(fn_out, 'wb') as f:
                            pickle.dump([evals, evecs, ZX], f, protocol=2)
                        print('eigendecomposition stored in {}'.format(fn_out))
                    bad_evals = evals < np.amax(evals) * 1e-10
                    nbad = np.sum(bad_evals)
                    if self.rank == 0:
                        print('Discarding {} singular eigenmodes in the '
                              'template covariance matrix.\n'
                              'All eigenvalues: {}'.format(nbad, evals))
                    if nbad > 0:
                        # Regularized inverse
                        evals[bad_evals] = 0
                        good_evals = np.logical_not(bad_evals)
                        evals[good_evals] = 1 / evals[good_evals]
                        ZXinv = np.dot(evecs, np.dot(np.diag(evals), evecs.T))
                    else:
                        ZXinv = np.linalg.inv(ZX)
            except np.linalg.linalg.LinAlgError as e:
                raise Exception(
                    'ERROR: failed to invert the {} template '
                    'covariance matrix:\n{}\n'.format(
                        self.ntemplate, ZX)
                ) from e
            x = np.dot(ZXinv, Zy)
        else:
            ZXinv = np.zeros_like(ZX)
            x = np.zeros(nn)

        self.comm.Bcast(x, root=0)
        self.comm.Bcast(ZXinv, root=0)
        self.time_invert_cov += MPI.Wtime() - t1

        amplitudes = np.zeros(self.ndet * self.ntemplate, dtype=float)
        amplitudes[ioff:] = x
        amplitudes = amplitudes.reshape([self.ndet, self.ntemplate])

        # Measure the improvement in chi-squared

        old_chisq = np.sum(clean_toi ** 2)
        for idet in range(self.ndet - ioff_det):
            for template, amplitude in zip(
                    clean_template_tois[idet], amplitudes[idet + ioff_det]):
                clean_toi -= amplitude * template
            del template
        new_chisq = np.sum(clean_toi ** 2)

        old_chisq = self.comm.allreduce(old_chisq, op=MPI.SUM)
        new_chisq = self.comm.allreduce(new_chisq, op=MPI.SUM)
        if self.rank == 0:
            print('old chisq = {},\nnew chisq = {}.\nnew / old = {}'.format(
                old_chisq, new_chisq, new_chisq / old_chisq), flush=True)

        del clean_template_tois
        del clean_IV_template_tois
        self.cache.clear('.*clean_template.*')

        ioff = 0

        if do_offset:
            if self.nnz != 2:
                mean_offset = np.median(amplitudes[:, 0])
                amplitudes[:, ioff] -= mean_offset
            ioff += 1

        if do_gain:
            # Translate the template amplitude to a gain correction
            gains = 1 - amplitudes[:, ioff]
            self.gainmean = np.mean(gains)
            if self.verbose and self.rank == 0:
                print('Removing mean gain correction: {}'.format(
                    self.gainmean))
            # Recalibrate templates with the mean gain
            amplitudes[:, ioff] = gains / self.gainmean
            ZXinv /= self.gainmean
            ioff += 1
        else:
            self.gainmean = 1

        if do_pol_eff:
            # Translate the template amplitude to a polarization
            # efficiency correction
            pol_effs = 1 + amplitudes[:, ioff]
            self.pol_eff_mean = np.mean(pol_effs)
            if self.verbose and self.rank == 0:
                print('Removing mean pol efficiency: {}'.format(
                    self.pol_eff_mean))
            amplitudes[:, ioff] = pol_effs / self.pol_eff_mean
            ioff += 1

        if do_pol_angle:
            # Translate the template amplitude to a polarization
            # angle correction
            pol_angles = amplitudes[:, ioff] / degree
            self.pol_ang_mean = np.mean(pol_angles)
            if self.verbose and self.rank == 0:
                print('Removing mean pol angle: {}'.format(
                    self.pol_ang_mean))
            amplitudes[:, ioff] = pol_angles - self.pol_ang_mean
            ioff += 1

        # Minimize the correction by subtracting the mean amplitude for
        # each template.

        amplitudes[:, ioff:] /= self.gainmean

        if self.ndet > 1 and self.nnz != 2:
            for itemplate in range(ioff, self.ntemplate):
                amplitudes[:, itemplate] -= np.mean(amplitudes[:, itemplate])

        self.time_solve_amplitudes += MPI.Wtime() - t1

        destriped_toi = self.subtract_baselines(
            toi, weights, amplitudes, templates, do_offset, do_gain,
            do_pol_eff, do_pol_angle, in_place=in_place)

        # Measure the RMS of the baseline-subtracted TOI by subtracting the
        # signal estimate (includes any possible overall offset).
        # This produces a slight overestimate of the noise RMS as the binned
        # and scanned signal estimate is also noisy.

        rms = self.get_rms(destriped_toi, flag, pixels, weights)

        self.time_destripe += MPI.Wtime() - t0

        # report

        if self.verbose:
            destriped_map = self.toi2map(destriped_toi, flag, pixels, weights)
            destriped_map = self.cc_multiply(destriped_map)
            if save_maps:
                destriped_map = self.collect_map(destriped_map).T
            if self.rank == 0:
                if save_maps:
                    print('Destriped map RMS:')
                    for m in destriped_map:
                        print('{}'.format(np.nanstd(m.ravel())))
                    print('')
                print('residual TOI RMS = ', rms)
                print('TOI template amplitudes: ')
                for idet in range(self.ndet):
                    print('  detector # {}: {}'.format(idet, amplitudes[idet]))
                print('', flush=True)
                if save_maps:
                    t2 = MPI.Wtime()
                    fn = os.path.join(self.dir,
                                      'destriped_map{}.fits'.format(siter))
                    try:
                        hp.write_map(fn, destriped_map, nest=True)
                    except Exception:
                        hp.write_map(fn, destriped_map, nest=True,
                                     overwrite=True)
                    print('Saved destriped map to {}'.format(fn), flush=True)
                    self.time_write_map += MPI.Wtime() - t2
            self.report_timing()

        del self.cc
        del self.rcond
        del self.pixels
        self.cache.clear()
        self.cc = None
        self.rcond = None
        self.my_npix = None
        self.local2global = None
        self.global2local = None
        self.pixels = None
        self.my_own = None
        self.commonpix = None
        self.counts = None
        self.displs = None
        self.bufsize = None

        if return_baselines:
            return destriped_toi, amplitudes, ZXinv * rms ** 2
        else:
            return destriped_toi

    def initialize_alltoallv(self, pixels, flag):

        """Build the auxiliary arrays to facilitate fast map reduction"""

        t1 = MPI.Wtime()

        my_hitmap = self.cache.create('my_hitmap', np.int32, (self.npix,))
        for idet in range(self.ndet):
            good = np.logical_not(flag[idet])
            destripe_tools.fast_hit_binning(
                pixels[idet][good].astype(np.int32) // self.ndegrade, my_hitmap)
        my_hits = my_hitmap.astype(np.bool).copy()
        del my_hitmap
        self.cache.destroy('my_hitmap')

        self.local2global = np.arange(self.npix,
                                      dtype=np.int32)[my_hits].copy()
        self.my_npix = self.local2global.size
        self.global2local = -np.ones(self.npix, dtype=np.int32)
        self.global2local[my_hits] = np.arange(self.my_npix, dtype=np.int32)
        del my_hits

        self.pixels = []
        for idet in range(self.ndet):
            good = np.logical_not(flag[idet])
            local_pixels = destripe_tools.fast_scanning_int32(
                pixels[idet][good].astype(np.int32) // self.ndegrade,
                self.global2local)
            cachename = 'local_pixels_{}'.format(idet)
            if len(local_pixels) > 0:
                self.pixels.append(self.cache.put(
                        cachename, local_pixels, replace=True))
            else:
                self.pixels.append(local_pixels)

        all_hitpix = self.comm.allgather(self.local2global)

        self.commonpix = []
        counts = []
        displs = []
        self.my_own = self.local2global.copy()
        for itask, hitpix in enumerate(all_hitpix):
            if itask < self.rank and len(self.my_own) > 0:
                # Each pixel is assigned to the process with the lowest
                # rank that sees it
                self.my_own = np.setdiff1d(self.my_own, hitpix).copy()
            # Pixels seen by both processes are the ones that need to
            # be communicated
            common = np.intersect1d(self.local2global, hitpix,
                                    assume_unique=True)
            common = self.global2local[common]
            self.commonpix.append(common.copy())
            ncommon = common.size
            counts.append(ncommon)
            if len(displs) == 0:
                displs.append(0)
            else:
                displs.append(displs[-1] + counts[-2])
        del all_hitpix

        self.counts = np.array(counts)
        self.displs = np.array(displs)
        self.bufsize = np.sum(counts)
        self.time_init_alltoallv += MPI.Wtime() - t1

    def collect_map(self, sigmap):

        """ Collect the distributed map to the root process """

        t1 = MPI.Wtime()

        nnz = sigmap.shape[1]
        full_map = np.zeros([self.npix, nnz], dtype=np.float64)

        for my_pix in self.my_own:
            local_pix = self.global2local[my_pix]
            full_map[my_pix] = sigmap[local_pix]

        self.comm.Allreduce(MPI.IN_PLACE, full_map, op=MPI.SUM)

        self.time_collect_map += MPI.Wtime() - t1

        return full_map

    def get_rms(self, toi, flag, pixels, weights):

        """ Measure signal-subtracted RMS. """

        t1 = MPI.Wtime()
        noise_toi = self.apply_Z(toi, flag, pixels, weights)
        n = len(noise_toi)
        nnan = np.sum(np.isnan(noise_toi))
        if nnan != 0:
            print('{:4} : WARNING: noise TOI contains {} NaNs'
                  ''.format(self.rank, nnan))
            noise_toi[np.isnan(noise_toi)] = 0
        sqsum = np.sum(noise_toi ** 2)
        t2 = MPI.Wtime()
        n = self.comm.allreduce(n, op=MPI.SUM)
        sqsum = self.comm.allreduce(sqsum, op=MPI.SUM)
        self.time_mpi += MPI.Wtime() - t2
        if n == 0:
            rms = 0
        else:
            rms = np.sqrt(sqsum / n)
        self.time_rms += MPI.Wtime() - t1

        return rms

    def get_cc(self, flag, pixels, weights):

        """ Compute the pixel matrices. """

        t1 = MPI.Wtime()
        if self.cc is None:
            self.cc = np.zeros([self.my_npix, self.nnz, self.nnz],
                               dtype=np.float64)
            if self.my_npix > 0:
                self.cc = self.cache.put('cc', self.cc, replace=True)
        for idet in range(self.ndet):
            good = np.logical_not(flag[idet])
            destripe_tools.fast_weight_binning(
                self.pixels[idet], weights[idet][good].astype(np.float32),
                self.cc)
        self.time_accumulate_cc += MPI.Wtime() - t1

        nnan = np.sum(np.isnan(self.cc))
        if nnan != 0:
            print('{:4} : WARNING: accumulated cc contains {} NaNs'.format(
                    self.rank, nnan))
            self.cc[np.isnan(self.cc)] = 0

        self.allreduce(self.cc)

        nnan = np.sum(np.isnan(self.cc))
        if nnan != 0:
            print('{:4} : WARNING: allreduced cc contains {} NaNs'.format(
                    self.rank, nnan))
            self.cc[np.isnan(self.cc)] = 0

        # Now invert the pixel matrices

        t1 = MPI.Wtime()
        if self.rcond is None:
            self.rcond = np.zeros(self.my_npix, dtype=np.float64)
            if self.my_npix > 0:
                self.rcond = self.cache.put('rcond', self.rcond, replace=True)
        destripe_tools.fast_cc_invert(
            self.cc, self.cc, self.rcond, self.threshold, 0, 1)
        self.time_invert_cc += MPI.Wtime() - t1

        # Set the uninvertible pixel matrices to NaN so we can track them

        self.cc[self.rcond < self.threshold, :] = np.nan

        return

    def cc_multiply(self, sigmap_in):

        """ Multiply map with precomputed pixel matrices. """

        t1 = MPI.Wtime()
        sigmap = np.zeros([self.my_npix, self.nnz])
        destripe_tools.fast_cc_multiply(sigmap, self.cc, sigmap_in, 0, 1)
        self.time_cc_multiply += MPI.Wtime() - t1

        return sigmap

    def apply_Z(self, toi, flag, pixels, weights, det=None, hstack=True):

        """Apply Z = I - P ( P^T P )^-1 P^T to the TOI"""

        t1 = MPI.Wtime()

        zmap = self.toi2map(toi, flag, pixels, weights, det=det)
        if np.any(np.isnan(zmap)):
            nan = np.isnan(zmap)
            nnan = np.sum(nan)
            print('{:4} : WARNING : nulling {} NaNs in zmap PRIOR to '
                  'cc_multiply'.format(self.rank, nnan))
            zmap[nan] = 0

        # Singular pixel matrices are set to NaN and will flag the samples
        # that fall into them

        zmap = self.cc_multiply(zmap)

        ztoi = self.map2toi(zmap, flag, pixels, weights)

        """
        if det is None:
            for idet in range(self.ndet):
                nan = np.isnan(ztoi[idet])
                #nan[flag[det]] = False
                nnan = np.sum(nan[flag[idet]==0])
                if nnan != 0:
                    # Because the destriping flag is interpolated from the
                    # mask, we will have a few samples falling into pixels
                    # intended to be flagged.
                    #print('{:4} : ztoi has {} NaNs, det = {}'.format(
                        self.rank, nnan, idet), flush=True)
                    #ztoi[idet][nan] = 0
                    #flag[idet][nan] = True

            for idet in range(self.ndet):
                nan = np.isnan(toi[idet])
                #nan[flag[det]] = False
                nnan = np.sum(nan[flag[idet]==0])
                if nnan != 0:
                    print('{:4} : toi has {} NaNs, det = {}'.format(
                        self.rank, nnan, idet), flush=True)
                    #toi[idet][nan] = 0
                    #flag[idet][nan] = True
        """

        if det is None:
            cleaned_toi = []
            for idet in range(self.ndet):
                cleaned_toi.append(toi[idet] - ztoi[idet])
        else:
            cleaned_toi = []
            for idet in range(self.ndet):
                if idet == det:
                    cleaned_toi.append(toi - ztoi[idet])
                else:
                    cleaned_toi.append(-ztoi[idet])

        # Only return valid (downsampled) samples so no further flagging
        # is necessary

        t2 = MPI.Wtime()
        if self.lowpassfreq is not None and self.lowpassfreq != 0:
            for idet in range(self.ndet):
                # Mask out samples falling into failed pixels
                flg = flag[idet].copy()
                flg[np.isnan(cleaned_toi[idet])] = True
                (cleaned_toi[idet], hits
                 ) = destripe_tools.flagged_running_average_with_downsample32(
                        cleaned_toi[idet], flg.astype(np.uint8), self.naverage)
                if hstack:
                    cleaned_toi[idet] = cleaned_toi[idet][hits != 0].copy()
                if np.any(np.isnan(cleaned_toi[idet])):
                    nan = np.isnan(cleaned_toi[idet])
                    nnan = np.sum(nan)
                    print('{:4} : WARNING : nulling {} NaNs in cleaned toi '
                          'AFTER flagged_running_average, det = {}'.format(
                              self.rank, nnan, idet), flush=True)
                    cleaned_toi[idet][nan] = 0
        else:
            for idet in range(self.ndet):
                # Mask out samples falling into failed pixels
                good = np.logical_not(flag[idet])
                good[np.isnan(cleaned_toi[idet])] = False
                cleaned_toi[idet] = cleaned_toi[idet][good].astype(np.float32)
        self.time_lowpass += MPI.Wtime() - t2

        if hstack:
            cleaned_toi = np.hstack(cleaned_toi)

        self.time_apply_Z += MPI.Wtime() - t1

        return cleaned_toi

    def allreduce(self, sigmap):

        """Sum over sigmap and broadcast the results"""

        t1 = MPI.Wtime()

        nnz = np.prod(np.shape(sigmap)[1:])
        sendbuf = np.zeros(self.bufsize * nnz, dtype=np.float64) + np.nan
        recvbuf = np.zeros(self.bufsize * nnz, dtype=np.float64) + np.nan

        # Pack the send buffer

        offset = 0
        for common in self.commonpix:
            n = common.size * nnz
            sendbuf[offset:offset + n] = sigmap[common].ravel()
            offset += n

        # Communicate

        self.comm.Alltoallv(
            (sendbuf, self.counts * nnz, self.displs * nnz, MPI.DOUBLE),
            (recvbuf, self.counts * nnz, self.displs * nnz, MPI.DOUBLE))

        # Unpack the recv buffer

        sigmap_out = np.zeros_like(sigmap)
        offset = 0
        for common in self.commonpix:
            n = common.size * nnz
            shp = list(sigmap.shape)
            shp[0] = -1
            sigmap_out[common] += recvbuf[offset:offset + n].reshape(shp)
            offset += n

        sigmap[:] = sigmap_out

        self.time_alltoallv += MPI.Wtime() - t1

        return sigmap

    def toi2map(self, toi, flag, pixels, weights, det=None):

        """Bin the TOI based on the pointing in pixels."""

        t1 = MPI.Wtime()
        sigmap = np.zeros([self.my_npix, self.nnz])
        for idet in range(self.ndet):
            if det is not None and idet != det:
                continue
            good = np.logical_not(flag[idet])
            if det is not None:
                ttemp = toi[good]
            else:
                ttemp = toi[idet][good]
            destripe_tools.fast_binning_pol(
                ttemp.astype(np.float64), self.pixels[idet],
                weights[idet][good].astype(np.float32), sigmap)
        self.time_toi2map += MPI.Wtime() - t1

        self.allreduce(sigmap)

        return sigmap

    def map2toi(self, sigmap, flag, pixels, weights, det=None,
                pol_eff=False, pol_angle=False):

        """Scan TOI from the map based on a list of pixel vectors."""

        t1 = MPI.Wtime()
        if det is None:
            toi = np.zeros([self.ndet, self.nsamp], dtype=np.float64)
        else:
            toi = np.zeros(self.nsamp, dtype=np.float64)
        for idet in range(self.ndet):
            if det is not None and idet != det:
                continue
            good = np.logical_not(flag[idet])
            ngood = np.sum(good)
            toivec = np.zeros(ngood, dtype=np.float64)
            if pol_eff:
                destripe_tools.fast_scanning_pol_eff(
                    toivec, self.pixels[idet],
                    weights[idet][good].astype(np.float64),
                    sigmap.astype(np.float64))
            elif pol_angle:
                destripe_tools.fast_scanning_pol_angle(
                    toivec, self.pixels[idet],
                    weights[idet][good].astype(np.float64),
                    sigmap.astype(np.float64))
            else:
                destripe_tools.fast_scanning_pol(
                    toivec, self.pixels[idet],
                    weights[idet][good].astype(np.float32),
                    sigmap.astype(np.float64))
            if det is None:
                toi[idet, good] = toivec
            else:
                toi[good] = toivec
        self.time_map2toi += MPI.Wtime() - t1

        return toi

    def subtract_baselines(
            self, toi, weights, amplitudes, templates, do_offset, do_gain,
            do_pol_eff, do_pol_angle, in_place=False):

        """Subtract the templates with the best fit amplitudes."""

        # FIXME: The polarization efficiency and angle templates are not
        # subtracted. The right way to handle them is to modify the
        # pointing weights.

        t1 = MPI.Wtime()

        if in_place:
            clean_toi = toi
        else:
            clean_toi = np.zeros([self.ndet, self.nsamp], dtype=np.float64)
            for idet in range(self.ndet):
                clean_toi[idet][:] = toi[idet]

        # templates

        ioff = 0
        if do_offset:
            ioffset = ioff
            ioff += 1
        if do_gain:
            igain = ioff
            ioff += 1
        if do_pol_eff:
            ipol_eff = ioff
            ioff += 1
        if do_pol_angle:
            ipol_angle = ioff
            ioff += 1

        for idet in range(self.ndet):
            for amplitude, template in zip(
                    amplitudes[idet][ioff:], templates[idet]):
                clean_toi[idet] -= amplitude * template
            if do_gain:
                clean_toi[idet] *= amplitudes[idet][igain]
            if do_offset:
                # This is not exactly correct if the data are being
                # recalibrated but the error is only as large as the
                # calibration factor
                clean_toi[idet] -= amplitudes[idet][ioffset]
            if do_pol_eff and in_place:
                # Adjust the pointing weights accordingly
                pol_eff = amplitudes[idet][ipol_eff]
                weights[idet][:, 1:] *= pol_eff
            if do_pol_angle and in_place:
                # Adjust the pointing weights accordingly
                pol_ang = amplitudes[idet][ipol_angle]
                qweight = weights[idet][:, 1].ravel().copy()
                uweight = weights[idet][:, 2].ravel().copy()
                costerm = np.cos(2 * pol_ang * degree)
                sinterm = np.sin(2 * pol_ang * degree)
                weights[idet][:, 1] = qweight * costerm - uweight * sinterm
                weights[idet][:, 2] = uweight * costerm + qweight * sinterm

        self.time_subtract_baselines += MPI.Wtime() - t1

        return clean_toi

    def report_timing(self):

        """Report timing."""

        def report(name, t):
            ttot = np.array(self.comm.gather(t, root=0))
            if self.rank == 0:
                print('{} time mean {:7.2f} s min {:7.2f} s,'
                      ' max {:7.2f}s'.format(name, np.mean(ttot),
                                             np.amin(ttot), np.amax(ttot)))

        report('Total destriping .......', self.time_destripe)
        report('  - accumulate_cc ......', self.time_accumulate_cc)
        report('  - invert_cc ..........', self.time_invert_cc)
        report('  - clean templates ....', self.time_clean_templates)
        report('  - build covariance ...', self.time_build_cov)
        report('  - invert covariance ..', self.time_invert_cov)
        report('  - solve amplitudes ...', self.time_solve_amplitudes)
        report('  - apply Z ............', self.time_apply_Z)
        report('    - toi2map ..........', self.time_toi2map)
        report('    - cc_multiply ..... ', self.time_cc_multiply)
        report('    - map2toi ..........', self.time_map2toi)
        report('    - lowpass ..........', self.time_lowpass)
        report('  - subtract baselines .', self.time_subtract_baselines)
        report('  - get rms ............', self.time_rms)
        report('  - MPI ................', self.time_mpi)
        report('  - alltoallv ..........', self.time_alltoallv)
        report('  - init_alltoallv .....', self.time_init_alltoallv)
        report('  - collect_map ........', self.time_collect_map)
        report('  - write_map ..........', self.time_write_map)
