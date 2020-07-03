# Copyright (c) 2017-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
from toast_planck.preproc_modules.filters import flagged_running_average

from scipy.signal import fftconvolve
import scipy.sparse.linalg

import astropy.io.fits as pf
import healpy as hp
import numpy as np
import toast.cache as tc
from toast.mpi import MPI
import toast.timing as timing

from .destripe_tools import (
    fast_hit_binning, fast_scanning_int32, fast_weight_binning_with_mult,
    fast_cc_invert, fast_cc_multiply, collect_buf, sample_buf,
    fast_binning_pol_with_mult, fast_scanning_pol64)


# from memory_profiler import profile
class UltimateDestriper():
    """
    Polarized, multi-detector destriper that uses ring-binned data and accepts
    varying lenghts of baselines for the different templates.
    """

    def __init__(
            self, npix, nnz, mpicomm, threshold=1e-2, itermax=100, cglimit=1e-6,
            ndegrade=1, dir_out='./', precond=True,
            stationary_templates=None, stationary_detectors=None,
            prefix='destriper: '):
        """
        Initialize the Ultimate Destriper.

        Args:
            npix (int): Number of pixels in one sky component.
            nnz (int): Number of sky components.
            mpicomm (MPIComm): MPI communicator.
            threshold (float): Reciprocal condition number limit for
                pixels.
            itermax (int): Maximum number of CG iterations.
            cglimit (float): Required improvement in the CG residual.
            ndegrade (int): Downgrading factor to apply to the pixel
                numbers.
            dir_out (str): Output directory for maps.
            precond (bool): Construct and apply a simple diagonal
                preconditioner.
            stationary_templates (vector of bool): Which templates have
                an unconstrained mean.  Mean will be subtracted.
            stationary_detectors (vector of bool): Which detectors to
                consider for the mean subtraction of stationary templates.

        """
        self.ndegrade = np.int(ndegrade)
        if self.ndegrade < 1:
            raise Exception('ERROR: ndegrade cannot be smaller than one: {}'
                            ''.format(ndegrade))
        self.npix = npix // ndegrade
        self.nside = hp.npix2nside(self.npix)
        self.nnz = nnz
        self.threshold = threshold
        self.itermax = itermax
        self.cglimit = cglimit
        self.comm = mpicomm
        self.dir = dir_out
        self.precond = precond
        if stationary_templates is None:
            self.stationary_templates = []
        else:
            self.stationary_templates = stationary_templates
        if stationary_detectors is None:
            self.stationary_detectors = None
        else:
            self.stationary_detectors = np.array(stationary_detectors)
        self.prefix = prefix

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
        self.weights = None
        self.hits = None
        self.noiseweights = None
        self.my_own = None
        self.commonpix = None
        self.counts = None
        self.displs = None
        self.bufsize = None

        self.cache = tc.Cache()

        # Profiling

        self.reset_timers()

    def __del__(self):
        try:
            self.cache.clear()
        except Exception:
            pass

    def reset_timers(self):

        self.time_subtract_baselines = 0.
        self.time_apply_noisematrix = 0.
        self.time_apply_Z = 0.
        self.time_map2toi = 0.
        self.time_toi2map = 0.
        self.time_toi2base = 0.
        self.time_reducebase = 0.
        self.time_base2toi = 0.
        self.time_destripe = 0.
        self.time_lhs = 0.
        self.time_rhs = 0.
        self.time_precond_init = 0.
        self.time_iterate = 0.
        self.time_precond = 0.
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
        self.time_alltoallv_MPI = 0.
        self.time_alltoallv_pack = 0.
        self.time_alltoallv_unpack = 0.
        self.time_init_alltoallv = 0.
        self.time_collect_map = 0.
        self.time_write_map = 0.

    def flag_outliers(
            self, rings, dets, templates, namplitude, namplitude_tot,
            template_offsets,
            verbose=False, save_maps=False, siter='', threshold=10.):
        """
        Subtract binned signal and find outliers from the residual TOD
        """

        self.verbose = verbose

        self.reset_timers()

        if len(rings) == 0:
            raise Exception(
                'ERROR: the destriper cannot handle empty ring sets')

        self.my_rings = sorted(rings.keys())
        self.my_dets = dets

        self.ndet = len(self.my_dets)
        self.namplitude = namplitude
        self.namplitude_tot = namplitude_tot
        self.ndet_tot = namplitude_tot // namplitude

        self.template_names = sorted(template_offsets.keys())
        self.ntemplate = len(self.template_names)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                t1 = MPI.Wtime()
                print(self.prefix + 'Initializing alltoallv', flush=True)
            self.comm.Barrier()

        self.initialize_alltoallv(rings)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                t2 = MPI.Wtime()
                print(self.prefix + 'Initialized in {:.2f} s. Computing the '
                      'pixel covariance matrices'.format(t2 - t1), flush=True)
            self.comm.Barrier()

        self.get_cc()

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                t1 = MPI.Wtime()
                print(self.prefix + 'Computed in {:.2f} s. Binning full '
                      'initial map'.format(t1 - t2), flush=True)
            self.comm.Barrier()

        # Put the signal references in a standalone TOI object

        toi = []
        for iring, ring in enumerate(self.my_rings):
            toi.append([])
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    toi[iring].append(None)
                    continue
                toi[iring].append(rings[iring][det].signal)

        initial_map = self.toi2map(toi)
        initial_map = self.cc_multiply(initial_map)

        if save_maps:
            global_map = self.collect_map(initial_map).T

        if self.verbose:
            if self.rank == 0:
                t2 = MPI.Wtime()
                print(self.prefix + 'Binned in {:.2f} s. Computing initial RMS'
                      ''.format(t2 - t1), flush=True)
            self.comm.Barrier()

            rms = self.get_rms(toi, rings)

            if self.rank == 0:
                print(self.prefix + '   Input TOI RMS = ', rms, flush=True)
                if save_maps:
                    print(self.prefix + 'Initial map RMS:', flush=True)
                    for m in global_map:
                        good = m != 0
                        print(self.prefix + '{}'.format(np.std(m[good])))
                    print(self.prefix + '', flush=True)
                    t2 = MPI.Wtime()
                    fn = os.path.join(self.dir,
                                      'full_initial_map{}.fits'.format(siter))
                    try:
                        hp.write_map(fn, global_map, nest=True)
                    except Exception:
                        hp.write_map(fn, global_map, nest=True, overwrite=True)
                    print(self.prefix +
                          'Saved full initial map to {}'.format(fn),
                          flush=True)
                    self.time_write_map += MPI.Wtime() - t2

        if save_maps:
            del global_map
            self.cache.destroy('global_map')

        if self.verbose:
            if self.rank == 0:
                print(self.prefix + 'Cleaning TOI', flush=True)
            self.comm.Barrier()

        clean_toi = self.apply_Z(toi, rings)

        for idet, det in enumerate(self.my_dets):
            my_vars = []
            for iring, ring in enumerate(self.my_rings):
                if self.pixels[iring][idet] is None:
                    continue
                ring_hits = rings[ring][det].hits * rings[ring][det].mask
                if ring_hits.size < 10:
                    my_vars.append(1e10)
                    continue
                hitsum = np.sum(ring_hits)
                ring_toi = clean_toi[iring][idet]
                # The ring might have a large offset from preprocessing
                weighted_mean = np.sum(ring_toi * ring_hits) / hitsum
                var_mean = np.mean((ring_toi - weighted_mean) ** 2 * ring_hits)
                my_vars.append(var_mean)
            my_n = len(my_vars)
            all_ranks = np.hstack(
                self.comm.allgather(np.ones(my_n) * self.rank))
            all_vars = np.hstack(self.comm.allgather(my_vars))
            wmean = min(101, len(all_vars) // 10 + 1)
            werr = min(1001, len(all_vars) // 10 + 1)
            good = np.isfinite(all_vars)
            for _ in range(10):
                # Smooth mean
                var_mean = flagged_running_average(
                    all_vars, np.logical_not(good), wmean)
                # Smooth error estimate
                var_err = np.sqrt(flagged_running_average(
                    (all_vars - var_mean) ** 2, np.logical_not(good), werr))
                # Detect outliers
                ngood1 = np.sum(good)
                good[np.abs(all_vars - var_mean) > var_err * threshold] = False
                ngood2 = np.sum(good)
                if ngood1 == ngood2:
                    break
            # Flag the odd intervals that passed the test but are embedded
            # in a suspicious region.
            bad = np.logical_not(good)
            if self.rank == 0:
                nbad = np.sum(bad)
                print(self.prefix + '{} / {} rings for {} are {} sigma outliers'
                      ''.format(nbad, bad.size, det, threshold), flush=True)
            kernel = np.ones(5) / 5
            rbad = fftconvolve(bad, kernel, mode='same')
            good[rbad > 0.7] = False
            """
            # Flag lonely steps
            bad = np.logical_not(good)
            nbad = np.sum(bad)
            kernel = np.ones(10) / 10
            rbad = fftconvolve(bad, kernel, mode='same')
            good[rbad > 0.8] = False
            """
            if self.rank == 0:
                fn = os.path.join(self.dir,
                                  'ring_var{}_{}.pck'.format(siter, det))
                with open(fn, 'wb') as f:
                    pickle.dump([all_vars, good], f, protocol=2)
                print(self.prefix + 'step variances stored in {}'.format(fn))
                print(self.prefix + 'Flagging {} / {} outlier steps for {}'
                      ''.format(nbad, all_vars.size, det), flush=True)

            mine = all_ranks == self.rank
            my_good = good[mine]
            i = -1
            for iring, ring in enumerate(self.my_rings):
                if self.pixels[iring][idet] is None:
                    continue
                i += 1
                if not my_good[i]:
                    del rings[ring][det]
                    del templates[ring][det]

        del self.cc
        del self.rcond
        del self.pixels
        del self.weights
        del self.hits
        self.cache.clear()
        self.cc = None
        self.rcond = None
        self.my_npix = None
        self.local2global = None
        self.global2local = None
        self.pixels = None
        self.weights = None
        self.hits = None
        self.my_own = None
        self.commonpix = None
        self.counts = None
        self.displs = None
        self.bufsize = None
        return

    def normalize_baselines(self, baselines, template_offsets):
        """Normalize baselines that correspond to stationary templates.

        The frequency mean of these templates is unconstrained by the
        destriping equation.
        """

        if self.rank == 0:
            baselines = baselines.reshape([-1, self.namplitude])
            gainmean = None

            # Stationary templates are sky-synchronous.  Only
            # differences between detectors are meaningful so
            # we subtract the mean.

            if 'gain' in self.stationary_templates and ('gain' in
                                                        template_offsets):
                offset, namp = template_offsets['gain']
                amplitudes = baselines[:, offset:offset + namp]
                good = amplitudes != 0
                gainmean = np.mean(amplitudes[good])
                amplitudes[good] -= gainmean
                baselines[:, offset:offset + namp] = amplitudes
                if self.verbose and self.rank == 0:
                    print(self.prefix + 'Normalizing gain mean by {}'
                          ''.format(gainmean), flush=True)
                # Removing power from the gain template changes the
                # other templates that have been co-added into the
                # gain template.
                for name in template_offsets:
                    for idet, det in enumerate(self.my_dets):
                        if name not in self.best_fits[det]:
                            # name was not co-added into the gain
                            # template
                            continue
                        corr = self.best_fits[det][name] * gainmean
                        # When detectors share an amplitude, the correction
                        # only applies to the leading detector, others have
                        # zero amplitude.
                        offset, namp = template_offsets[name]
                        amplitudes = baselines[idet, offset:offset + namp]
                        good = (amplitudes != 0)
                        if np.any(good) and self.verbose:
                            print(self.prefix +
                                  'Normalizing {:8} {} mean by {}'
                                  ''.format(det, name, corr), flush=True)
                        amplitudes[good] += corr
                        baselines[idet, offset:offset + namp] = amplitudes

            # Treat all templates except the gain
            for name in self.stationary_templates:
                if name == 'gain':
                    continue
                if name in template_offsets:
                    offset, namp = template_offsets[name]
                    amplitudes = baselines[:, offset:offset + namp]
                    good = amplitudes != 0
                    freqmean = np.mean(amplitudes[good])
                    amplitudes[good] -= freqmean
                    baselines[:, offset:offset + namp] = amplitudes
                    if self.verbose and self.rank == 0:
                        print(self.prefix + 'Normalizing {} mean by {}'
                              ''.format(name, freqmean), flush=True)

            baselines = baselines.ravel()

        self.comm.Bcast(baselines)
        return

    def get_precond(self, templates):
        """
        Build the diagonal preconditioning function.
        """

        t1 = MPI.Wtime()

        self.nhitinv = np.zeros(self.namplitude_tot)
        for iring, ring in enumerate(self.my_rings):
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                hits = self.noiseweights[iring][idet]
                for template in templates[ring][det].values():
                    if template.offset is None:
                        continue
                    nn = np.dot(template.template * hits, template.template)
                    self.nhitinv[template.offset] += nn
        self.comm.Allreduce(MPI.IN_PLACE, self.nhitinv, op=MPI.SUM)
        good = self.nhitinv != 0
        self.nhitinv[good] = 1 / self.nhitinv[good]

        self.comm.Barrier()
        self.time_precond_init += MPI.Wtime() - t1

        def precond(baselines):
            t1 = MPI.Wtime()
            precond_baselines = self.nhitinv * baselines
            self.time_precond += MPI.Wtime() - t1
            return precond_baselines

        return precond

    def get_precond2(self, templates):
        """
        Build the dense preconditioning function.
        """

        t1 = MPI.Wtime()

        # Count the number of baseline amplitudes not associated with
        # offsets (offsets were projected out from other templates)
        self.non_offset = np.zeros(self.namplitude_tot, dtype=np.bool)
        self.is_offset = np.zeros(self.namplitude_tot, dtype=np.bool)
        self.is_gain = np.zeros(self.namplitude_tot, dtype=np.bool)
        for iring, ring in enumerate(self.my_rings):
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                for name, template in templates[ring][det].items():
                    if template.offset is None:
                        continue
                    if name == 'offset':
                        self.is_offset[template.offset] = True
                    elif name in ['gain', 'distortion']:
                        self.is_gain[template.offset] = True
                    else:
                        self.non_offset[template.offset] = True
        self.comm.Allreduce(MPI.IN_PLACE, self.is_offset, op=MPI.LOR)
        self.comm.Allreduce(MPI.IN_PLACE, self.is_gain, op=MPI.LOR)
        self.comm.Allreduce(MPI.IN_PLACE, self.non_offset, op=MPI.LOR)
        self.nnon_offset = np.sum(self.non_offset)
        self.ngain = np.sum(self.is_gain)
        if self.verbose and self.rank == 0:
            print(self.prefix + 'nnon_offset = {}, ngain = {}'.format(
                self.nnon_offset, self.ngain), flush=True)
        only_diagonal = ['offset']
        # if self.nnon_offset + self.ngain < 2:
        if self.nnon_offset + self.ngain < 2000:
            # Include the gain templates in the dense precoditioner
            if self.verbose and self.rank == 0:
                print(self.prefix +
                      'Including gains in the dense preconditioner',
                      flush=True)
            self.non_offset = np.logical_or(self.non_offset, self.is_gain)
            self.nnon_offset += self.ngain
        else:
            # Too many gains to invert preconditioner
            only_diagonal.append('gain')
            only_diagonal.append('distortion')
            if self.verbose and self.rank == 0:
                print(self.prefix +
                      'NOT including gains in the dense preconditioner',
                      flush=True)
            self.is_offset = np.logical_or(self.is_offset, self.is_gain)

        # Compress the non-offset indices

        self.non_offset_index = -np.ones(self.namplitude_tot, dtype=np.int)
        self.non_offset_index[self.non_offset] = np.arange(
            self.nnon_offset, dtype=np.int)

        # Then build the preconditioner matrix from dot products of
        # the destriping templates
        self.precond_matrix = np.zeros([self.nnon_offset, self.nnon_offset])
        # This is the diagonal preconditioner for offsets and
        # optionally for gains
        self.nhitinv = np.zeros(self.namplitude_tot)
        for iring, ring in enumerate(self.my_rings):
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                hits = self.noiseweights[iring][idet]
                for rowname, rowtemplate in templates[ring][det].items():
                    if rowtemplate.offset is None:
                        continue
                    weighted_row = rowtemplate.template * hits
                    if rowname in only_diagonal:
                        # Add to the diagonal preconditioner
                        self.nhitinv[rowtemplate.offset] += np.dot(
                            weighted_row, rowtemplate.template)
                    else:
                        # Add to the dense preconditioner
                        rowindex = self.non_offset_index[rowtemplate.offset]
                        for colname, coltemplate in templates[ring][det
                                                                    ].items():
                            if colname in only_diagonal or (coltemplate.offset
                                                            is None):
                                continue
                            colindex = self.non_offset_index[
                                coltemplate.offset]
                            if colindex > rowindex:
                                # Don't take the dot products twice
                                # for a symmetric matrix
                                continue
                            dp = np.dot(weighted_row, coltemplate.template)
                            self.precond_matrix[rowindex, colindex] += dp
                            if colindex != rowindex:
                                self.precond_matrix[colindex, rowindex] += dp

        self.comm.Allreduce(MPI.IN_PLACE, self.precond_matrix, op=MPI.SUM)
        if self.verbose and self.rank == 0:
            print(self.prefix + 'Inverting preconditioner')
        if False:
            # Dense matrix inverse
            self.precond_matrix = np.linalg.inv(self.precond_matrix)
        else:
            # Sparse matrix inverse
            mat = scipy.sparse.csc_matrix(self.precond_matrix)
            try:
                self.precond_matrix = scipy.sparse.linalg.inv(mat)
            except Exception as e:
                if self.comm.rank == 0:
                    self.precond_matrix.tofile('failed_precond_matrix.dat')
                    print(
                        'ERROR: failed to invert the preconditioner matrix: '
                        '{}'.format(e), flush=True)
                    self.comm.Abort()
                self.comm.Barrier()

        self.comm.Allreduce(MPI.IN_PLACE, self.nhitinv, op=MPI.SUM)
        good = self.nhitinv != 0
        self.nhitinv[good] = 1 / self.nhitinv[good]

        self.comm.Barrier()
        self.time_precond_init += MPI.Wtime() - t1

        def precond(baselines):
            t1 = MPI.Wtime()
            precond_baselines = np.zeros_like(baselines)
            precond_baselines[self.is_offset] = self.nhitinv[
                self.is_offset] * baselines[self.is_offset]
            precond_baselines[self.non_offset] = \
                self.precond_matrix.dot(baselines[self.non_offset])
            self.time_precond += MPI.Wtime() - t1
            return precond_baselines

        return precond

    def iterate(self, lhs, rhs, precond):
        """
        Perform the PCG iteration
        """

        t2 = MPI.Wtime()

        tt1 = MPI.Wtime()
        x0 = np.zeros(self.namplitude_tot, dtype=np.float64)
        r0 = rhs - lhs(x0)  # r stands for the residual
        z0 = precond(r0)
        x = x0.copy()
        r = r0.copy()
        p = z0.copy()
        rz0 = np.dot(r0, z0)
        if rz0 < 1e-10:
            if self.rank == 0:
                print(self.prefix + 'Anomalously low initial residual: {}. '
                      'NO ITERATION'.format(rz0), flush=True)
        else:
            rz = rz0
            if self.rank == 0:
                print(self.prefix + 'Initial residual = {}'.format(rz0))
            for iiter in range(self.itermax):
                if np.isnan(rz):
                    raise Exception(
                        'ERROR: rz is NaN on iteration {}'.format(iiter))
                Ap = lhs(p)
                denom = np.dot(p, Ap)
                if denom == 0:
                    raise Exception(
                        'ERROR: denom is zero on iteration {}'.format(iiter))
                alpha = rz / denom
                x += alpha * p
                r -= alpha * Ap
                z = precond(r)
                new_rz = np.dot(r, z)
                if self.rank == 0:
                    tt2 = MPI.Wtime()
                    print(self.prefix + 'Iteration {:03} : relative residual = '
                          '{:25}, {:.2f} s'.format(
                              iiter, new_rz / rz0, tt2 - tt1), flush=True)
                    tt1 = MPI.Wtime()
                if rz / rz0 < self.cglimit:
                    break
                beta = new_rz / rz
                p = z + beta * p
                rz = new_rz

        self.comm.Barrier()
        self.time_iterate += MPI.Wtime() - t2
        return x

    def destripe(
            self, rings, dets, templates, namplitude, namplitude_tot,
            template_offsets,
            verbose=False, in_place=False, siter='', save_maps=False,
            return_baselines=False, best_fits=None):
        """
        Fit the provided multi-detector polarized TOI with the given
        templates.
        """

        self.verbose = verbose

        self.reset_timers()

        t0 = MPI.Wtime()

        if len(rings) == 0:
            raise Exception(
                'ERROR: the destriper cannot handle empty ring sets')

        self.my_rings = sorted(rings.keys())
        self.my_dets = dets
        self.best_fits = best_fits

        self.ndet = len(self.my_dets)
        self.namplitude = namplitude
        self.namplitude_tot = namplitude_tot
        self.ndet_tot = namplitude_tot // namplitude

        self.template_names = sorted(template_offsets.keys())
        self.ntemplate = len(self.template_names)

        if self.ntemplate == 0:
            if self.rank == 0:
                print(self.prefix + 'No templates to fit. No destriping done.',
                      flush=True)
            return

        if self.verbose and self.rank == 0:
            print(self.prefix + 'ndet = {}, ntemplate = {}, namplitude = {} '
                  'ndet_tot = {}, namplitude_tot = {}'.format(
                      self.ndet, self.ntemplate, self.namplitude,
                      self.ndet_tot, self.namplitude_tot), flush=True)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print(self.prefix + 'Initializing alltoallv', flush=True)
            self.comm.Barrier()

        self.initialize_alltoallv(rings)

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print(self.prefix + 'Computing the pixel covariance matrices',
                      flush=True)
            self.comm.Barrier()

        self.get_cc()

        if self.verbose:
            self.comm.Barrier()
            if self.rank == 0:
                print(self.prefix + 'Binning initial map', flush=True)
            self.comm.Barrier()

        # Put the signal references in a standalone TOI object

        toi = []
        for iring, ring in enumerate(self.my_rings):
            toi.append([])
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    toi[iring].append(None)
                    continue
                toi[iring].append(rings[ring][det].signal)

        initial_map = self.toi2map(toi)
        initial_map = self.cc_multiply(initial_map)

        if save_maps:
            global_map = self.collect_map(initial_map).T

        if self.verbose:
            if self.rank == 0:
                print(self.prefix + 'Computing initial RMS', flush=True)
            self.comm.Barrier()

            rms = self.get_rms(toi, rings)

            if self.rank == 0:
                print(self.prefix + '   Input TOI RMS = ', rms, flush=True)
                if save_maps:
                    print(self.prefix + 'Initial map RMS:', flush=True)
                    for m in global_map:
                        good = m != 0
                        print(self.prefix + '{}'.format(np.std(m[good])))
                    print(self.prefix + '', flush=True)
                    t2 = MPI.Wtime()
                    fn = os.path.join(self.dir,
                                      'initial_map{}.fits'.format(siter))
                    try:
                        hp.write_map(fn, global_map, nest=True)
                    except Exception:
                        hp.write_map(fn, global_map, nest=True, overwrite=True)
                    print(self.prefix + 'Saved initial map to {}'.format(fn),
                          flush=True)
                    self.time_write_map += MPI.Wtime() - t2

        if save_maps:
            del global_map
            self.cache.destroy('global_map')

        if self.verbose:
            if self.rank == 0:
                print(self.prefix + 'Cleaning TOI', flush=True)
            self.comm.Barrier()

        # Evaluate the RHS of the destriping equation

        t1 = MPI.Wtime()
        rhs = self.toi2base(
                self.apply_noisematrix(self.apply_Z(toi, rings)),
                templates)
        self.comm.Barrier()
        self.time_rhs += MPI.Wtime() - t1

        # Define the linear operator that applies the sparse matrix
        # on the Left Hand Side

        def lhs(baselines):
            t1 = MPI.Wtime()
            b = self.toi2base(
                    self.apply_noisematrix(
                        self.apply_Z(
                            self.base2toi(baselines, rings, templates), rings)),
                    templates)
            self.comm.Barrier()
            self.time_lhs += MPI.Wtime() - t1
            return b

        # Generate the preconditioner from F^T F, the primary component of LHS

        # precond = self.get_precond(templates)
        precond = self.get_precond2(templates)

        # CG-iterate to solve lhs(baselines) = rhs

        baselines = self.iterate(lhs, rhs, precond)

        self.normalize_baselines(baselines, template_offsets)

        # Apply the normalized baselines

        destriped_toi = self.subtract_baselines(rings, templates, baselines)

        # report

        if self.verbose:
            if self.rank == 0:
                print(self.prefix + 'Computing final RMS', flush=True)
            self.comm.Barrier()

            destriped_map = self.toi2map(destriped_toi)
            destriped_map = self.cc_multiply(destriped_map)
            if save_maps:
                destriped_map = self.collect_map(destriped_map).T
            rms = self.get_rms(destriped_toi, rings)

            if self.rank == 0:
                if save_maps:
                    print(self.prefix + 'Destriped map RMS:')
                    for m in destriped_map:
                        good = m != 0
                        print(self.prefix + '{}'.format(np.std(m[good])))
                    print(self.prefix + '')
                print(self.prefix + 'residual TOI RMS = ', rms)
                print(self.prefix + '', flush=True)
                if save_maps:
                    t2 = MPI.Wtime()
                    fn = os.path.join(self.dir,
                                      'destriped_map{}.fits'.format(siter))
                    try:
                        hp.write_map(fn, destriped_map, nest=True)
                    except Exception:
                        hp.write_map(fn, destriped_map, nest=True,
                                     overwrite=True)
                    print(self.prefix + 'Saved destriped map to {}'.format(fn),
                          flush=True)
                    self.time_write_map += MPI.Wtime() - t2

                if save_maps:
                    del destriped_map
                    self.cache.destroy('global_map')

                # write the baselines to file

                hdulist = [pf.PrimaryHDU()]
                fn_base = os.path.join(self.dir,
                                       'baselines{}.fits'.format(siter))

                for name in sorted(template_offsets.keys()):
                    offset, namp = template_offsets[name]
                    cols = []
                    for idet, det in enumerate(self.my_dets):
                        col = pf.Column(
                            name=det, format='D',
                            array=baselines[offset:offset + namp])
                        cols.append(col)
                        offset += self.namplitude

                    hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols))
                    hdu.header['extname'] = name
                    hdulist.append(hdu)

                pf.HDUList(hdulist).writeto(fn_base, overwrite=True)
                print(self.prefix + 'Saved baselines to {}'.format(fn_base),
                      flush=True)

            self.comm.Barrier()
            self.time_destripe += MPI.Wtime() - t0

            self.report_timing()

        del self.cc
        del self.rcond
        del self.pixels
        del self.weights
        del self.hits
        del self.noiseweights
        self.cache.clear()
        self.cc = None
        self.rcond = None
        self.my_npix = None
        self.local2global = None
        self.global2local = None
        self.pixels = None
        self.weights = None
        self.hits = None
        self.noiseweights = None
        self.my_own = None
        self.commonpix = None
        self.counts = None
        self.displs = None
        self.bufsize = None
        if return_baselines:
            return destriped_toi, baselines
        else:
            return destriped_toi

    def toi2base(self, toi, templates):
        """
        Apply the baseline matrix to TOD.
        """

        time1 = MPI.Wtime()

        baselines = np.zeros(self.namplitude_tot)
        for iring, ring in enumerate(self.my_rings):
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                signal = toi[iring][idet]
                for template in templates[ring][det].values():
                    if template.offset is None:
                        continue
                    baselines[template.offset] += np.sum(template.template
                                                         * signal)
        self.comm.Barrier()
        time2 = MPI.Wtime()
        self.comm.Allreduce(MPI.IN_PLACE, baselines, op=MPI.SUM)
        self.time_reducebase += MPI.Wtime() - time2

        self.time_toi2base += MPI.Wtime() - time1
        return baselines

    def base2toi(self, baselines, rings, templates):
        """
        Scan baselines to TOI
        """

        t1 = MPI.Wtime()
        basetoi = []
        for iring, ring in enumerate(self.my_rings):
            basetoi.append([])
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    basetoi[iring].append(None)
                    continue
                n = self.pixels[iring][idet].size
                basetoi[iring].append(np.zeros(n, dtype=np.float64))
                for template in templates[ring][det].values():
                    if template.offset is None:
                        continue
                    amp = baselines[template.offset]
                    basetoi[iring][idet] += amp * template.template
        self.time_base2toi += MPI.Wtime() - t1
        return basetoi

    def get_global2local(self, rings):
        my_hitmap = self.cache.create('my_hitmap', np.int32, (self.npix,))
        my_hitmap[:] = 0
        for ring in self.my_rings:
            for det in self.my_dets:
                if det not in rings[ring]:
                    continue
                pixels = rings[ring][det].pixels
                fast_hit_binning(
                    pixels.astype(np.int32) // self.ndegrade, my_hitmap)
        my_hits = my_hitmap.astype(np.bool).copy()
        del my_hitmap
        self.cache.destroy('my_hitmap')

        self.local2global = np.arange(
            self.npix, dtype=np.int32)[my_hits].copy()
        self.my_npix = self.local2global.size
        self.global2local = -np.ones(self.npix, dtype=np.int32)
        self.global2local[my_hits] = np.arange(self.my_npix, dtype=np.int32)
        del my_hits
        return

    def get_hits(self, rings):
        # Create a version of the pixel numbers that references
        # the local map and collect weight and hit references into
        # lists instead of dictionaries
        self.pixels = []
        self.noiseweights = []
        self.weights = []
        self.hits = []
        for iring, ring in enumerate(self.my_rings):
            self.pixels.append([])
            self.noiseweights.append([])
            self.weights.append([])
            self.hits.append([])
            for det in self.my_dets:
                if det not in rings[ring]:
                    self.pixels[iring].append(None)
                    self.noiseweights[iring].append(None)
                    self.weights[iring].append(None)
                    self.hits[iring].append(None)
                    continue
                pixels = rings[ring][det].pixels
                local_pixels = fast_scanning_int32(
                    pixels.astype(np.int32) // self.ndegrade,
                    self.global2local)
                if len(local_pixels) > 0:
                    cachename = 'local_pixels_{}_{}'.format(ring, det)
                    local_pixels = self.cache.put(
                        cachename, local_pixels, replace=True)
                self.pixels[iring].append(local_pixels)
                self.noiseweights[iring].append(rings[ring][det].hits
                                                * rings[ring][det].mask)
                self.weights[iring].append(
                    rings[ring][det].weights[:, :self.nnz])
                self.hits[iring].append(rings[ring][det].hits)
        return

    def initialize_alltoallv(self, rings):
        """
        Build the auxiliary arrays to facilitate fast map reduction.
        """

        t1 = MPI.Wtime()

        self.get_global2local(rings)

        self.get_hits(rings)

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

        if self.verbose:
            if self.rank == 0:
                print(self.prefix + '')
                print(self.prefix + '{:23} {:>12} {:>12} {:>10} {:>10} {:>10}'
                      ''.format('item', 'mean', 'stdev', 'median',
                                'min', 'max'))

            def report(name, t):
                ttot = self.comm.gather(t, root=0)
                if self.rank == 0:
                    ttot = np.array(ttot)
                    print(
                        self.prefix +
                        '{:>23} {:12.1f} {:12.1f} {:10} {:10} {:10}'
                        ''.format(
                            name, np.mean(ttot), np.std(ttot),
                            int(np.median(ttot)), np.amin(ttot), np.amax(ttot)))

            report('Message buffer size ...', self.bufsize)
            report('Number of messages ....', np.sum(self.counts != 0))
            report('Largest message .......', np.amax(self.counts))
            report('Local npix ............', np.amax(self.my_npix))
            if self.rank == 0:
                print(self.prefix + '')

        self.time_init_alltoallv += MPI.Wtime() - t1
        return

    def collect_map(self, sigmap):
        """
        Collect the distributed map to the root process
        """

        t1 = MPI.Wtime()

        nnz = sigmap.shape[1]
        full_map = self.cache.put(
            'global_map', np.zeros([self.npix, nnz], dtype=np.float64),
            replace=True)

        for my_pix in self.my_own:
            local_pix = self.global2local[my_pix]
            full_map[my_pix] = sigmap[local_pix]

        self.comm.Allreduce(MPI.IN_PLACE, full_map, op=MPI.SUM)

        self.time_collect_map += MPI.Wtime() - t1
        return full_map

    def get_rms(self, toi, rings):
        """
        Measure signal-subtracted RMS.
        """

        t1 = MPI.Wtime()
        noise_toi = self.apply_Z(toi, rings)
        sqsum = 0
        n = 0
        for iring, _ in enumerate(self.my_rings):
            for idet, _ in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                hits = self.noiseweights[iring][idet]
                sqsum += np.sum(noise_toi[iring][idet] ** 2 * hits)
                n += np.sum(hits)
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

    def get_cc(self):
        """
        Compute the pixel matrices:
            (P^T N^-1 P)^-1
        """

        # First accumulate (P^T N-1 P)

        t1 = MPI.Wtime()
        if self.cc is None:
            self.cc = np.zeros([self.my_npix, self.nnz, self.nnz],
                               dtype=np.float64)
            if self.my_npix > 0:
                self.cc[:, :, :] = 0
                self.cc = self.cache.put('cc', self.cc, replace=True)
        for iring, _ in enumerate(self.my_rings):
            for idet, _ in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                pixels = self.pixels[iring][idet]
                weights = self.weights[iring][idet]
                hits = self.noiseweights[iring][idet]
                fast_weight_binning_with_mult(pixels, weights, hits, self.cc)
        self.comm.Barrier()
        self.time_accumulate_cc += MPI.Wtime() - t1
        self.allreduce(self.cc)

        # Now invert for (P^T N^-1 P)^-1

        t1 = MPI.Wtime()
        if self.rcond is None:
            self.rcond = np.zeros(self.my_npix, dtype=np.float64)
            if self.my_npix > 0:
                self.rcond = self.cache.put('rcond', self.rcond, replace=True)
        fast_cc_invert(self.cc, self.cc, self.rcond, self.threshold, 0, 1)
        self.time_invert_cc += MPI.Wtime() - t1

        # Set the uninvertible pixel matrices to zero to null samples
        # falling into them

        self.cc[self.rcond < self.threshold, :] = 0
        return

    def cc_multiply(self, sigmap_in):
        """
        Multiply map with precomputed pixel matrices:

            (P^T N^-1 P)^-1 m
        """

        t1 = MPI.Wtime()
        sigmap = np.zeros([self.my_npix, self.nnz], dtype=np.float64)
        fast_cc_multiply(sigmap, self.cc, sigmap_in, 0, 1)
        self.time_cc_multiply += MPI.Wtime() - t1
        return sigmap

    def apply_Z(self, toi, rings):
        """
        Apply Z = I - P ( P^T N^-1 P )^-1 P^T N^-1 to the TOI

        """

        t1 = MPI.Wtime()

        zmap = self.toi2map(toi)

        # Singular pixel matrices are set to 0 and will null the samples
        # that fall into them

        zmap = self.cc_multiply(zmap)

        ztoi = self.map2toi(zmap)

        cleaned_toi = []
        for iring, _ in enumerate(self.my_rings):
            cleaned_toi.append([])
            for idet, _ in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    cleaned_toi[iring].append(None)
                    continue
                zt = ztoi[iring][idet]
                ct = toi[iring][idet] - zt
                ct[zt == 0] = 0
                cleaned_toi[iring].append(ct)

        self.time_apply_Z += MPI.Wtime() - t1
        return cleaned_toi

    def apply_noisematrix(self, toi):
        """
        This operator applies the noise weighting as a diagonal
        inverse noise matrix, N^-1.  The weights are otherwise uniform
        but modulated with the hit counts and destriping mask.

            N^-1 y
        """

        t1 = MPI.Wtime()

        for iring, _ in enumerate(self.my_rings):
            for idet, _ in enumerate(self.my_dets):
                if toi[iring][idet] is None:
                    continue
                hits = self.noiseweights[iring][idet]
                toi[iring][idet] *= hits

        self.time_apply_noisematrix += MPI.Wtime() - t1
        return toi

    def allreduce(self, sigmap):
        """
        Sum over sigmap and broadcast the results
        """

        time1 = MPI.Wtime()

        nnz = np.prod(np.shape(sigmap)[1:])
        sendbuf = np.zeros(self.bufsize * nnz, dtype=np.float64)
        recvbuf = np.zeros(self.bufsize * nnz, dtype=np.float64)

        # Pack the send buffer

        self.comm.Barrier()
        time2 = MPI.Wtime()
        collect_buf(sendbuf, self.commonpix, nnz, sigmap.reshape([-1, nnz]))
        self.time_alltoallv_pack += MPI.Wtime() - time2

        # Communicate

        self.comm.Barrier()
        time2 = MPI.Wtime()
        self.comm.Alltoallv(
            (sendbuf, self.counts * nnz, self.displs * nnz, MPI.DOUBLE),
            (recvbuf, self.counts * nnz, self.displs * nnz, MPI.DOUBLE))
        self.time_alltoallv_MPI += MPI.Wtime() - time2

        # Unpack the recv buffer

        self.comm.Barrier()
        time2 = MPI.Wtime()
        sample_buf(recvbuf, self.commonpix, nnz, sigmap.reshape([-1, nnz]))
        self.time_alltoallv_unpack += MPI.Wtime() - time2

        self.comm.Barrier()
        self.time_alltoallv += MPI.Wtime() - time1
        return sigmap

    def toi2map(self, toi):
        """
        Bin the TOI based on the pointing in pixels:
            P^T N^-1 y
        Applies the noise weights to the signal while binning
        """

        time1 = MPI.Wtime()
        sigmap = np.zeros([self.my_npix, self.nnz])
        for iring, _ in enumerate(self.my_rings):
            for idet, _ in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                signal = toi[iring][idet]
                pixels = self.pixels[iring][idet]
                weights = self.weights[iring][idet]
                hits = self.noiseweights[iring][idet]
                fast_binning_pol_with_mult(
                    signal, pixels, weights, hits, sigmap)
        self.comm.Barrier()
        self.allreduce(sigmap)
        self.time_toi2map += MPI.Wtime() - time1
        return sigmap

    def map2toi(self, sigmap):
        """
        Scan TOI from the map based on a list of pixel vectors:
            P m
        """

        t1 = MPI.Wtime()
        toi = []
        for iring, _ in enumerate(self.my_rings):
            toi.append([])
            for idet, _ in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    toi[iring].append(None)
                    continue
                pixels = self.pixels[iring][idet]
                weights = self.weights[iring][idet]
                nbin = pixels.size
                toivec = np.zeros(nbin, dtype=np.float64)
                fast_scanning_pol64(
                    toivec, pixels, weights, sigmap)
                toi[iring].append(toivec)
        self.comm.Barrier()
        self.time_map2toi += MPI.Wtime() - t1
        return toi

    def subtract_baselines(
            self, rings, templates, baselines, in_place=False):
        """
        Subtract the templates with the best fit amplitudes:
            y - F a
        """

        t1 = MPI.Wtime()

        clean_toi = []
        for iring, ring in enumerate(self.my_rings):
            clean_toi.append([])
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    clean_toi[iring].append(None)
                    continue
                if in_place:
                    clean_toi[iring].append(rings[ring][det].signal)
                else:
                    clean_toi[iring].append(rings[ring][det].signal.copy())

        for iring, ring in enumerate(self.my_rings):
            for idet, det in enumerate(self.my_dets):
                if self.pixels[iring][idet] is None:
                    continue
                for template in templates[ring][det].values():
                    if template.offset is None:
                        continue
                    amp = baselines[template.offset]
                    clean_toi[iring][idet] -= amp * template.template

        self.comm.Barrier()
        self.time_subtract_baselines += MPI.Wtime() - t1
        return clean_toi

    def report_timing(self):
        """
        Report timing.
        """

        if self.rank == 0:
            print(self.prefix + '')
            print(self.prefix + '{:32} {:>10} {:>10} {:>10} {:>10} {:>10}'
                  ''.format('item', 'mean [s]', 'stdev [s]', 'median [s]',
                            'min [s]', 'max [s]'))

        def report(name, t):
            ttot = self.comm.gather(t, root=0)
            if self.rank == 0:
                ttot = np.array(ttot)
                print(
                    self.prefix +
                    '{:>32} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'
                    ''.format(
                        name, np.mean(ttot), np.std(ttot), np.median(ttot),
                        np.amin(ttot), np.amax(ttot)))

        report('Total destriping ...............', self.time_destripe)
        report('  - accumulate_cc ..............', self.time_accumulate_cc)
        report('  - invert_cc ..................', self.time_invert_cc)
        report('  - clean templates ............', self.time_clean_templates)
        report('  - build covariance ...........', self.time_build_cov)
        report('  - invert covariance ..........', self.time_invert_cov)
        report('  - solve amplitudes ...........', self.time_solve_amplitudes)
        report('  - RHS ........................', self.time_rhs)
        report('  - precond_init ...............', self.time_precond_init)
        report('  - CG iterate .................', self.time_iterate)
        report('    - precond...................', self.time_precond)
        report('    - LHS ......................', self.time_lhs)
        report('      o toi2base ...............', self.time_toi2base)
        report('        - reduce ...............', self.time_reducebase)
        report('      o apply noisematrix ......', self.time_apply_noisematrix)
        report('      o apply Z ................', self.time_apply_Z)
        report('        - toi2map ..............', self.time_toi2map)
        report('          * allreduce ..........', self.time_alltoallv)
        report('            + pack .............', self.time_alltoallv_pack)
        report('            + MPI ..............', self.time_alltoallv_MPI)
        report('            + unpack ...........', self.time_alltoallv_unpack)
        report('        - cc_multiply ..........', self.time_cc_multiply)
        report('        - map2toi ..............', self.time_map2toi)
        report('      o base2toi ...............', self.time_base2toi)
        report('  - subtract baselines .........', self.time_subtract_baselines)
        report('  - get rms ....................', self.time_rms)
        report('  - MPI ........................', self.time_mpi)
        report('  - init_alltoallv .............', self.time_init_alltoallv)
        report('  - collect_map ................', self.time_collect_map)
        report('  - write_map ..................', self.time_write_map)

        if self.rank == 0:
            print(self.prefix + '', flush=True)
        return
