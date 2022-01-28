# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

import os
from toast_planck.reproc_modules.destripe_tools import (fast_hit_binning,
                                                        fast_binning)

import scipy.signal
from toast import qarray as qa
from toast.mpi import MPI
from toast_planck.preproc_modules import MapSampler, flagged_running_average

import astropy.io.fits as pf
import numpy as np
import toast.fod as tf
import toast.timing as timing


class OpNoiseEstim():

    def __init__(
            self, signal=None, flags=None, detmask=1, commonmask=3, out=None,
            maskfile=None, mapfile=None, rimo=None, pol=True, nbin_psd=1000,
            lagmax=100000, stationary_period=86400., nosingle=False,
            no_spin_harmonics=False, calibrate_signal_estimate=False,
            nsum=10, naverage=100):
        self._signal = signal
        self._flags = flags
        self._detmask = detmask
        self._commonmask = commonmask
        self._out = out
        self._maskfile = maskfile
        self._mapfile = mapfile
        if rimo is None:
            raise RuntimeError('OpNoiseEstim: You must provide a RIMO')
        self._rimo = rimo
        self._pol = pol
        self._nbin_psd = nbin_psd
        self._lagmax = lagmax
        self._stationary_period = stationary_period
        self._nosingle = nosingle
        self._no_spin_harmonics = no_spin_harmonics
        self._calibrate_signal_estimate = calibrate_signal_estimate
        # Parameters for downsampling the data
        self._nsum = nsum
        self._naverage = naverage

    def exec(self, data):

        cworld = data.comm.comm_world
        rank = cworld.Get_rank()

        masksampler = None
        if self._maskfile:
            masksampler = MapSampler(self._maskfile, comm=cworld)
        mapsampler = None
        if self._mapfile:
            mapsampler = MapSampler(self._mapfile, comm=cworld, pol=True)

        for obs in data.obs:
            tod = obs['tod']
            local_intervals = tod.local_intervals(obs['intervals'])
            dets = tod.local_dets
            ndet = len(dets)

            timestamps = tod.local_timestamps()
            commonflags = tod.local_common_flags()
            commonflags = (commonflags & self._commonmask != 0)

            fsample = self.subtract_signal(
                tod, cworld, rank, masksampler, mapsampler, local_intervals)

            # Extend the gap between intervals to prevent sample pairs
            # that cross the gap.

            intervals = obs['intervals']
            gap_min = int(self._lagmax) + 1
            # Downsampled data requires longer gaps
            gap_min_nsum = int(self._lagmax * self._nsum) + 1
            offset, nsamp = tod.local_samples
            gapflags = np.zeros_like(commonflags)
            gapflags_nsum = np.zeros_like(commonflags)
            for ival1, ival2 in zip(intervals[:-1], intervals[1:]):
                gap_start = ival1.last + 1
                gap_stop = max(gap_start + gap_min, ival2.first)
                gap_stop_nsum = max(gap_start + gap_min_nsum, ival2.first)
                if gap_start < offset + nsamp and gap_stop > offset:
                    gap_start = max(0, gap_start - offset)
                    gap_stop = min(offset + nsamp, gap_stop - offset)
                    gapflags[gap_start:gap_stop] = True
                    gap_stop_nsum = min(offset + nsamp, gap_stop_nsum - offset)
                    gapflags_nsum[gap_start:gap_stop_nsum] = True

            for idet1 in range(ndet):
                for idet2 in range(idet1, ndet):
                    det1 = dets[idet1]
                    det2 = dets[idet2]
                    if det1 == det2 and self._nosingle:
                        continue
                    signal1 = tod.local_signal(det1)
                    flags1 = tod.local_flags(det1, name=self._flags)
                    flags = (flags1 & self._detmask != 0)
                    signal2 = None
                    flags2 = None
                    if det1 != det2:
                        signal2 = tod.local_signal(det2)
                        flags2 = tod.local_flags(det2, name=self._flags)
                        flags[flags2 & self._detmask != 0] = True
                    flags[commonflags] = True

                    self.process_noise_estimate(
                        signal1, signal2, flags, gapflags, gapflags_nsum,
                        timestamps, fsample, cworld, rank, 'noise', det1, det2,
                        local_intervals)

        return

    def subtract_signal(self, tod, cworld, rank, masksampler, mapsampler,
                        local_intervals):
        """ Subtract a signal estimate from the TOD and update the
        flags for noise estimation.
        """

        start_signal_subtract = MPI.Wtime()
        for det in tod.local_dets:
            if rank == 0:
                print('Subtracting signal for {}'.format(det), flush=True)
                tod.cache.report()
            fsample = self._rimo[det].fsample
            epsilon = self._rimo[det].epsilon
            eta = (1 - epsilon) / (1 + epsilon)
            signal = tod.local_signal(det, name=self._signal)
            flags = tod.local_flags(det, name=self._flags)
            flags &= self._detmask
            for ival in local_intervals:
                ind = slice(ival.first, ival.last + 1)
                sig = signal[ind]
                flg = flags[ind]
                quat = tod.local_pointing(det)[ind]
                if self._pol:
                    theta, phi, psi = qa.to_angles(quat)
                    iw = np.ones_like(theta)
                    qw = eta * np.cos(2 * psi)
                    uw = eta * np.sin(2 * psi)
                    iquw = np.column_stack([iw, qw, uw])
                else:
                    theta, phi = qa.to_position(quat)
                if masksampler is not None:
                    maskflg = masksampler.at(theta, phi) < 0.5
                    flg[maskflg] |= 255
                if mapsampler is not None:
                    if self._pol:
                        bg = mapsampler.atpol(theta, phi, iquw)
                    else:
                        bg = mapsampler.at(theta, phi)
                    if self._calibrate_signal_estimate:
                        good = flg == 0
                        ngood = np.sum(good)
                        if ngood > 1:
                            templates = np.vstack([np.ones(ngood), bg[good]])
                            invcov = np.dot(templates, templates.T)
                            cov = np.linalg.inv(invcov)
                            proj = np.dot(templates, sig[good])
                            coeff = np.dot(cov, proj)
                            bg = coeff[0] + coeff[1] * bg
                    sig -= bg
        cworld.barrier()
        stop_signal_subtract = MPI.Wtime()
        if rank == 0:
            print('TOD signal-subtracted in {:.2f} s'.format(
                stop_signal_subtract - start_signal_subtract),
                flush=True)
        return fsample

    def decimate(self, x, flg, gapflg, local_intervals):
        # Low-pass filter with running average, then downsample
        xx = x.copy()
        flags = flg.copy()
        for ival in local_intervals:
            ind = slice(ival.first, ival.last + 1)
            xx[ind], flags[ind] = flagged_running_average(
                x[ind], flg[ind], self._naverage,
                return_flags=True)
        return xx[::self._nsum].copy(), (flags + gapflg)[::self._nsum].copy()

    """
    def highpass(self, x, flg):
        # Flagged real-space high pass filter
        xx = x.copy()

        j = 0
        while j < x.size and flg[j]: j += 1

        alpha = .999

        for i in range(j+1, x.size):
            if flg[i]:
                xx[i] = x[j]
            else:
                xx[i] = alpha*(xx[j] + x[i] - x[j])
                j = i

        xx /= alpha
        return xx
    """

    def log_bin(self, freq, nbin=100, fmin=None, fmax=None):
        if np.any(freq == 0):
            raise Exception('Logarithmic binning should not include '
                            'zero frequency')

        if fmin is None:
            fmin = np.amin(freq)
        if fmax is None:
            fmax = np.amax(freq)

        bins = np.logspace(np.log(fmin), np.log(fmax), num=nbin + 1,
                           endpoint=True, base=np.e)
        bins[-1] *= 1.01  # Widen the last bin not to have a bin with one entry

        locs = np.digitize(freq, bins).astype(np.int32)

        hits = np.zeros(nbin + 2, dtype=np.int32)
        fast_hit_binning(locs, hits)
        return locs, hits

    def bin_psds(self, my_psds, fmin=None, fmax=None):
        my_binned_psds = []
        my_times = []
        binfreq0 = None

        for i in range(len(my_psds)):
            t0, _, freq, psd = my_psds[i]

            good = freq != 0

            if self._no_spin_harmonics:
                # Discard the bins containing spin harmonics and their
                # immediate neighbors
                for i0 in range(1, 3):
                    f0 = i0 / 60.
                    for i in range(1, 30):
                        fmask = f0 * i
                        imin = np.argmin(np.abs(freq - fmask))
                        if i == 1:
                            # The first bin has a wider neighborhood
                            good[imin - 2:imin + 3] = False
                        else:
                            good[imin - 1:imin + 2] = False

            if self._nbin_psd is not None:
                locs, hits = self.log_bin(freq[good], nbin=self._nbin_psd,
                                          fmin=fmin, fmax=fmax)
                binfreq = np.zeros(hits.size)
                fast_binning(freq[good], locs, binfreq)
                binfreq = binfreq[hits != 0] / hits[hits != 0]
            else:
                binfreq = freq
                hits = np.ones(len(binfreq))

            if binfreq0 is None:
                binfreq0 = binfreq
            else:
                if np.any(binfreq != binfreq0):
                    raise Exception('Binned PSD frequencies change')

            if self._nbin_psd is not None:
                binpsd = np.zeros(hits.size)
                fast_binning(psd[good], locs, binpsd)
                binpsd = binpsd[hits != 0] / hits[hits != 0]
            else:
                binpsd = psd

            my_times.append(t0)
            my_binned_psds.append(binpsd)
        return my_binned_psds, my_times, binfreq0

    def discard_spin_harmonics(self, binfreq, all_psds):
        ind = binfreq != 0
        for i0 in range(1, 3):
            f0 = i0 / 60.
            for i in range(1, 10):
                fmask = f0 * i
                imin = np.argmin(np.abs(binfreq - fmask))
                if i == 1:
                    ind[imin - 1:imin + 2] = False
                else:
                    ind[imin] = False

        binfreq = binfreq[ind]
        all_psds = all_psds[:, ind]
        return binfreq, all_psds

    def discard_outliers(self, binfreq, all_psds, all_times):
        all_psds = list(all_psds)
        all_times = list(all_times)

        nrow, ncol = np.shape(all_psds)

        # Discard empty PSDs

        i = 1
        nbad = 0
        all_psds = list(all_psds)
        all_times = list(all_times)
        while i < nrow:
            p = all_psds[i]
            if np.all(p == 0) or np.any(np.isnan(p)):
                del all_psds[i]
                del all_times[i]
                nrow -= 1
                nbad += 1
            else:
                i += 1

        if nbad > 0:
            print('Discarded {} empty or NaN psds'.format(nbad), flush=True)

        # Throw away outlier PSDs by comparing the PSDs in specific bins

        if nrow > 10:

            all_good = np.isfinite(np.sum(all_psds, 1))
            for col in range(ncol - 1):
                if binfreq[col] < .001:
                    continue

                # Local outliers

                psdvalues = np.array([x[col] for x in all_psds])
                smooth_values = scipy.signal.medfilt(psdvalues, 11)
                good = np.ones(psdvalues.size, dtype=np.bool)
                good[psdvalues == 0] = False

                for i in range(10):
                    # Local test
                    diff = np.zeros(psdvalues.size)
                    diff[good] = np.log(psdvalues[good]) - \
                        np.log(smooth_values[good])
                    sdev = np.std(diff[good])
                    good[np.abs(diff) > 5 * sdev] = False
                    # Global test
                    diff = np.zeros(psdvalues.size)
                    diff[good] = np.log(psdvalues[good]) - \
                        np.mean(np.log(psdvalues[good]))
                    sdev = np.std(diff[good])
                    good[np.abs(diff) > 5 * sdev] = False

                all_good[np.logical_not(good)] = False

            bad = np.logical_not(all_good)
            nbad = np.sum(bad)
            if nbad > 0:
                for ii in np.argwhere(bad).ravel()[::-1]:
                    del all_psds[ii]
                    del all_times[ii]

            if nbad > 0:
                print('Masked extra {} psds due to outliers.'
                      ''.format(nbad))
        return all_psds, all_times

    def save_psds(self, binfreq, all_psds, all_times, det1, det2, fsample,
                  rootname):
        if det1 == det2:
            fn_out = os.path.join(self._out,
                                  '{}_{}.fits'.format(rootname, det1))
        else:
            fn_out = os.path.join(self._out,
                                  '{}_{}_{}.fits'.format(rootname, det1, det2))
        all_psds = np.vstack([binfreq, all_psds])

        cols = []
        cols.append(pf.Column(name='OBT', format='D', array=all_times))
        coldefs = pf.ColDefs(cols)
        hdu1 = pf.BinTableHDU.from_columns(coldefs)
        hdu1.header['RATE'] = fsample, 'Sampling rate'

        cols = []
        cols.append(pf.Column(name='PSD', format='{}E'.format(binfreq.size),
                              array=all_psds))
        coldefs = pf.ColDefs(cols)
        hdu2 = pf.BinTableHDU.from_columns(coldefs)
        hdu2.header['EXTNAME'] = det1, 'Detector'
        hdu2.header['DET1'] = det1, 'Detector1'
        hdu2.header['DET2'] = det2, 'Detector2'

        hdu0 = pf.PrimaryHDU()
        hdulist = pf.HDUList([hdu0, hdu1, hdu2])

        if os.path.isfile(fn_out):
            os.remove(fn_out)
        hdulist.writeto(fn_out)
        print('Detector {} vs. {} PSDs stored in {}'.format(
            det1, det2, fn_out))
        return

    def process_noise_estimate(
            self, signal1, signal2, flags, gapflags, gapflags_nsum,
            timestamps, fsample, cworld, rank, fileroot, det1, det2,
            local_intervals):
        # High pass filter the signal to avoid aliasing
        # self.highpass(signal1, noise_flags)
        # self.highpass(signal2, noise_flags)

        # Compute the autocovariance function and the matching
        # PSD for each stationary interval

        start = MPI.Wtime()
        if signal2 is None:
            my_psds1 = tf.autocov_psd(
                timestamps, signal1, flags + gapflags, self._lagmax,
                self._stationary_period, fsample, comm=cworld)
        else:
            my_psds1 = tf.crosscov_psd(
                timestamps, signal1, signal2, flags + gapflags, self._lagmax,
                self._stationary_period, fsample, comm=cworld)

        # Get another PSD for a down-sampled TOD to measure the
        # low frequency power

        timestamps_decim = timestamps[::self._nsum]
        # decimate() will smooth and downsample the signal in
        # each valid interval separately
        signal1_decim, flags_decim = self.decimate(
            signal1, flags, gapflags_nsum, local_intervals)
        if signal2 is not None:
            signal2_decim, flags_decim = self.decimate(
                signal2, flags, gapflags_nsum, local_intervals)

        if signal2 is None:
            my_psds2 = tf.autocov_psd(
                timestamps_decim, signal1_decim, flags_decim,
                min(self._lagmax, timestamps_decim.size),
                self._stationary_period, fsample / self._nsum, comm=cworld)
        else:
            my_psds2 = tf.crosscov_psd(
                timestamps_decim, signal1_decim, signal2_decim, flags_decim,
                min(self._lagmax, timestamps_decim.size),
                self._stationary_period, fsample / self._nsum, comm=cworld)

        # Ensure the two sets of PSDs are of equal length

        my_new_psds1 = []
        my_new_psds2 = []
        i = 0
        while i < min(len(my_psds1), len(my_psds2)):
            t1 = my_psds1[i][0]
            t2 = my_psds2[i][0]
            if np.isclose(t1, t2):
                my_new_psds1.append(my_psds1[i])
                my_new_psds2.append(my_psds2[i])
                i += 1
            else:
                if t1 < t2:
                    del my_psds1[i]
                else:
                    del my_psds2[i]
        my_psds1 = my_new_psds1
        my_psds2 = my_new_psds2

        if len(my_psds1) != len(my_psds2):
            while my_psds1[-1][0] > my_psds2[-1][0]:
                del my_psds1[-1]
            while my_psds1[-1][0] < my_psds2[-1][0]:
                del my_psds2[-1]

        # frequencies that are usable in the down-sampled PSD
        fcut = fsample / 2 / self._naverage / 100

        stop = MPI.Wtime()

        if rank == 0:
            print('Correlators and PSDs computed in {:.2f} s'
                  ''.format(stop - start), flush=True)

        # Now bin the PSDs

        fmin = 1 / self._stationary_period
        fmax = fsample / 2

        start = MPI.Wtime()
        my_binned_psds1, my_times1, binfreq10 = self.bin_psds(
            my_psds1, fmin, fmax)
        my_binned_psds2, _, binfreq20 = self.bin_psds(
            my_psds2, fmin, fmax)
        stop = MPI.Wtime()

        """
        # DEBUG begin
        import pdb
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(my_psds2[0][2], my_psds2[0][3], 'r.')
        plt.loglog(my_psds1[0][2], my_psds1[0][3], 'b.')
        plt.loglog(binfreq20, my_binned_psds2[0], 'r-')
        plt.loglog(binfreq10, my_binned_psds1[0], 'b-')
        plt.gca().axvline(fcut, color='k')
        plt.draw()
        plt.show()
        pdb.set_trace()
        # DEBUG end
        """

        # concatenate

        if binfreq10 is None or binfreq20 is None:
            my_times = []
            my_binned_psds = []
            binfreq0 = None
        else:
            my_times = my_times1
            ind1 = binfreq10 > fcut
            ind2 = binfreq20 <= fcut
            binfreq0 = np.hstack([binfreq20[ind2], binfreq10[ind1]])
            my_binned_psds = []
            for psd1, psd2 in zip(my_binned_psds1, my_binned_psds2):
                my_binned_psds.append(np.hstack([psd2[ind2], psd1[ind1]]))

        # Collect and write the PSDs.  Start by determining the first
        # process to have a valid PSD to determine binning

        start = MPI.Wtime()
        have_bins = binfreq0 is not None
        have_bins_all = cworld.allgather(have_bins)
        root = 0
        if np.any(have_bins_all):
            while not have_bins_all[root]:
                root += 1
        else:
            raise RuntimeError('None of the processes have valid PSDs')
        binfreq = cworld.bcast(binfreq0, root=root)
        if binfreq0 is not None and np.any(binfreq != binfreq0):
            raise Exception(
                '{:4} : Binned PSD frequencies change. len(binfreq0)={}'
                ', len(binfreq)={}, binfreq0={}, binfreq={}. '
                'len(my_psds)={}'.format(
                    rank, binfreq0.size, binfreq.size, binfreq0,
                    binfreq, len(my_psds1)))
        if len(my_times) != len(my_binned_psds):
            raise Exception(
                'ERROR: Process {} has len(my_times) = {}, len(my_binned_psds)'
                ' = {}'.format(rank, len(my_times), len(my_binned_psds)))
        all_times = cworld.gather(my_times, root=0)
        all_psds = cworld.gather(my_binned_psds, root=0)
        stop = MPI.Wtime()

        if rank == 0:
            if len(all_times) != len(all_psds):
                raise Exception(
                    'ERROR: Process {} has len(all_times) = {}, len(all_psds)'
                    ' = {} before deglitch'.format(
                        rank, len(all_times), len(all_psds)))
            # De-glitch the binned PSDs and write them to file
            i = 0
            while i < len(all_times):
                if len(all_times[i]) == 0:
                    del all_times[i]
                    del all_psds[i]
                else:
                    i += 1

            all_times = np.hstack(all_times)
            all_psds = np.vstack(all_psds)

            if len(all_times) != len(all_psds):
                raise Exception(
                    'ERROR: Process {} has len(all_times) = {}, len(all_psds)'
                    ' = {} AFTER deglitch'.format(
                        rank, len(all_times), len(all_psds)))

            # if self._no_spin_harmonics:
            #    binfreq, all_psds = self.discard_spin_harmonics(binfreq, all_psds)

            good_psds, good_times = self.discard_outliers(
                binfreq, all_psds, all_times)

            self.save_psds(
                binfreq, all_psds, all_times, det1, det2, fsample, fileroot)

            self.save_psds(
                binfreq, good_psds, good_times, det1, det2, fsample,
                fileroot + '_good')
        return
