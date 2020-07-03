# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
from toast_planck.reproc_modules import Differentiator
from toast_planck.reproc_modules import destripe_tools

import scipy.optimize
import scipy.signal
from toast.mpi import MPI
from toast.tod import calibrate

import astropy.io.fits as pf
import numpy as np
import toast.timing as timing

from .filters import flagged_running_average


class InputEstimator():

    def __init__(self, det, transf1, gaincorrector, nphase4k, wbin, comm,
                 out=None, input_correction=None, nharm=4, fsample=180.3737):
        self._det = det
        self._transf1 = transf1
        self._gaincorrector = gaincorrector
        self._nphase4k = nphase4k
        self._wbin = wbin
        self.comm = comm
        self.ntask = comm.size
        self.rank = comm.rank
        self.corrections = None
        self._out = out
        self.reset_cache()
        if input_correction is not None:
            self.load_correction(input_correction)
        self._nharm = nharm
        self._fsample = fsample
        self._differentiator = Differentiator(nharm=self._nharm + 3,
                                              fsample=self._fsample)

    def load_correction(self, fn):

        fn_in = fn.replace('DETECTOR', self._det)
        h = pf.open(fn_in, 'readonly')

        self.corrections = []

        # It is possible to split phases into subphases in new
        # corrections but only if the number of new phases is
        # divisible by the number of old phases. Say we have 2 phases
        # in the input file and want to estimate 4 this time. The old
        # phase 0 is copied as input to phase 0 and phase 1. Old phase
        # 1 seeds new phases 2 and 3.

        nphase = len(h[1:])
        nskip = self._nphase4k // nphase
        if nphase * nskip != self._nphase4k:
            raise RuntimeError('Inputestimator is set to {} phases but {} has '
                               '{}'.format(self._nphase4k, fn_in, nphase))
        for hdu in h[1:]:
            for _ in range(nskip):
                self.corrections.append([hdu.data.field(0).copy(),
                                         hdu.data.field(1).copy()])
        return

    def reset_cache(self):
        self.ring_number = []
        self.signal_ADU = []
        self.signal_estimate = []
        self.signal_offset = []
        for _ in range(self._nphase4k):
            self.signal_ADU.append([])
            self.signal_estimate.append([])
            self.signal_offset.append([])

    def estimate(self, ring_number, timestamps=None, signal_estimate=None,
                 noise_estimate=None, line_estimate=None, decorr_template=None,
                 glitch_estimate=None, jump_estimate=None, signal_demod=None,
                 flags=None, baselineND=None, parity=None, sigND=None,
                 sigADU=None, gains=None, phase4k=None):
        """
        Construct an estimate of the modulated signal in co-added
        digitized units that lie in range 0..(2**16-1)*40
        """

        if jump_estimate is not None:
            print('Skipping ring {} in ADC NL estimation due to jump(s).'
                  ''.format(ring_number), flush=True)
            return

        if timestamps is None:
            raise Exception('You must supply the time stamps in timestamps.')
        if flags is None:
            raise Exception('You must supply the flags in flags.')
        if gains is None:
            raise Exception('You must supply the gains in gains.')
        if signal_estimate is None:
            raise Exception('You must supply the signal estimate in '
                            'signal_estimate.')
        if sigND is None:
            raise Exception('You must supply the undemodulated signal in '
                            'sigND.')
        if sigADU is None:
            raise Exception('You must supply the digitized signal in sigADU.')
        if baselineND is None:
            raise Exception('You must supply the undemodulated signal '
                            'baseline in baselineND.')

        raw_flags = flags.copy()

        # Now construct an estimate of the original TOI in Volts

        raw_signal_estimate = signal_estimate.copy()
        if noise_estimate is not None:
            raw_signal_estimate += noise_estimate
        if line_estimate is not None:
            raw_signal_estimate += line_estimate
        if decorr_template is not None:
            raw_signal_estimate += decorr_template
        if glitch_estimate is not None:
            raw_signal_estimate += glitch_estimate
        if jump_estimate is not None:
            raw_signal_estimate += jump_estimate

        good = raw_flags == 0

        minvalKCMB = np.amin(raw_signal_estimate[good])
        maxvalKCMB = np.amax(raw_signal_estimate[good])

        # From K_CMB to Watt

        raw_signal_estimate = calibrate(
            timestamps, raw_signal_estimate, gains[0], 1. / gains[1])

        # From Watt to Volt

        raw_signal_estimate = self._gaincorrector.uncorrect(
            raw_signal_estimate, flags)

        """
        delta = np.mean(signal_demod[good] - raw_signal_estimate[good])

        if np.isnan(delta):
            raise Exception('Raw signal offset is NaN')

        print('Correcting for biased baseline: {:.3f} nV'.format(delta*1e9))

        raw_signal_estimate += delta
        """

        # Transfer function residuals bias the measurement.  Project the
        # associated mismatch out from the raw signal estimate.  The
        # signal estimate is in K_CMB but that does not matter for
        # filtering.

        derivs = self._differentiator.differentiate(
            signal_estimate, do_bands=False, do_derivs=True)[1]

        templates = [np.ones(good.size)[good]]
        for i in range(self._nharm):
            templates.append(derivs[i][good])
        templates = np.vstack(templates)
        invcov = np.dot(templates, templates.T)
        proj = np.dot(templates, (raw_signal_estimate - signal_demod)[good])
        try:
            cov = np.linalg.inv(invcov)
        except Exception:
            # The matrix failed to invert, abandon this ring.
            print('{:4} : ring {} failed to invert the transfer function '
                  'residual covariance matrix.'.format(
                      self.rank, ring_number), flush=True)
            return
        coeffs = np.dot(cov, proj)
        raw_signal_estimate -= coeffs[0]
        for i in range(self._nharm):
            raw_signal_estimate -= coeffs[1 + i] * derivs[i]

        # Now modulate the signal and add back the subtracted baseline

        raw_signal_estimate = raw_signal_estimate * (1. - 2.*parity) \
            + baselineND

        # The baselineND measurement was also biased by glitches.
        # Determine extra offset to add.

        delta = np.mean(sigND[good] - raw_signal_estimate[good])

        print('Correcting for biased baselineND: {:.3f} nV'.format(delta * 1e9),
              flush=True)

        raw_signal_estimate += delta

        # From Volt to DSP

        raw_signal_estimate = self._transf1.convert(
            raw_signal_estimate, timestamps, self._det, unconvert=True)

        raw_signal_estimate[raw_flags != 0] = np.nan

        # Avoid overflows binning the signal estimate later on

        minval = np.amin(raw_signal_estimate[good])
        maxval = np.amax(raw_signal_estimate[good])
        delta = maxval - minval
        if delta / self._wbin > 2 ** 32:
            print('{:4} : ring {} has a too large signal estimate range: '
                  '{} - {} 40DSP ({} - {} KCMB)'.format(
                      self.rank, ring_number, minval, maxval,
                      minvalKCMB, maxvalKCMB), flush=True)
            return

        self.ring_number.append(ring_number)

        for phase in range(self._nphase4k):
            good = np.logical_and(phase4k == phase, raw_flags == 0)
            self.signal_ADU[phase].append(sigADU[good].copy())
            self.signal_estimate[phase].append(raw_signal_estimate[good].copy())
            self.signal_offset[phase].append(np.mean(raw_signal_estimate[good]))
        return

    def remove_outliers(self, outliers):
        """Exclude outlier rings from the measurement

        """
        for outlier in outliers:
            if outlier in self.ring_number:
                i = self.ring_number.index(outlier)
                del self.ring_number[i]
                for phase in range(self._nphase4k):
                    del self.signal_ADU[phase][i]
                    del self.signal_estimate[phase][i]
                    del self.signal_offset[phase][i]
        return

    def measure_correction(self, fn=None, gain=None):
        """
        Estimate the ADC correction per bin for each 4K phase
        """

        self.corrections = []

        if fn is not None and self.rank == 0:
            fn_out = os.path.join(self._out, fn)
            hdulist = [pf.PrimaryHDU()]
        else:
            fn_out = None
            hdulist = None

        if gain is not None and gain != 0:
            for phase in self.signal_estimate:
                for estimate in phase:
                    mean_before = np.mean(estimate)
                    estimate[:] = (estimate - mean_before) / gain + mean_before

        for phase in range(self._nphase4k):

            tstart_phase = MPI.Wtime()

            if self.rank == 0:
                print('Estimating ADC correction for phase {}'.format(phase),
                      flush=True)

            signal_ADU = self.signal_ADU[phase]
            signal_estimate = self.signal_estimate[phase]
            signal_offset = self.signal_offset[phase]

            nl = []  # Additive offset between input and output signal
            for sig_ADU, estimate in zip(signal_ADU, signal_estimate):
                nl.append(sig_ADU - estimate)

            bins = []
            for sig in signal_estimate:
                bins.append(np.floor(sig / self._wbin).astype(np.int32))
            my_binmin = 99999999
            my_binmax = -99999999
            for binvec in bins:
                my_binmin = min(my_binmin, np.amin(binvec))
                my_binmax = max(my_binmax, np.amax(binvec))
            binmin = self.comm.allreduce(my_binmin, MPI.MIN)
            binmax = self.comm.allreduce(my_binmax, MPI.MAX)
            bin_offset = binmin
            nbin = binmax - binmin + 1
            if nbin < 1:
                raise RuntimeError(
                    '{} : ERROR in measure_correction: No valid bins '
                    'for phase = {} / {}. binmin, binmax = {}, {}, '
                    'my_binmin, my_binmax = {}, {}, nbin = {}'.format(
                        self.rank, phase + 1, self._nphase4k, binmin, binmax,
                        my_binmin, my_binmax, nbin))

            for binvec in bins:
                binvec -= bin_offset

            my_rings = []
            for binvec, sampvec, offset in zip(bins, nl, signal_offset):
                hitmap = np.zeros(nbin, dtype=np.int32)
                sigmap = np.zeros(nbin, dtype=np.float64)

                destripe_tools.fast_hit_binning(binvec, hitmap)
                destripe_tools.fast_binning(sampvec, binvec, sigmap)

                bin_centers = (0.5 + np.arange(nbin) + bin_offset) * self._wbin

                hit = hitmap != 0
                hitmap = hitmap[hit].copy()
                bin_centers = bin_centers[hit].copy()
                sigmap = sigmap[hit].copy()

                sigmap /= hitmap

                # Fit a line to the ring measurement of NL

                coeff, cov = self.fit_line(bin_centers, sigmap, hitmap)
                slope = coeff[1]
                slope_err = np.sqrt(cov[1, 1])

                my_rings.append((bin_centers, sigmap, hitmap,
                                 offset, slope, slope_err))

            # The ring offset optimization is done serially until we can
            # find a nonlinear parallel solver

            rings = self.comm.gather(my_rings, root=0)

            if self.rank == 0:
                # Flatten the ring list
                rings = [ring for ringlist in rings for ring in ringlist]

                if fn_out.endswith('.fits'):
                    fn = fn_out.replace(
                        '.fits', '_ring_data_phase{:02}.pck'.format(phase))
                else:
                    fn = fn_out + '_ring_data_phase{:02}.pck'.format(phase)

                pickle.dump(rings, open(fn, 'wb'), protocol=2)

                print('ADC ring data saved in {}'.format(fn), flush=True)

                polyorder = 10

                ring_rms = []
                for ring in rings:
                    ring_rms.append(np.std(ring[1]))
                ring_rms = np.array(ring_rms)

                outliers = np.isnan(ring_rms)
                if ring_rms.size > 100:
                    for i in range(10):
                        good = np.logical_not(outliers)
                        mn = np.mean(ring_rms[good])
                        rms = np.std(ring_rms[good])
                        bad = np.abs(ring_rms - mn) > 4 * rms
                        bad[outliers] = False
                        nbad = np.sum(bad)
                        if nbad == 0:
                            break
                        outliers[bad] = True
                        nout = np.sum(outliers)
                        print('iter = {}: Discarding {} outlier RMS out of {}. '
                              'Total outliers: {}.'.format(
                                i, nbad, ring_rms.size, nout))

                ring_len = []
                for ring in rings:
                    ring_len.append(ring[0].size)
                ring_len = np.array(ring_len)

                if ring_len.size > 100:
                    for i in range(10):
                        good = np.logical_not(outliers)
                        smooth = flagged_running_average(ring_len,
                                                         outliers, 100)
                        rms = np.std((ring_len - smooth)[good])
                        bad = np.abs(ring_len - smooth) > 4 * rms
                        bad[outliers] = False
                        nbad = np.sum(bad)
                        if nbad == 0:
                            break
                        outliers[bad] = True
                        nout = np.sum(outliers)
                        print('iter = {}: Discarding {} outlier lengths out of '
                              '{}. Total outliers: {}.'.format(
                                i, nbad, ring_len.size, nout))

                # Build an initial correction by integrating the
                # ring-by-ring derivatives (slopes)

                offset = []
                deriv = []
                err = []
                for ring in rings:
                    offset_ring, deriv_ring, deriv_err = ring[3:6]
                    offset.append(offset_ring)
                    deriv.append(deriv_ring)
                    err.append(deriv_err)
                offset = np.hstack(offset)
                deriv = np.hstack(deriv)
                err = np.hstack(err)

                if offset.size > 100:
                    for i in range(10):
                        good = np.logical_not(outliers)
                        smooth = flagged_running_average(deriv, outliers, 100)
                        rms = np.std((deriv - smooth)[good])
                        bad = np.abs(deriv - smooth) > 4 * rms
                        bad[outliers] = False
                        nbad = np.sum(bad)
                        if nbad == 0:
                            break
                        outliers[bad] = True
                        nout = np.sum(outliers)
                        print('iter = {}: Discarding {} outlier slopes out of '
                              '{}. Total outliers: {}.'.format(
                                i, nbad, offset.size, nout))

                good = np.logical_not(outliers)
                offset = offset[good]
                deriv = deriv[good]

                ind = np.argsort(offset)
                offset = offset[ind]
                deriv = deriv[ind]

                total_offset = np.zeros(offset.size, dtype=np.float64)
                for i in range(offset.size - 1):
                    total_offset[i + 1] = total_offset[i] + deriv[i] * (
                        offset[i + 1] - offset[i])

                w = offset[-1] - offset[0]
                # Domain will shift with the offset correction
                domain = [offset[0] - w / 10, offset[-1] + w / 10]

                polyorder = min(polyorder, offset.size - 1)

                # Omit the offset term
                x0 = np.polynomial.legendre.Legendre.fit(
                    offset, total_offset, polyorder, domain=domain).coef[1:]

                # Collapse pointing periods that have the same offset

                collapsed_ring_lists = {}
                wbin_offset = (domain[1] - domain[0]) / 1000
                for (ring, outlier) in zip(rings, outliers):
                    if outlier:
                        continue
                    bin_centers, sigmap, hitmap, offset = ring[:4]
                    offset_bin = np.int(np.floor(offset / wbin_offset))
                    if offset_bin not in collapsed_ring_lists:
                        collapsed_ring_lists[offset_bin] = []
                    collapsed_ring_lists[offset_bin].append(ring)

                collapsed_rings = []
                for offset_bin, offset_rings in collapsed_ring_lists.items():
                    # co-add the rings that have the same offset
                    center_min = offset_rings[0][0][0]
                    center_max = offset_rings[0][0][-1]
                    for ring in offset_rings[1:]:
                        bin_centers = ring[0]
                        center_min = min(center_min, bin_centers[0])
                        center_max = max(center_max, bin_centers[-1])
                    nbin = np.int(
                        np.rint((center_max - center_min) / self._wbin)) + 1
                    all_bin_centers = center_min + np.arange(nbin) * self._wbin
                    all_sigmap = np.zeros(nbin, dtype=np.float64)
                    all_hitmap = np.zeros(nbin, dtype=np.float64)
                    for ring in offset_rings:
                        bin_centers, sigmap, hitmap, offset = ring[:4]
                        ind = np.searchsorted(all_bin_centers, bin_centers,
                                              side='left')
                        all_hitmap[ind] += hitmap
                        all_sigmap[ind] += sigmap * hitmap
                    good = all_hitmap != 0
                    all_bin_centers = all_bin_centers[good]
                    all_sigmap = all_sigmap[good]
                    all_hitmap = all_hitmap[good]
                    all_sigmap /= all_hitmap
                    collapsed_rings.append(
                        (all_bin_centers, all_sigmap, all_hitmap,
                         (offset_bin + .5) * wbin_offset))

                # Collect the ring-by-ring vectors into single vectors

                all_bin_centers = []
                all_sigmaps = []
                all_hitmaps = []
                all_offsets = []
                all_ranges = []
                istart = 0

                for ring in collapsed_rings:
                    bin_centers, sigmap, hitmap, offset = ring
                    all_bin_centers.append(bin_centers)
                    all_sigmaps.append(sigmap)
                    all_hitmaps.append(hitmap)
                    all_offsets.append(offset)
                    all_ranges.append(slice(istart, istart + sigmap.size))
                    istart += sigmap.size

                all_bin_centers = np.hstack(all_bin_centers).astype(np.float64)
                all_sigmaps = np.hstack(all_sigmaps).astype(np.float64)
                all_hitmaps = np.hstack(all_hitmaps).astype(np.float64)
                all_offsets = np.hstack(all_offsets).astype(np.float64)

                def get_nl(param, all_bin_centers, all_sigmaps, all_hitmaps,
                           all_offsets, domain):

                    # Add zeroth term, this is not a free parameter

                    pfull = np.append([0], param)
                    get_offset_nl = np.polynomial.legendre.Legendre(
                        pfull, domain=domain)

                    # Adjust the zeroth term to minimize offset

                    x = np.linspace(domain[0], domain[1], 100)
                    get_offset_nl.coef[0] = -np.median(get_offset_nl(x))

                    all_bins = all_bin_centers.copy()
                    all_nl = all_sigmaps.copy()
                    for ind, off in zip(all_ranges, all_offsets):
                        delta = get_offset_nl(off)
                        all_bins[ind] -= delta
                        all_nl[ind] += delta
                    all_bins = np.floor(all_bins / self._wbin).astype(np.int64)
                    all_hits = all_hitmaps

                    binmin = np.amin(all_bins)
                    binmax = np.amax(all_bins)
                    nbin = binmax - binmin + 1
                    all_bins -= binmin

                    hitmap = np.zeros(nbin, dtype=np.float64)
                    nlmap = np.zeros(nbin, dtype=np.float64)

                    destripe_tools.fast_binning(
                        all_hits, all_bins.astype(np.int32), hitmap)
                    destripe_tools.fast_binning(
                        all_nl * all_hits, all_bins.astype(np.int32), nlmap)

                    hit = hitmap != 0
                    nlmap[hit] /= hitmap[hit]

                    bin_centers = (0.5 + np.arange(nlmap.size)
                                   + binmin) * self._wbin
                    nlmap_offset = get_offset_nl(bin_centers)

                    return (nlmap, nlmap_offset, hitmap, all_nl, all_hits,
                            all_bins, binmin)

                def get_nl_resid(param, all_bin_centers, all_sigmaps,
                                 all_hitmaps, all_offsets, domain):

                    # Measure the residual between signal/estimate
                    # difference and a binned+unrolled version of the
                    # difference

                    (nlmap, nlmap_offset, hitmap, nl, hits, bins,
                     binmin) = get_nl(
                        param, all_bin_centers, all_sigmaps, all_hitmaps,
                        all_offsets, domain)

                    nl_from_map = nlmap[bins]
                    nl_from_offset_map = nlmap_offset[bins]

                    return np.hstack([
                            (nl - nl_from_map) * np.log(hits),
                            (nl - nl_from_offset_map) * np.log(hits)])

                start = MPI.Wtime()
                try:
                    xopt, _, infodict, mesg, ierr = scipy.optimize.leastsq(
                        get_nl_resid, x0,
                        args=(all_bin_centers, all_sigmaps, all_hitmaps,
                              all_offsets, domain),
                        full_output=True, Dfun=None, maxfev=1000)
                except Exception as e:
                    print('leastsq failed with {}'.format(e))
                    raise
                if ierr not in [1, 2, 3, 4]:
                    raise RuntimeError('leastsq failed with {}'.format(mesg))
                stop = MPI.Wtime()

                print('Nonlinear optimization finished in {:.2f} s after {} '
                      'evaluations.'.format(stop - start, infodict['nfev']),
                      flush=True)

                print('Uncorrected residual: {}'.format(np.std(get_nl_resid(
                                x0 * 0, all_bin_centers, all_sigmaps,
                                all_hitmaps, all_offsets, domain))))
                print('First guess residual: {}'.format(np.std(get_nl_resid(
                                x0, all_bin_centers, all_sigmaps, all_hitmaps,
                                all_offsets, domain))))
                print('  Optimized residual: {}'.format(np.std(get_nl_resid(
                                xopt, all_bin_centers, all_sigmaps, all_hitmaps,
                                all_offsets, domain))),
                      flush=True)

                nlmap, _, hitmap, nl, _, bins, binmin = get_nl(
                    xopt, all_bin_centers, all_sigmaps, all_hitmaps,
                    all_offsets, domain)
                bin_centers = (0.5 + np.arange(nlmap.size) + binmin) \
                    * self._wbin

                good = hitmap > 100
                nlmap = nlmap[good].copy()
                hitmap = hitmap[good].copy()
                bin_centers = bin_centers[good].copy()

                # Apply a median filter to remove bin-to-bin variation
                # and make the poorly sampled end points usable

                nwin = min(101, (nlmap.size // 8) * 4 + 1)
                good = np.ones(bin_centers.size, dtype=np.bool)
                good[:nwin] = False
                good[-nwin:] = False
                steps = np.diff(bin_centers)
                step = np.median(steps)
                for i in np.argwhere(steps > nwin * step / 10).ravel():
                    good[i - nwin // 2:i + nwin // 2] = False
                bin_centers = scipy.signal.medfilt(bin_centers, nwin)[good]
                nlmap = scipy.signal.medfilt(nlmap, nwin)[good]
            else:
                bin_centers = None
                hitmap = None
                nlmap = None

            bin_centers = self.comm.bcast(bin_centers, root=0)
            hitmap = self.comm.bcast(hitmap, root=0)
            sigmap = self.comm.bcast(nlmap)

            V_in, V_delta = bin_centers, sigmap
            V_out = V_in + V_delta

            # Stretch the last two bins to get a rough extrapolation

            V_out[0] = 0
            V_out[-1] = 1e7

            self.corrections.append([V_out, -V_delta])

            if hdulist is not None:
                phasename = 'phase{:02}'.format(phase)
                cols = []
                cols.append(pf.Column(name='V_out', format='D', array=V_out))
                cols.append(pf.Column(name='V_corr',
                                      format='D', array=-V_delta))
                hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols))
                hdu.header['extname'] = phasename
                hdulist.append(hdu)

            tstop_phase = MPI.Wtime()

            if self.rank == 0:
                print('ADC correction for phase {} done in {:.2f} s'.format(
                        phase, tstop_phase - tstart_phase), flush=True)

        if hdulist is not None:
            if os.path.isfile(fn_out):
                os.remove(fn_out)
            pf.HDUList(hdulist).writeto(fn_out)
            print('ADC correction saved in {}'.format(fn_out), flush=True)
        return

    def apply_correction(self, toi, phase4k, in_place=False):
        if self.corrections is None:
            raise Exception('Cannot apply ADC NL correction without '
                            'measuring it first.')
        if in_place:
            toi_out = toi
        else:
            toi_out = toi.copy()
        for phase in range(self._nphase4k):
            x, y = self.corrections[phase]
            good = phase4k == phase
            toi_out[good] += np.interp(toi[good], x, y)
        return toi_out

    def fit_line(self, xx, yy, nn):
        """
        Fit a line to the (xx,yy) data, taking into account
        the hit counts in nn
        """
        invvar = 1.*nn
        invcov = np.zeros([2, 2])
        invcov[0, 0] = np.sum(invvar)
        invcov[0, 1] = np.sum(invvar * xx)
        invcov[1, 0] = invcov[0, 1]
        invcov[1, 1] = np.sum(invvar * xx ** 2)
        cov = np.linalg.inv(invcov)

        proj = np.zeros(2)
        proj[0] = np.sum(invvar * (yy))
        proj[1] = np.sum(xx * invvar * (yy))

        coeff = np.dot(cov, proj)
        var = np.var(yy - coeff[0] - coeff[1] * xx)
        cov *= var
        return coeff, cov
