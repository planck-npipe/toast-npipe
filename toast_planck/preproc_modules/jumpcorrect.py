# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import scipy.signal

from toast._libtoast import filter_polynomial as polyfilter

from .signal_estimation import SignalEstimator


class JumpCorrector():

    def __init__(self, filterlen, nbin=10000, tol=1.0, threshold=4.0, order=0):
        """
        Instantiate a jump correction object. Parameters:
        filterlen -- length of the matched filter in samples
            (typically a few minutes)
        nbin   -- number of phase bins in the signal estimate.
        tol    -- size of the flagged region (in samples) to put around
            a detected jump
        threshold -- jump detection limit in units of the filtered signal RMS
        """
        self.filterlen = filterlen
        self.nbin = nbin
        self.tol = tol
        self.threshold = threshold
        self.estim = SignalEstimator(nbin=self.nbin)
        self.order = order

    def _find_gaps(self, flag):
        # Find continuous flagged regions left by the despiker and
        # save the start and stop locations so that we don't falsely
        # detect jumps at the gap boundaries.
        flagdiff = np.diff(flag.astype(int))
        gap_starts = np.where(flagdiff > 0)[0]
        gap_stops = np.where(flagdiff < 0)[0]
        if flag[0] and gap_starts[0] != 0:
            gap_starts = np.append([0], gap_starts)
        if flag[-1] and gap_stops[-1] != flag.size - 1:
            gap_stops = np.append(gap_stops, [flag.size - 1])
        gap_lengths = gap_stops - gap_starts
        ind = gap_lengths > self.filterlen
        positions = list(sorted(np.hstack([gap_starts[ind], gap_stops[ind]])))
        return positions

    def _subtract_signal(self, signal_estimate, signal, good, phase,
                         cleaned_signal):
        """ subtract signal
        """
        if signal_estimate is not None:
            unrolled_signal = signal_estimate + \
                np.ma.median(signal[good] - signal_estimate[good])
            signal_estimate_is_binned = False
        else:
            try:
                self.estim.fit(phase[good], signal[good])
            except Exception as e:
                raise Exception('jump_corrector: Signal estimation '
                                'failed: {}'.format(e))
            unrolled_signal = self.estim.eval(phase)
            signal_estimate_is_binned = True
        cleaned_signal[good] -= unrolled_signal[good]
        return signal_estimate_is_binned

    def _fill_gaps(self, good, cleaned_signal):
        """ simple gap fill
        """
        bad = np.logical_not(good)
        nbad = np.sum(bad)
        rms = np.std(cleaned_signal[good])
        cleaned_signal[bad] = np.random.randn(nbad) * rms
        return

    def _apply_filter(self, cleaned_signal, step_filter):
        """ Apply the matched filter
        """
        filtered_signal = scipy.signal.fftconvolve(
            cleaned_signal, step_filter, mode='same')
        return filtered_signal

    def _suppress_gaps(self, filtered_signal, positions):
        """ Suppress the filtered signal near gaps not to have false
        detections

        """
        for pos in positions:
            start = pos - self.filterlen
            stop = pos + self.filterlen
            ind = slice(start, stop)
            filtered_signal[ind] = 0
        filtered_signal[:self.filterlen] = 0
        filtered_signal[-self.filterlen:] = 0
        return

    def correct(self, signal, flag, phase, dark=False, signal_estimate=None,
                signal_estimate_is_binned=True):
        """
        Perform jump correction on given signal. Inputs:
        signal -- demodulated and gap-filled signal to be corrected
            (masked array)
        phase -- spin phase in RADIANS
        dark   -- enable dark bolometer mode (disable signal subtraction)
        signal_estimate(None) -- estimate of the total sky emission in
            the same units as signal
        """
        corrected_signal = signal.copy()
        flag_out = flag.copy()

        step_filter = self._get_stepfilter(self.filterlen)
        amplitudes = []
        positions = self._find_gaps(flag)
        njump = 0

        while True:
            cleaned_signal = corrected_signal.copy()
            good = flag_out == 0
            if not dark:
                signal_estimate_is_binned = self._subtract_signal(
                    signal_estimate, signal, good, phase, cleaned_signal)
            else:
                signal_estimate_is_binned = False

            polyfilter(
                self.order,
                flag_out.astype(np.uint8),
                [cleaned_signal],
                np.array([0]),
                np.array([signal.size]),
            )

            self._fill_gaps(good, cleaned_signal)
            filtered_signal = self._apply_filter(cleaned_signal, step_filter)
            self._suppress_gaps(filtered_signal, positions)

            # find the peaks in the filtered TOI and correct for the
            # jumps accordingly

            peaks = self._find_peaks(filtered_signal, flag, flag_out,
                                     lim=self.threshold,
                                     tol=self.filterlen // 2)

            if len(peaks) == 0:
                break

            njump += len(peaks)
            if njump > 10:
                # Prevent indefinite iterations. 10 jumps is too much anyways.
                break

            if not dark and signal_estimate_is_binned:
                self._correct_for_signal_subtraction(good, peaks, phase,
                                                     len(signal))
            corrected_signal, flag_out = self._remove_jumps(
                corrected_signal, flag_out, peaks, self.tol)

            for peak, _, amplitude in peaks:
                positions.append(peak)
                amplitudes.append(amplitude)

        return corrected_signal, flag_out, njump

    def _get_stepfilter(self, m):
        """
        Return the time domain matched filter kernel of length m.
        """
        h = np.zeros(m)
        h[:m // 2] = 1
        h[m // 2:] = -1
        # This turns the interpretation of the peak amplitude directly
        # into the step amplitude
        h /= m // 2
        return h

    def _find_peaks(self, toi, flag, flag_out, lim=3.0, tol=1e4, sigma_in=None):
        """
        Find the peaks and their amplitudes in the match-filtered TOI.
        Inputs:
        lim -- threshold for jump detection in units of filtered TOI RMS.
        tol -- radius of a region to mask from further peak finding upon
            detecting a peak.
        sigma_in -- an estimate of filtered TOI RMS that overrides the
             sample variance otherwise used.

        """
        peaks = []
        mytoi = np.ma.masked_array(toi)
        # Do not accept jumps at the ends due to boundary effects
        lbound = tol
        rbound = tol
        mytoi[:lbound] = np.ma.masked
        mytoi[-rbound:] = np.ma.masked
        if sigma_in is None:
            sigma = self._get_sigma(mytoi, flag_out, tol)
        else:
            sigma = sigma_in

        if np.isnan(sigma) or sigma == 0:
            npeak = 0
        else:
            npeak = np.ma.sum(np.abs(mytoi) > sigma * lim)

        # Only one jump per iteration
        if npeak > 0:
            imax = np.argmax(np.abs(mytoi))
            amplitude = mytoi[imax]
            significance = np.abs(amplitude) / sigma

            # Mask the peak for taking mean and finding additional peaks
            istart = max(0, imax - tol)
            istop = min(len(mytoi), imax + tol)
            # mask out the vicinity not to have false detections near the peak
            mytoi[istart:istop] = np.ma.masked
            flag_out[istart:istop] = True
            if sigma_in is None:
                sigma = self._get_sigma(mytoi, flag_out, tol)

            # Excessive flagging is a sign of false detection
            if significance > 5 or (float(np.sum(flag[istart:istop]))
                                    / (istop - istart) < .5):
                peaks.append((imax, significance, amplitude))

            npeak = np.sum(np.abs(mytoi) > sigma * lim)
        return peaks

    def _get_sigma(self, toi, flag, tol):

        full_flag = np.logical_or(flag, toi == 0)

        sigmas = []
        nn = len(toi)
        for start in range(tol, nn - tol, 2 * tol):
            stop = start + 2 * tol
            if stop > nn - tol:
                break
            ind = slice(start, stop)
            x = toi[ind][full_flag[ind] == 0]
            if len(x) != 0:
                rms = np.sqrt(np.mean(x.data ** 2))
                sigmas.append(rms)

        if len(sigmas) != 0:
            sigma = np.median(sigmas)
        else:
            sigma = 0.
        return sigma

    def _unroll_jumps(self, peaks, nn):
        """
        Returns a timeline representation of the jumps contained in
        peaks, an output of _find_peaks.
        nn is the length of the returned TOI vector.

        """
        y = np.zeros(nn)
        for peak, _, amplitude in peaks:
            y[peak:] -= amplitude

        y -= np.mean(y)
        return y

    def _remove_jumps(self, signal, flag, peaks, tol):
        """
        Removes the jumps described by peaks from x.
        Adds a buffer of flags with radius of tol.

        """
        corrected_signal = signal.copy()
        flag_out = flag.copy()
        for peak, _, amplitude in peaks:
            corrected_signal[peak:] -= amplitude
            flag_out[peak - int(tol):peak + int(tol)] = True
        return corrected_signal, flag_out

    def _correct_for_signal_subtraction(self, good, peaks, phase, nn):
        """
        Correct the peak (jump) estimate for suppression that comes from
        signal subtraction.
        nn is the length of the unrolled TOI.

        """
        jump_toi = np.ma.masked_array(self._unroll_jumps(peaks, nn))
        jump_toi_orig = copy.deepcopy(jump_toi)
        estim = self.estim
        try:
            estim.fit(phase[good], jump_toi[good])
        except Exception as e:
            print('WARNING Spline estimation failed in jump '
                  'correction::_correct_for_signal_subtraction: {}. '
                  'Leaving the small bias in the jump sizes.'.format(e))
            return

        jump_toi -= estim.eval(phase)

        for ii in range(len(peaks)):
            peak, significance, amplitude = peaks[ii]
            jump0 = np.diff(jump_toi_orig[peak - 1:peak + 1])[0]
            jump = np.diff(jump_toi[peak - 1:peak + 1])[0]
            peaks[ii] = (peak, significance, amplitude * jump0 / jump)
        return
