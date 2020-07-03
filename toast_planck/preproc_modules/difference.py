# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from toast_planck.preproc_modules.filters import flagged_running_average
from toast_planck.reproc_modules.destripe_tools import linear_regression

import scipy.optimize
from toast._libtoast import filter_polynomial as polyfilter

import numpy as np
import toast.timing as timing

from .signal_estimation import SignalEstimator


class Differencer():

    def __init__(self, nbin=1000):
        """
        Instantiate a differencing object. Parameters:
        nbin   -- number of phase bins in the signal estimate.
        """
        self.nbin = nbin
        self.estim = SignalEstimator(self.nbin)

    def _apply_polyfilter(self, signals, good, order=0):
        """ This method fits and removes a low order polynomial from
        all of the signals.

        """
        polyfilter(order,
                   np.logical_not(good).astype(np.uint8),
                   signals,
                   np.array([0]),
                   np.array([good.size]))
        return

    def difference_lfi(
            self, signal0, flag0, signal1, flag1, fsample,
            pntflag=None, ssoflag=None, maskflag=None, bg=None, dipole=None,
            weight0=None, weight1=None, coadd_diodes=True,
            lowpass0=None, lowpass1=None):
        """
        Perform R-factor differencing on given signal and co-add them for
        optimal noise. Inputs:
        signal0 -- diode 0 signal+reference to be corrected (2-column array)
        signal1 -- diode 1 signal+reference to be corrected (2-column array)
        flag -- extra processing flags not present in signal.mask
        bg, dipole -- estimate of the total sky emission in the same units as
            signal
        coadd_diodes(True) -- Return only one, co-added signal
        """

        test = [weight0 is None, weight1 is None,
                lowpass0 is None, lowpass1 is None]
        if np.any(test) and not np.all(test):
            raise RuntimeError('Co-add weights can only be supplied with the '
                               'low pass filters.')

        flag = np.logical_or(flag0, flag1)

        ind_fit = np.logical_not(flag)
        for flg in [pntflag, ssoflag, maskflag]:
            if flg is not None:
                ind_fit[flg] = False
        n_fit = np.sum(ind_fit)
        if n_fit == 0:
            raise RuntimeError('No samples left to difference')

        n = ind_fit.size

        sky0 = signal0[0].copy()
        load0 = signal0[1].copy()
        sky1 = signal1[0].copy()
        load1 = signal1[1].copy()

        self._apply_polyfilter([sky0, sky1, load0, load1], ind_fit)

        # Process the reference load TODs into noise estimates

        stack = []
        fullstack = []

        # Start with the full reference load TOD
        resid0 = load0.copy()
        resid1 = load1.copy()

        # Add signal offset template

        offset = np.ones(n, dtype=np.float64)

        stack.append(offset[ind_fit].copy())
        fullstack.append(offset)

        nn = resid0.size
        npad = 2
        while npad < nn:
            npad *= 2
        freq = np.fft.rfftfreq(npad, 1 / fsample)
        fresid0 = np.fft.rfft(resid0, n=npad)
        fresid1 = np.fft.rfft(resid1, n=npad)

        if lowpass0 is None or lowpass1 is None:
            filter_params = []
        else:
            # Use the provided lowpass filters
            noise0 = self._lowpass_lfi(freq, fresid0, lowpass0, npad, nn)
            sky0 -= noise0
            noise1 = self._lowpass_lfi(freq, fresid1, lowpass1, npad, nn)
            sky1 -= noise1
            filter_params = [lowpass0, lowpass1]

        # Marginalize over the global signal estimate

        stack.append((bg + dipole)[ind_fit].copy())

        # Measure diode gains and remove offset

        gains = []
        for sky in [sky0, sky1]:
            coeff, _, _, _ = linear_regression(
                stack, np.ascontiguousarray(sky[ind_fit]))
            gains.append(1 / coeff[-1])
            for cc, tod in zip(coeff[:2], fullstack):
                sky -= cc * tod

        if not coadd_diodes:
            if lowpass0 is None:
                gains = []
                for sky, fresid in zip([sky0, sky1], [fresid0, fresid1]):
                    gain, opt, offset = self._fit_lowpass_lfi_single(
                        sky, fresid, npad, nn, bg, dipole, ind_fit)
                    # Low pass filter the Load signal and subtract
                    lowpassed = self._lowpass_lfi(freq, fresid, opt, npad, nn)
                    sky -= lowpassed + offset
                    gains.append(1 / gain)
                    filter_params.append(opt)

            gain0, gain1 = np.array(gains)

            cleaned0 = gain0 * sky0 - (bg + dipole)
            cleaned1 = gain1 * sky1 - (bg + dipole)

            good = np.logical_not(flag)
            if maskflag is not None:
                good[maskflag] = False
            if pntflag is not None:
                good[pntflag] = False

            rms0 = np.std(cleaned0[good])
            rms1 = np.std(cleaned1[good])

            return [(sky0, sky1), gain0, gain1, rms0, rms1,
                    filter_params]

        if weight0 is None:
            (weight0, weight1, opt0, opt1, gain, offset,
             ) = self._fit_lowpass_lfi(
                    sky0, sky1, fresid0, fresid1, freq, npad, nn, bg, dipole,
                    ind_fit)
            # Low pass filter the Load signals and subtract
            lowpassed0 = self._lowpass_lfi(freq, fresid0, opt0, npad, nn)
            lowpassed1 = self._lowpass_lfi(freq, fresid1, opt1, npad, nn)
            sky0 -= lowpassed0
            sky1 -= lowpassed1
            filter_params = [opt0, opt1]
            signal = weight0 * sky0 + weight1 * sky1 - offset
        else:
            signal = weight0 * sky0 + weight1 * sky1
            # Subtract the offset that is compatible with the
            # Load low-pass filters.
            coeff, _, _, _ = linear_regression(
                stack, signal[ind_fit].copy())
            gain = 1 / coeff[-1]
            for cc, template in zip(coeff[:2], fullstack):
                signal -= cc * template

        gain0, gain1 = np.array(gains)

        cleaned0 = gain0 * sky0 - (bg + dipole)
        cleaned1 = gain1 * sky1 - (bg + dipole)

        good = np.logical_not(flag)
        if maskflag is not None:
            good[maskflag] = False
        if pntflag is not None:
            good[pntflag] = False

        rms0 = np.std(cleaned0[good])
        rms1 = np.std(cleaned1[good])

        """
        # Finally, construct and remove a 4-minute thermal baseline
        thermal = flagged_running_average(
            weight0 * cleaned0 + weight1 * cleaned1, flag,
            np.int(240 * fsample))
        signal -= thermal
        """


        return (signal, flag, gain0, gain1, rms0, rms1,
                filter_params, weight0, weight1)

    def difference_hfi(
            self, signal, flag, dark1, darkflag1, dark2, darkflag2,
            phase, fsample, pntflag=None, ssoflag=None, maskflag=None,
            signal_estimate=None, fast=True):
        """
        Perform R-factor differencing on given signal and coadd them for
        optimal noise. Inputs:
        phase -- spin phase in RADIANS
        flag -- extra processing flags not present in signal.mask
        signal_estimate(None) -- estimate of the total sky emission in
            the same units as signal

        """

        if not np.all(np.isfinite(dark1)):
            raise RuntimeError('Dark-1 has {} NaNs.'.format(
                np.sum(np.logical_not(np.isfinite(dark1)))))
        if not np.all(np.isfinite(dark2)):
            raise RuntimeError('Dark-2 has {} NaNs.'.format(
                np.sum(np.logical_not(np.isfinite(dark2)))))

        ind_fit = np.logical_not(flag)
        for flg in [darkflag1, darkflag2, pntflag, ssoflag, maskflag]:
            if flg is not None:
                ind_fit[flg] = False
        nn = ind_fit.size
        n_fit = np.sum(ind_fit)
        if n_fit == 0:
            raise RuntimeError('No samples left to difference')

        # Process the reference load TODs into noise estimates

        sky = signal.copy()
        resid1 = dark1.copy()
        resid2 = dark2.copy()

        self._apply_polyfilter([sky, resid1, resid2], ind_fit)

        if signal_estimate is None:
            # Subtract signal
            good = (flag == 0)
            self.estim.fit(phase[good], signal[good])
            signal_estimate = self.estim.eval(phase)

        npad = 2
        while npad < nn:
            npad *= 2
        freq = np.fft.rfftfreq(npad, 1 / fsample)

        fresid1 = np.fft.rfft(resid1, n=npad)
        fresid2 = np.fft.rfft(resid2, n=npad)

        if fast:
            (weight1, weight2, filter_params, gain, offset,
             lowpassed) = self._fit_lowpass_hfi_fast(
                    sky, fresid1, fresid2, freq, npad, nn, signal_estimate,
                    ind_fit)
        else:
            (weight1, weight2, filter_params, gain, offset,
             ) = self._fit_lowpass_hfi(
                    sky, fresid1, fresid2, freq, npad, nn, signal_estimate,
                    ind_fit)
            # Low pass filter the co-added dark signal and subtract
            lowpassed = self._lowpass_hfi(
                freq, weight1 * fresid1 + weight2 * fresid2,
                filter_params, npad, nn)

        sky -= lowpassed + offset

        return sky, flag, gain, filter_params, weight1, weight2

    def _fit_lowpass_lfi_single(self, sky, fresid, freq, npad, nn, bg, dipole,
                                ind_fit):
        """ Fit for optimal co-add weights and low-pass filter parameters

        """
        p0 = [1, 0,  # w, gain, offset
              1, 1e-1, -1.5]  # R, sigma, alpha
        result = scipy.optimize.least_squares(
            single_residuals_lfi, p0, method='lm',
            args=(sky, fresid, freq, npad, nn, (bg + dipole), ind_fit,
                  self._lowpass_lfi), max_nfev=1000)
        if not result.success:
            raise RuntimeError(
                'least_squares failed: {}'.format(result.message))
        gain, offset = result.x[:2]
        opt = result.x[2:]
        return gain, opt, offset

    def _lowpass_lfi(self, freq, fresid, params, npad, nn):
        """ Construct a low-pass filter from params, apply to fresid and
        return the filtered signal.
        """
        iszero = freq == 0
        nonzero = np.logical_not(iszero)
        lowpassfilter = np.ones_like(freq)
        R, sigma, alpha = params
        lowpassfilter[iszero] = R ** 2
        correlated = freq[nonzero] ** alpha
        lowpassfilter[nonzero] = R ** 2 * correlated / (correlated + sigma ** 2)
        noise = np.fft.irfft(fresid * lowpassfilter, n=npad)[:nn]
        return noise

    def _fit_lowpass_lfi(self, sky0, sky1, fresid0, fresid1, freq, npad, nn, bg,
                         dipole, ind_fit):
        """ Fit for optimal co-add weights and low-pass filter parameters

        """
        p0 = [0.5, 1, 0,  # w, gain, offset
              1, 1e-1, -1.5,  # R0, sigma0, alpha0
              1, 1e-1, -1.5]  # R1, sigma1, alpha1
        result = scipy.optimize.least_squares(
            full_residuals_lfi, p0, method='lm',
            args=(sky0, sky1, fresid0, fresid1, freq, npad, nn,
                  (bg + dipole), ind_fit, self._lowpass_lfi),
            max_nfev=200)
        if not result.success:
            raise RuntimeError(
                'least_squares failed: {}'.format(result.message))
        w, gain, offset = result.x[:3]
        weight0 = w
        weight1 = 1 - w
        opt0, opt1 = result.x[3:].reshape([2, -1])
        return weight0, weight1, opt0, opt1, gain, offset

    def _fit_lowpass_hfi(self, sky, fresid1, fresid2, freq, npad, nn,
                         signal_estimate, ind_fit):
        """ Fit for optimal co-add weights and low-pass filter parameters

        """
        p0 = [0.5, 1, 0,  # w, gain, offset
              1, -5]  # R, logfcut
        bounds = ([0, 0, -np.inf, -np.inf, -np.inf, -10],
                  [1, 2, np.inf, np.inf, np.inf, 0])
        result = scipy.optimize.least_squares(
            full_residuals_hfi, p0, method='trf', bounds=bounds,
            args=(sky, fresid1, fresid2, freq, npad, nn,
                  signal_estimate, ind_fit, self._lowpass_hfi),
            max_nfev=100)
        if not result.success:
            raise RuntimeError(
                'least_squares failed: {}'.format(result.message))
        w, invgain, offset = result.x[:3]
        weight1 = w
        weight2 = 1 - w
        filter_params = result.x[3:]
        return weight1, weight2, filter_params, 1 / invgain, offset

    def _fit_lowpass_hfi_fast(self, sky, fresid1, fresid2, freq, npad, nn,
                              signal_estimate, ind_fit):
        """ Fit for co-add weights and low-pass filter parameters
        assuming fixed fcut.

        """
        logfcut = np.log(.01)  # median value from earlier runs
        resid1 = self._lowpass_hfi(freq, fresid1, [1, logfcut], npad, nn)
        resid2 = self._lowpass_hfi(freq, fresid2, [1, logfcut], npad, nn)
        templates = np.vstack([
            np.ones(nn)[ind_fit], signal_estimate[ind_fit],
            resid1[ind_fit], resid2[ind_fit]])
        coeff, _, _, _ = linear_regression(
            templates, np.ascontiguousarray(sky[ind_fit]))
        (offset, invgain, weight1, weight2) = coeff
        filter_params = [1, logfcut]
        lowpassed = weight1 * resid1 + weight2 * resid2
        return (weight1, weight2, filter_params, 1 / invgain, offset,
                lowpassed)

    def _lowpass_hfi(self, freq, fresid, params, npad, nn):
        """ Construct a low-pass filter from params, apply to fresid and
        return the filtered signal.

        """
        iszero = freq == 0
        nonzero = np.logical_not(iszero)
        # Construct and apply a Fourier domain filter
        lowpassfilter = np.ones_like(freq)
        R, logfcut = params
        fcut = np.exp(logfcut)
        alpha = -3
        lowpassfilter[iszero] = R ** 2
        freqalpha = freq[nonzero] ** alpha
        lowpassfilter[nonzero] = R ** 2 * freqalpha ** 3 / (
            (freqalpha + np.abs(fcut) ** alpha)
            * (freqalpha + np.abs(10 * fcut) ** alpha)
            * (freqalpha + np.abs(100 * fcut) ** alpha))
        noise = np.fft.irfft(fresid * lowpassfilter, n=npad)[:nn]
        return noise


def single_residuals_lfi(p, sky, fresid, freq, npad, nn,
                         signal_estimate, ind_fit, lowpass):
    """ Nonlinear fitting of the diode weights.
    """
    gain, offset = p[:2]
    opt = p[2:]

    lowpassed = lowpass(freq, fresid, opt, npad, nn)
    return (sky - lowpassed - gain * signal_estimate -
            offset)[ind_fit]


def full_residuals_lfi(p, sky0, sky1, fresid0, fresid1, freq, npad, nn,
                       signal_estimate, ind_fit, lowpass):
    """ Nonlinear fitting of the diode weights.
    """
    w, gain, offset = p[:3]
    opt0, opt1 = p[3:].reshape([2, -1])

    lowpassed0 = lowpass(freq, fresid0, opt0, npad, nn)
    lowpassed1 = lowpass(freq, fresid1, opt1, npad, nn)

    cleaned0 = sky0 - lowpassed0
    cleaned1 = sky1 - lowpassed1

    return (w * cleaned0 + (1 - w) * cleaned1 - gain * signal_estimate -
            offset)[ind_fit]


def full_residuals_hfi(p, sky, fresid1, fresid2, freq, npad, nn,
                       signal_estimate, ind_fit, lowpass):
    """ Nonlinear fitting of dark bolometer filters.
    """
    w, invgain, offset = p[:3]
    opt = p[3:]

    lowpassed = lowpass(freq, w * fresid1 + (1 - w) * fresid2, opt, npad, nn)

    return (sky - lowpassed - invgain * signal_estimate -
            offset)[ind_fit]
