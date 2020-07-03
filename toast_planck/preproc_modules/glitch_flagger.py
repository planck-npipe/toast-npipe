# Copyright (c) 2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import toast.timing as timing


class GlitchFlagger():

    def __init__(self, fwhm=5, threshold=4.0, fsample=180.3737, twice=False,
                 wkernel=3):
        """
        Instantiate a glitch flagging object. Parameters:
        fwhm (float): Beam width [arc minutes]
        threshold (float): glitch detection limit in units of the
            filtered signal RMS
        twice(bool): Run the glitch detection on regular and convolved TOD
        """
        self.fwhm = fwhm
        wbin = self.fwhm
        self.order = 6  # order of signal model across 3 bins
        nbin_min = np.int(2 * np.pi / np.radians(wbin / 60))
        nbin = 2
        while nbin < nbin_min:
            nbin *= 2
        wbin = 2 * np.pi / nbin
        self.nbin = nbin
        self.wbin = wbin
        self.threshold = threshold
        self.fsample = fsample
        self.twice = twice
        self.wkernel = wkernel

    def flag_glitches(self, signal_in, flag_in, phase=None, dark=False,
                      pntflag=None):
        """
        Find and flag glitches.
        """
        if not dark:
            if phase is None:
                raise RuntimeError('Optical detectors must provide phase')
            if pntflag is None:
                raise RuntimeError(
                    'Optical detectors must provide pointing flags')

        signal_in = signal_in.copy()
        self.subtract_trend(signal_in)
        flag = flag_in.copy()

        if dark:
            self.flag_outliers(signal_in, flag)
            flag_intense = np.zeros_like(flag)
        else:
            # POD = phase-ordered data
            ind = np.argsort(phase)
            reverse_ind = np.argsort(ind)
            POD_signal_in = signal_in[ind]
            POD_signal = POD_signal_in.copy()
            POD_flag = flag[ind]
            POD_pntflag = pntflag[ind]
            POD_phase = phase[ind]
            bin_lim = np.arange(self.nbin) * self.wbin
            bin_ind = np.searchsorted(POD_phase, bin_lim)
            bin_ranges = [
                (bin_ind[i], bin_ind[i + 1]) for i in range(self.nbin - 1)]
            bin_ranges.append((bin_ind[-1], POD_phase.size))

            # Identify outliers in each phase bin
            POD_signal, POD_flag, bin_rms = self.flag_outliers_by_phase(
                POD_signal, POD_phase, POD_flag, POD_pntflag, bin_ranges)
            POD_signal_estimate = POD_signal_in - POD_signal
            POD_flag_intense = self.get_intense(
                bin_ranges, bin_rms, POD_signal, POD_signal_estimate, POD_flag)

            if self.twice:
                POD_signal2 = np.convolve(signal_in, [.25, .5, .25],
                                          mode='same')[ind]
                # POD_signal2_in = POD_signal2.copy()
                flag2_in = POD_flag[reverse_ind]
                flag2_in = np.convolve(flag2_in, np.ones(3), mode='same') != 0
                POD_flag2_in = flag2_in[ind]
                POD_flag2 = POD_flag2_in.copy()
                POD_signal2, POD_flag2, bin_rms = self.flag_outliers_by_phase(
                    POD_signal2, POD_phase, POD_flag2, POD_pntflag, bin_ranges)
            """
            # DEBUG begin
            import matplotlib.pyplot as plt
            import pdb
            plt.figure()
            good = flag_in[ind] + POD_pntflag == 0
            plt.plot(POD_phase[good] / self.wbin, POD_signal_in[good], '.',
                     label='input')
            good = POD_flag + POD_pntflag == 0
            plt.plot(POD_phase[good] / self.wbin, POD_signal_in[good],
                     label='unflagged')
            plt.plot(POD_phase[good] / self.wbin, POD_signal_estimate[good],
                     label='model')
            good[POD_flag_intense == 0] = False
            plt.plot(POD_phase[good] / self.wbin, POD_signal_in[good], '.',
                     label='unflagged intense')
            # plt.plot(POD_phase[good] / self.wbin, POD_signal[good], '.',
            #         label='unflagged - model')
            if self.twice:
                plt.legend(loc='best')
                plt.figure()
                POD_signal_estimate2 = POD_signal2_in - POD_signal2
                good = POD_flag2_in + POD_pntflag == 0
                plt.plot(POD_phase[good] / self.wbin, POD_signal2_in[good], '.',
                         label='input')
                good = POD_flag2 + POD_pntflag == 0
                plt.plot(POD_phase[good] / self.wbin, POD_signal2_in[good],
                         label='unflagged')
                plt.plot(POD_phase[good] / self.wbin,
                         POD_signal_estimate2[good], label='model')
                good[POD_flag_intense == 0] = False
                plt.plot(POD_phase[good] / self.wbin, POD_signal2_in[good], '.',
                         label='unflagged intense')
                # plt.plot(POD_phase[good] / self.wbin, POD_signal2[good], '.',
                #         label='unflagged - model')
            plt.legend(loc='best')
            plt.show()
            pdb.set_trace()
            # DEBUG end
            """
            if self.twice:
                POD_flag2[POD_flag2_in] = False
                POD_flag[POD_flag2] = True
            flag = POD_flag[reverse_ind]
            # flag = POD_flag[reverse_ind]
            flag_intense = POD_flag_intense[reverse_ind]
            signal_estimate = POD_signal_estimate[reverse_ind]

        if self.wkernel:
            # Extend the flagging
            flag[flag_in] = False
            flag = np.convolve(flag, np.ones(self.wkernel), mode='same') != 0
            flag = np.roll(flag, self.wkernel // 2 - 1)
            flag[flag_in] = True

        return flag, flag_intense, signal_estimate

    def subtract_trend(self, signal):
        """
        subtract a simple trend
        """
        istart = 0
        step = np.int(60 * self.fsample)
        while istart < signal.size:
            istop = istart + step
            if istop + step > signal.size:
                istop += step
            ind = slice(istart, istop)
            offset = np.median(signal[ind])
            signal[ind] -= offset
            istart = istop
        return

    def flag_outliers(self, signal, flag):
        """
        Find outliers in offset-removed signal
        """
        for _ in range(10):
            offset = np.median(signal[flag == 0])
            signal -= offset
            rms = np.mean(signal[flag == 0] ** 2) ** .5
            bad = np.abs(signal) > self.threshold * rms
            bad[flag != 0] = False
            nbad = np.sum(bad)
            if nbad == 0:
                break
            flag[bad] = True
        return

    def _get_bin(self, ibin, signal, phase, flag, pntflag, bin_ranges):
        """
        Return signal in the current bin with margins
        """
        nbin = len(bin_ranges)
        signals = []
        phases = []
        flags = []
        pntflags = []
        # previous bin
        if ibin == 0:
            bin_start, bin_stop = bin_ranges[-1]
        else:
            bin_start, bin_stop = bin_ranges[ibin - 1]
        ind = slice(bin_start, bin_stop)
        signals.append(signal[ind])
        if ibin == 0:
            phases.append(phase[ind] - 2 * np.pi)
        else:
            phases.append(phase[ind])
        flags.append(flag[ind])
        pntflags.append(pntflag[ind])
        # current bin
        bin_start, bin_stop = bin_ranges[ibin]
        ind = slice(bin_start, bin_stop)
        signals.append(signal[ind])
        phases.append(phase[ind])
        flags.append(flag[ind])
        pntflags.append(pntflag[ind])
        # next bin
        if ibin < nbin - 1:
            bin_start, bin_stop = bin_ranges[ibin + 1]
        else:
            bin_start, bin_stop = bin_ranges[0]
        ind = slice(bin_start, bin_stop)
        signals.append(signal[ind])
        if ibin < nbin - 1:
            phases.append(phase[ind])
        else:
            phases.append(phase[ind] + 2 * np.pi)
        flags.append(flag[ind])
        pntflags.append(pntflag[ind])
        center = slice(signals[0].size, signals[0].size + signals[1].size)
        # concatenate
        signals = np.hstack(signals)
        phases = np.hstack(phases)
        flags = np.hstack(flags)
        pntflags = np.hstack(pntflags)
        return signals, phases, flags, pntflags, center

    def robust_rms(self, x):
        """
        Measure the sample variance using the interquartile range (IQR) method
        """
        if len(x) < 4:
            return np.std(x)
        xsorted = np.sort(x)
        nx = x.size
        i1 = np.int(0.25 * nx)
        i2 = np.int(0.75 * nx)
        iqr = xsorted[i2] - xsorted[i1]
        rms = iqr * 0.7412
        return rms

    def flag_outliers_by_phase(self, signal, phase, flag, pntflag, bin_ranges):
        """
        Find outliers in the de-trended signal and derive a signal estimate.
        """
        bin_rms = []
        nbin = len(bin_ranges)
        signal_out = np.zeros_like(signal)
        flag_out = np.zeros_like(flag)
        for ibin in range(nbin):
            bin_start, bin_stop = bin_ranges[ibin]
            ind = slice(bin_start, bin_stop)
            sig, phse, flg, pntflg, center = self._get_bin(
                ibin, signal, phase, flag, pntflag, bin_ranges)
            rms = 0
            for iiter in range(10):
                good_ind = flg + pntflg == 0
                ngood = np.sum(good_ind)
                if ngood < 10:
                    # This bin is beyond hope
                    flg[:] = True
                    break
                if iiter < 2:
                    # Signal model is an offset
                    offset = np.median(sig[good_ind])
                else:
                    # Signal model is a polynomial
                    offset = self.fit_poly(phse, sig, good_ind)
                sig -= offset
                rms = self.robust_rms(sig[good_ind])
                bad = np.abs(sig) > self.threshold * rms
                bad[flg != 0] = False
                nbad = np.sum(bad)
                if nbad == 0 and iiter > 2:
                    break
                flg[bad] = True
            signal_out[ind] = sig[center]
            flag_out[ind] = flg[center]
            bin_rms.append(rms)

        return signal_out, flag_out, np.array(bin_rms)

    def get_intense(self, bin_ranges, bin_rms, noise, estimate, flag):
        """
        Flag all samples falling into bins with extreme RMS as intense
        """
        snr = []
        for ibin, ((bin_start, bin_stop), rms) in enumerate(zip(bin_ranges,
                                                                bin_rms)):
            ind = slice(bin_start, bin_stop)
            good = flag[ind] == 0
            rms_signal = np.std(estimate[ind][good])
            rms_noise = np.std(noise[ind][good])
            snr.append(rms_signal / rms_noise)
        flag_intense = np.zeros_like(flag)
        good = bin_rms != 0
        for _ in range(10):
            ngood = np.sum(good)
            good_rms = bin_rms[good]
            rms_median = np.median(good_rms)
            rms_rms = (np.sum((good_rms - rms_median) ** 2) / (ngood - 1)) ** .5
            for ibin, ((bin_start, bin_stop), rms) in enumerate(zip(bin_ranges,
                                                                    bin_rms)):
                if rms < max(2 * rms_median,
                             rms_median + 5 * rms_rms) and snr[ibin] < 1:
                    continue
                good[ibin] = False
                ind = slice(bin_start, bin_stop)
                flag_intense[ind] = True

        return flag_intense

    def fit_poly(self, x, y, ind_fit):
        templates = []
        xx = (x - np.mean(x)) / np.ptp(x)
        for iorder in range(self.order + 1):
            templates.append(xx[ind_fit] ** iorder)
        templates = np.vstack(templates)
        invcov = np.dot(templates, templates.T)
        cov = np.linalg.inv(invcov)
        proj = np.dot(templates, y[ind_fit])
        coeff = np.dot(cov, proj)
        poly = np.zeros_like(y)
        for iorder, cc in enumerate(coeff):
            poly += cc * xx ** iorder
        return poly
