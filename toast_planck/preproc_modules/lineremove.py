# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from toast_planck.reproc_modules.destripe_tools import build_4k_templates

import numpy as np
import toast.timing as timing

from .filters import flagged_running_average


class LineRemover():

    def __init__(self, fsample=180.3737, nline=15, nline_intense=6,
                 threshold=-1, fittime=300):
        """
        Instantiate a line remover object. Parameters:
        fsample -- sampling frequency
        threshold -- required signal-to-noise for a given line to
            perform subtraction
        nline -- number of lines to correct. The templates are
            approximately ordered by intensity
        nline_intense -- number of lines treated as intense: fit is done
            on a very short time scale to track the evolution of the
            line amplitude.
        fittime -- maximum length to fit at a time [seconds] for the
            intense lines
        """
        self.fsample = fsample
        self.nline = nline
        self.nline_intense = nline_intense
        self.threshold = threshold
        self.ntemplate = 2  # cosine and sine per line
        # self.ntemplate = 4 # cosine, sine and derivatives per line
        # self.ntemplate = 6 # cosine, sine and 2 derivatives per line
        self.fitlen = int(fittime * fsample)

    def remove(self, start, stop, signal, flag, phase,
               dark=False, maskflag=None, signal_estimate=None,
               return_line_estimate=False, pntflag=None):
        """
        Perform thermal decorrelation. Inputs:
        signal -- demodulated and gap-filled signal to be corrected
        phase -- spin phase in RADIANS
        timestamp -- sample times for creating the templates
        fsample -- approximate sampling frequency for this ring
        dark -- enable dark bolometer mode (disable signal estimation)
        signal_estimate(None) -- estimate of the total sky emission in
            the same units as signal
        return_line_estimate(False) -- Return a timeline of the
            estimated 4K line component
        """

        if return_line_estimate:
            line_estimate = np.zeros(len(signal))
        else:
            line_estimate = None

        if signal_estimate is None and not dark:
            raise RuntimeError(
                'signal estimate is required for optical bolometers.')

        # Make a working copy of the TOI

        cleaned_signal = signal.copy()
        corrected_signal = signal.copy()

        # Auxiliary masks

        ind_fit = np.logical_not(flag)
        for flg in [pntflag, maskflag]:
            if flg is not None:
                ind_fit[flg] = False

        # Subtract the signal and a trending polynomial

        self.__remove_signal(
            signal, flag, cleaned_signal, dark,
            signal_estimate=signal_estimate)

        # Estimate noise level

        rms = np.std(cleaned_signal[ind_fit])

        # Fit and subtract the templates in self.fitlen-sized chunks

        first_line = 0
        if self.nline_intense > 0:
            coeffs_intense, errs_intense = self.__remove_lines_by_interval(
                signal, cleaned_signal, corrected_signal, flag, start, stop,
                self.fitlen, line_estimate,
                ind_fit, first_line, self.nline_intense, rms)
            first_line += self.nline_intense
        else:
            coeffs_intense, errs_intense = None, None

        if first_line < self.nline:
            coeffs, errs = self.__remove_lines_by_interval(
                signal, cleaned_signal, corrected_signal, flag, start, stop,
                100000, line_estimate,
                ind_fit, first_line, self.nline - first_line, rms)
        else:
            coeffs, errs = None, None

        # Get the average amplitude and phase

        if coeffs_intense is not None:
            coeff_intense = np.mean(coeffs_intense, 0)
            err_intense = np.mean(errs_intense, 0)
        else:
            coeff_intense = []
            err_intense = []

        if coeffs is not None:
            coeff = np.mean(coeffs, 0)
            err = np.mean(errs, 0)
        else:
            coeff = []
            err = []

        coeff = np.hstack([coeff_intense, coeff])
        err = np.hstack([err_intense, err])

        frequencies = []
        amplitudes = []

        for iline in range(self.nline):
            i0 = self.ntemplate * iline
            freq = np.array([10, 30, 50, 70, 17, 20, 40, 60, 80,
                             16, 25, 43, 46, 48, 57])[iline]
            a = coeff[i0]
            b = coeff[i0 + 1]
            amplitudes.append((a, b))
            frequencies.append(freq)

        return corrected_signal, flag, line_estimate, frequencies, amplitudes

    def __remove_lines_by_interval(self, signal, cleaned_signal,
                                   corrected_signal, flag, start, stop,
                                   fitlen, line_estimate, ind_fit, first_line,
                                   nline, rms):

        coeffs = []
        errs = []
        istart = 0
        istep = 0

        while istart < signal.size:
            istep += 1
            # Set the interval by requiring a given number of
            # unflagged samples
            istop = istart
            n = 0
            while istop < signal.size and n < fitlen:
                if ind_fit[istop]:
                    n += 1
                istop += 1

            # Should there be less than fitlen samples left,
            # merge this and the last interval
            if istop + fitlen > signal.size:
                istop = signal.size

            ind = slice(istart, istop)
            ind_fit_slice = ind_fit[ind]
            good = flag[ind] == 0

            # Construct the template matrix using only unflagged samples
            # for efficiency

            full_templates = build_4k_templates(
                start + istart, start + istop, first_line, nline,
                self.ntemplate)

            templates = full_templates[:, ind_fit_slice].copy()

            # linear regression

            xx = np.dot(templates, templates.T)
            xxinv = np.linalg.inv(xx)

            coeff = np.dot(xxinv, np.dot(templates,
                                         cleaned_signal[ind][ind_fit_slice]))
            coeffs.append(coeff)

            # Error estimate on each coefficient

            err = np.diag(xxinv) ** .5 * rms
            errs.append(err)

            # cast the template amplitudes into a single amplitude and
            # a phase shift of the form A.cos(2.pi.f.t-alpha)

            for iline in range(nline):
                i0 = self.ntemplate * iline
                a = coeff[i0]
                b = coeff[i0 + 1]
                a_err = err[i0]
                b_err = err[i0 + 1]
                amplitude = np.sqrt(a ** 2 + b ** 2)
                amplitude_err = (np.abs(a * a_err) + np.abs(b * b_err)) \
                    / amplitude

                remove = amplitude / amplitude_err > self.threshold

                if remove:
                    atemplate = full_templates[i0]
                    btemplate = full_templates[i0 + 1]
                    line_template = a * atemplate + b * btemplate
                    if self.ntemplate > 2:
                        a_deriv = coeff[i0 + 2]
                        b_deriv = coeff[i0 + 3]
                        atemplate_deriv = full_templates[i0 + 2]
                        btemplate_deriv = full_templates[i0 + 3]
                        line_template += a_deriv * atemplate_deriv \
                            + b_deriv * btemplate_deriv
                    if self.ntemplate > 4:
                        a_deriv2 = coeff[i0 + 4]
                        b_deriv2 = coeff[i0 + 5]
                        atemplate_deriv2 = full_templates[i0 + 4]
                        btemplate_deriv2 = full_templates[i0 + 5]
                        line_template += a_deriv2 * atemplate_deriv2 \
                            + b_deriv2 * btemplate_deriv2
                    if line_estimate is not None:
                        line_estimate[ind] += line_template
                    # Subtract the template.
                    # Gap-filled samples don't have 4K lines.
                    cleaned_signal[ind][good] -= line_template[good]
                    corrected_signal[ind][good] -= line_template[good]

            istart = istop
        return coeffs, errs

    def __remove_signal(self, signal, flag, cleaned_signal, dark,
                        signal_estimate=None):
        """
        Create and subtract a smoothed noise estimate from the
        signal-removed TOI
        """

        good = flag == 0

        if not dark:
            # subtract signal
            unrolled_signal = signal_estimate \
                              + np.median(signal[good] - signal_estimate[good])
            cleaned_signal -= unrolled_signal
        else:
            unrolled_signal = None

        trend = flagged_running_average(cleaned_signal, flag, 10822)
        cleaned_signal -= trend
        return unrolled_signal, trend
