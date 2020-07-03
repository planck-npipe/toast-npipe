# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from toast._libtoast import filter_polynomial as polyfilter

from toast_planck.reproc_modules.destripe_tools import (
    build_spike_templates, linear_regression)


class LineRemoverLFI():

    def __init__(self):
        """
        Instantiate a lineremover object.
        """
        pass

    def apply_polyfilter(self, signals, good):
        """ Remove the sub-harmonic modes by fitting and removing a
        linear trend.  This effectively high-pass filters the signal.

        """
        polyfilter(0,
                   np.logical_not(good).astype(np.uint8),
                   list(signals),
                   np.array([0]),
                   np.array([good.size]),
        )
        return

    def get_templates(self, obt, fsample, ind_fit, signal_estimate):
        # Process the reference load TODs into noise estimates
        stack = []
        # Add 1Hz spike templates
        spike_templates = build_spike_templates(obt, fsample)
        for spike_template in spike_templates:
            stack.append(spike_template[ind_fit].copy())
        estimate = signal_estimate[ind_fit].astype(np.float64).copy()
        stack.append(estimate)
        return stack, spike_templates

    def fit_templates(self, stack, ind_fit, sig):
        # Marginalize over the global signal estimate
        sig_fit = sig[ind_fit].astype(np.float64).copy()
        coeff, _, _, _ = linear_regression(stack, sig_fit)
        spike_params = []
        for (a, b) in coeff[:-1].reshape([-1, 2]):
            spike_params.append((a, b))
        return spike_params

    def assemble_line_template(self, sig, spike_params, spike_templates):
        line_template = np.zeros_like(sig)
        iline = 0
        for (a, b) in spike_params:
            if a == 0 and b == 0:
                break
            line_template += a * spike_templates[2 * iline]
            line_template += b * spike_templates[2 * iline + 1]
            iline += 1
        return line_template

    def remove(self, signal, flag, fsample, obt, signal_estimate,
               pntflag=None, ssoflag=None, maskflag=None,
               spike_params=None, fit_only=False):
        """ Measure and remove the 1Hz housekeeping lines in the signal.

        """
        ind_fit = np.logical_not(flag)
        for flg in [pntflag, ssoflag, maskflag]:
            if flg is not None:
                ind_fit[flg] = False
        # This method works with both raw and diffenced LFI data
        signals = np.atleast_2d(signal.T).copy()
        self.apply_polyfilter(signals, ind_fit)

        stack, spike_templates = self.get_templates(
            obt, fsample, ind_fit, signal_estimate)

        if spike_params is None:
            new_params = []
        else:
            new_params = spike_params

        for isig, sig in enumerate(signals):
            if spike_params is None:
                params = self.fit_templates(stack, ind_fit, sig)
                new_params.append(params)
            else:
                params = spike_params[isig]
            if not fit_only:
                line_template = self.assemble_line_template(sig, params,
                                                            spike_templates)
                sig -= line_template

        if len(signals) == 1:
            signals = signals.ravel()
            new_params = new_params[0]

        return signals.T.copy(), flag, new_params
