# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import toast.timing as timing

from .filters import flagged_running_average


class GainCorrector():
    """
    Operator for applying (nonlinear) gain correction to Planck HFI
    timelines. The model is
        signal_corrected = g0 * signal * ( 1 + smooth_signal / v0 )
    The operator does not correct for dynamical nonlinearity that is
    associated with, say, planet transits.

    Args:
        IMO (imo):  IMO XML object containing detector parameters such
            as the gain and nonlinearity.
        bolo_id (str):  Planck HFI bolometer ID such as 00_100_1a used
            to search the IMO
        nwin_hp (int):  Number of samples in the running average width
            for lowpass-filtered signal
        nwin_lp (int):  Number of samples in the running average width
            for highpass-filtered signal
        linear (bool):  Reduce the operator to simple calibration without
            nonlinearity correction.
        g0 (float): gain parameter; if specified, overrides read from IMO.
        v0 (float): nonlinear parameter; if specified, overrides read
            from IMO.
    """

    def __init__(self, IMO, bolo_id, nwin_hp=10822, nwin_lp=201, linear=False,
                 g0=None, v0=None):
        if g0 is None:
            self.g0 = IMO.get(
                'IMO:HFI:DET:Phot_Pixel Name="{}":NoiseAndSyst:NonLinearity:g0'
                ''.format(bolo_id), float)
        else:
            self.g0 = g0
        if v0 is None:
            self.v0 = IMO.get(
                'IMO:HFI:DET:Phot_Pixel Name="{}":NoiseAndSyst:NonLinearity:v0'
                ''.format(bolo_id), float)
        else:
            self.v0 = v0

        self.nwin_hp = nwin_hp
        self.nwin_lp = nwin_lp
        self.linear = linear

    def correct(self, signal, flag):
        """
        Convert from Volts to Watts and apply the non-linearity correction.
        Will bandpass the signal for bolometric non-linearity correction.
        """
        if self.linear:
            out = signal * self.g0
        else:
            lp_signal = flagged_running_average(signal, flag, self.nwin_lp)
            hp_signal = flagged_running_average(lp_signal, flag, self.nwin_hp)
            bp_signal = lp_signal - hp_signal
            out = signal * self.g0 * (1 + bp_signal / self.v0)
        return out

    def uncorrect(self, signal, flag):
        """
        Convert from Watts to Volts and apply the non-linearity factor.
        Will bandpass the signal for bolometric non-linearity correction.
        """
        if self.linear:
            out = signal / self.g0
        else:
            # This form assumes that V / v0 = W / g0 / v0 is small
            lp_signal = flagged_running_average(signal, flag, self.nwin_lp)
            hp_signal = flagged_running_average(lp_signal, flag, self.nwin_hp)
            bp_signal = lp_signal - hp_signal
            out = signal / self.g0 * (1 - bp_signal / self.g0 / self.v0)
        return out
