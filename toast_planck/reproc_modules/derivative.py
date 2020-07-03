# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import toast.timing as timing


class Differentiator():

    def __init__(self, nharm=20, fsample=180.3737):
        """
        Instantiate the band-derivator object
        Args:
            nharm (int):  Number of harmonic bands
            fsample (float):  Sampling frequency
        """

        self._nharm = nharm
        self._fsample = fsample
        self.fsig = None

    def get_npad(self, sig):
        npad = 2
        while npad < sig.size:
            npad *= 2
        self.n = sig.size
        self.npad = npad
        self.freq = np.fft.rfftfreq(self.npad, 1 / self._fsample)
        return

    def set_signal(self, sig):
        """ FFT and store the provided signal so that user may
        manipulate individual bands
        """
        self.get_npad(sig)
        self.fsig = np.fft.rfft(sig, n=self.npad)
        return

    def correct_band(self, amp, harm):
        """ Multiply a harmonic band with amp.
        """
        if self.fsig is None:
            raise RuntimeError(
                'Cannot manipulate signal before calling set_signal')
        fmin, fmax = self.get_band(harm)
        ind = np.logical_and(np.abs(self.freq) >= fmin,
                             np.abs(self.freq) < fmax)
        self.fsig[ind] *= amp
        pass

    def get_signal(self):
        """ Inverse FFT the manipulated signal and return it.
        """
        if self.fsig is None:
            raise RuntimeError(
                'Cannot return signal before calling set_signal')
        sig = np.fft.irfft(self.fsig, n=self.npad)[:self.n]
        self.fsig = None
        return sig

    def get_band(self, harm):
        """ Return the frequency limits of this band
        """
        fmax = 0.5 / 60
        for band in range(1, harm + 2):
            fmin = fmax
            # fstep = (1 + band // 4) / 60
            fstep = np.exp(band / 4).astype(np.int) / 60
            if band < self._nharm:
                fmax = fmin + fstep
            else:
                fmax = 1000.
        return fmin, fmax

    def differentiate(self, sig, do_bands=False, do_derivs=True, cache=None):
        """
        Return the time (sample index) derivatives of the signal in
        disjoint passbands.  If do_bands is True, then also return the
        signal itself in those passbands.

        Args:
            sig (float): Signal to differentiate.
            do_bands (bool):  Return the band-passed signal.
            do_derivs (bool):  Return the differentiated,
                 band-passed signal.
        """
        self.get_npad(sig)

        fsig = np.fft.rfft(sig, n=self.npad)

        bands = []
        derivs = []

        if cache is None:
            if do_bands:
                bands = np.zeros([self._nharm, self.n], dtype=np.float32)
            if do_derivs:
                derivs = np.zeros([self._nharm, self.n], dtype=np.float32)
        else:
            if do_bands:
                bands = cache.create('bands', np.float32,
                                     [self._nharm, self.n])
            if do_derivs:
                derivs = cache.create('derivs', np.float32,
                                      [self._nharm, self.n])

        for harm in range(self._nharm):
            fmin, fmax = self.get_band(harm)
            cut = np.logical_or(np.abs(self.freq) <= fmin,
                                np.abs(self.freq) > fmax)
            fsig_band = fsig.copy()
            fsig_band[cut] = 0
            if do_bands:
                sig_band = np.fft.irfft(fsig_band, n=self.npad)[:self.n]
                bands[harm][:] = np.array(sig_band, dtype=np.float32)
            if do_derivs:
                fsig_band_deriv = 1j * fsig_band
                sig_band_deriv = np.fft.irfft(fsig_band_deriv,
                                              n=self.npad)[:self.n]
                derivs[harm][:] = np.array(sig_band_deriv, dtype=np.float32)
        return bands, derivs
