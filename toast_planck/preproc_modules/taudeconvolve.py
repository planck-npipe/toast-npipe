# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from scipy import optimize
from scipy.signal import fftconvolve
from toast.mpi import MPI

import numpy as np
import toast.timing as timing

from . import time_response_tools


class TauDeconvolver():

    def __init__(self, bolo_id, IMO, filterlen=2 ** 20, fsample=180.3737,
                 lfer='LFER8', overlap=10000, extra_global_offset=None,
                 filterfile=None, tabulated_tf=None, fnorm=0.016, comm=None,
                 normalize_filter=True):
        """
        Instantiate the deconvolution object
        bolo_id -- Bolometer ID (e.g. 00_100_1a)
        IMO -- Either an IMO object or a path to IMO XML dump
        filterlen -- Fourier transform length, actual length will be a
             power of 2 AT LEAST as long as this
        fsample -- fixed sampling frequency
        lfer -- Transfer function to seek from the IMO and deconvolve
        overlap -- number of samples read for boundary effects.  These
             are not written into the filtered TOI
        notch -- vector of line frequencies to notch out
        wnotch -- relative width of the notch
        extra_global_offset -- add another phase shift by hand; in same
             units as global_offset in IMO.
        tabulated_tf(None) -- When set, overrides LFER and IMO and
             filterfile.  A 3-element tuple containing frequency, real,
              imaginary
        filterfile(None) -- When set, overrides LFER and IMO. A 3 column
             ASCII file containing the transfer function to convolve with
        fnorm -- the frequency at which the transfer function is
             normalized to 1.0.  default is the dipole frequency.

        """
        self.bolo_id = bolo_id
        self.IMO = IMO
        self.filterlen = 2
        while self.filterlen < filterlen or self.filterlen < 3 * overlap:
            self.filterlen *= 2
        self.overlap = overlap

        self.comm = comm
        if self.comm is None:
            self.ntask = 1
            self.rank = 0
        else:
            self.ntask = self.comm.size
            self.rank = self.comm.rank
        self.normalize_filter = normalize_filter
        # DEBUG begin
        if self.rank == 0:
            print("Initializing TauDeconvolver. bolo_id = {}, IMO = {}, filterlen = {}, fsample = {}, lfer = {}, filterfile = {}".format(bolo_id, IMO, filterlen, fsample, lfer, filterfile), flush=True)
        # DEBUG end

        freq = np.fft.rfftfreq(self.filterlen, 1. / fsample)
        self.freq = freq

        if tabulated_tf is not None:
            self.tf = np.interp(
                self.freq, tabulated_tf[0],
                tabulated_tf[1]) + 1j * np.interp(self.freq, tabulated_tf[0],
                                                  tabulated_tf[2])
            if self.normalize_filter:
                norm = np.abs(
                    np.interp(fnorm, tabulated_tf[0], tabulated_tf[1]) +
                    1j * np.interp(fnorm, tabulated_tf[0], tabulated_tf[2]))
                self.tf = self.tf / norm

            self.tfinv = 1. / self.tf
            self.tfinv[np.abs(self.tf) < 1e-4] = 0
            self.lowpass = time_response_tools.filter_function(freq)
            self.filter = self.lowpass * self.tfinv

            self.fsample = fsample

            if extra_global_offset is not None:
                if extra_global_offset != 0.0:
                    phase = -2. * np.pi * extra_global_offset * freq / fsample
                    shift_tf = np.cos(phase) + 1j * np.sin(phase)

                    self.filter /= shift_tf
                    self.tf *= shift_tf
        else:
            self.filterfile = filterfile
            if self.filterfile is not None:
                if self.rank == 0:
                    try:
                        filt = np.genfromtxt(filterfile).T
                    except Exception as e:
                        raise Exception('Failed to load filter function from '
                                        '{}: {}'.format(filterfile, e))
                else:
                    filt = None
                if self.comm is not None:
                    filt = self.comm.bcast(filt)
                self.filter = np.interp(self.freq, filt[0], filt[1]) + \
                    1j * np.interp(self.freq, filt[0], filt[2])
                if self.normalize_filter:
                    norm = np.abs(np.interp(fnorm, filt[0], filt[1]) +
                                  1j * np.interp(fnorm, filt[0], filt[2]))
                    self.filter = self.filter / norm
                # Invert the filter to allow convolving
                self.tf = self.filter.copy()
                good = self.filter != 0
                self.tf[good] = 1. / self.filter[good]
                self.tf[np.abs(self.filter) < 1e-4] = 0
            else:
                self.global_offset = self.IMO.get(
                    'IMO:HFI:DET:Phot_Pixel Name="{}":NoiseAndSyst:TimeResp:'
                    'LFER8:global_offset'.format(bolo_id), np.float64)

                if extra_global_offset is not None:
                    self.global_offset += extra_global_offset

                self.pars = {}
                npole = 0
                if lfer == 'LFER8':
                    prefix = 'IMO:HFI:DET:Phot_Pixel Name="{}":NoiseAndSyst:' \
                        'TimeResp:LFER8:'.format(bolo_id)
                    self.pars['a1'] = self.IMO.get(prefix + 'par1', np.float64)
                    self.pars['a2'] = self.IMO.get(prefix + 'par2', np.float64)
                    self.pars['a3'] = self.IMO.get(prefix + 'par3', np.float64)
                    self.pars['a4'] = self.IMO.get(prefix + 'par9', np.float64)
                    self.pars['a5'] = self.IMO.get(prefix + 'par11', np.float64)
                    self.pars['a6'] = self.IMO.get(prefix + 'par13', np.float64)
                    self.pars['a7'] = self.IMO.get(prefix + 'par15', np.float64)
                    self.pars['a8'] = self.IMO.get(prefix + 'par17', np.float64)
                    self.pars['tau1'] = self.IMO.get(prefix + 'par4',
                                                     np.float64)
                    self.pars['tau2'] = self.IMO.get(prefix + 'par5',
                                                     np.float64)
                    self.pars['tau3'] = self.IMO.get(prefix + 'par6',
                                                     np.float64)
                    self.pars['tau4'] = self.IMO.get(prefix + 'par10',
                                                     np.float64)
                    self.pars['tau5'] = self.IMO.get(prefix + 'par12',
                                                     np.float64)
                    self.pars['tau6'] = self.IMO.get(prefix + 'par14',
                                                     np.float64)
                    self.pars['tau7'] = self.IMO.get(prefix + 'par16',
                                                     np.float64)
                    self.pars['tau8'] = self.IMO.get(prefix + 'par18',
                                                     np.float64)
                    self.pars['tau_stray'] = self.IMO.get(prefix + 'par7',
                                                          np.float64)
                    self.pars['Sphase'] = self.IMO.get(prefix + 'par8',
                                                       np.float64)

                    prefix = 'IMO:HFI:DET:Phot_Pixel Name="{}":NoiseAndSyst:' \
                        'TimeResp:SallenKeyHPF:'.format(bolo_id)
                    self.pars['tauhp1'] = self.IMO.get(prefix + 'tauhp1',
                                                       np.float64)
                    self.pars['tauhp2'] = self.IMO.get(prefix + 'tauhp2',
                                                       np.float64)

                    npole = 8
                    for i in range(8, 0, -1):
                        if self.pars['tau' + str(i)] != 0:
                            break
                        npole -= 1

                    if self.pars['tauhp1'] != self.pars['tauhp2']:
                        raise Exception(
                            'Don\'t know how to handle the case where tauhp1 '
                            '({}) is not equal to tauhp2 ({})'.format(
                                self.pars['tauhp1'], self.pars['tauhp2']))
                elif lfer == 'LFER1':
                    npole = 1
                    self.pars['a1'] = 1.0
                    self.pars['tau1'] = 0.01
                    self.pars['tau_stray'] = 2.095108e-03
                    self.pars['Sphase'] = 0.0
                else:
                    raise Exception(
                        'Don\'t know how to parse {} transfer function '
                        'parameters from IMO'.format(lfer))

                norm_f = np.array([0.0, fnorm])
                norm_tf = time_response_tools.LFERn(norm_f, npole, self.pars)
                phase = -2. * np.pi * self.global_offset * norm_f / fsample
                shift_tf = np.cos(phase) + 1j * np.sin(phase)
                norm_tf = norm_tf * (np.cos(phase) + 1j * np.sin(phase))
                norm = np.abs(norm_tf[1])

                tstart = MPI.Wtime()
                if self.ntask == 1:
                    self.tf = time_response_tools.LFERn(freq, npole, self.pars) \
                              / norm
                else:
                    nfreq = len(freq)
                    nfreq_task = np.int(np.ceil(nfreq / self.ntask))
                    # First frequency must be zero for normalization
                    my_freq = np.hstack(
                        [[0.0],
                         freq[nfreq_task * self.rank:
                              nfreq_task * (self.rank + 1)]])
                    # Discard the extra frequency bin here
                    my_tf = time_response_tools.LFERn(
                        my_freq, npole, self.pars)[1:] / norm
                    self.tf = np.hstack(self.comm.allgather(my_tf))
                tstop = MPI.Wtime()
                if self.rank == 0:
                    print('Computed the LFER transfer function in {:.2f} s.'
                          ''.format(tstop - tstart), flush=True)
                self.tfinv = 1. / self.tf
                self.tfinv[np.abs(self.tf) < 1e-4] = 0
                self.lowpass = time_response_tools.filter_function(freq)
                self.filter = self.lowpass * self.tfinv

                self.fsample = fsample

                phase = -2. * np.pi * self.global_offset * freq / fsample
                shift_tf = np.cos(phase) + 1j * np.sin(phase)

                self.filter /= shift_tf
                self.tf *= shift_tf
        self.init_flag_kernels()
        return

    def _trim_flag_kernel(self, kernel, center, tol=.1):
        """
        Extract the center of the kernel
        """
        ind = np.abs(np.arange(kernel.size) - center, dtype=np.int)
        kernel = np.abs(kernel) > np.abs(np.amax(kernel)) * tol
        wkernel = np.amax(kernel * ind)
        ind = slice(center - wkernel, center + wkernel + 1)
        return kernel[ind]

    def init_flag_kernels(self):
        """
        When (de)convolving the signal, some number of unflagged samples
        become compromised by the flagged samples.  Here we determine the
        time-domain kernels to convolve the flags with.
        """
        x = np.zeros(self.filterlen)
        center = self.filterlen // 2
        x[center] = 1
        tfkernel = np.fft.irfft(np.fft.rfft(x) * self.tf, self.filterlen)
        filterkernel = np.fft.irfft(np.fft.rfft(x) * self.filter,
                                    self.filterlen)
        self.tfkernel = self._trim_flag_kernel(tfkernel, center)
        self.filterkernel = self._trim_flag_kernel(filterkernel, center)
        return

    def convolve(self, signal, flag):
        return self.deconvolve(signal, flag, convolve_instead=True)

    def deconvolve(self, signal_in, flag_in, convolve_instead=False):
        """
        Deconvolve the precomputed transfer function.
        Extend the flags appropriately.
        """

        ntot = signal_in.size
        signal_out = np.zeros(ntot)
        buf = np.zeros(self.filterlen)

        istart = 0
        while istart < ntot:
            nleft = len(signal_in) - istart
            nprocess = min(nleft, self.filterlen)
            istop = istart + nprocess

            buf[:nprocess] = signal_in[istart:istop]
            buf[nprocess:] = 0

            bufstart = 0
            bufstop = nprocess
            if istart != 0:
                istart += self.overlap
                bufstart += self.overlap
            if istop != signal_in.size:
                istop -= self.overlap
                bufstop -= self.overlap

            if convolve_instead:
                signal_out[istart:istop] = np.fft.irfft(
                    np.fft.rfft(buf) * self.tf,
                    self.filterlen)[bufstart:bufstop]
            else:
                signal_out[istart:istop] = np.fft.irfft(
                    np.fft.rfft(buf) * self.filter,
                    self.filterlen)[bufstart:bufstop]

            if istop == ntot:
                break

            istart = istop - self.overlap

        signal_out = signal_out.astype(signal_in.dtype)
        if flag_in is not None:
            if convolve_instead:
                flag_out = fftconvolve(
                    flag_in != 0, self.tfkernel, mode='same') > 1e-3
            else:
                flag_out = fftconvolve(
                    flag_in != 0, self.filterkernel, mode='same') > 1e-3
            flag_out = flag_out.astype(flag_in.dtype)
        else:
            flag_out = None

        return signal_out, flag_out

    def fit_relative_offset(self, npole, new_pars, sim_fwhm_arcmin=4.5,
                            sim_length=1.0):
        """
        Compute the relative phase shift between this time response
        and a new one.  The new time response function is parameterized
        as a different LFERn function with npole poles.

        The return value is the shift in number of samples; should be
        compatible with the input to global_offset defined in the
        constructor method.

        npole : number of poles in the filter
        new_pars : parameters specified in the same dictionary format as
            in the constructor

        sim_fwhm_arcmin : simulated Gaussian signal FWHM in arcmin
        sim_length : simulation length in seconds
        sim_time_step : simulation time step between samples

        """
        # generate the trial transfer function
        freq = np.fft.rfftfreq(self.filterlen, 1. / self.fsample)

        trial_filter = time_response_tools.LFERn(freq, npole, new_pars)

        # generate a Gaussian timeline to use as an input
        sigma_seconds = sim_fwhm_arcmin / 60 / 6 / np.sqrt(8 * np.log(2))
        sim_time_step = sim_length / self.filterlen
        time = np.arange(0.0, sim_length, sim_time_step)

        model_tod = np.exp(-(time - (sim_length / 2)) ** 2
                           / 2 / sigma_seconds ** 2)

        filtered_tod = np.fft.irfft(
            np.fft.rfft(model_tod) * self.filter / trial_filter,
            self.filterlen).real

        # internal error functions
        def func_gauss(p, xin):
            return p[0] * np.exp(-(xin - p[1]) ** 2 / (2.*p[2] ** 2))

        def chi2_nosigma(p, xin, d):
            return ((func_gauss(p, xin) - d) ** 2).sum()

        par_guess = [1., 0.5, 0.1]
        par_fit = optimize.fmin(
            chi2_nosigma, par_guess, args=(time, filtered_tod),
            disp=False, maxiter=10000, maxfun=10000, xtol=0.01)

        relative_shift = -(sim_length / 2.0 - par_fit[1]) * self.fsample
        return relative_shift

    def fit_absolute_offset(self, sim_fwhm_arcmin=4.5, sim_length=1.0):
        """
        Compute and correct the phase shift between this filter
        function and NO filtering.  This function is useful only for a
        simulated time stream.

        sim_fwhm_arcmin : simulated Gaussian signal FWHM in arcmin
        sim_length : simulation length in seconds

        """
        # generate the trial transfer function
        freq = np.fft.rfftfreq(self.filterlen, 1. / self.fsample)

        # generate a Gaussian timeline to use as an input
        sigma_seconds = sim_fwhm_arcmin / 60 / 6 / np.sqrt(8 * np.log(2))
        sim_time_step = sim_length / self.filterlen
        time = np.arange(0.0, sim_length, sim_time_step)

        model_tod = np.exp(-(time - (sim_length / 2.0)) ** 2 / 2.0
                           / sigma_seconds ** 2)

        filtered_tod = np.fft.irfft(
            np.fft.rfft(model_tod) * self.filter, self.filterlen).real

        # internal error functions
        def func_gauss(p, xin):
            return p[0] * np.exp(-(xin - p[1]) ** 2 / (2.*p[2] ** 2))

        def chi2_nosigma(p, xin, d):
            return ((func_gauss(p, xin) - d) ** 2).sum()

        par_guess = [1., 0.5, 0.1]
        par_fit = optimize.fmin(
            chi2_nosigma, par_guess, args=(time, filtered_tod), disp=False,
            maxiter=10000, maxfun=10000, xtol=0.01)

        relative_shift = -(sim_length / 2.0 - par_fit[1]) * self.fsample

        phase = -2 * np.pi * relative_shift * freq / self.fsample
        shift_tf = np.cos(phase) + 1j * np.sin(phase)

        self.filter /= shift_tf
        self.tf *= shift_tf
        return relative_shift
