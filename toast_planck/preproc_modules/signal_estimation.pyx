#cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=True, cdivision=True

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

import numpy as np
import toast.timing as timing

cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack
cimport cython

np.import_array()

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport sin, cos, M_PI, fabs

f64 = np.float64
i64 = np.int64
i32 = np.int32
uint8 = np.uint8

ctypedef np.float64_t f64_t
ctypedef np.int64_t i64_t
ctypedef np.int32_t i32_t

cdef extern from "medianmap.h":
    void medianmap(double * signal,
                   unsigned char * flg,
                   double * ph,
                   double * ring_out,
                   double * outphase,
                   long ndata,
                   long nring)


class SignalEstimator():
    """
    Fit and evaluate a periodic interpolant of a signal.
    Args:
        nmode (int):  Number of basis functions to fit.
        tol (float):  Minimum required improvement in the RMS to add
            more basis functions.
        mode (str):
    """

    allowed_modes = ['Fourier']

    def __init__(self, nbin=10000, nmodemax=10000, tol=1e-4, mode='Fourier',
                 intense_threshold=0.01):

        if mode not in self.allowed_modes:
            raise Exception('SignalEstimator only supports these modes: {}'
                            ''.format(self.allowed_modes))
        self.nbin = nbin
        self.nmodemax = nmodemax
        self.tol = tol
        self.nmode = None
        self.coeff = None
        self.bin_phase = None
        self.bin_value = None
        self.bin_fit = None
        self.bin_hit = None
        self.intense_threshold = intense_threshold

    def fit(self, double[:] phase not None, double[:] signal not None):
        """
        Fit the basis functions to the signal and store the coefficients
        for future interpolation.  The fast fast version uses the
        approximate orthogonality of the basis functions and fits each
        separately.  It also bins the signal into bins to reduce the
        number of samples to fit.

        Args:
            phase (float):  Signal phase in radians.
            signal (float):  Signal.

        """
        cdef long n = self.nmodemax
        cdef long nbin = self.nbin
        cdef long nn = len(phase)

        cdef double a, b, rms, rms_old = 1e30
        cdef int mode, i, ii, ncoeff

        # Sort the signal by phase

        cdef int[:] inds = np.argsort(phase).astype(np.int32)
        cdef int[:] inds_view = inds
        # Medianmap expects this array as double so we cast to int later.
        cdef double[:] sorted_phase = np.zeros(nn, dtype=np.float64)
        cdef double[:] sorted_signal = np.zeros(nn, dtype=np.float64)
        cdef unsigned char[:] flag = np.zeros(nn, dtype=np.uint8)

        cdef double phase2bin = nbin / (2*M_PI)
        for i in range(nn):
            ii = inds_view[i]
            sorted_phase[i] = phase[ii] * phase2bin
            sorted_signal[i] = signal[ii]

        # bin the signal

        cdef np.ndarray bin_hit = np.zeros(nbin, dtype=np.int32)
        cdef int[:] bin_hit_view = bin_hit
        cdef double[:] bin_phase = np.zeros(nbin, dtype=np.float64)
        cdef double[:] bin_value = np.zeros(nbin, dtype=np.float64)
        cdef np.ndarray[double] bin_var = np.zeros(nbin, dtype=np.float64)
        cdef double[:] bin_var_view = bin_var

        medianmap(&sorted_signal[0], &flag[0], &sorted_phase[0],
                  &bin_value[0], &bin_phase[0], nn, nbin)

        cdef int ibin
        for i in range(nn):
            ibin = <int>sorted_phase[i]
            bin_hit_view[ibin] += 1

        for ibin in range(nbin):
            if bin_hit_view[ibin] != 0:
                bin_var_view[ibin] = 1. / bin_hit_view[ibin]

        self.bin_hit = bin_hit
        self.bin_value = bin_value
        self.bin_var = bin_var

        cdef double bin2phase = (2 * M_PI) / nbin
        cdef np.ndarray[double] bin_invvar = np.zeros(nbin, dtype=f64)
        cdef double[:] bin_invvar_view = bin_invvar
        for ibin in range(nbin):
            bin_phase[ibin] *= bin2phase
            if bin_var_view[ibin] != 0:
                bin_invvar_view[ibin] = 1. / bin_var_view[ibin]

        # do the fit against the bins

        cdef double cov, tol = self.tol
        cdef double proj
        cdef np.ndarray[double] residual = np.zeros(nbin, dtype=f64)
        cdef double[:] residual_view = residual
        cdef np.ndarray[double] template = np.zeros(nbin, dtype=f64)
        cdef double[:] template_view = template
        cdef np.ndarray[double] fit = np.zeros(nbin, dtype=f64)
        cdef double[:] fit_view = fit
        # Smooth fit will capture the dipole modulation and will
        # be used to identify exceptionally hot regions
        cdef np.ndarray[double] smooth_fit = np.zeros(nbin, dtype=f64)
        cdef double[:] smooth_fit_view = smooth_fit

        for ibin in range(nbin):
            residual_view[ibin] = bin_value[ibin]

        cdef np.ndarray[double] coeff = np.zeros(n, dtype=f64)

        for mode in range(n):
            ncoeff = mode + 1
            a = (mode + 1) // 2

            if mode % 2 == 0:
                for ibin in range(nbin):
                    template_view[ibin] = cos(a * bin_phase[ibin])
            else:
                for ibin in range(nbin):
                    template_view[ibin] = sin(a * bin_phase[ibin])

            proj = 0
            cov = 0
            for ibin in range(nbin):
                cov += template_view[ibin] * template_view[ibin] \
                    * bin_invvar_view[ibin]
                proj += template_view[ibin] * residual_view[ibin] \
                    * bin_invvar_view[ibin]

            if cov == 0:
                raise Exception('cov is zero!')
            cc = proj / cov
            coeff[mode] = cc

            for ibin in range(nbin):
                residual_view[ibin] -= cc * template_view[ibin]
                fit_view[ibin] += cc * template_view[ibin]
            if mode < 5:
                for ibin in range(nbin):
                    smooth_fit_view[ibin] += cc * template_view[ibin]

            # Don't check the RMS on every iteration
            if mode % 10 == 0:
                rms = np.std(residual)
                if rms_old / rms - 1 < tol:
                    break
                rms_old = rms

        # If there are bins where the residual is very high, we replace the
        # smooth value with the binned value. These bins are driven by intense
        # and sharp signal such as planets and bright point sources

        intense = np.zeros(nbin, dtype=bool)
        # Very hot regions are always considered intense
        for ibin in range(nbin):
            intense[ibin] = self.bin_hit[ibin] == 0 \
                or (self.bin_value[ibin]
                    - smooth_fit_view[ibin]) > self.intense_threshold
        # Now add intense flags as long as the residual RMS improves
        for i in range(10):
            if np.all(intense):
                break
            good = np.logical_not(intense)
            rms = np.std(residual[good])
            # mn = np.mean(self.bin_value[good])
            # intense[np.abs(self.bin_value-mn) > 3.0*rms] = True
            intense[np.abs(residual) > 3 * rms] = True

        # smooth the intense flag so that nearby bins are also covered
        cdef int w = nbin / 1000  # 21 arc minutes
        if w % 2 == 0:
            w += 1
        if w > 1:
            ivec = np.hstack([intense * 1., intense * 1.])  # Avoid boundaries
            kernel = np.ones(w) / w
            ivec = fftconvolve(ivec, kernel, mode='same')
            ivec = np.roll(ivec[nbin // 2:nbin // 2 + nbin], nbin // 2 - nbin)
            intense[ivec > .1] = True
            intense[ivec < .01] = False  # Reduce spurious detections

        intense[self.bin_hit == 0] = False

        for i in range(fit.size):
            if intense[i]:
                fit_view[i] = self.bin_value[i]

        self.nmode = ncoeff
        self.coeff = coeff[:ncoeff]
        # Extend the estimate by one bin from both ends to enable interpolation
        cdef long first_good = 0
        while self.bin_hit[first_good] == 0 and first_good < len(self.bin_hit):
            first_good += 1
        cdef long last_good = len(self.bin_hit) - 1
        while self.bin_hit[last_good] == 0 and last_good > 0:
            last_good -= 1
        self.bin_hit = np.hstack(
            [self.bin_hit[last_good], self.bin_hit, self.bin_hit[first_good]])
        self.bin_phase = np.hstack(
            [bin_phase[last_good] - 2 * M_PI, bin_phase,
             bin_phase[first_good] + 2 * M_PI])
        self.bin_intense = np.hstack(
            [intense[last_good], intense, intense[first_good]])
        self.bin_fit = np.hstack([fit[last_good], fit, fit[first_good]])
        self.residual_residual = np.hstack(
            [residual[last_good], residual, residual[first_good]])
        hit = self.bin_hit != 0
        self.cspline = CubicSpline(self.bin_phase[hit], self.bin_fit[hit])
        return

    def fit_full(self, double[:] phase not None, double[:] signal not None):
        """
        Fit the basis functions to the signal and store the coefficients
        for future interpolation.  The fast version uses the approximate
        orthogonality of the basis functions and fits each separately.

        Args:
            phase (float):  Signal phase in radians.
            signal (float):  Signal.

        """
        cdef int n = self.nmodemax
        cdef int nn = len(phase)
        cdef double cov, tol = self.tol
        cdef double proj
        cdef np.ndarray[double] residual = np.zeros(nn, dtype=f64)
        cdef np.ndarray[double] template = np.zeros(nn, dtype=f64)

        cdef double a, b, rms, rms_old = 1e30
        cdef int mode, i, ncoeff

        for i in range(nn):
            residual[i] = signal[i]

        cdef np.ndarray[double] coeff = np.zeros(n, dtype=f64)

        for mode in range(n):
            ncoeff = mode + 1
            a = (mode + 1) // 2

            if mode % 2 == 0:
                for i in range(nn):
                    template[i] = cos(a * phase[i])
            else:
                for i in range(nn):
                    template[i] = sin(a * phase[i])

            proj = 0
            cov = 0
            for i in range(nn):
                cov += template[i] * template[i]
                proj += template[i] * residual[i]

            cc = proj / cov
            coeff[mode] = cc

            for i in range(nn):
                residual[i] -= cc * template[i]

            # Don't check the RMS on every iteration
            if mode % 100 == 0:
                rms = np.std(residual)
                if rms_old / rms - 1 < tol:
                    break
                rms_old = rms

        self.nmode = ncoeff
        self.coeff = coeff[:ncoeff]
        return

    def fit_exact(self, np.ndarray[f64_t, ndim=1] phase not None,
                  np.ndarray[f64_t, ndim=1] signal not None):
        """
        Fit the basis functions to the signal and store the coefficients
        for future interpolation.  Since the basis functions are only
        orthogonal without gaps, we do here a full linear regression
        (expensive).

        Args:
            phase (float):  Signal phase in radians.
            signal (float):  Signal.

        """
        # Templates themselves are not kept as the template matrix
        # could be too large.

        cdef int n = self.nmode
        cdef int nn = len(phase)
        cdef double * cov = <double*>malloc(n*n*sizeof(double))
        cdef double * proj = <double*>malloc(n*sizeof(double))
        cdef np.ndarray[double] rowtemplate = np.zeros(nn, dtype=f64)
        cdef np.ndarray[double] coltemplate = np.zeros(nn, dtype=f64)

        memset(cov, 0, n*n*sizeof(f64_t))
        memset(proj, 0, n*sizeof(f64_t))

        cdef double a, b
        cdef int row, col, i

        for row in range(n):
            a = (row + 1) // 2

            if row % 2 == 0:
                for i in range(nn):
                    rowtemplate[i] = cos(a*phase[i])
            else:
                for i in range(nn):
                    rowtemplate[i] = sin(a*phase[i])

            for i in range(nn):
                proj[row] += rowtemplate[i] * signal[i]

            for col in range(row, n):
                b = (col + 1) // 2

                if col % 2 == 0:
                    for i in range(nn):
                        coltemplate[i] = cos(b*phase[i])
                else:
                    for i in range(nn):
                        coltemplate[i] = sin(b*phase[i])

                for i in range(nn):
                    cov[row*n+col] += rowtemplate[i] * coltemplate[i]

        cdef int info
        cdef char uplo = b'L'

        # factorize
        info = 0
        cython_lapack.dpotrf(&uplo, &n, cov, &n, &info)
        if info != 0:
            raise Exception('Decomposition failed: info = {}'.format(info))

        # invert
        cython_lapack.dpotri(&uplo, &n, cov, &n, &info)
        if info != 0:
            raise Exception('Inversion failed: info = {}'.format(info))

        cdef np.ndarray[double] coeff = np.zeros(n, dtype=f64)

        for row in range(n):
            for col in range(row):
                coeff[row] += cov[row*n+col]*proj[col]
            for col in range(row, n):
                coeff[row] += cov[col*n+row]*proj[col]

        self.coeff = coeff

        free(cov)
        free(proj)

    def eval_exact(self, np.ndarray[f64_t, ndim=1] phase not None):
        """
        Evaluate previously fitted interpolant at given points.
        Args:
            phase (float):  Phase in radians.

        """

        if self.coeff is None:
            raise Exception('SignalEstimator cannot evaluate interpolant '
                            'without first fitting it.')

        cdef int mode, i, n = self.nmode, nn = len(phase)
        cdef double a, cc

        cdef np.ndarray[double] signal = np.zeros(nn, dtype=f64)
        cdef np.ndarray[double] coeff = self.coeff

        for mode in range(n):
            cc = coeff[mode]
            a = (mode + 1) // 2

            if mode % 2 == 0:
                for i in range(nn):
                    signal[i] += cc * cos(a * phase[i])
            else:
                for i in range(nn):
                    signal[i] += cc * sin(a * phase[i])

        return signal

    def eval(self, double[:] phase not None):
        """
        Evaluate previously fitted interpolant at given points.
        Args:
            phase (float):  Phase in radians.

        """
        if self.bin_phase is None or self.bin_fit is None:
            raise Exception('SignalEstimator cannot evaluate interpolant '
                            'without first fitting it.')

        if False:
            # Interpolate the binned function
            good = self.bin_hit != 0
            bin_phase = self.bin_phase[good]
            bin_fit = self.bin_value[good]

            return np.interp(phase, bin_phase, bin_fit)
        else:
            # Interpolate the smooth function
            return (self.cspline(phase),
                    np.interp(phase, self.bin_phase, self.bin_intense) > .5)
