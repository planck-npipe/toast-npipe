#cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=True, cdivision=True

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""destripe_tools contains a number of high performance computational kernels.

"""

from toast_planck import Ring

import toast.qarray

from cython.parallel import prange
import numpy as np


cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas
from libc.math cimport sqrt, fabs, sin, cos, M_PI
cimport cython


def masked_linear_regression(templates, double[:] signal not None,
                             unsigned char[:] good not None):
    """ Perform a least-squares fit of the templates against the signal.
    """
    cdef long ntemplate = len(templates)
    if ntemplate > 1024:
        raise RuntimeError(
            'Maximum number of templates in masked_linear_regression is 1024')
    cdef long nsamp = signal.size
    if good.size != nsamp:
        raise RuntimeError(
            'Flags have {} samples but the signal has {} samples'
            ''.format(good.size, nsamp))

    cdef long i, row, col
    cdef double * ptemplate[1024]
    cdef double[:] template_view
    for row in range(ntemplate):
        if templates[row].size != nsamp:
            raise RuntimeError(
                'Template # {} has {} samples but the signal has {} samples'
                ''.format(row, templates[row].size, nsamp))
        template_view = templates[row]
        ptemplate[row] = &(template_view[0])

    cdef np.ndarray invcov = np.zeros([ntemplate, ntemplate], dtype=np.float64)
    cdef double[:, :] invcov_view = invcov
    cdef np.ndarray proj = np.zeros(ntemplate, dtype=np.float64)
    cdef double[:] proj_view = proj

    cdef double dotprod
    cdef double * rowtemplate
    cdef double * coltemplate

    cdef long ngood = 0
    for i in range(nsamp):
        if good[i] != 0:
            ngood += 1
    cdef unsigned long[:] ind = np.zeros(ngood, np.uint64)
    cdef unsigned long igood = 0
    for i in range(nsamp):
        if good[i] != 0:
            ind[igood] = i
            igood += 1

    for row in prange(ntemplate, nogil=True, schedule='static', chunksize=1):
        rowtemplate = ptemplate[row]
        for col in range(row, ntemplate):
            coltemplate = ptemplate[col]
            dotprod = 0
            for i in range(ngood):
                igood = ind[i]
                dotprod = dotprod + rowtemplate[igood]*coltemplate[igood]
            invcov_view[row, col] = dotprod
            invcov_view[col, row] = dotprod

    for row in prange(ntemplate, nogil=True, schedule='static', chunksize=4):
        dotprod = 0
        rowtemplate = ptemplate[row]
        for i in range(ngood):
            igood = ind[i]
            dotprod = dotprod + rowtemplate[igood]*signal[igood]
        proj_view[row] = dotprod

    cov = np.linalg.inv(invcov)
    coeff = np.dot(cov, proj)

    return coeff, invcov, cov, proj


def linear_regression(templates, double[:] signal not None):
    """ Perform a least-squares fit of the templates against the signal.
    """
    cdef long ntemplate = len(templates)
    if ntemplate > 1024:
        raise RuntimeError(
            'Maximum number of templates in linear_regression is 1024')
    cdef long nsamp = signal.size

    cdef long i, row, col
    cdef double * ptemplate[1024]
    cdef double[:] template_view
    for row in range(ntemplate):
        if templates[row].size != nsamp:
            raise RuntimeError(
                'Template # {} has {} samples but the signal has {} samples'
                ''.format(row, templates[row].size, nsamp))
        template_view = templates[row]
        ptemplate[row] = &(template_view[0])

    cdef np.ndarray invcov = np.zeros([ntemplate, ntemplate], dtype=np.float64)
    cdef double[:, :] invcov_view = invcov
    cdef np.ndarray proj = np.zeros(ntemplate, dtype=np.float64)
    cdef double[:] proj_view = proj

    cdef double dotprod
    cdef double * rowtemplate
    cdef double * coltemplate
    cdef int n = nsamp, one = 1

    # for row in prange(ntemplate, nogil=True, schedule='static', chunksize=10):
    for row in range(ntemplate):
        rowtemplate = ptemplate[row]
        for col in range(row, ntemplate):
            coltemplate = ptemplate[col]
            dotprod = cython_blas.ddot(&n, rowtemplate, &one, coltemplate, &one)
            invcov_view[row, col] = dotprod

    for row in range(ntemplate):
        for col in range(row):
            invcov_view[row, col] = invcov_view[col, row]

    cdef double *psignal = &(signal[0])
    # for row in prange(ntemplate, nogil=True, schedule='static', chunksize=10):
    for row in range(ntemplate):
        rowtemplate = ptemplate[row]
        dotprod = cython_blas.ddot(&n, rowtemplate, &one, psignal, &one)
        proj_view[row] = dotprod

    cov = np.linalg.inv(invcov)
    coeff = np.dot(cov, proj)

    return coeff, invcov, cov, proj


def build_spike_templates(double[:] obt not None, double fsample):
    """ Evaluate 1Hz spike templates.

    """
    cdef long nsamp = obt.size
    cdef double frac = fsample - int(fsample)
    cdef long nline = int(fsample/2)
    cdef long ntemplate = nline*3
    if nline+frac >= fsample/2:
        ntemplate -= 1
    ntemplate *= 2

    cdef np.ndarray templates = np.zeros([ntemplate, nsamp],
                                         dtype=np.float64)
    cdef double[:, :] templates_view = templates
    cdef double[:] argvec = np.zeros(nsamp, dtype=np.float64)

    # Templates at fline +- frac are for aliased modes

    cdef double fline, prefac, offset
    cdef long itemplate, iline, i, ioffset

    for iline in prange(nline, nogil=True):
        fline = iline + 1
        itemplate = 6*iline
        offset = -frac
        for ioffset in range(3):
            prefac = 2 * M_PI * (fline + offset)
            for i in range(nsamp):
                argvec[i] = prefac * obt[i]
            for i in range(nsamp):
                templates_view[itemplate, i] = cos(argvec[i])
            itemplate = itemplate + 1
            for i in range(nsamp):
                templates_view[itemplate, i] = sin(argvec[i])
            itemplate = itemplate + 1
            if itemplate == ntemplate:
                break
            offset = offset + frac

    return templates


def build_4k_templates(long start, long stop, long first_line,
                       long nline, long ntemplate):
    """
    Build the sine and cosine templates that match the 4K harmonics
    """
    if ntemplate not in [2, 4, 6]:
        raise RuntimeError('Unrecognized number of templates: {}'
                           ''.format(ntemplate))

    cdef long nsamp = stop-start
    cdef double meanpoint = (start+stop) / 2

    cdef np.ndarray templates = np.zeros([nline*ntemplate, nsamp],
                                         dtype=np.float64)
    cdef double[:, :] templates_view = templates
    cdef double[:] argvec = np.zeros(nsamp, dtype=np.float64)

    # arrange the lines by intensity:
    # 70, 30, 10, 50, 16.7, 20, 40, 60 Hz

    cdef long iline, line, ii, offset, row, col
    cdef double arg, x

    for iline in range(first_line, first_line+nline):
        if iline < 4:
            # line * 10Hz
            line = [1, 3, 5, 7][iline]
            arg = line / 18. * 2.0 * M_PI
        elif iline == 4:
            # 16.70Hz
            arg = 5 / 54. * 2.0 * M_PI
        elif iline < 9:
            # line * 10Hz
            line = [2, 4, 6, 8][iline-5]
            arg = line / 18. * 2.0 * M_PI
        elif iline == 9:
            # 16.03Hz
            arg = 4 / 45. * 2.0 * M_PI
        elif iline == 10:
            # 25.052Hz
            arg = 5 / 36. * 2.0 * M_PI
        elif iline == 11:
            # 43.423Hz
            arg = 13 / 54. * 2.0 * M_PI
        elif iline == 12:
            # 46.274Hz
            arg = 2489 / 9702. * 2.0 * M_PI
        elif iline == 13:
            # 47.599Hz
            arg = 19 / 76. * 2.0 * M_PI
        elif iline == 14:
            # 56.784Hz
            arg = 17 / 54. * 2.0 * M_PI
        else:
            raise RuntimeError('Unknown line # {}'.format(iline))

        # Standard sine and cosine templates

        offset = ntemplate*(iline-first_line)

        col = 0
        for ii in range(start, stop):
            argvec[col] = (arg*ii) % (2*M_PI)
            col += 1

        row = offset
        for col in range(nsamp):
            templates_view[row][col] = cos(argvec[col])
        row = offset + 1
        for col in range(nsamp):
            templates_view[row][col] = sin(argvec[col])

        # Derivatives for the amplitude

        if ntemplate > 2:
            col = 0
            for ii in range(start, stop):
                argvec[col] = ii - meanpoint
                col += 1
            row = offset + 2
            for col in range(nsamp):
                templates_view[row][col] = argvec[col] \
                                           * templates_view[offset][col]
            row = offset + 3
            for col in range(nsamp):
                templates_view[row][col] = argvec[col] \
                                           * templates_view[offset+1][col]

        if ntemplate > 4:
            for i in range(nsamp):
                argvec[i] = argvec[i] * argvec[i]
            row = offset + 4
            for col in range(nsamp):
                templates[row][col] = argvec[col] \
                                      * templates_view[offset][col]
            row = offset + 5
            for col in range(nsamp):
                templates[row][col] = argvec[col] \
                                      * templates_view[offset+1][col]

    return templates


def lowpass_ring(double[:] signal not None, int[:] pixels not None,
                 double[:] hits not None, int ndegrade):
    """
    Average neighboring pixels.
    """
    cdef int istart, istop, n, pix, i
    cdef double sigsum
    cdef double hitsum
    if signal.size != pixels.size:
        raise ValueError('Signal and pixels must be same size')
    n = signal.size
    istart = 0
    while istart < n:
        istop = istart
        sigsum = 0
        hitsum = 0
        pix = pixels[istart] / ndegrade
        while istop < n and (pixels[istop] / ndegrade == pix):
            sigsum += signal[istop]*hits[istop]
            hitsum += hits[istop]
            istop += 1
        if hitsum != 0 and istop - istart > 1:
            sigsum /= hitsum
            for i in range(istart, istop):
                signal[i] = sigsum
        istart = istop

    return


def collect_buf(double[:] sendbuf not None, commonpix,
                int nnz, double[:, :] sigmap not None):
    """
    Sample the sigmap at common pixels and pack into sendbuf.
    """
    cdef long offset = 0, i, j, pix
    cdef int[:] common_view
    for common in commonpix:
        common_view = common
        for i in range(common_view.size):
            pix = common_view[i]
            for j in range(nnz):
                sendbuf[offset] = sigmap[pix, j]
                offset += 1
    return


def sample_buf(double[:] recvbuf not None, commonpix, int nnz,
               double[:, :] sigmap not None):
    """
    Sample the recvbuf into sigmap.
    """
    cdef long offset = 0, i, j, pix
    cdef int[:] common_view
    sigmap[:, :] = 0
    for common in commonpix:
        common_view = common
        for i in range(common_view.size):
            pix = common_view[i]
            for j in range(nnz):
                sigmap[pix, j] += recvbuf[offset]
                offset += 1
    return


def bin_ring(int[:] pixels not None, double[:] signal not None,
             double[:, :] weights not None, double[:, :] quat not None,
             double[:] phase not None, unsigned char[:] flag not None,
             float[:] mask not None):
    """
    Bin TOD and pointing into rings.
    """

    cdef long nsamp, i, pix
    cdef int nnz

    nsamp, nnz = np.shape(weights)

    # Get the range of observed pixels

    cdef long pixmin = int(1e10)
    cdef long pixmax = int(-1e10)
    for i in range(nsamp):
        if flag[i] == 0:
            if pixmin > pixels[i]:
                pixmin = pixels[i]
            if pixmax < pixels[i]:
                pixmax = pixels[i]
    cdef long npix = pixmax - pixmin + 1

    if npix < 0:
        return None

    # Bin hits

    cdef int[:] hit = np.zeros(npix, dtype=np.int32)

    for i in range(nsamp):
        if flag[i] == 0:
            pix = pixels[i] - pixmin
            hit[pix] += 1

    # Build pixel index map

    cdef long nhit = 0
    for pix in range(npix):
        if hit[pix]:
            nhit += 1

    cdef int[:] glob2loc = np.zeros(npix, dtype=np.int32)
    cdef np.ndarray loc2glob = np.zeros(nhit, dtype=np.int32)
    cdef int[:] loc2glob_view = loc2glob

    i = 0
    for pix in range(npix):
        if hit[pix]:
            glob2loc[pix] = i
            loc2glob_view[i] = pix + pixmin
            i += 1

    # Bin the hits, signal and weights

    cdef np.ndarray ring_hits = np.zeros(nhit, dtype=np.int32)
    cdef np.ndarray ring_signal = np.zeros(nhit, dtype=np.float64)
    cdef np.ndarray ring_centroid = np.zeros([nhit, 4], dtype=np.float64)
    cdef np.ndarray ring_weights = np.zeros([nhit, nnz], dtype=np.float64)
    cdef np.ndarray ring_phase = np.zeros(nhit, dtype=np.float64)

    cdef int[:] ring_hits_view = ring_hits
    cdef double[:] ring_signal_view = ring_signal
    cdef double[:, :] ring_centroid_view = ring_centroid
    cdef double[:, :] ring_weights_view = ring_weights
    cdef double[:] ring_phase_view = ring_phase

    cdef long globpix, locpix
    cdef double vtest

    cdef double[:] first_phase = np.zeros(nhit) - 1000
    cdef double phasetemp

    for i in range(nsamp):
        if flag[i] == 0:
            globpix = pixels[i] - pixmin
            locpix = glob2loc[globpix]
            ring_hits_view[locpix] += 1
            ring_signal_view[locpix] += signal[i]
            # Averaging quaternions is sensitive business because
            # quat and -quat are the same rotation
            vtest = quat[i, 0]
            if fabs(vtest) < 1e-6:
                vtest = quat[i, 3]
            if vtest < 0:
                for j in range(4):
                    ring_centroid_view[locpix, j] += quat[i, j]
            else:
                for j in range(4):
                    ring_centroid_view[locpix, j] -= quat[i, j]
            for j in range(nnz):
                ring_weights_view[locpix, j] += weights[i, j]
            # Must ensure all phases that we add have the same branch
            phasetemp = phase[i]
            if first_phase[locpix] == -1000:
                first_phase[locpix] = phasetemp
            if first_phase[locpix] - phasetemp > M_PI:
                phasetemp += 2*M_PI
            elif first_phase[locpix] - phasetemp < -M_PI:
                phasetemp -= 2*M_PI
            ring_phase_view[locpix] += phasetemp

    cdef long n

    for pix in range(nhit):
        n = ring_hits_view[pix]
        if n > 0:
            ring_signal_view[pix] /= n
            for j in range(4):
                ring_centroid_view[pix, j] /= n
            for j in range(nnz):
                ring_weights_view[pix, j] /= n
            ring_phase_view[pix] /= n

    ring_centroid = toast.qarray.norm(ring_centroid)
    phaseorder = np.argsort(ring_phase).astype(np.int32)

    # Sample the mask

    ring_mask = np.zeros(nhit, dtype=np.float64)
    for pix in range(nhit):
        globpix = loc2glob_view[pix]
        ring_mask[pix] = mask[globpix]

    nbytes = (loc2glob.nbytes + ring_hits.nbytes + ring_signal.nbytes
              + ring_centroid.nbytes + ring_weights.nbytes + ring_phase.nbytes
              + ring_mask.nbytes)

    ring = Ring(pixels=loc2glob, hits=ring_hits, signal=ring_signal,
                quat=ring_centroid, weights=ring_weights, phase=ring_phase,
                nbytes=nbytes, phaseorder=phaseorder, mask=ring_mask)

    return ring


def get_glob2loc(int[:] loc2glob not None):
    """
    Build the inverse mapping of loc2glob
    """

    cdef long i, pix

    cdef long pixmin = np.amin(loc2glob)
    cdef long pixmax = np.amax(loc2glob)
    cdef long nhit = loc2glob.size
    cdef long npix = pixmax - pixmin + 1

    cdef int[:] hit = np.zeros(npix, dtype=np.int32)
    for i in range(nhit):
        pix = loc2glob[i] - pixmin
        hit[pix] = 1

    cdef np.ndarray[int] glob2loc = -np.ones(npix, dtype=np.int32)
    cdef int[:] glob2loc_view = glob2loc

    i = 0
    for pix in range(npix):
        if hit[pix]:
            glob2loc_view[pix] = i
            i += 1

    return glob2loc


def bin_ring_extra(int[:] pixels not None, double[:] signal not None,
                   unsigned char[:] flag not None,
                   int[:] loc2glob not None, int[:] glob2loc not None):
    """
    Using inputs from a previous call to bin_ring, bin an extra signal
    to a ring.
    """

    cdef long i, pix, n

    cdef long nsamp = signal.size
    cdef long pixmin = np.amin(loc2glob)
    cdef long nhit = loc2glob.size

    # Bin the signal

    cdef np.ndarray ring_signal = np.zeros(nhit, dtype=np.float64)
    cdef double[:] ring_signal_view = ring_signal
    # We count the hits again because the hit counts may have been
    # modulated by a mask
    cdef int[:] ring_hits = np.zeros(nhit, dtype=np.int32)
    cdef long globpix, locpix, nhit1 = 0, nhit2 = 0

    for i in range(nsamp):
        if flag[i] == 0:
            globpix = pixels[i] - pixmin
            locpix = glob2loc[globpix]
            ring_signal_view[locpix] += signal[i]
            ring_hits[locpix] += 1

    for pix in range(nhit):
        if ring_hits[pix] > 0:
            ring_signal_view[pix] /= ring_hits[pix]

    return ring_signal


def sample_ring(int[:] pixels not None, double[:] ring_signal not None,
                int[:] loc2glob not None, unsigned char[:] flag not None):
    """
    Sample given ring signal to full TOD.
    """

    cdef long i, pix

    # Reconstruct glob2loc

    cdef long nsamp = pixels.size
    cdef long pixmin = np.amin(loc2glob)
    cdef long pixmax = np.amax(loc2glob)
    cdef long nhit = loc2glob.size
    cdef long npix = pixmax - pixmin + 1

    cdef int[:] hit = np.zeros(npix, dtype=np.int32)
    for i in range(nhit):
        pix = loc2glob[i] - pixmin
        hit[pix] = 1

    cdef int[:] glob2loc = -np.ones(npix, dtype=np.int32)

    i = 0
    for pix in range(npix):
        if hit[pix]:
            glob2loc[pix] = i
            i += 1

    # Sample the signal

    cdef np.ndarray signal = np.zeros(nsamp, dtype=np.float64)
    cdef double[:] signal_view = signal

    cdef long globpix, locpix

    for i in range(nsamp):
        if flag[i] != 0:
            continue
        globpix = pixels[i]
        # if globpix < pixmin or globpix > pixmax:
        #     raise RuntimeError('globpix out of range: {} not in [{}, {}]'
        #                        ''.format(globpix, pixmin, pixmax))
        if globpix < 0:
            continue
        globpix -= pixmin
        locpix = glob2loc[globpix]
        if locpix >= 0:
            signal_view[i] += ring_signal[locpix]

    return signal


def fast_scanning_int32(
        int[:] pixels_in not None, int[:] map_in not None):
    cdef long n = pixels_in.size
    cdef np.ndarray tod = np.zeros(n, dtype=np.int32)
    cdef int[:] tod_view = tod
    cdef long pix
    cdef long i
    for i in range(n):
        pix = pixels_in[i]
        if pix < 0:
            continue
        tod_view[i] = map_in[pix]

    return tod


def flagged_running_average_with_downsample32(
        double[:] signal not None,
        np.ndarray[np.uint8_t, cast=True, ndim=1] flag not None,
        long wkernel):

    cdef unsigned char[:] flag_view = flag

    cdef long n = signal.size
    cdef long nn = int(n / wkernel)
    cdef np.ndarray out = np.zeros(nn, dtype=np.float32)
    cdef np.ndarray hit = np.zeros(nn, dtype=np.int32)
    cdef float[:] out_view = out
    cdef int[:] hit_view = hit

    cdef long ibin, istart, istop, i, nhit
    cdef double val
    for ibin in range(nn):
        istart = ibin * wkernel
        istop = istart + wkernel
        if istop > n:
            istop = n
        val = 0
        nhit = 0
        for i in range(istart, istop):
            if flag_view[i] == 0:
                val += signal[i]
                nhit += 1
        if nhit != 0:
            out_view[ibin] = val / nhit
            hit_view[ibin] = nhit

    return out, hit


def fast_cc_invert(
        double[:, :, :] cc_out not None, double[:, :, :] cc_in not None,
        double[:] rcond not None, double threshold, long itask, long ntask):

    cdef long npix
    cdef int nnz, nnz2

    npix, nnz, nnz2 = np.shape(cc_in)

    cdef long npix_task
    if ntask > 1:
        npix_task = npix // ntask + 1
    else:
        npix_task = npix

    cdef long my_first = itask * npix_task
    cdef long my_last = my_first + npix_task
    if my_last > npix:
        my_last = npix

    cdef int nnzsq = nnz*nnz
    cdef double[:] evals = np.zeros(nnzsq, dtype=np.float64)
    cdef double[:] ftemp = np.zeros(nnzsq, dtype=np.float64)
    cdef double[:] finv = np.zeros(nnzsq, dtype=np.float64)
    cdef double[:] fdata = np.zeros(nnzsq, dtype=np.float64)

    cdef int NB = 256
    cdef int lwork = NB * 2 + nnz
    cdef double[:] work = np.zeros(lwork, dtype=np.float64)

    cdef double emin
    cdef double emax
    cdef double rc

    cdef int info
    cdef double fzero = 0.0
    cdef double fone = 1.0

    cdef char jobz_vec = b'V'
    cdef char uplo = b'L'
    cdef char transN = b'N'
    cdef char transT = b'T'

    cdef long pix
    cdef long imap, jmap, kmap
    cdef long offset

    cdef double temp

    for pix in range(npix):
        if pix < my_first or pix >= my_last:
            for imap in range(nnz):
                for jmap in range(nnz):
                    cc_out[pix, imap, jmap] = 0
            rcond[pix] = 0
            continue
        if nnz == 1:
            # shortcut
            if cc_in[pix, 0, 0] == 0:
                rcond[pix] = 0
                cc_out[pix, 0, 0] = 0
            else:
                rcond[pix] = 1
                cc_out[pix, 0, 0] = 1 / cc_in[pix, 0, 0]
            continue
        offset = 0
        for imap in range(nnz):
            for jmap in range(nnz):
                fdata[offset + jmap] = cc_in[pix, imap, jmap]
            offset += nnz
        cython_lapack.dsyev(&jobz_vec, &uplo, &nnz, &(fdata[0]), &nnz,
                            &(evals[0]), &(work[0]), &lwork, &info)
        rc = 0
        for imap in range(nnz):
            for jmap in range(imap, nnz):
                cc_out[pix, imap, jmap] = 0
        if info == 0:
            emin = 1.0e100
            emax = 0
            for imap in range(nnz):
                if evals[imap] < emin:
                    emin = evals[imap]
                if evals[imap] > emax:
                    emax = evals[imap]
            if emax > 0:
                rc = emin / emax
            if rc >= threshold:
                # Scale the eigenvectors to invert the matrix
                offset = 0
                for imap in range(nnz):
                    evals[imap] = 1.0 / sqrt(evals[imap])
                    for jmap in range(nnz):
                        fdata[offset + jmap] *= evals[imap]
                    offset += nnz
                # transpose for faster access
                for imap in range(nnz):
                    for jmap in range(imap+1, nnz):
                        temp = fdata[imap*nnz + jmap]
                        fdata[imap*nnz+jmap] = fdata[jmap*nnz + imap]
                        fdata[jmap*nnz+imap] = temp
                # Multiply the eigenvectors to create the inverse
                offset = -1
                for imap in range(nnz):
                    for jmap in range(imap, nnz):
                        offset = imap*nnz + jmap
                        finv[offset] = 0
                        for kmap in range(nnz):
                            finv[offset] += fdata[imap*nnz + kmap] \
                                            * fdata[jmap*nnz + kmap]
                # Store the inverted matrix
                offset = 0
                for imap in range(nnz):
                    for jmap in range(imap, nnz):
                        cc_out[pix, imap, jmap] = finv[offset + jmap]
                        if imap != jmap:
                            cc_out[pix, jmap, imap] = finv[offset + jmap]
                    offset += nnz
        rcond[pix] = rc

    return


def fast_cc_multiply(
        double[:, :] map_out not None, double[:, :, :] cc not None,
        double[:, :] map_in not None, long itask, long ntask):

    cdef long npix
    cdef int nnz

    npix, nnz = np.shape(map_in)

    cdef long npix_task
    if ntask > 1:
        npix_task = npix // ntask + 1
    else:
        npix_task = npix

    cdef long my_first = itask * npix_task
    cdef long my_last = my_first + npix_task
    if my_last > npix:
        my_last = npix

    # Temporary storage for the result to allow
    # map_out and map_in be the same array
    cdef double[:] temp = np.zeros(nnz, dtype=np.float64)

    cdef long pix, imap, jmap
    for pix in range(npix):
        if pix < my_first or pix >= my_last:
            for imap in range(nnz):
                map_out[pix, imap] = 0
            continue
        if nnz == 1:
            # shortcut
            map_out[pix, 0] = cc[pix, 0, 0] * map_in[pix, 0]
            continue
        for imap in range(nnz):
            temp[imap] = 0
            for jmap in range(nnz):
                temp[imap] += cc[pix, imap, jmap] * map_in[pix, jmap]
        for imap in range(nnz):
            map_out[pix, imap] = temp[imap]

    return


def fast_scanning_pol(
        double[:] toi not None, int[:] pixels not None,
        float[:, :] weights not None, double[:, :] bmap not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix
    cdef long isamp, imap
    for isamp in range(nsamp):
        pix = pixels[isamp]
        if pix < 0:
            continue
        toi[isamp] = bmap[pix, 0] * weights[isamp, 0]
        for imap in range(1, nnz):
            toi[isamp] += bmap[pix, imap] * weights[isamp, imap]

    return


def fast_scanning_pol64(
        double[:] toi not None, int[:] pixels not None,
        double[:, :] weights not None, double[:, :] bmap not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix
    cdef long isamp, imap
    for isamp in prange(nsamp, nogil=True):
        pix = pixels[isamp]
        if pix < 0:
            continue
        toi[isamp] = bmap[pix, 0] * weights[isamp, 0]
        for imap in range(1, nnz):
            toi[isamp] += bmap[pix, imap] * weights[isamp, imap]

    return


def fast_scanning_pol_eff(
        double[:] toi not None, int[:] pixels not None,
        double[:, :] weights not None, double[:, :] bmap not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix
    cdef long isamp, imap
    for isamp in range(nsamp):
        pix = pixels[isamp]
        if pix < 0:
            continue
        toi[isamp] = (bmap[pix, 1] * weights[isamp, 1]
                      + bmap[pix, 2] * weights[isamp, 2])

    return


def fast_scanning_pol_angle(
        double[:] toi not None, int[:] pixels not None,
        double[:, :] weights not None, double[:, :] bmap not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix
    cdef long isamp, imap
    for isamp in range(nsamp):
        pix = pixels[isamp]
        if pix < 0:
            continue
        toi[isamp] = 2*(bmap[pix, 2] * weights[isamp, 1]
                        - bmap[pix, 1] * weights[isamp, 2])

    return


def fast_weight_binning(
        int[:] pixels not None, float[:, :] weights not None,
        double[:, :, :] cc not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix
    cdef long isamp, imap, jmap
    for isamp in range(nsamp):
        pix = pixels[isamp]
        if pix < 0:
            continue
        for imap in range(nnz):
            for jmap in range(nnz):
                cc[pix, imap, jmap] += weights[isamp, imap] \
                                       * weights[isamp, jmap]

    return


def fast_weight_binning_with_mult(
        int[:] pixels not None, double[:, :] weights not None,
        double[:] mult not None, double[:, :, :] cc not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix, isamp, imap, jmap
    cdef double m
    for isamp in range(nsamp):
        pix = pixels[isamp]
        m = mult[isamp]
        if pix < 0 or m == 0:
            continue
        for imap in range(nnz):
            for jmap in range(nnz):
                cc[pix, imap, jmap] += weights[isamp, imap] \
                                       * weights[isamp, jmap] * m

    return


def fast_binning_pol(
        double[:] toi not None, int[:] pixels not None,
        float[:, :] weights not None, double[:, :] bmap not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix
    cdef long isamp, imap
    for isamp in range(nsamp):
        pix = pixels[isamp]
        if pix < 0:
            continue
        for imap in range(nnz):
            bmap[pix, imap] += toi[isamp] * weights[isamp, imap]

    return


def fast_binning_pol_with_mult(
        double[:] toi not None, int[:] pixels not None,
        double[:, :] weights not None, double[:] mult not None,
        double[:, :] bmap not None):

    cdef long nsamp, nnz
    nsamp, nnz = np.shape(weights)

    cdef long pix, isamp, imap
    cdef double m
    for isamp in range(nsamp):
        pix = pixels[isamp]
        m = mult[isamp]
        if pix < 0 or m == 0:
            continue
        for imap in range(nnz):
            bmap[pix, imap] += toi[isamp] * weights[isamp, imap] * m
    return


def fast_binning(
        double[:] toi not None, int[:] pixels not None,
        double[:] bmap not None):

    cdef long pix
    cdef long i
    for i in range(len(toi)):
        pix = pixels[i]
        if pix < 0:
            continue
        bmap[pix] += toi[i]

    return


def fast_binning32(
        float[:] toi not None, int[:] pixels not None,
        float[:] bmap not None):

    cdef long pix
    cdef long i
    for i in xrange(len(toi)):
        pix = pixels[i]
        if pix < 0:
            continue
        bmap[pix] += toi[i]

    return


def fast_hit_binning(int[:] pixels not None, int[:] hmap not None):

    cdef long pix
    cdef long i
    for i in range(len(pixels)):
        pix = pixels[i]
        if pix < 0:
            continue
        hmap[pix] += 1

    return


def fast_scanning(
        double[:] toi not None, int[:] pixels not None,
        double[:] bmap not None):

    cdef long pix
    cdef long i, n = len(toi)
    for i in prange(n, nogil=True, schedule='static', chunksize=1000):
        pix = pixels[i]
        if pix < 0:
            toi[i] = 0
        else:
            toi[i] = bmap[pix]

    return


def fast_scanning32(
        double[:] toi not None, long[:, :] pixels not None,
        double[:, :] weights not None, float[:] bmap not None):
    """ Fast scanning for bilinear interpolation
    """
    cdef long pix
    cdef long row, col, nrow, ncol
    nrow, ncol = np.shape(pixels)
    for row in range(nrow):
        for col in prange(ncol, nogil=True, schedule='static', chunksize=1000):
            pix = pixels[row, col]
            toi[col] += bmap[pix] * weights[row, col]

    return


def fast_masked_dot_product(
        double[:] toi1 not None, double[:] toi2 not None,
        int[:] flag not None):

    cdef double val = 0
    cdef long i
    for i in range(len(toi1)):
        if flag[i] == 0:
            val += toi1[i] * toi2[i]

    return val


def fast_dot_product(double[:] toi1 not None, double[:] toi2 not None):

    cdef double val = 0
    cdef long i
    for i in range(len(toi1)):
        val += toi1[i] * toi2[i]

    return val


def fast_dot_product32(float[:] toi1 not None, float[:] toi2 not None):

    cdef double val = 0
    cdef long i
    cdef long imax = len(toi1)
    for i in range(imax):
        val += toi1[i] * toi2[i]

    return val


def fast_masked_sum(double[:] toi not None, int[:] flag not None):

    cdef double val = 0
    cdef long i
    for i in range(len(toi)):
        if flag[i] == 0:
            val += toi[i]

    return val
