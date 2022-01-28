# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import adc_inverter_helpers
import astropy.io.fits as pyfits
import numpy as np
import toast.timing as timing


def get_dpc_ring_slices(dpc_iring_table, dpc_obt_table, npipe_obt):
    """
    This helper function takes a lookup table and determines up which
    samples fall inside which dpc ring.

    Args:
      dpc_iring_table -- array containing list of Planck HFI DPC ring
          index numbers
      dpc_obt_table -- array containing the onboard time (OBT) for the
          start of each DPC ring
      npipe_obt -- the time series of on-board times to look up.

    Returns:
      list of DPC ring numbers, list of OBT start times of the rings,
      list of start sample index in input array, list of end sample index
          in input array
    """
    # first find closest ring to start of npipe obt data
    time_diff = np.abs(npipe_obt[0] - dpc_obt_table)
    ii = np.where(time_diff == time_diff.min())[0][0]
    if dpc_obt_table[ii] > npipe_obt[0]:
        ii -= 1

    # now step through possible DPC rings making sure we don't go past the end
    dpc_ring_list = []
    dpc_obt_list = []
    ibegin_list = []
    iend_list = []

    while ((dpc_obt_table[ii] < npipe_obt[-1])
           and (dpc_iring_table[ii] < dpc_iring_table[-1])):
        dpc_ring_list.append(dpc_iring_table[ii])
        dpc_obt_list.append(dpc_obt_table[ii])
        idxs = np.where(np.logical_and(npipe_obt >= dpc_obt_table[ii],
                                       npipe_obt < dpc_obt_table[ii + 1]))
        ibegin_list.append(idxs[0][0])
        iend = idxs[0][-1]
        if iend >= len(npipe_obt):
            iend = len(npipe_obt) - 1
        iend_list.append(iend)
        ii += 1
    return dpc_ring_list, dpc_obt_list, ibegin_list, iend_list


def dist(n, m):
    """
    This is a rewrite of an IDL function.
    """
    x = np.arange(n)
    x = (np.minimum(x, (n * np.ones(n) - x))) ** 2
    a = np.zeros((m, n))
    for i in np.arange(m / 2 + 1):
        y = np.sqrt(x + i ** 2)
        a[i, :] = y
        if i != 0:
            a[m - i, :] = y
    return a


def invert_adc_table(inlint, npts=10000000, adu_noise=4.0):
    """
    This is a helper function that inverts an INL lookup table.

    Args:
      inlint -- the forward nonlinearity table
      npts -- the number of points in the output inverse table
      adu_noise -- Gaussian noise smoothing kernel width in ADU

    Returns:
      noise-smoothed inverse lookup table.
    """
    sss = np.linspace(0, npts - 1, num=npts) / npts * 32768 * 2 - 32768
    ip_list = np.linspace(10 * npts / 32768,
                          np.size(sss) - 1 - 10 * npts / 32768,
                          num=npts).astype('int64')
    adu = np.zeros(npts)
    icoulist = np.zeros(np.size(ip_list), dtype='int16') - 10
    ok = np.zeros(np.size(icoulist), dtype='int16')
    while np.size(np.where(ok == 0)) > 0:
        ok[np.where((inlint[(sss[ip_list] + icoulist).astype('int64') + 32768]
                     - sss[ip_list] - 32768) >= 0)] = 1
        icoulist[np.where(ok == 0)] += 1
    adu[ip_list] = (sss[ip_list] + icoulist - 1).astype(int)

    if (adu_noise != 0.0):
        dist_nadu = dist(npts, 1)
        electronic_noise = np.fft.fft(
            np.exp(-dist_nadu ** 2 / 2
                   / (adu_noise * np.size(adu) / 65536) ** 2))
        adu = np.fft.fft(
            np.fft.ifft(adu) * electronic_noise) / np.amax(electronic_noise)
        adu = adu.real[0, :]
    return adu


class ADC_Inverter():

    def __init__(self, bolometer, sphase, adc_inllookupfile,
                 adc_rawgainfile, adc_4kharmonicsfile,
                 adc_gainvstimefile, adc_ringtimefile, adc_noise=4.0,
                 comm=None):
        """
        Instantiate the ADC inverter object.

        Args:
        bolometer -- bolometer name
        sphase -- Sphase parameter
        adc_inllookupfile -- fits file containing INL table
        adc_rawgainfile -- fits file containing raw gain
        adc_4kharmonicsfile -- fits file containing 4K harmonics
        adc_gainvstimefile -- fits file containing raw constant
        adc_ringtimefile -- DPC ring index
        adc_noise -- assumed ADC noise smoothing width
        comm -- an MPI communicator instance (optional)
        """
        if comm is None:
            rank = 0
            ntask = 1
        else:
            rank = comm.Get_rank()
            ntask = comm.Get_size()

        self.adc_data = {}
        if rank == 0:

            # read ADC lookup table
            hdulist = pyfits.open(adc_inllookupfile)
            self.adc_data['inlint'] = hdulist[bolometer].data.field(0)
            hdulist.close()

            hdulist = pyfits.open(adc_rawgainfile)
            self.adc_data['raw_gain'] = hdulist[bolometer].data.field(0)
            hdulist.close()

            # get 4k harmonics
            hdulist = pyfits.open(adc_4kharmonicsfile)
            self.adc_data[
                'harmonics_4k_real'] = hdulist[bolometer].data.field(0)
            self.adc_data[
                'harmonics_4k_imag'] = hdulist[bolometer].data.field(1)

            # do we have 6 4K harmonics?
            if hdulist[bolometer].header['n_harmonics'] == 6:
                self.adc_data['harm_list'] = np.array([1, 4, 6, 8, 10, 14])
                self.nharm = 6
            else:
                # default is three 4K harmonics
                self.adc_data['harm_list'] = np.array([1, 8, 10])
                self.nharm = 3
            hdulist.close()

            hdulist = pyfits.open(adc_gainvstimefile)
            self.adc_data[
                'constant_gain'] = hdulist[bolometer].data.field(0) - 2 ** 15
            hdulist.close()

            hdulist = pyfits.open(adc_ringtimefile)
            self.adc_data[
                'dpc_iring_table'] = hdulist['RINGTIMES'].data.field(0)
            self.adc_data['dpc_obt_table'] = hdulist['RINGTIMES'].data.field(1)
            igood = np.where(self.adc_data['dpc_obt_table'] != 0)
            self.adc_data[
                'dpc_iring_table'] = self.adc_data['dpc_iring_table'][igood]
            self.adc_data[
                'dpc_obt_table'] = self.adc_data['dpc_obt_table'][igood]
            hdulist.close()

        if ntask > 1:
            self.adc_data = comm.bcast(self.adc_data, root=0)

        # ERROR : pars is undefined
        self.inv_dnl_vct = invert_adc_table(self.adc_data['inlint'],
                                            adu_noise=pars.adu_noise)
        self.rw_gn_vct = self.adc_data['raw_gain']

        self.s_phs = sphase
        self.n_hrm_rw = 9

        self.sgnl_stp = 25
        self.n_sgnl_ttl = 100000
        self.alph_hrm_rw = 2 * np.pi / self.n_hrm_rw
        self.n_pnt_sgnl = self.n_sgnl_ttl / self.sgnl_stp

        # ERROR : rw_gn_vct is undefined
        self.n_fst_smpl = np.size(rw_gn_vct)
        self.n_hlf_fst_smpl = self.n_fst_smpl / 2

        self.adc_rng = 2 ** 16

        self.cnv_fct = float(
            np.size(self.inv_dnl_vct)) / float(self.adc_rng)

        self.s_phs_arr = (np.arange(self.n_fst_smpl, dtype='int16')
                          + self.s_phs) % self.n_fst_smpl
        return

    def nw_rw_dt(self,
                 corr_1_arg,
                 corr_0_arg,
                 rw_dt_arg,
                 sgn_arg):
        """
        Correct a given raw data value.
        """
        i_ps = 1
        dff_tmp = np.round((rw_dt_arg - corr_0_arg[i_ps]) / self.sgnl_stp)
        if sgn_arg == -1:
            if dff_tmp >= 0:
                i_ps = 1
            else:
                i_ps -= dff_tmp
                i_ps -= 5
                if i_ps < 0:
                    i_ps = 0
                if i_ps >= np.size(corr_0_arg):
                    i_ps = np.size(corr_0_arg) - 1
                while ((rw_dt_arg - corr_0_arg[i_ps]) / self.sgnl_stp) < 0:
                    i_ps += 1
                    if i_ps >= self.n_pnt_sgnl - 1:
                        i_ps = self.n_pnt_sgnl - 1
                        break
        if sgn_arg == 1:
            if dff_tmp <= 0:
                i_ps = 0
            else:
                i_ps += dff_tmp
                i_ps -= 5
                if i_ps < 0:
                    i_ps = 0
                if i_ps >= np.size(corr_0_arg):
                    i_ps = np.size(corr_0_arg) - 2
                while ((rw_dt_arg - corr_0_arg[i_ps]) / self.sgnl_stp) >= 0:
                    i_ps += 1
                    if i_ps >= self.n_pnt_sgnl - 1:
                        i_ps = self.n_pnt_sgnl - 2
                        break
                i_ps -= 1
        return (corr_1_arg[i_ps + sgn_arg] - corr_1_arg[i_ps]) \
            * (rw_dt_arg - corr_0_arg[i_ps]) \
            / (corr_0_arg[i_ps + sgn_arg] - corr_0_arg[i_ps]) \
            + corr_1_arg[i_ps]

    def correct_hrm(self,
                    i_hrm_arg,
                    rw_dt_arg,
                    rw_cst_vct_arg,
                    indx_rng_mod_dbl_arg,
                    prty_vct_arg,
                    hrm_4k_vct_arg,
                    rw_4k_re_vct_arg,
                    rw_4k_im_vct_arg,
                    n_hlf_pnt_arg,
                    n_dt_hrm_arg):
        """
        Make the correction for a particular 4K harmonic.
        """
        n_pnt_arg = self.n_pnt_sgnl
        corrp_0 = np.zeros(n_pnt_arg)
        corrp_1 = np.zeros(n_pnt_arg)
        corrm_0 = np.zeros(n_pnt_arg)
        corrm_1 = np.zeros(n_pnt_arg)
        pp_gp = (int(indx_rng_mod_dbl_arg) + 2 * i_hrm_arg
                 + int(prty_vct_arg[0])) % 18288

        rw_4K_arry = np.zeros(self.n_fst_smpl)
        inv_hlf_fst_smpl_dbl = 2.0 / float(self.n_fst_smpl)

        for i_hrm in np.arange(np.size(hrm_4k_vct_arg)):
            phs_4k_hrm = hrm_4k_vct_arg[i_hrm] * self.alph_hrm_rw
            for i_fst in np.arange(self.n_fst_smpl):
                phs_4k_fst = (float(i_fst) * inv_hlf_fst_smpl_dbl + pp_gp) \
                             * phs_4k_hrm
                rw_4K_arry[i_fst] += (
                    rw_4k_re_vct_arg[i_hrm] * np.cos(phs_4k_fst)
                    + rw_4k_im_vct_arg[i_hrm] * np.sin(phs_4k_fst))

        sgnl = -8.0e5
        for i_fst in np.arange(self.n_hlf_fst_smpl):
            rw_w = rw_cst_vct_arg[self.s_phs_arr[i_fst]] + rw_4K_arry[i_fst] \
                   + sgnl * self.rw_gn_vct[self.s_phs_arr[i_fst]]
            rw_m = self.inv_dnl_vct[int((rw_w + 32768) * self.cnv_fct)]
            corrp_0[0] += rw_m
            corrp_1[0] += rw_w

            rw_w = rw_cst_vct_arg[self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]] \
                + rw_4K_arry[i_fst + self.n_hlf_fst_smpl] \
                + sgnl * self.rw_gn_vct[
                    self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]]

            rw_m = self.inv_dnl_vct[int((rw_w + 32768) * self.cnv_fct)]
            corrm_0[0] += rw_m
            corrm_1[0] += rw_w
        sgnl = 8.0e5
        for i_fst in np.arange(self.n_hlf_fst_smpl):
            rw_w = rw_cst_vct_arg[self.s_phs_arr[i_fst]] + rw_4K_arry[i_fst] \
                   + sgnl * self.rw_gn_vct[self.s_phs_arr[i_fst]]
            rw_m = self.inv_dnl_vct[int((rw_w + 32768) * self.cnv_fct)]
            corrp_0[n_pnt_arg - 1] += rw_m
            corrp_1[n_pnt_arg - 1] += rw_w

            rw_w = rw_cst_vct_arg[self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]] \
                + rw_4K_arry[i_fst + self.n_hlf_fst_smpl] \
                + sgnl * self.rw_gn_vct[
                    self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]]
            rw_m = self.inv_dnl_vct[int((rw_w + 32768) * self.cnv_fct)]
            corrm_0[n_pnt_arg - 1] += rw_m
            corrm_1[n_pnt_arg - 1] += rw_w

        if False:
            for i_pnt in np.arange(1, n_pnt_arg - 1):
                sgnl = (i_pnt - n_hlf_pnt_arg) * self.sgnl_stp
                for i_fst in np.arange(self.n_hlf_fst_smpl):
                    rw_w = rw_cst_vct_arg[self.s_phs_arr[i_fst]] \
                       + rw_4K_arry[i_fst] \
                       + sgnl * self.rw_gn_vct[self.s_phs_arr[i_fst]]
                    rw_m = self.inv_dnl_vct[
                        int((rw_w + 32768) * self.cnv_fct)]
                    corrp_0[i_pnt] += rw_m
                    corrp_1[i_pnt] += rw_w

                    rw_w = rw_cst_vct_arg[
                        self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]] \
                        + rw_4K_arry[i_fst + self.n_hlf_fst_smpl] \
                        + sgnl * self.rw_gn_vct[
                            self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]]
                    rw_m = self.inv_dnl_vct[
                        int((rw_w + 32768) * self.cnv_fct)]
                    corrm_0[i_pnt] += rw_m
                    corrm_1[i_pnt] += rw_w
        else:
            i_pnt = np.arange(1, n_pnt_arg - 1)
            abunchofones = np.ones(np.size(i_pnt), dtype=int)
            sgnl = (i_pnt - n_hlf_pnt_arg) * self.sgnl_stp
            for i_fst in np.arange(self.n_hlf_fst_smpl):
                rw_w = rw_cst_vct_arg[self.s_phs_arr[i_fst]] * abunchofones \
                       + rw_4K_arry[i_fst] * abunchofones \
                       + sgnl * self.rw_gn_vct[self.s_phs_arr[i_fst]]
                rw_m = self.inv_dnl_vct[((rw_w + 32768)
                                         * self.cnv_fct).astype(int)]
                corrp_0[i_pnt] += rw_m
                corrp_1[i_pnt] += rw_w

                rw_w = rw_cst_vct_arg[
                    self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]] \
                    * abunchofones \
                    + rw_4K_arry[i_fst + self.n_hlf_fst_smpl] \
                    + sgnl * self.rw_gn_vct[
                        self.s_phs_arr[i_fst + self.n_hlf_fst_smpl]]
                rw_m = self.inv_dnl_vct[((rw_w + 32768)
                                         * self.cnv_fct).astype(int)]
                corrm_0[i_pnt] += rw_m
                corrm_1[i_pnt] += rw_w
        slctp = np.zeros(n_dt_hrm_arg, dtype='int')
        n_dt_rw_18 = np.size(rw_dt_arg) - 18
        offst_tmp = 2 * i_hrm_arg + int(prty_vct_arg[0])
        slctp[0] = offst_tmp
        slctp[1] = offst_tmp + 1
        i_tmp = 1
        while (slctp[2 * i_tmp - 2] < n_dt_rw_18):
            slctp[2 * i_tmp] = 18 * i_tmp + offst_tmp
            slctp[2 * i_tmp + 1] = slctp[2 * i_tmp] + 1
            i_tmp += 1

        ttl_4k_p = 0
        ttl_4k_m = 0
        for i_fst in np.arange(self.n_hlf_fst_smpl):
            ttl_4k_p += rw_4K_arry[i_fst]
            ttl_4k_m += rw_4K_arry[i_fst + self.n_hlf_fst_smpl]

        shft_p = 0
        shft_m = 1
        sgn_p = 1
        sgn_m = -1
        if (prty_vct_arg[slctp[0]] == 1):
            shft_p = 1
            shft_m = 0
            sgn_p = -1
            sgn_m = 1
        for i_p in np.arange(i_tmp):
            if slctp[2 * i_p + shft_p] < np.size(rw_dt_arg):
                rw_dt_arg[slctp[2 * i_p + shft_p]] = self.nw_rw_dt(
                    corrp_1, corrp_0, rw_dt_arg[slctp[2 * i_p + shft_p]],
                    sgn_p) - ttl_4k_p
            if slctp[2 * i_p + shft_m] < np.size(rw_dt_arg):
                rw_dt_arg[slctp[2 * i_p + shft_m]] = self.nw_rw_dt(
                    corrm_1, corrm_0, rw_dt_arg[slctp[2 * i_p + shft_m]],
                    sgn_m) - ttl_4k_m

        # First element of the TOI (NB: this condition is expected to
        # be the usual case)
        if (i_hrm_arg == self.n_hrm_rw - 1) and (prty_vct_arg[slctp[0]] == 0):
                rw_dt_arg[0] = self.nw_rw_dt(corrm_1, corrm_0,
                                             rw_dt_arg[0], -1) - ttl_4k_m
        return rw_dt_arg

    def correct_adc(self,
                    raw_data_in,
                    rw_cst_vct_arg,
                    indx_rng_mod_dbl_arg,
                    prty_vct_arg,
                    hrm_4k_vct_arg,
                    rw_4k_re_vct_arg,
                    rw_4k_im_vct_arg, asint16=True):
        """
        This is the main function that takes input raw data and corrects it

        raw_data_in = input raw data
        rw_cst_vct_arg = constant signal  vector
        prty_vct_arg = parity data
        hrm_4k_re_vct_arg = real part of 4K harmonics
        hrm_4k_im_vct_arg = imaginary part of 4K harmonics

        Returns corrected raw data.

        """
        # half the number of samples in a fast sample chunk
        n_hlf_pnt = float(self.n_pnt_sgnl) / 2

        n_dt_hrm = np.size(raw_data_in) / 9 + 4

        # shift data
        result = raw_data_in.copy() - 40 * 2 ** 15

        result = result.astype(float)

        for i_hrm in np.arange(self.n_hrm_rw):
            # native python
            # result = self.correct_hrm(
            #    i_hrm, result, rw_cst_vct_arg, indx_rng_mod_dbl_arg,
            #    prty_vct_arg, hrm_4k_vct_arg, rw_4k_re_vct_arg,
            #    rw_4k_im_vct_arg, n_hlf_pnt, n_dt_hrm)
            # cython:
            result = adc_inverter_helpers.correct_hrm(
                i_hrm, result, rw_cst_vct_arg, indx_rng_mod_dbl_arg,
                prty_vct_arg, hrm_4k_vct_arg, rw_4k_re_vct_arg,
                rw_4k_im_vct_arg, n_hlf_pnt, n_dt_hrm, self.n_pnt_sgnl,
                self.n_fst_smpl, self.alph_hrm_rw, self.n_hlf_fst_smpl,
                self.s_phs_arr, self.rw_gn_vct, self.cnv_fct,
                self.inv_dnl_vct, self.sgnl_stp, self.n_pnt_sgnl, self.n_hrm_rw)
            # print i_hrm,result[:10]

        # if flag is set, return integer
        if asint16:
            result = (np.floor(result)).astype(np.int16) + 40 * 2 ** 15
        else:
            # otherwise, leave it as a double
            result += 40 * 2 ** 15
        return result

    def correct(self, data_in, parity, obt, mod4k, asint16=False):
        # figure out which DPC rings correspond to the samples we just got
        # ERROR : adc_data undefined
        (dpc_ring_list, dpc_obt_list, ibegin_list,
         iend_list) = get_dpc_ring_slices(
                adc_data['dpc_iring_table'], adc_data['dpc_obt_table'], obt)
        result = data_in.copy()
        for ii, ring_number_dpc in enumerate(dpc_ring_list):
            if not np.isnan(adc_data['constant_gain'][ring_number_dpc * 80]):
                # shift the 4K line modulation phase
                mod4k_slice = (mod4k + ibegin_list[ii]) % 18288
                result[ibegin_list[ii]:iend_list[ii]] = self.correct_adc(
                    data_in[ibegin_list[ii]:iend_list[ii]],
                    adc_data['constant_gain'][
                        ring_number_dpc * 80:(ring_number_dpc * 80 + 80)],
                    mod4k_slice, parity_in[ibegin_list[ii]:iend_list[ii]],
                    adc_data['harm_list'],
                    adc_data['harmonics_4k_real'][
                        ring_number_dpc * nharm:(ring_number_dpc * nharm + nharm)],
                    adc_data['harmonics_4k_imag'][
                        ring_number_dpc * nharm:(ring_number_dpc * nharm + nharm)],
                    asint16=asint16)
        return result
