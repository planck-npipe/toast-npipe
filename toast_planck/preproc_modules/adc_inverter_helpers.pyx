# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


import numpy as np
cimport numpy as np

DTYPE = np.double


def nw_rw_dt(np.ndarray corr_1_arg,
             np.ndarray corr_0_arg,
             float rw_dt_arg,
             int sgn_arg,
             int sgnl_stp, int n_pnt_sgnl):
    """
        Correct a given raw data value.
    """
    if rw_dt_arg == 0.0:
        return rw_dt_arg
    cdef int i_ps = 1
    cdef int dff_tmp
    # BPC: debug test
    try:
        dff_tmp = np.int(np.round((rw_dt_arg - corr_0_arg[i_ps])
                                  / np.float(sgnl_stp)))
    except Exception:
        return 0.0
    if (sgn_arg == -1):
        if dff_tmp >= 0:
            i_ps = 1
        else:
            i_ps -= dff_tmp
            i_ps -= 5
            if i_ps < 0:
                i_ps = 0
            if i_ps > (np.size(corr_0_arg)):
                i_ps = np.size(corr_0_arg)-1
                # print i_ps,np.size(corr_0_arg),sgnl_stp
            while ((rw_dt_arg - corr_0_arg[i_ps]) / np.float(sgnl_stp)) < 0:
                i_ps += 1
                if i_ps >= n_pnt_sgnl - 1:
                    i_ps = n_pnt_sgnl - 1
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
                i_ps = np.size(corr_0_arg)-2
            while ((rw_dt_arg - corr_0_arg[i_ps]) / np.float(sgnl_stp)) >= 0:
                i_ps += 1
                if i_ps >= n_pnt_sgnl - 1:
                    i_ps = n_pnt_sgnl - 2
                    break
            i_ps -= 1
    return ((corr_1_arg[i_ps + sgn_arg] - corr_1_arg[i_ps])
            * (rw_dt_arg - corr_0_arg[i_ps])
            / (corr_0_arg[i_ps + sgn_arg] - corr_0_arg[i_ps])
            + corr_1_arg[i_ps])


def correct_hrm(int i_hrm_arg,
                np.ndarray rw_dt_arg,
                np.ndarray rw_cst_vct_arg,
                int indx_rng_mod_dbl_arg,
                np.ndarray prty_vct_arg,
                np.ndarray hrm_4k_vct_arg,
                np.ndarray rw_4k_re_vct_arg,
                np.ndarray rw_4k_im_vct_arg,
                int n_hlf_pnt_arg,
                int n_dt_hrm_arg,
                int n_pnt_arg,
                int n_fst_smpl,
                double alph_hrm_rw,
                int n_hlf_fst_smpl,
                np.ndarray s_phs_arr,
                np.ndarray rw_gn_vct,
                double cnv_fct,
                np.ndarray inv_dnl_vct,
                double sgnl_stp,
                int n_pnt_sgnl,
                int n_hrm_rw):
        """
        Make the correction for a particular 4K harmonic.
        """
        cdef int i_fst
        cdef np.ndarray corrp_0 = np.zeros(n_pnt_arg)
        cdef np.ndarray corrp_1 = np.zeros(n_pnt_arg)
        corrm_0 = np.zeros(n_pnt_arg)
        corrm_1 = np.zeros(n_pnt_arg)
        cdef double pp_gp = np.float((np.int(indx_rng_mod_dbl_arg)
                                      + 2 * i_hrm_arg
                                      + np.int(prty_vct_arg[0])) % 18288)

        rw_4K_arry = np.zeros(n_fst_smpl)
        inv_hlf_fst_smpl_dbl = 2.0 / np.float(n_fst_smpl)

        for i_hrm in np.arange(np.size(hrm_4k_vct_arg)):
            phs_4k_hrm = hrm_4k_vct_arg[i_hrm] * alph_hrm_rw
            for i_fst in np.arange(n_fst_smpl):
                phs_4k_fst = (np.float(i_fst) * inv_hlf_fst_smpl_dbl
                              + pp_gp) * phs_4k_hrm
                rw_4K_arry[i_fst] += (
                    rw_4k_re_vct_arg[i_hrm] * np.cos(phs_4k_fst)
                    + rw_4k_im_vct_arg[i_hrm] * np.sin(phs_4k_fst))

        sgnl = -8.0e5
        for i_fst in np.arange(n_hlf_fst_smpl):
            rw_w = rw_cst_vct_arg[s_phs_arr[i_fst]] + rw_4K_arry[i_fst] \
                + sgnl * (rw_gn_vct[s_phs_arr[i_fst]])
            rw_m = inv_dnl_vct[np.int((rw_w + 32768) * cnv_fct)]
            corrp_0[0] += rw_m
            corrp_1[0] += rw_w

            rw_w = rw_cst_vct_arg[s_phs_arr[i_fst + n_hlf_fst_smpl]] \
                + rw_4K_arry[i_fst + n_hlf_fst_smpl] \
                + sgnl * (rw_gn_vct[s_phs_arr[i_fst + n_hlf_fst_smpl]])

            rw_m = inv_dnl_vct[np.int((rw_w + 32768) * cnv_fct)]
            corrm_0[0] += rw_m
            corrm_1[0] += rw_w
        sgnl = 8.0e5
        for i_fst in np.arange(n_hlf_fst_smpl):
            rw_w = rw_cst_vct_arg[s_phs_arr[i_fst]] + rw_4K_arry[i_fst] \
                + sgnl * (rw_gn_vct[s_phs_arr[i_fst]])

            rw_m = inv_dnl_vct[np.int((rw_w + 32768) * cnv_fct)]
            corrp_0[n_pnt_arg - 1] += rw_m
            corrp_1[n_pnt_arg - 1] += rw_w

            rw_w = rw_cst_vct_arg[s_phs_arr[i_fst + n_hlf_fst_smpl]] \
                + rw_4K_arry[i_fst + n_hlf_fst_smpl] \
                + sgnl * (rw_gn_vct[s_phs_arr[i_fst + n_hlf_fst_smpl]])

            rw_m = inv_dnl_vct[np.int((rw_w + 32768) * cnv_fct)]
            corrm_0[n_pnt_arg - 1] += rw_m
            corrm_1[n_pnt_arg - 1] += rw_w

        cdef np.ndarray i_pnt = np.arange(1, n_pnt_arg - 1)
        cdef np.ndarray abunchofones = np.ones(np.size(i_pnt), dtype=int)
        sgnl = (i_pnt - n_hlf_pnt_arg) * sgnl_stp
        for i_fst in np.arange(n_hlf_fst_smpl):
                rw_w = rw_cst_vct_arg[s_phs_arr[i_fst]]*abunchofones \
                    + rw_4K_arry[i_fst]*abunchofones \
                    + sgnl * (rw_gn_vct[s_phs_arr[i_fst]])
                rw_m = inv_dnl_vct[((rw_w + 32768) * cnv_fct).astype('int64')]
                corrp_0[i_pnt] += rw_m
                corrp_1[i_pnt] += rw_w

                rw_w = rw_cst_vct_arg[s_phs_arr[i_fst + n_hlf_fst_smpl]] \
                    * abunchofones + rw_4K_arry[i_fst + n_hlf_fst_smpl] \
                    + sgnl * (rw_gn_vct[s_phs_arr[i_fst + n_hlf_fst_smpl]])
                rw_m = inv_dnl_vct[((rw_w + 32768) * cnv_fct).astype('int64')]
                corrm_0[i_pnt] += rw_m
                corrm_1[i_pnt] += rw_w

        # print "sum corrm_0",np.sum(corrm_0)
        # print "sum corrm_1",np.sum(corrm_1)
        # print "sum corrp_0",np.sum(corrp_0)
        # print "sum corrp_1",np.sum(corrp_1)
        cdef np.ndarray slctp = np.zeros(n_dt_hrm_arg, dtype='int')
        cdef int n_dt_rw_18 = np.size(rw_dt_arg) - 18
        cdef int offst_tmp = 2 * i_hrm_arg + np.int(prty_vct_arg[0])
        slctp[0] = offst_tmp
        slctp[1] = offst_tmp + 1
        cdef int i_tmp = 1
        while (slctp[2*i_tmp-2] < n_dt_rw_18):
            slctp[2 * i_tmp] = 18 * i_tmp + offst_tmp
            slctp[2 * i_tmp + 1] = slctp[2 * i_tmp] + 1
            i_tmp += 1

        cdef double ttl_4k_p = 0
        cdef double ttl_4k_m = 0
        for i_fst in np.arange(n_hlf_fst_smpl):
            ttl_4k_p += rw_4K_arry[i_fst]
            ttl_4k_m += rw_4K_arry[i_fst + n_hlf_fst_smpl]

        shft_p = 0
        shft_m = 1
        sgn_p = 1
        sgn_m = -1
        if prty_vct_arg[slctp[0]] == 1:
            shft_p = 1
            shft_m = 0
            sgn_p = -1
            sgn_m = 1
        for i_p in np.arange(i_tmp):
            if slctp[2 * i_p + shft_p] < np.size(rw_dt_arg):
                rw_dt_arg[slctp[2 * i_p + shft_p]] = nw_rw_dt(
                    corrp_1, corrp_0, rw_dt_arg[slctp[2 * i_p + shft_p]],
                    sgn_p, sgnl_stp, n_pnt_sgnl) - ttl_4k_p
            # else:
            #     print('i_hrm_arg = {} data size = {} i_p = {} shft_p = {}'
            #           ''.format(i_hrm_arg,np.size(rw_dt_arg),i_p,shft_p))

            if slctp[2 * i_p + shft_m] < np.size(rw_dt_arg):
                rw_dt_arg[slctp[2 * i_p + shft_m]] = nw_rw_dt(
                    corrm_1, corrm_0, rw_dt_arg[slctp[2 * i_p + shft_m]],
                    sgn_m, sgnl_stp, n_pnt_sgnl) - ttl_4k_m

        # First element of the TOI (NB: this condition is expected to
        # be the usual case)
        if (i_hrm_arg == (n_hrm_rw - 1)) and (prty_vct_arg[slctp[0]] == 0):
                rw_dt_arg[0] = nw_rw_dt(corrm_1, corrm_0, rw_dt_arg[0],
                                        -1, sgnl_stp, n_pnt_sgnl) - ttl_4k_m

        return rw_dt_arg
