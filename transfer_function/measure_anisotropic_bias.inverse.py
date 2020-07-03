# This script measures how the smoothed and foreground-cleaned
# maps should be manipulated to recover the input CMB map.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import pickle

from scipy.optimize import least_squares

import healpy as hp
import astropy.io.fits as pf
from planck_util import log_bin

from measure_bias_tools import *

force = False
do_plot = True
almcache = "alms"
mapcache = "maps"
clcache = "cls"
fwhm_deg = 3
fwhm_arcmin = fwhm_deg * 60
fwhm_rad = np.radians(fwhm_deg)
lmax = 256
nside = 256
nbin = 100
nell = 40
gbeam = hp.gauss_beam(fwhm_rad, lmax=lmax, pol=True).T
if len(sys.argv) > 1:
    freqs = []
    for f in sys.argv[1:]:
        freqs.append(int(f))
else:
    freqs = (100, 143)  # 30, 44, 70, 100, 143, 217, 353
norm = 1e15
# npipedir = '/project/projectdirs/planck/data/npipe'
npipedir = "/global/cscratch1/sd/keskital/npipe_maps"
quickpoldir = "/global/cscratch1/sd/keskital/npipe_maps/npipe6v20/quickpol"
almdir = "/global/cscratch1/sd/keskital/toast_hfi/sims/cmb_cache"
ffpver = "ffp10_cmb"
mcstart, mcstop = 300, 800
ver = "npipe6v20"
# mcstart, mcstop = 900, 902  # Test TFMODE

fsky = 60  # 60 # 27, 33, 39, 46, 53, 60, 67, 74, 82, 90
lmin_fit = 2
lmax_fit = 20
nside_fit = 32

fit_t = False
fit_e = True
fit_b = False


def truncate_alm(alm, lmax_in, lmax_out):
    nalm = len(alm)
    sz = hp.Alm.getsize(lmax_out, lmax_out)
    new_alm = np.zeros([nalm, sz], dtype=np.complex)
    for ell in range(lmax_out + 1):
        for m in range(ell + 1):
            i = hp.Alm.getidx(lmax_in, ell, m)
            j = hp.Alm.getidx(lmax_out, ell, m)
            new_alm[:, j] = alm[:, i]
    return new_alm


def get_mask(fsky, nside):
    fn_mask = "clmask_{:02}fsky_nside{:04}.fits".format(fsky, 256)
    mask = hp.read_map(fn_mask) > 0.5
    mask = hp.ud_grade(mask, nside) > 0.5
    return mask


for subset in "", "A", "B":
    # for subset in '',: #, 'A', 'B':
    if subset == "":
        sname = "GHz"
    else:
        sname = subset
    simdir = "{}/{}{}_sim".format(npipedir, ver, subset)
    for freq in freqs:
        if freq < 100:
            nside_in = 1024
        else:
            nside_in = 2048

        #fn_dipo = (
        #    "/global/cscratch1/sd/keskital/hfi_pipe/dipole_nside{:04}.fits"
        #    "".format(nside)
        #)
        #print("Reading", fn_dipo)
        #dipo = hp.read_map(fn_dipo, verbose=False)

        if freq == 0:
            fn_beam_in = os.path.join(
                quickpoldir,
                "Bl_TEB_npipe6v20_{:03}{}x{:03}{}.fits".format(
                    143, sname, 143, sname
                ),
            )
        else:
            fn_beam_in = os.path.join(
                quickpoldir,
                "Bl_TEB_npipe6v20_{:03}{}x{:03}{}.fits".format(
                    freq, sname, freq, sname
                ),
            )
        # Load the Beam
        print("Loading", fn_beam_in)
        bl = hp.read_cl(fn_beam_in)

        maps = []
        alms = []

        for mc in range(mcstart, mcstop):
            print("MC =", mc)
            if freq == 0:
                fn_alm_cmb = os.path.join(
                    almcache,
                    "{:04}".format(mc),
                    "smooth_alm_{:04}_{:03}_cmb.fits".format(mc, 143),
                )
            else:
                fn_alm_cmb = os.path.join(
                    almcache,
                    "{:04}".format(mc),
                    "smooth_alm_{:04}_{:03}_cmb.fits".format(mc, freq),
                )

            if not os.path.isfile(fn_alm_cmb):
                # Create a truncated version of the CMB alm expansion
                # that already includes the beam and additional smoothing
                fn_alm_cmb_in = os.path.join(
                    almdir, "ffp10_cmb_{:03}_alm_mc_{:04}.fits".format(freq, mc)
                )
                print("Reading", fn_alm_cmb_in)
                alm, mmax_file = hp.read_alm(fn_alm_cmb_in, return_mmax=True)
                nalm_file = len(alm)
                lmax_file = hp.Alm.getlmax(nalm_file, mmax_file)
                alm = [alm]
                alm.append(hp.read_alm(fn_alm_cmb_in, 2))
                alm.append(hp.read_alm(fn_alm_cmb_in, 3))
                alm = np.vstack(alm)
                # Truncate the expansion
                alm = truncate_alm(alm, lmax_file, lmax)
                # Apply TEB beam
                for i in range(3):
                    hp.almxfl(
                        alm[i], bl[i][: lmax + 1] * gbeam[i], mmax=lmax, inplace=True
                    )
                print("Writing", fn_alm_cmb)
                dir_out = os.path.dirname(fn_alm_cmb)
                os.makedirs(dir_out, exist_ok=True)
                hp.write_alm(fn_alm_cmb, alm, mmax_in=lmax)

            fn_clean = "{}/{:04}/cleaned_{:04}_{:03}{}.fits".format(
                mapcache, mc, mc, freq, subset
            )
            if not os.path.isfile(fn_clean):
                print(fn_clean, "does not exist")
                continue

            if freq == 0:
                fn_alm_clean = os.path.join(
                    almcache,
                    "{:04}".format(mc),
                    "cleaned_alm_{:04}_{:03}.fits".format(mc, 143),
                )
            else:
                fn_alm_clean = os.path.join(
                    almcache,
                    "{:04}".format(mc),
                    "cleaned_alm_{:04}_{:03}.fits".format(mc, freq),
                )

            if not os.path.isfile(fn_alm_clean):
                print('Reading', fn_clean)
                m = hp.read_map(fn_clean, None, verbose=False)
                alm = hp.map2alm(m, lmax=lmax, pol=True)
                hp.write_alm(fn_alm_clean, alm, mmax_in=lmax)

            # Read the CMB alm, convert and cache the map
            alm = []
            for i in range(1, 4):
                alm.append(hp.read_alm(fn_alm_cmb, i))
            alm = np.array(alm)
            cmb = hp.alm2map(alm, nside, pol=True)
            maps.append(cmb)
            # Read the clean map a_lm
            alm = []
            for i in range(1, 4):
                alm.append(hp.read_alm(fn_alm_clean, i))
            alm = np.array(alm)
            alms.append(alm)

        # Flag the a_lm we want to measure the transfer function for
        nmc = len(alms)
        ncomp, nalm = np.shape(alms[0])
        ind = np.zeros([ncomp, nalm], dtype=np.bool)
        for ell in range(lmin_fit, lmax_fit + 1):
            for m in range(ell + 1):
                ix = hp.Alm.getidx(lmax, ell, m)
                for i, fit in enumerate([fit_t, fit_e, fit_b]):
                    ind[i, ix] = fit

        # Remove the modes that are not fitted from the maps
        indinv = np.logical_not(ind)
        fit_maps = []
        fit_alms = []
        for cmb, alm in zip(maps, alms):
            alm2 = alm.copy()
            alm2[ind] *= 0
            clean = hp.alm2map(alm2, nside, verbose=False)
            fit_maps.append(hp.ud_grade(cmb - clean, nside_fit))
            alm2 = alm.copy()
            alm2[indinv] = 0
            fit_alms.append(truncate_alm(alm2, lmax, lmax_fit))

        # Flag the a_lm we want to measure the transfer function for
        # (truncated version)
        nmc = len(alms)
        ncomp, nalm = np.shape(fit_alms[0])
        ind = np.zeros([ncomp, nalm], dtype=np.bool)
        lparamreal = []
        mparamreal = []
        lparamimag = []
        mparamimag = []
        compparamreal = []
        compparamimag = []
        nreal = 0
        nimag = 0
        iscomplex = []
        for comp, fit in enumerate([fit_t, fit_e, fit_b]):
            if not fit:
                continue
            for ix in range(nalm):
                ell, m = hp.Alm.getlm(lmax_fit, ix)
                if ell < lmin_fit or ell > lmax_fit:
                    continue
                ind[comp, ix] = True
                lparamreal.append(ell)
                mparamreal.append(m)
                compparamreal.append(comp)
                nreal += 1
                if m == 0:
                    iscomplex.append(False)
                else:
                    lparamimag.append(ell)
                    mparamimag.append(m)
                    compparamimag.append(comp)
                    nimag += 1
                    iscomplex.append(True)
        nfit = np.sum([fit_t, fit_e, fit_b])
        iscomplex = np.array(iscomplex)  # .reshape([nfit, -1])
        lparam = np.hstack([lparamreal, lparamimag])
        mparam = np.hstack([mparamreal, mparamimag])
        nparam = nreal + nimag
        print("There are {} TF params".format(nparam))
        x0 = np.hstack([np.ones(nreal), np.zeros(nimag)])

        mask = get_mask(fsky, nside_fit)

        def get_tf(x, full=False):
            global fit_maps, fit_alms, ind, nreal, nimag, iscomplex, fit_t, fit_e, fit_b
            tf = x[:nreal].astype(np.complex)
            tf[iscomplex] += 1j * x[nreal:]
            if full:
                full_tf = np.ones_like(fit_alms[0])
                full_tf[ind] = tf
                tf = full_tf
            return tf

        def get_resid(x):
            global fit_maps, fit_alms, ind, nreal, nimag, iscomplex
            # print('Evaluating resid, x =', x)
            tf = get_tf(x)
            resid = []
            nside = hp.get_nside(fit_maps[0])
            for cmb, alm in zip(fit_maps, fit_alms):
                alm2 = alm.copy()
                alm2[ind] *= tf
                clean = hp.alm2map(alm2, nside, verbose=False)
                for i in [1, 2]:
                    resid.append((cmb[i] - clean[i])[mask])
            resid = np.hstack(resid)
            return resid

        chisq0 = get_resid(x0)
        chisq0 = np.dot(chisq0, chisq0)

        print("Minimizing least squares")
        result = least_squares(
            get_resid,
            x0,
            bounds=(-10, 10),
            method="trf",
            verbose=2,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        if result.success:
            x = result.x
            chisq = get_resid(x)
            chisq = np.dot(chisq, chisq)
            print(
                'Minimization done: "{}". chisq0 = {}, chisq0 - chisq = {}'.format(
                    result.message, chisq0, chisq0 - chisq
                )
            )

        # Error estimates
        jac = result.jac
        hess = np.dot(jac.T, jac)
        invhess = np.linalg.inv(hess)
        sigma = np.std(result.fun)
        err = sigma * np.sqrt(np.diag(invhess))

        iplot = 0
        plt.figure(figsize=[18, 12])
        plt.suptitle("{}{}, fsky = {}%".format(freq, sname, fsky))
        for comp, fit in enumerate([fit_t, fit_e, fit_b]):
            if not fit:
                continue
            iplot += 1
            plt.subplot(1, nfit, iplot)
            lreal = lparam[:nreal].reshape([nfit, -1])[iplot - 1]
            mreal = mparam[:nreal].reshape([nfit, -1])[iplot - 1]
            xreal = x[:nreal].reshape([nfit, -1])[iplot - 1]
            ereal = err[:nreal].reshape([nfit, -1])[iplot - 1]
            plt.errorbar(lreal + mreal * 0.02, xreal, ereal, fmt=".")
            limag = lparam[nreal:].reshape([nfit, -1])[iplot - 1]
            mimag = mparam[nreal:].reshape([nfit, -1])[iplot - 1]
            ximag = x[nreal:].reshape([nfit, -1])[iplot - 1]
            eimag = err[nreal:].reshape([nfit, -1])[iplot - 1]
            plt.errorbar(limag + mimag * 0.02, ximag, eimag, fmt=".")

            ax = plt.gca()
            ax.axhline(0, color="k")
            ax.axhline(1, color="k")
            ax.set_ylim([-1.5, 1.5])
        plt.savefig("anisotropic_inverse_tf_{:03}{}_fsky{:02}.png".format(freq, sname, fsky))

        tf = get_tf(result.x, full=True)
        tferr = get_tf(err, full=True)
        tf = np.vstack([tf, tferr])
        fnout = "anisotropic_inverse_tf_{:03}{}_fsky{:02}.fits".format(freq, sname, fsky)
        if os.path.isfile(fnout):
            os.remove(fnout)
        hp.write_alm(fnout, tf)
        print("Wrote", fnout, flush=True)
