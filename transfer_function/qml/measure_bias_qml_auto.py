import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import pickle
from time import time

import healpy as hp
import astropy.io.fits as pf
import xqml
from planck_util import log_bin

import npipeqml

from measure_bias_tools import *


do_plot = True


mapcache = "../maps"
clcache = "../cls"
fwhm_deg = 3
fwhm_arcmin = fwhm_deg * 60
fwhm_rad = np.radians(fwhm_deg)
lmax = 256
nside = 256
nbin = 100
nell = 40
# freqs = 30, 44, 70, 100, 143, 217, 353
#freqs = 70, 100, 143, 217
freqs = 217,
norm = 1e15

nsideqml = 16
lmaxqml = 45

clth = np.genfromtxt(
    "/project/projectdirs/planck/data/ffp10/sky/CMB/inputs/ffp10_lensedCls.dat"
).T.copy()
ell = clth[0]
normth = ell * (ell + 1) / (2 * np.pi) * 1e12
clth = np.vstack([clth[1], clth[2], clth[3], clth[4]]) / normth  # TT, EE, BB, TE

bl = npipeqml.cosbeam(nsideqml)

fsky = 60  # 27, 33, 39, 46, 53, 60, 67, 74, 82, 90

fn_mask = "../clmask_{:02}fsky_nside{:04}.fits".format(fsky, nside)
mask = hp.ud_grade(hp.read_map(fn_mask, verbose=False), nsideqml) > 0.5

# construct QML
ellbins = np.arange(2, lmaxqml + 2)
cross = xqml.xQML(
    mask, ellbins, clth, bell=bl, lmax=lmaxqml, temp=False, polar=True, corr=False
)

last_freq = -1


def set_estimator(cross, freq):
    global last_freq
    if last_freq == freq:
        return
    print("Constructing estimator for {} x cmb".format(freq))
    deltaN = 1e-16
    dN = np.identity(cross.npix * cross.nstokes) * deltaN
    N1 = npipeqml.readcov(
        "/project/projectdirs/planck/data/npipe/npipe6v20/lowres/"
        "npipe6v20_ncm_ns0016_smoothed_{:03}_bin.dat".format(freq1),
        mask,
    )
    N1 += dN

    N2 = N1 * 1e-6 + deltaN

    t1 = time()
    print("Constructing xQML estimator ...", flush=True, end="")
    cross.construct_esti(N1, N2)
    print(" constructed in {:.1f} s".format(time() - t1), flush=True)

    last_freq = freq
    return


factors = {"EE": {}, "BB": {}}
errors = {"EE": {}, "BB": {}}

for freq in freqs:

    cl_in = []
    cl_out = []

    for mc in range(200, 800):
        fn_cl_clean = "{}/{:04}/clqml_{:04}_{:03}_x_cmb_cleaned_{:02}fsky.fits".format(
            clcache, mc, mc, freq, fsky
        )
        fn_cl_cmb = "{}/{:04}/clqml_{:04}_{:03}_x_{:03}_cmb_{:02}fsky.fits".format(
            clcache, mc, mc, freq, freq, fsky
        )
        if not os.path.isfile(fn_cl_clean):
            fn_clean = "{}/{:04}/cleaned_{:04}_{:03}.fits".format(
                mapcache, mc, mc, freq
            )
            if not os.path.isfile(fn_clean):
                print("File not found: " + fn_clean)
                continue
            fn_cmb = "{}/{:04}/smoothed_cmb_{:04}_{:03}.fits".format(mapcache, mc, mc, freq)
            if not os.path.isfile(fn_cmb):
                print("File not found: " + fn_cmb)
                continue
            m = npipeqml.smooth_and_degrade(fn_clean, bl, nsideqml)
            cmb = npipeqml.smooth_and_degrade(fn_cmb, bl, nsideqml)
            if freq == 0:
                set_estimator(cross, 143)
            else:
                set_estimator(cross, freq)
            t1 = time()
            print("Estimating C_ell ...", flush=True, end="")
            ee, bb = cross.get_spectra(m, cmb)
            cl_clean = np.zeros([4, lmaxqml + 1])
            cl_clean[1, 2:] = ee
            cl_clean[2, 2:] = bb
            print(" estimated in {:.1f} s".format(time() - t1), flush=True)
            hp.write_cl(fn_cl_clean, cl_clean)  # EE and BB
        if not os.path.isfile(fn_cl_cmb):
            fn_cmb = "{}/{:04}/smoothed_cmb_{:04}_{:03}.fits".format(mapcache, mc, mc, freq)
            if not os.path.isfile(fn_cmb):
                print("File not found: " + fn_cmb)
                continue
            if freq == 0:
                set_estimator(cross, 143)
            else:
                set_estimator(cross, freq)
            cmb = npipeqml.smooth_and_degrade(fn_cmb, bl, nsideqml)
            ee, bb = cross.get_spectra(cmb, cmb)
            cl_cmb = np.zeros([4, lmaxqml + 1])
            cl_cmb[1, 2:] = ee
            cl_cmb[2, 2:] = bb
            hp.write_cl(fn_cl_cmb, cl_cmb)  # EE and BB

        print("Loading", fn_cl_cmb)
        cl_cmb = hp.read_cl(fn_cl_cmb)  # EE and BB

        print("Loading", fn_cl_clean)
        cl_clean = hp.read_cl(fn_cl_clean)  # EE and BB

        cl_in.append(cl_cmb[1:3])
        cl_out.append(cl_clean[1:3])

    cl_in = np.array(cl_in)
    cl_out = np.array(cl_out)

    nrow = 4
    ncol = 4
    for imode, mode in enumerate(["EE", "BB"]):
        if do_plot:
            plt.figure(figsize=[4 * ncol, 3 * nrow])
            plt.suptitle("{:03}GHz {} fsky = {}%".format(freq, mode, fsky))
        factors[mode][freq] = np.zeros(nell + 2)
        errors[mode][freq] = np.zeros(nell + 2)
        for ell in range(2, nell + 2):
            x = cl_in[:, imode, ell] * norm
            y = cl_out[:, imode, ell] * norm
            c, cerr = get_corr_and_var(x, y, cross=False)
            factors[mode][freq][ell] = c
            errors[mode][freq][ell] = cerr

            if ell - 1 > nrow * ncol or not do_plot:
                continue

            plt.subplot(nrow, ncol, ell - 1)
            ax = plt.gca()
            ax.set_title("$\ell$ = {}".format(ell))

            plt.plot(x, y, ".")

            vmin = min(0, np.amin(x))
            vmax = np.amax(x)
            xx = np.array([vmin, vmax])
            plt.plot(xx, xx, color="k")

            if (ell - 2) // ncol == nrow - 1:
                ax.set_xlabel("Input C$_\ell$ x 1e15")
            if (ell - 2) % ncol == 0:
                ax.set_ylabel("Output C$_\ell$ x 1e15")

            plt.plot(xx, c * xx, label="k = {:.3f} $\pm$ {:.3f}".format(c, cerr))
            plt.legend(loc="best")

        if do_plot:
            plt.subplots_adjust(hspace=0.25, wspace=0.2)
            plt.savefig("{}_bias_{:03}GHz.png".format(mode, freq))
            plt.close()

pickle.dump(
    [factors, errors],
    open("suppression_factors_{:02}fsky_AxB.pck".format(fsky), "wb"),
    protocol=2,
)

# Plot the transfer functions

for imode, mode in enumerate(["EE", "BB"]):
    if do_plot:
        plt.figure(figsize=[18, 12])
        plt.suptitle("{} bias fsky = {} AxB".format(mode, fsky))
    ell = np.arange(nell + 2)
    ifreq = -1
    for freq in freqs:
        if freq in [30, 353]:
            continue
        ifreq += 1
        c = factors[mode][freq]
        err = errors[mode][freq]
        if do_plot:
            plt.errorbar(
                ell[2:] + (ifreq - 2) / 20,
                c[2:],
                err[2:],
                label="{:03}GHz".format(freq),
            )
    if do_plot:
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_ylim([0, 2])
        ax.axhline(1, color="k")
        plt.legend(loc="upper right")
        plt.savefig("{}_bias_{:02}fsky.png".format(mode, fsky))
        plt.show()

# Save the transfer functions

for freq in freqs:
    if freq == 0:
        fname_bl_in = (
            "/project/projectdirs/planck/data/npipe/npipe6v20/"
            "quickpol/"
            "Bl_TEB_npipe6v20_{:03}GHzx{:03}GHz.fits".format(143, 143)
        )
    else:
        fname_bl_in = (
            "/project/projectdirs/planck/data/npipe/npipe6v20/"
            "quickpol/"
            "Bl_TEB_npipe6v20_{:03}GHzx{:03}GHz.fits".format(freq, freq)
        )
    hdulist = pf.open(fname_bl_in, "readonly")
    # Augment header
    hdulist[1].header["biasfsky"] = (fsky, "fsky used to evaluate E mode bias")
    # Make sure all columns are loaded
    hdulist[1].data.field("T")[:] *= 1
    hdulist[1].data.field("E")[:] *= 1
    hdulist[1].data.field("B")[:] *= 1
    for w in [False]:
        if w:
            fname_bl = "Bl_TEB_xQML_npipe6v20_{:03}GHzx{:03}GHz_with_E_tf.fits".format(
                freq, freq
            )
        else:
            fname_bl = "Bl_TEB_xQML_npipe6v20_{:03}GHzx{:03}GHz_only_E_tf.fits".format(
                freq, freq
            )
            # null the original TF
            hdulist[1].data.field("T")[:] = 1
            hdulist[1].data.field("E")[:] = 1
            hdulist[1].data.field("B")[:] = 1
        for imode, mode in enumerate(["E"]):
            c = factors[mode + mode][freq]
            err = errors[mode + mode][freq]
            # Apply the transfer function between ell=2 and ell=10
            n = c.size
            use_c = np.zeros(n, dtype=np.bool)
            use_c[2:11] = True
            # use_c[c > 1] = False
            #if mode == "E":
            #    # use_c[err > .1] = False
            #    use_c[err > err[2]] = False
            #elif mode == "B":
            #    use_c[err > 0.2] = False
            tf = np.ones(n)
            tf[use_c] = c[use_c]
            # Combine the transfer functions
            hdulist[1].data.field(mode)[:n] *= tf
        hdulist.writeto(fname_bl, overwrite=True)
        print("Wrote", fname_bl)
