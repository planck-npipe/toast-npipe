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


do_plot = False


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
freqs = 44, 70, 100, 143, 217
norm = 1e15

nsideqml = 16
lmaxqml = 45

freqpairs = []
for ifreq1, freq1 in enumerate(freqs):
    if len(sys.argv) > 1:
        if freq1 != int(sys.argv[1]):
            continue
    for freq2 in freqs:
        if freq1 < freq2:
            freqpairs.append((freq1, freq2))
# freqpairs = [(70, 70), (100, 100), (143, 143), (217, 217)]

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

last_freqs = [0, 0]


def set_estimator(cross, freq1, freq2):
    global last_freqs
    if last_freqs[0] == freq1 and last_freqs[1] == freq2:
        return
    print('Constructing estimator for {} x {}'.format(freq1, freq2))
    deltaN = 1e-16
    dN = np.identity(cross.npix * cross.nstokes) * deltaN
    N1 = npipeqml.readcov(
        "/project/projectdirs/planck/data/npipe/npipe6v20/lowres/"
        "npipe6v20_ncm_ns0016_smoothed_{:03}_bin.dat".format(freq1),
        mask,
    )
    N1 += dN

    N2 = npipeqml.readcov(
        "/project/projectdirs/planck/data/npipe/npipe6v20/lowres/"
        "npipe6v20_ncm_ns0016_smoothed_{:03}_bin.dat".format(freq2),
        mask,
    )
    N2 += dN

    t1 = time()
    print("Constructing xQML estimator ...", flush=True, end="")
    cross.construct_esti(N1, N2)
    print(" constructed in {:.1f} s".format(time() - t1), flush=True)

    last_freqs[0] = freq1
    last_freqs[1] = freq2
    return


factors = {"EE": {}, "BB": {}}
errors = {"EE": {}, "BB": {}}

for freqpair in freqpairs:
    freq1, freq2 = freqpair

    cl_in = []
    cl_out = []

    for mc in range(200, 800):
        fn_cl_clean = "{}/{:04}/clqml_{:04}_{:03}_x_{:03}_cleaned_{:02}fsky.fits".format(
            clcache, mc, mc, freq1, freq2, fsky
        )
        fn_cl_cmb = "{}/{:04}/clqml_{:04}_{:03}_x_{:03}_cmb_{:02}fsky.fits".format(
            clcache, mc, mc, freq1, freq2, fsky
        )
        if not os.path.isfile(fn_cl_clean):
            set_estimator(cross, freq1, freq2)
            fn_clean1 = "{}/{:04}/cleaned_{:04}_{:03}.fits".format(mapcache, mc, mc, freq1)
            fn_clean2 = "{}/{:04}/cleaned_{:04}_{:03}.fits".format(mapcache, mc, mc, freq2)
            there = True
            for fn_clean in [fn_clean1, fn_clean2]:
                if not os.path.isfile(fn_clean):
                    print("File not found: " + fn_clean)
                    there = False
            if not there:
                continue
            m1 = npipeqml.smooth_and_degrade(fn_clean1, bl, nsideqml)
            m2 = npipeqml.smooth_and_degrade(fn_clean2, bl, nsideqml)
            t1 = time()
            print("Estimating C_ell ...", flush=True, end="")
            ee, bb = cross.get_spectra(m1, m2)
            cl_clean = np.zeros([4, lmaxqml + 1])
            cl_clean[1, 2:] = ee
            cl_clean[2, 2:] = bb
            print(" estimated in {:.1f} s".format(time() - t1), flush=True)
            hp.write_cl(fn_cl_clean, cl_clean)  # EE and BB
        if not os.path.isfile(fn_cl_cmb):
            set_estimator(cross, freq1, freq2)
            fn_cmb1 = "{}/{:04}/smoothed_cmb_{:04}_{:03}.fits".format(mapcache, mc, mc, freq1)
            fn_cmb2 = "{}/{:04}/smoothed_cmb_{:04}_{:03}.fits".format(mapcache, mc, mc, freq2)
            m1 = npipeqml.smooth_and_degrade(fn_cmb1, bl, nsideqml)
            m2 = npipeqml.smooth_and_degrade(fn_cmb2, bl, nsideqml)
            ee, bb = cross.get_spectra(m1, m2)
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
            plt.suptitle("{:03}x{:03} {} fsky = {}%".format(freq1, freq2, mode, fsky))
        factors[mode][freqpair] = np.zeros(nell + 2)
        errors[mode][freqpair] = np.zeros(nell + 2)
        for ell in range(2, nell + 2):
            x = cl_in[:, imode, ell] * norm
            y = cl_out[:, imode, ell] * norm
            if True:
                c, cerr = get_corr_and_var(x, y, cross=True)
            else:
                # Remove 10% of the largest input values
                xsorted = np.sort(x)
                lim = xsorted[int(x.size * 0.9)]
                good = x < lim
                c, cerr = get_corr_and_var(x[good], y[good], cross=True)
            factors[mode][freqpair][ell] = c
            errors[mode][freqpair][ell] = cerr

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
            plt.savefig("{}_bias_{:03}x{:03}.png".format(mode, freq1, freq2))
            plt.close()

pickle.dump(
    [factors, errors],
    open("suppression_factors_{:02}fsky.pck".format(fsky), "wb"),
    protocol=2,
)

# Plot the transfer functions

for imode, mode in enumerate(["EE", "BB"]):
    if do_plot:
        plt.figure(figsize=[18, 12])
        plt.suptitle("{} bias fsky = {}".format(mode, fsky))
    ell = np.arange(nell + 2)
    ifreq = -1
    for freqpair in freqpairs:
        freq1, freq2 = freqpair
        if freq1 in [30, 353] or freq2 in [30, 353]:
            continue
        ifreq += 1
        c = factors[mode][freqpair]
        err = errors[mode][freqpair]
        if do_plot:
            plt.errorbar(
                ell[2:] + (ifreq - 2) / 20,
                c[2:],
                err[2:],
                label="{:03}x{:03}".format(freq1, freq2),
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

for freqpair in freqpairs:
    freq1, freq2 = freqpair
    fname_bl_in = (
        "/project/projectdirs/planck/data/npipe/npipe6v20/"
        "quickpol/"
        "Bl_TEB_npipe6v20_{:03}GHzx{:03}GHz.fits".format(freq1, freq2)
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
                freq1, freq2
            )
        else:
            fname_bl = "Bl_TEB_xQML_npipe6v20_{:03}GHzx{:03}GHz_only_E_tf.fits".format(
                freq1, freq2
            )
            # null the original TF
            hdulist[1].data.field("T")[:] = 1
            hdulist[1].data.field("E")[:] = 1
            hdulist[1].data.field("B")[:] = 1
        for imode, mode in enumerate(["E"]):
            c = factors[mode + mode][freqpair]
            err = errors[mode + mode][freqpair]
            n = c.size
            if False:
                # Apply the transfer function between ell=2 and ell=10
                use_c = np.zeros(n, dtype=np.bool)
                use_c[2:11] = True
            else:
                use_c = np.ones(n, dtype=np.bool)
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
