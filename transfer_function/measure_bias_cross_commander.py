import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import pickle

import healpy as hp
import astropy.io.fits as pf
from spice import ispice
from planck_util import log_bin

from measure_bias_tools import *

do_plot = True

mapcache = "maps"
clcache = "cls"
fwhm_deg = 3
fwhm_arcmin = fwhm_deg * 60
fwhm_rad = np.radians(fwhm_deg)
lmax = 256
nside = 256
nbin = 100
nell = 40
freqs = 30, 44, 70, 100, 143, 217, 353
norm = 1e15
freq = 0
freqref = 143

fsky = 60  # 27, 33, 39, 46, 53, 60, 67, 74, 82, 90

factors = {"EE": {}, "BB": {}, "TE": {}}
errors = {"EE": {}, "BB": {}, "TE": {}}
# simdir = '/project/projectdirs/planck/data/npipe/npipe6v19_sim'
simdir = "/global/cscratch1/sd/keskital/npipe_maps/npipe6v20_sim"
freqpairs = [(freq, freq)]

for freqpair in freqpairs:
    freq1, freq2 = freqpair

    fn_mask = "clmask_{:02}fsky_nside{:04}.fits".format(fsky, nside)

    fn_dipo = "/global/cscratch1/sd/keskital/hfi_pipe/dipole_nside{:04}.fits".format(
        nside
    )
    print("Reading", fn_dipo)
    dipo = hp.read_map(fn_dipo, verbose=False)

    cl_in = []
    cl_out = []

    lmax_out = 50

    for mc in range(500, 600):
        fn_cl_clean = (
            "{}/{:04}/clcross_{:04}_{:03}A_x_{:03}B_cleaned_{:02}fsky.fits"
            "".format(clcache, mc, mc, freq1, freq2, fsky)
        )
        fn_cl_cmb = "{}/{:04}/clcross_{:04}_{:03}_x_{:03}_cmb_{:02}fsky.fits".format(
            clcache, mc, mc, freq1, freq2, fsky
        )
        there = True
        if not os.path.isfile(fn_cl_clean):
            print("File not found: " + fn_cl_clean)
            there = False
        if not os.path.isfile(fn_cl_cmb):
            print("File not found: " + fn_cl_cmb)
            there = False
        if not there:
            fn_clean1 = "{}/{:04}/cleaned_{:04}_{:03}A.fits".format(
                mapcache, mc, mc, freq1
            )
            fn_clean2 = "{}/{:04}/cleaned_{:04}_{:03}B.fits".format(
                mapcache, mc, mc, freq2
            )
            there = True
            for fn_clean in [fn_clean1, fn_clean2]:
                if not os.path.isfile(fn_clean):
                    print("File not found: " + fn_clean)
                    there = False
                    break
            if not there:
                continue
            run_spice(fn_clean1, fn_clean2, fn_cl_clean, lmax, fwhm_arcmin, fn_mask)
        if not os.path.isfile(fn_cl_cmb):
            fn_cmb1 = "{}/{:04}/smoothed_cmb_{:04}_{:03}.fits".format(
                mapcache, mc, mc, freq1
            )
            fn_cmb2 = "{}/{:04}/smoothed_cmb_{:04}_{:03}.fits".format(
                mapcache, mc, mc, freq2
            )
            for freq, fn_cmb in [(freq1, fn_cmb1), (freq2, fn_cmb2)]:
                if not os.path.isfile(fn_cmb):
                    nside_in = 2048
                    wbeam = freq_to_fwhm(freq)
                    fn_in = (
                        "{}/{:04}/input/"
                        "ffp10_cmb_{:03}_alm_mc_{:04}_nside{:04}_quickpol.fits"
                        "".format(simdir, mc, freqref, mc, nside_in, wbeam)
                    )
                    load_map(fn_in, fn_cmb, nside, fwhm_rad, lmax)
            run_spice(fn_cmb1, fn_cmb2, fn_cl_cmb, lmax, fwhm_arcmin, fn_mask)

        print("Loading", fn_cl_cmb)
        cl_cmb = hp.read_cl(fn_cl_cmb)

        print("Loading", fn_cl_clean)
        cl_clean = hp.read_cl(fn_cl_clean)

        cl_in.append(cl_cmb[1:4][: lmax_out + 1])
        cl_out.append(cl_clean[1:4][: lmax_out + 1])

    cl_in = np.array(cl_in)
    cl_out = np.array(cl_out)

    nrow = 4
    ncol = 4
    for imode, mode in enumerate(["EE", "BB", "TE"]):
        if do_plot:
            plt.figure(figsize=[4 * ncol, 3 * nrow])
            plt.suptitle("{:03}Ax{:03}B {} fsky = {}%".format(freq1, freq2, mode, fsky))
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
            plt.savefig("{}_bias_{:03}Ax{:03}B.png".format(mode, freq1, freq2))
            plt.close()

pickle.dump(
    [factors, errors],
    open("suppression_factors_{:02}fsky_AxB.pck".format(fsky), "wb"),
    protocol=2,
)

# Plot the transfer functions

for imode, mode in enumerate(["EE", "BB", "TE"]):
    if do_plot:
        plt.figure(figsize=[18, 12])
        plt.suptitle("{} bias fsky = {} AxB".format(mode, fsky))
    ell = np.arange(nell + 2)
    ifreq = -1
    for freqpair in freqpairs:
        freq1, freq2 = freqpair
        ifreq += 1
        c = factors[mode][freqpair]
        err = errors[mode][freqpair]
        if do_plot:
            plt.errorbar(
                ell[2:] + (ifreq - 2) / 20,
                c[2:],
                err[2:],
                label="{:03}Ax{:03}B".format(freq1, freq2),
            )
    if do_plot:
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_ylim([0, 2])
        ax.axhline(1, color="k")
        plt.legend(loc="upper right")
        plt.savefig("{}_bias_{:02}fsky_AxB.commander.png".format(mode, fsky))
        plt.show()

# Save the transfer functions

for freqpair in freqpairs:
    freq1, freq2 = freqpair
    fname_bl_in = (
        "/project/projectdirs/planck/data/npipe/npipe6v20/"
        "quickpol/"
        "Bl_TEB_npipe6v20_{:03}Ax{:03}B.fits".format(freqref, freqref)
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
            fname_bl = "Bl_TEB_npipe6v20_{:03}Ax{:03}B_with_E_tf.fits".format(
                freq1, freq2
            )
        else:
            fname_bl = "Bl_TEB_npipe6v20_{:03}Ax{:03}B_only_E_tf.fits".format(
                freq1, freq2
            )
            # null the original TF
            hdulist[1].data.field("T")[:] = 1
            hdulist[1].data.field("E")[:] = 1
            hdulist[1].data.field("B")[:] = 1
        for imode, mode in enumerate(["E"]):
            c = factors[mode + mode][freqpair]
            err = errors[mode + mode][freqpair]
            # Apply the transfer function between ell=2 and ell=10
            n = c.size
            use_c = np.zeros(n, dtype=bool)
            use_c[2:11] = True
            # use_c[c > 1] = False
            # if mode == 'E':
            #    #use_c[err > .1] = False
            #    use_c[err > err[2]] = False
            # elif mode == 'B':
            #    use_c[err > .2] = False
            tf = np.ones(n)
            tf[use_c] = c[use_c]
            # Combine the transfer functions
            hdulist[1].data.field(mode)[:n] *= tf
        hdulist.writeto(fname_bl, overwrite=True)
        print("Wrote", fname_bl)
