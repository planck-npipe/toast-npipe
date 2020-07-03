import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import pickle

import healpy as hp
import astropy.io.fits as pf
from planck_util import log_bin

from measure_bias_tools import *

force = False
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
if len(sys.argv) > 1:
    freqs = []
    for f in sys.argv[1:]:
        freqs.append(int(f))
else:
    freqs = 100,
norm = 1e15
# npipedir = '/project/projectdirs/planck/data/npipe'
npipedir = "/global/cscratch1/sd/keskital/npipe_maps"
ffpver = "ffp10_cmb"
mcstart, mcstop = 200, 600
ver = "npipe6v20"
#mcstart, mcstop = 900, 901  # Test TFMODE

fsky = 60  # 60 # 27, 33, 39, 46, 53, 60, 67, 74, 82, 90

for fsky in 10, 20, 30, 40, 50, 60, 70, 80, 90:
    subset = ""
    if subset == "":
        sname = "GHz"
    else:
        sname = subset
    factors = {"EE": {}, "BB": {}, "TE": {}}
    errors = {"EE": {}, "BB": {}, "TE": {}}
    simdir = "{}/{}{}_sim".format(npipedir, ver, subset)
    for freq in freqs:
        if freq < 100:
            nside_in = 1024
        else:
            nside_in = 2048

        fn_mask = "clmask_{:02}fsky_nside{:04}.fits".format(fsky, nside)

        fn_dipo = (
            "/global/cscratch1/sd/keskital/hfi_pipe/dipole_nside{:04}.fits"
            "".format(nside)
        )
        print("Reading", fn_dipo)
        dipo = hp.read_map(fn_dipo, verbose=False)

        cl_in = []
        cl_out = []

        lmax_out = 50

        for mc in range(mcstart, mcstop):
            fn_cl_clean = (
                "{}/{:04}/clcross_{:04}_{:03}{}_x_cmb_cleaned_{:02}fsky.fits"
                "".format(clcache, mc, mc, freq, subset, fsky)
            )
            fn_cl_cmb = "{}/{:04}/clcross_{:04}_{:03}{}_cmb_{:02}fsky.fits".format(
                clcache, mc, mc, freq, subset, fsky
            )
            if (
                not os.path.isfile(fn_cl_clean)
                or not os.path.isfile(fn_cl_cmb)
                or force
            ):
                fn_cmb = "{}/{:04}/smoothed_cmb_{:04}_{:03}.fits".format(
                    mapcache, mc, mc, freq
                )
                if not os.path.isfile(fn_cmb) or force:
                    fn_in = (
                        "{}/{:04}/input/"
                        "{}_{:03}_alm_mc_{:04}_nside{:04}_"
                        "quickpol.fits"
                        "".format(simdir, mc, ffpver, freq, mc, nside_in)
                    )
                    cmb = load_map(fn_in, fn_cmb, nside, fwhm_rad, lmax)
                    if cmb is None:
                        continue
                else:
                    print("Reading", fn_cmb, flush=True)
                    cmb = hp.read_map(fn_cmb, None, verbose=False)
                fn_clean = "{}/{:04}/cleaned_{:04}_{:03}{}.fits".format(
                    mapcache, mc, mc, freq, subset
                )
                if not os.path.isfile(fn_clean) or force:
                    fgmaps = []
                    for name in ["sky_model", "sky_model_deriv"]:
                        fn_in = (
                            "{}/skymodel_cache/"
                            "{}_{:03}GHz_nside{}_quickpol"
                            "_cfreq_zodi_smoothed.fits"
                            "".format(simdir, name, freq, nside_in)
                        )
                        fn_out = "{}/smoothed_{}_{:03}_quickpol.fits".format(
                            mapcache, name, freq
                        )
                        fgmap = load_map(fn_in, fn_out, nside, fwhm_rad, lmax)
                        if fgmap is None:
                            # This realization does not yet exist
                            continue
                        fgmaps.append(fgmap)

                    fn_in = "{}/{:04}/{}{}_{:03}_map.fits".format(
                        simdir, mc, ver, subset, freq
                    )
                    fn_out = "{}/{:04}/smoothed_{:04}_{:03}{}.fits".format(
                        mapcache, mc, mc, freq, subset
                    )
                    m = load_map(fn_in, fn_out, nside, fwhm_rad, lmax)
                    cleaned = clean_map(m, fgmaps, dipo, cmb)
                    print("Writing", fn_clean)
                    write_map(fn_clean, cleaned)
                run_spice(fn_cmb, fn_clean, fn_cl_clean, lmax, fwhm_arcmin, fn_mask)
                run_spice(fn_cmb, fn_cmb, fn_cl_cmb, lmax, fwhm_arcmin, fn_mask)

            print("Loading", fn_cl_clean)
            cl_cmb = hp.read_cl(fn_cl_cmb)

            print("Loading", fn_cl_cmb)
            cl_clean = hp.read_cl(fn_cl_clean)

            if mc >= 900:
                # 900-series was run in the special tfmode with
                # very low noise.
                # Scale the C_ell up so that the noise matches the
                # earlier series.
                cl_cmb *= 1e6
                cl_clean *= 1e6

            cl_in.append(cl_cmb[1:4][: lmax_out + 1])
            cl_out.append(cl_clean[1:4][: lmax_out + 1])

        cl_in = np.array(cl_in)
        cl_out = np.array(cl_out)

        nrow = 4
        ncol = 4
        for imode, mode in enumerate(["EE", "BB", "TE"]):
            if do_plot:
                plt.figure(figsize=[4 * ncol, 3 * nrow])
                plt.suptitle("{}{} {} fsky = {}%".format(freq, sname, mode, fsky))
            factors[mode][freq] = np.zeros(nell + 2)
            errors[mode][freq] = np.zeros(nell + 2)
            for ell in range(2, nell + 2):
                x = cl_in[:, imode, ell] * norm
                y = cl_out[:, imode, ell] * norm
                if True:
                    c, cerr = get_corr_and_var(x, y, cross=False)
                else:
                    # Remove 10% of the largest input values
                    xsorted = np.sort(x)
                    lim = xsorted[int(x.size * 0.9)]
                    good = x < lim
                    c, cerr = get_corr_and_var(x[good], y[good], cross=False)
                print("c = {} +- {}".format(c, cerr), flush=True)
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
                plt.savefig("{}_bias_{:03}{}_{:02}fsky.png".format(mode, freq, subset, fsky))
                plt.close()

    pickle.dump(
        [factors, errors],
        open("suppression_factors_{:02}fsky{}.pck".format(fsky, subset), "wb"),
        protocol=2,
    )

    # Plot the transfer functions

    if do_plot:
        for imode, mode in enumerate(["EE", "BB", "TE"]):
            plt.figure(figsize=[18, 12])
            plt.suptitle("{} bias fsky = {} {}".format(mode, fsky, subset))
            ell = np.arange(nell + 2)
            for ifreq, freq in enumerate(freqs[2:-1]):
                c = factors[mode][freq]
                err = errors[mode][freq]
                plt.errorbar(
                    ell[2:] + (ifreq - 2) / 20,
                    c[2:],
                    err[2:],
                    label="{}GHz".format(freq),
                )
            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_ylim([0, 2])
            ax.axhline(1, color="k")
            plt.legend(loc="best")
            plt.savefig("{}_bias_{:02}fsky{}.png".format(mode, fsky, subset))
            # plt.show()

    # Save the transfer functions

    for ifreq, freq in enumerate(freqs):
        fname_bl_in = (
            "/project/projectdirs/planck/data/npipe/npipe6v20{}/"
            "quickpol/"
            "Bl_TEB_npipe6v20_{:03}{}x{:03}{}.fits".format(
                subset, freq, sname, freq, sname
            )
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
                fname_bl = "Bl_TEB_{}_{:03}{}x{:03}{}_with_E_tf_{:02}fsky.fits".format(
                    ver, freq, sname, freq, sname, fsky
                )
            else:
                fname_bl = "Bl_TEB_{}_{:03}{}x{:03}{}_only_E_tf_{:02}fsky.fits".format(
                    ver, freq, sname, freq, sname, fsky
                )
                # null the original TF
                hdulist[1].data.field("T")[:] = 1
                hdulist[1].data.field("E")[:] = 1
                hdulist[1].data.field("B")[:] = 1
            for imode, mode in enumerate(["E"]):
                c = factors[mode + mode][freq]
                err = errors[mode + mode][freq]
                # Apply the transfer function between ell=2 and ell=10
                # and only when it is less than 1.
                n = c.size
                use_c = np.zeros(n, dtype=np.bool)
                use_c[2:11] = True
                # use_c[c > 1] = False
                # if mode == "E":
                #    # use_c[err > .1] = False
                #    use_c[err > err[2]] = False
                # elif mode == "B":
                #    use_c[err > 0.2] = False
                tf = np.ones(n)
                tf[use_c] = c[use_c]
                # Combine the transfer functions
                hdulist[1].data.field(mode)[:n] *= tf
            hdulist.writeto(fname_bl, overwrite=True)
            print("Wrote", fname_bl)
