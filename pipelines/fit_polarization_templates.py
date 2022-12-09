# This script fits the polarization template maps against the
# single detector/full frequency difference to find corrections
# to the polarization parameters

import glob
import os
import sys

import astropy.io.fits as pf
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


separate_plots = True

fname_rimo = "/global/cfs/cdirs/cmb/data/planck2020/npipe/aux/RIMO_HFI_npipe5v16_symmetrized.fits"
fname_mask = "/global/cfs/cdirs/cmb/data/planck2020/npipe/aux/destriping_mask_857.fits"

nside = 1024
fwhm = np.radians(1)
lmax = 512


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <reproc output directory>")
    sys.exit()

indir = sys.argv[1]
if not os.path.isdir(indir):
    print(f"No such directory: {indir}")
    sys.exit()


def plug_holes(m, verbose=False, in_place=True, nest=False):
    """
    Use simple downgrading to derive estimates of the missing pixel values

    Especially useful if you mask out point sources but need to fill the
    missing pixels before smoothing.
    """
    nbad_start = np.sum(np.isclose(m, hp.UNSEEN))

    if nbad_start == m.size:
        if verbose:
            print('plug_holes: All map pixels are empty. Cannot plug holes',
                  flush=True)
        return

    if nbad_start == 0:
        return

    nside = hp.get_nside(m)
    npix = m.size
    if nest:
        mnest = m.copy()
    else:
        mnest = hp.reorder(m, r2n=True)

    lowres = mnest
    nside_lowres = nside
    bad = np.isclose(mnest, hp.UNSEEN)
    while np.any(bad) and nside_lowres > 1:
        nside_lowres //= 2
        lowres = hp.ud_grade(lowres, nside_lowres, order_in='NESTED')
        hires = hp.ud_grade(lowres, nside, order_in='NESTED')
        bad = np.isclose(mnest, hp.UNSEEN)
        mnest[bad] = hires[bad]

    nbad_end = np.sum(bad)

    if nbad_end != 0:
        mn = np.mean(mnest[np.logical_not(bad)])
        mnest[bad] = mn

    if not in_place:
        m = m.copy()
    if nest:
        m[:] = mnest
    else:
        m[:] = hp.reorder(mnest, n2r=True)

    if verbose and nbad_start != 0:
        print(
            "Filled {} missing pixels ({:.2f}%), lowest resolution was "
            "Nside={}.".format(nbad_start, (100.0 * nbad_start) // npix, nside_lowres)
        )

    return

rimo = pf.open(fname_rimo, "readonly")

print(f"Reading {fname_mask}...")
full_mask = hp.read_map(fname_mask, nest=True)

# List polarization amplitude templates
polmaps = sorted(glob.glob(f"{indir}/pol/*fits"))
# List polarization angle templates
derivmaps = sorted(glob.glob(f"{indir}/pol_deriv/*fits"))
# List single detector maps
singlemaps = sorted(glob.glob(f"{indir}/bandpass_corrected/*_map*fits"))
# List full frequency maps
fullmaps = sorted(glob.glob(f"{indir}/*_map_???.fits"))
if len(fullmaps) == 0:
    fullmaps = sorted(glob.glob(f"{indir}/*_map.fits"))

ndet = max(len(polmaps), len(derivmaps), len(singlemaps))
print(f"Found {ndet} detectors")

# Make sure the number of found maps makes sense

if len(polmaps) != ndet:
    raise RuntimeError(
        f"Too few polarization maps! Expected {ndet} but found "
        f"{len(polmaps)} : {polmaps}"
    )

if len(derivmaps) != ndet:
    raise RuntimeError(
        f"Too few polarization derivative maps! Expected {ndet} but found "
        f"{len(derivmaps)} : {derivmaps}"
    )

if len(singlemaps) != ndet:
    raise RuntimeError(
        f"Too few single detector maps! Expected {ndet} but found "
        f"{len(singlemaps)} : {singlemaps}"
    )

fname = fullmaps[-1]
print(f"Reading {fname} ...")
freqmap_full = np.atleast_2d(hp.read_map(fname, None, nest=True))
if freqmap_full.shape[0] != 3:
    raise RuntimeError("Frequency map is not polarized.")

dets = set()
all_coeffs = {}

# Downgrade and smooth the full frequency map

sfreqmap_full = hp.ud_grade(freqmap_full[0], nside, order_in="NEST", order_out="RING")
sfreqmap_full = hp.smoothing(sfreqmap_full, fwhm=fwhm, lmax=lmax, iter=0)

# The NPIPE destriping mask has floating point values in the range [0, 1]
# We can generate different masks by thresholding the mask
# For every value of `limit` perform a full fit of the templates and see
# how they depend on the threshhold

for limit in 0.05, 0.10, 0.20, 0.30:
    print(f"Limit = {limit}")

    # Construct a mask for this iteration
    nside_mask = hp.get_nside(full_mask)
    bad = full_mask < limit
    good = hp.ud_grade(full_mask > limit, nside, order_in="NEST", order_out="RING") > 0.9
    ngood = np.sum(good)
    all_coeffs[limit] = {}

    # Mask, fill and smooth the full frequency map
    freqmap = hp.ud_grade(freqmap_full, nside_mask, order_in="NEST", order_out="NEST")
    freqmap[:, bad] = hp.UNSEEN
    plug_holes(freqmap, nest=True)
    freqmap = hp.ud_grade(freqmap, nside, order_in="NEST", order_out="RING")
    sfreqmap = hp.smoothing(freqmap, fwhm=fwhm, lmax=lmax, iter=0)

    if separate_plots:
        nrow, ncol = 2, 2
        fig1 = plt.figure(figsize=[4 * ncol, 4 * nrow])
        fig2 = plt.figure(figsize=[4 * ncol, 4 * nrow])
        fig3 = plt.figure(figsize=[4 * ncol, 4 * nrow])
        fig4 = plt.figure(figsize=[4 * ncol, 4 * nrow])
    else:
        nrow, ncol = 4, 4
        fig1 = plt.figure(figsize=[4 * ncol, 4 * nrow])
        fig2 = fig1
        fig3 = fig1
        fig4 = fig1
    num1 = fig1.number
    num2 = fig2.number
    num3 = fig3.number
    num4 = fig4.number
    iplot = 0

    # Fit each detector separately

    for polmap, derivmap, singlemap in zip(polmaps, derivmaps, singlemaps):
        det = os.path.basename(polmap).split("_")[-2]
        dets.add(det)

        print(f"det = {det}")

        # mask fill and smooth the single detector map and
        # the polarization templates
        
        print(f"Reading {polmap}...")
        pol_full = hp.read_map(polmap, nest=True)
        pol = hp.ud_grade(pol_full, nside_mask, order_in="NEST", order_out="NEST")
        pol[bad] = hp.UNSEEN
        plug_holes(pol, nest=True)
        pol = hp.ud_grade(pol, nside, order_in="NEST", order_out="RING")
        spol = hp.smoothing(pol, fwhm=fwhm, lmax=lmax, iter=0)

        print(f"Reading {derivmap}...")
        deriv_full = hp.read_map(derivmap, nest=True)
        deriv = hp.ud_grade(deriv_full, nside_mask, order_in="NEST", order_out="NEST")
        deriv[bad] = hp.UNSEEN
        plug_holes(deriv, nest=True)
        deriv = hp.ud_grade(deriv, nside, order_in="NEST", order_out="RING")
        sderiv = hp.smoothing(deriv, fwhm=fwhm, lmax=lmax, iter=0)

        print(f"Reading {singlemap}...")
        detmap_full = hp.read_map(singlemap, nest=True)
        detmap = hp.ud_grade(detmap_full, nside_mask, order_in="NEST", order_out="NEST")
        detmap[bad] = hp.UNSEEN
        plug_holes(detmap, nest=True)
        detmap = hp.ud_grade(detmap, nside, order_in="NEST", order_out="RING")
        sdetmap = hp.smoothing(detmap, fwhm=fwhm, lmax=lmax, iter=0)
        sresid = sdetmap - sfreqmap[0]

        # Simple linear regression to get the template amplitudes

        print("Fitting...")
        templates = np.vstack([np.ones(ngood), spol[good], sderiv[good]])
        invcov = np.dot(templates, templates.T)
        cov = np.linalg.inv(invcov)
        proj = np.dot(templates, sresid[good])
        coeff = np.dot(cov, proj)

        all_coeffs[limit][det] = coeff

        # plot maps without the masking

        print("Plotting...")
        detmap_full = hp.ud_grade(detmap_full, nside, order_in="NEST", order_out="RING")
        sdetmap_full = hp.smoothing(detmap_full, fwhm=fwhm, lmax=lmax, iter=0)
        pol_full = hp.ud_grade(pol_full, nside, order_in="NEST", order_out="RING")
        spol_full = hp.smoothing(pol_full, fwhm=fwhm, lmax=lmax, iter=0)
        deriv_full = hp.ud_grade(deriv_full, nside, order_in="NEST", order_out="RING")
        sderiv_full = hp.smoothing(deriv_full, fwhm=fwhm, lmax=lmax, iter=0)
        sresid = sdetmap_full - sfreqmap_full
        cleaned = sresid - coeff[0] - coeff[1] * spol_full - coeff[2] * sderiv_full
        rms1 = np.std(sresid[good])
        rms2 = np.std(cleaned[good])
        amp = 2 * rms1
        iplot += 1
        plt.figure(num1)
        hp.mollview(sresid,                 fig=num1, sub=[nrow, ncol, iplot],
                    title=f"Resid {det} RMS = {rms1:.6f}", min=-amp, max=amp, cmap="bwr")
        if not separate_plots:
            iplot += 1
        plt.figure(num2)
        hp.mollview(coeff[1] * spol_full,   fig=num2, sub=[nrow, ncol, iplot],
                    title=f"Pol fit {det} {coeff[1]:.6f}", min=-amp, max=amp, cmap="bwr")
        if not separate_plots:
            iplot += 1
        plt.figure(num3)
        hp.mollview(coeff[2] * sderiv_full, fig=num3, sub=[nrow, ncol, iplot],
                    title=f"Deriv fit {det} {coeff[2]:.6f}", min=-amp, max=amp, cmap="bwr")
        if not separate_plots:
            iplot += 1
        plt.figure(num4)
        hp.mollview(cleaned,                fig=num4, sub=[nrow, ncol, iplot],
                    title=f"Cleaned {det} RMS = {rms2:.6f}", min=-amp, max=amp, cmap="bwr")

    if separate_plots:
        fname_plot = f"{indir}/full_residuals_{limit}.png"
        print(f"Writing plot to {fname_plot}")
        fig1.savefig(fname_plot)
        fname_plot = f"{indir}/pol_fit_{limit}.png"
        print(f"Writing plot to {fname_plot}")
        fig2.savefig(fname_plot)
        fname_plot = f"{indir}/pol_deriv_fit_{limit}.png"
        print(f"Writing plot to {fname_plot}")
        fig3.savefig(fname_plot)
        fname_plot = f"{indir}/cleaned_residuals_{limit}.png"
        print(f"Writing plot to {fname_plot}")
        fig4.savefig(fname_plot)
    else:
        fname_plot = f"{indir}/residuals_{limit}.png"
        print(f"Writing plot to {fname_plot}")
        fig.savefig(fname_plot)
    sys.exit()

# Tabulate the fit values

for det in sorted(dets):
    idet = np.argwhere(rimo[1].data["detector"] == det).ravel()[0]
    epsilon = rimo[1].data["epsilon"][idet]
    eta = (1 - epsilon) / (1 + epsilon)
    psi_uv = rimo[1].data["psi_uv"][idet]
    psi_pol = rimo[1].data["psi_pol"][idet]
    psi = psi_uv + psi_pol
    print(f"\n{det} : eta = {eta:.3f}, psi_uv = {psi_uv:.2f}, psi_pol = {psi_pol:.2f}\n")

    print("{:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format("limit", "offset x 1e6", "pol", "angle", "new eta", "new psi"))
    for limit, coeffs in all_coeffs.items():
        offset, pol_amp, deriv_amp = coeffs[det]
        new_eta = eta * (1 + pol_amp)
        new_angle = psi + np.degrees(deriv_amp)
        print(f"{limit:8.2f} {offset * 1e6:12.2f} {pol_amp:8.4f} {deriv_amp:8.4f} {new_eta:8.4f} {new_angle:8.2f}")
