import os
import sys

import numpy as np

import healpy as hp
from spice import ispice

def write_map(fn, m):
    dir_out = os.path.dirname(fn)
    os.makedirs(dir_out, exist_ok=True)
    hp.write_map(fn, m, coord='g', overwrite=True)
    return

def run_spice(fn1, fn2, fn_cl, lmax, fwhm, mask):
    dir_out = os.path.dirname(fn_cl)
    os.makedirs(dir_out, exist_ok=True)
    ispice(
        fn1,
        fn_cl,
        nlmax=lmax,
        beam1=fwhm,
        beam2=fwhm,
        mapfile2=fn2,
        weightfile1=mask,
        weightfile2=mask,
        polarization="YES",
        subav="YES",
        subdipole="YES",
        symmetric_cl="YES",
    )
    return

def get_corr_and_var(x, y, cross=False):
    if len(x) > 1:
        rms = np.std(y - x)
    else:
        rms = 1
    templates = np.array([x])
    invcov = np.dot(templates, templates.T) / rms ** 2
    proj = np.dot(templates, y) / rms ** 2
    cov = np.linalg.inv(invcov)
    c = np.dot(cov, proj)[0]
    err = np.sqrt(cov[0, 0])
    if cross:
        # c should always be positive, unless the error is very large
        sgn = np.sign(c)
        c = np.sqrt(np.abs(c))
        err = 0.5 / c * err
        c *= sgn
    return c, err


"""
def get_corr_and_var(x, y):
    def get_corr(x, y):
        return np.dot(x, y) / np.dot(x, x)
    c0 = get_corr(x, y)
    # Use jackknife resampling to measure variance
    n = x.size
    cn = np.zeros(n)
    good = np.ones(n, dtype=np.bool)
    for i in range(n):
        good[i] = False
        cn[i] = get_corr(x[good], y[good])
        good[i] = True
    cvar = (n - 1) / n * np.sum((cn - c0)** 2)
    return c0, np.sqrt(cvar)
"""


def freq_to_fwhm(freq):
    ifreq = int(freq)
    fwhms = {30: 33, 44: 24, 70: 14,
             100: 10, 143: 7.1, 217: 5.5,
             353: 5, 545: 5, 857: 5}
    if ifreq not in fwhms:
        raise RuntimeError('Unknown frequency: {}'.format(freq))
    return fwhms[ifreq]


def load_map(fn_in, fn_out, nside, fwhm, lmax):
    if os.path.isfile(fn_out):
        print('Reading', fn_out, flush=True)
        m = hp.read_map(fn_out, None, verbose=False)
    else:
        if not os.path.isfile(fn_in):
            print('File not found:', fn_in)
            return None
        print('Reading', fn_in, flush=True)
        m = hp.read_map(fn_in, None, nest=True, verbose=False)
        m = hp.ud_grade(m, nside, order_in='nest', order_out='ring')
        m = hp.smoothing(m, fwhm=fwhm, lmax=lmax, iter=0, verbose=False)
        print('Writing', fn_out)
        write_map(fn_out, m)
    return m


def clean_map(m, fgmaps, dipo, cmb):
    nside = hp.get_nside(dipo)
    npix = 12 * nside ** 2
    good = np.zeros(npix, dtype=np.bool)
    bad = np.ones(npix, dtype=np.bool)
    # Derive a regression mask by excluding pixels with
    #   a) too much foreground intensity
    #   b) too little foreground polarization
    for fgmap in fgmaps:
        fgi = fgmap[0] - dipo
        fgp = np.sqrt(fgmap[1]**2 + fgmap[2]**2)
        print('Sorting')
        fgi_sorted = np.sort(fgi)
        fgp_sorted = np.sort(fgp)
        # Enough FG polarization
        good[fgp > fgp_sorted[int(0.50 * npix)]] = True
        # Too much FG intensity
        bad[fgi > fgi_sorted[int(0.85 * npix)]] = False
    mask = good * bad
    # Fitting templates are the masked QU maps
    templates = []
    fg, fgderiv = fgmaps
    #for fgmap in fgmaps:
    #    templates.append(
    #        np.hstack([fgmap[1][mask], fgmap[2][mask]]))
    templates = [np.hstack([fgderiv[1][mask], fgderiv[2][mask]])]
    templates = np.vstack(templates)
    invcov = np.dot(templates, templates.T)
    cov = np.linalg.inv(invcov)
    target = np.hstack([(m - fg - cmb)[1][mask], (m - fg - cmb)[2][mask]])
    proj = np.dot(templates, target)
    coeff = np.dot(cov, proj)
    print('Regression coefficients are {}'.format(coeff))
    cleaned = m.copy()
    #for cc, fgmap in zip(coeff, fgmaps):
    #    cleaned[1] -= cc * fgmap[1]
    #    cleaned[2] -= cc * fgmap[2]
    cleaned[0] -= fg[0] + dipo
    cleaned[1] -= fg[1] + coeff[0] * fgderiv[1]
    cleaned[2] -= fg[2] + coeff[0] * fgderiv[2]
    return cleaned

"""
def clean_map(m, fgmaps, dipo, cmb):
    nside = hp.get_nside(dipo)
    npix = 12 * nside ** 2
    good = np.zeros(npix, dtype=np.bool)
    bad = np.ones(npix, dtype=np.bool)
    # Derive a regression mask by excluding pixels with
    #   a) too much foreground intensity
    #   b) too little foreground polarization
    for fgmap in fgmaps:
        fgi = fgmap[0] - dipo
        fgp = np.sqrt(fgmap[1]**2 + fgmap[2]**2)
        print('Sorting')
        fgi_sorted = np.sort(fgi)
        fgp_sorted = np.sort(fgp)
        # Enough FG polarization
        good[fgp > fgp_sorted[int(0.70 * npix)]] = True
        # Too much FG intensity
        bad[fgi > fgi_sorted[int(0.90 * npix)]] = False
    mask = good * bad
    # Fitting templates are the masked QU maps
    templates = []
    for fgmap in fgmaps + [cmb]:
        templates.append(
            np.hstack([fgmap[1][mask], fgmap[2][mask]]))
    templates = np.vstack(templates)
    invcov = np.dot(templates, templates.T)
    cov = np.linalg.inv(invcov)
    target = np.hstack([m[1][mask], m[2][mask]])
    proj = np.dot(templates, target)
    coeff = np.dot(cov, proj)
    print('Regression coefficients are {}'.format(coeff))
    cleaned = m.copy()
    for cc, fgmap in zip(coeff, fgmaps):
        cleaned[1] -= cc * fgmap[1]
        cleaned[2] -= cc * fgmap[2]
    return cleaned
"""

