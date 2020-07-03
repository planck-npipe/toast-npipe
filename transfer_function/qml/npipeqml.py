# lib for NPIPE maps
import numpy as np
import healpy as hp


def readcov(filename, mask=None):
    f = open(filename, "rb")
    data = np.fromfile(f, float)
    f.close()

    npix = np.int(np.sqrt(len(data))) / 3
    nside = hp.npix2nside(npix)

    if mask is None:
        mask = np.ones(npix)

    mask = np.array(mask, bool)
    npix = len(mask)
    ip = np.arange(npix)
    ipnest = hp.ring2nest(nside, ip[mask])
    ipok = list(ipnest + npix) + list(ipnest + 2 * npix)

    return data.reshape(npix * 3, npix * 3)[ipok, :][:, ipok]


def cosbeam(nside, l1=1):
    ell1 = l1
    ell2 = 3 * nside
    ell = np.arange(ell1 + 1, ell2 + 1)
    bl = np.ones(ell2)
    bl[ell1:] = (1.0 + np.cos((ell - ell1) * np.pi / (ell2 - ell1))) / 2.0

    return bl


def smooth_and_degrade(filename, bl, nside):
    mymap = hp.read_map(filename, verbose=False, field=(0, 1, 2))
    mymap = hp.smoothing(mymap, beam_window=bl, lmax=3 * nside - 1, verbose=False)
    mymap = hp.ud_grade(mymap, nside_out=nside)
    return mymap


def read_coef(filenames):
    res = []
    for f in filenames:
        res.append(np.loadtxt(f, unpack=True))
    return np.array(res).T


def read_coefilc(filenames):
    res = []
    for f in filenames:
        res.append(
            [complex(s.replace("+-", "-")).real for s in np.loadtxt(f, dtype=str)]
        )
    res = np.array(res).T
    return -res[1:, :] / res[0]
