import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from spice import ispice
import healpy as hp
from glob import glob

from planck_util import log_bin

freq1 = 100
freq2 = 143
simdir = '/project/projectdirs/planck/data/npipe/npipe6v19_sim'
cachedir = 'smooth_maps'
fwhm_deg = 3
fwhm_arcmin = fwhm_deg * 60
fwhm_rad = np.radians(fwhm_deg)
lmax = 256
nside = 256
nbin = 100

fn_mask = 'clmask_60fsky_nside{:04}.fits'.format(nside)

fn_dipo = '/global/cscratch1/sd/keskital//hfi_pipe/dipole_nside{:04}.fits' \
          ''.format(nside)
print('Reading', fn_dipo)
dipo = hp.read_map(fn_dipo, verbose=False)


def load_map(fn_in, fn_out, nside, fwhm, lmax):
    if os.path.isfile(fn_out):
        print('Loading', fn_out)
        m = hp.read_map(fn_out, None, verbose=False)
    else:
        print('Reading', fn_in)
        m = hp.read_map(fn_in, None, nest=True, verbose=False)
        m = hp.ud_grade(m, nside, order_in='nest', order_out='ring')
        m = hp.smoothing(m, fwhm=fwhm, lmax=lmax, iter=0, verbose=False)
        print('Writing', fn_out)
        hp.write_map(fn_out, m, coord='g')
    return m


def clean_maps(maps, fgmaps, dipo):
    nside = hp.get_nside(dipo)
    npix = 12 * nside ** 2
    good = np.zeros(npix, dtype=np.bool)
    bad = np.ones(npix, dtype=np.bool)
    for fgmap in fgmaps:
        fgi = fgmap[0] - dipo
        fgp = np.sqrt(fgmap[1]**2 + fgmap[2]**2)
        print('Sorting')
        fgi_sorted = np.sort(fgi)
        fgp_sorted = np.sort(fgp)
        good[fgp > fgp_sorted[int(0.70 * npix)]] = True
        bad[fgi > fgi_sorted[int(0.90 * npix)]] = False
    mask = good * bad
    templates = []
    for fgmap in fgmaps:
        templates.append(
            np.hstack([fgmap[1][mask], fgmap[2][mask]]))
    templates = np.vstack(templates)
    invcov = np.dot(templates, templates.T)
    cov = np.linalg.inv(invcov)
    for m in maps:
        target = np.hstack([m[1][mask], m[2][mask]])
        proj = np.dot(templates, target)
        coeff = np.dot(cov, proj)
        print('Regression coefficients are {}'.format(coeff))
        for cc, fgmap in zip(coeff, fgmaps):
            m[1] -= cc * fgmap[1]
            m[2] -= cc * fgmap[2]
        norm = 1 - np.sum(coeff)
        m[1] /= norm
        m[2] /= norm
    return maps


cl_in = []
cl_out = []

lmax_out = 50

for mc in range(100):
    fn_cl_clean = '{}/clcross_{:04}_{:03}x{:03}_cleaned.fits' \
                  ''.format(cachedir, mc, freq1, freq2)
    if not os.path.isfile(fn_cl_clean):
        fn_clean1 = '{}/cleaned_{:04}_{:03}.fits'.format(cachedir, mc, freq1)
        fn_clean2 = '{}/cleaned_{:04}_{:03}.fits'.format(cachedir, mc, freq2)
        clean = []
        if os.path.isfile(fn_clean1) and os.path.isfile(fn_clean2):
            for fn in [fn_clean1, fn_clean2]:
                print('Loading', fn)
                clean.append(hp.read_map(fn, verbose=False))
        else:
            fgmaps = []
            for freq in [30, 353]:
                fn_in = '{}/{:04}/npipe6v19_{:03}_map.fits'.format(simdir, mc, freq)
                fn_out = '{}/smoothed_{:04}_{:03}.fits'.format(cachedir, mc, freq)
                fgmap = load_map(fn_in, fn_out, nside, fwhm_rad, lmax)
                fgmaps.append(fgmap)

            maps = []
            for freq in [freq1, freq2]:
                fn_in = '{}/{:04}/npipe6v19_{:03}_map.fits'.format(simdir, mc, freq)
                fn_out = '{}/smoothed_{:04}_{:03}.fits'.format(cachedir, mc, freq)
                m = load_map(fn_in, fn_out, nside, fwhm_rad, lmax)
                maps.append(m)
            clean = clean_maps(maps, fgmaps, dipo)
            for m, fn in zip(clean, [fn_clean1, fn_clean2]):
                print('Writing')
                hp.write_map(fn, m, coord='G', overwrite=True)
        ispice(fn_clean1, fn_cl_clean, nlmax=lmax,
               beam1=fwhm_arcmin, beam2=fwhm_arcmin,
               mapfile2=fn_clean2, maskfile1=fn_mask, maskfile2=fn_mask,
               polarization='YES', subav='YES', subdipole='YES',
               symmetric_cl='YES')
    print('Loading', fn_cl_clean)
    cl_clean = hp.read_cl(fn_cl_clean)

    fn_cl_cmb = '{}/clcross_{:04}_{:03}x{:03}_cmb.fits' \
                  ''.format(cachedir, mc, freq1, freq2)
    if not os.path.isfile(fn_cl_cmb):
        cmbmaps = []
        for freq in [freq1, freq2]:
            pattern = '{}/{:04}/input/ffp9_cmb_scl_{:03}_alm_mc_{:04}_*.fits' \
                      ''.format(simdir, mc, freq, mc)
            fn_in = glob(pattern)[0]
            fn_out = '{}/smoothed_cmb_{:04}_{:03}.fits'.format(cachedir, mc, freq)
            load_map(fn_in, fn_out, nside, fwhm_rad, lmax)
            cmbmaps.append(fn_out)
        ispice(cmbmaps[0], fn_cl_cmb, nlmax=lmax,
               beam1=fwhm_arcmin, beam2=fwhm_arcmin,
               mapfile2=cmbmaps[1], maskfile1=fn_mask, maskfile2=fn_mask,
               polarization='YES', subav='YES', subdipole='YES',
               symmetric_cl='YES')
    print('Loading', fn_cl_cmb)
    cl_cmb = hp.read_cl(fn_cl_cmb)

    cl_in.append(cl_cmb[1][:lmax_out + 1])
    cl_out.append(cl_clean[1][:lmax_out + 1])

cl_in = np.vstack(cl_in)
cl_out = np.vstack(cl_out)

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

nrow = 3
ncol = 4
plt.figure(figsize=[4*ncol, 3*nrow])
plt.suptitle('{} x {}'.format(freq1, freq2))
nell = nrow * ncol
for ell in range(2, nell + 2):
    plt.subplot(nrow, ncol, ell - 1)
    ax = plt.gca()
    ax.set_title('$\ell$ = {}'.format(ell))

    norm = 1e15

    x, y = cl_in[:, ell] * norm, cl_out[:, ell] * norm

    plt.plot(x, y, '.')

    vmin = min(0, np.amin(x))
    vmax = np.amax(x)
    xx = np.array([vmin, vmax])
    plt.plot(xx, xx, color='k')

    c, cerr = get_corr_and_var(x, y)
    # Remove 10% of the largest input values
    #xsorted = np.sort(x)
    #lim = xsorted[int(x.size*.9)]
    #good = x < lim
    #c2, cerr2 = get_corr_and_var(x[good], y[good])

    if (ell - 2) // ncol == nrow - 1:
        ax.set_xlabel('Input C$_\ell$ x 1e15')
    if (ell - 2) % ncol == 0:
        ax.set_ylabel('Output C$_\ell$ x 1e15')

    plt.plot(xx, c * xx, label='k = {:.3f} $\pm$ {:.3f}'.format(c, cerr))
    #plt.plot(xx, c2 * xx, label='k = {:.3f} $\pm$ {:.3f}'.format(c2, cerr2))
    plt.legend(loc='best')

plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.savefig('polarization_bias_{}_x_{}.png'.format(freq1, freq2))

plt.show()
