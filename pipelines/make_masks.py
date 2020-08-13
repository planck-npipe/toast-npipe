import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#from spice import ispice
import healpy as hp

fwhm_map_deg = 10.
fwhm_mask_deg = 5.
fwhm_map = np.radians(fwhm_map_deg)
fwhm_mask = np.radians(fwhm_mask_deg)
lmax_mask = 128
lmax = 512
nside = 512
npix = 12 * nside ** 2

fname_dipo = '/global/cscratch1/sd/keskital/hfi_pipe/dipole_nside{:04}.fits' \
    ''.format(nside)
print('Loading', fname_dipo)
dipo = hp.read_map(fname_dipo, verbose=False)

freqs = [30, 353]
imaps = []
pmaps = []
for freq in freqs:
    fname = '/global/cscratch1/sd/keskital/npipe_maps/npipe6v20/' \
            'npipe6v20_{:03}_map.fits'.format(freq)
    print('Loading ', fname)
    fgmap = hp.ud_grade(
        hp.read_map(fname, range(3), verbose=False, nest=True), nside,
        order_in='NEST', order_out='RING')
    print('Smoothing')
    fgmap = hp.smoothing(fgmap, fwhm=fwhm_map, lmax=lmax, iter=0, verbose=False)
    fgi = fgmap[0] - dipo
    fgp = np.sqrt(fgmap[1]**2 + fgmap[2]**2)
    print('Sorting')
    fgi_sorted = np.sort(fgi)
    fgp_sorted = np.sort(fgp)
    imaps.append((fgi, fgi_sorted))
    pmaps.append((fgp, fgp_sorted))

# Tabulate the sky fraction associated with each pixel limit
pixlims = np.arange(0, npix, npix//100)
fskies = np.zeros(pixlims.size)
mask = np.zeros(npix)
for ilim, pixlim in enumerate(pixlims):
    mask[:] = 1
    for imap, imap_sorted in imaps:
        mask[imap > imap_sorted[pixlim]] = False
    for pmap, pmap_sorted in pmaps:
        mask[pmap > pmap_sorted[pixlim]] = False
    mask = hp.smoothing(mask, fwhm=fwhm_mask, lmax=lmax_mask, verbose=False)
    fskies[ilim] = np.sum(mask) / npix
pixlims[0] = 0
fskies[0] = 0
pixlims[-1] = npix - 1
fskies[-1] = 1

#for cut in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
for cut in [55, 65]:
    pixlim = int(np.interp(cut/100, fskies, pixlims))
    mask[:] = 1
    for imap, imap_sorted in imaps:
        mask[imap > imap_sorted[pixlim]] = False
    for pmap, pmap_sorted in pmaps:
        mask[pmap > pmap_sorted[pixlim]] = False
    smask = hp.smoothing(mask, fwhm=fwhm_mask, lmax=lmax_mask, verbose=False)
    smask[mask == 0] = 0
    smask[smask < 0] = 0
    fsky = np.sum(mask) / npix
    print('cut = {}, fsky = {}'.format(cut, fsky))
    header = [
        ('fsky', fsky, 'Effective sky area'),
        ('cut', cut, 'Fractional cut in temperature and polarization'),
        ('fwhmmap', fwhm_map_deg, 'Map smoothing [degrees]'),
        ('fwhmmask', fwhm_mask_deg, 'Mask apodization [degrees]'),
    ]
    for nsideout in [256, 512, 1024, 2048]:
        maskout = hp.ud_grade(smask, nsideout)
        fname = 'clmask_{:02}fsky_nside{:04}.fits'.format(
            int(cut), nsideout)
        hp.write_map(fname, maskout, dtype=np.float32, extra_header=header, overwrite=True)
        print('Mask saved in {}'.format(fname))
