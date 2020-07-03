import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.style.use('classic')

hfi_pipe = '/global/cscratch1/sd/keskital/hfi_pipe/'
dipo_hires = hp.read_map(hfi_pipe + 'dipole_nside1024.fits')
cmb_hires = hp.read_map(
    hfi_pipe + 'sky_model/PR2/COM_CMB_IQU-commander_1024_R2.02_full.fits')
cmb_hires += dipo_hires

ver = 'npipe5v50'

# First threshold is for destriping, second for bandpass

threshholds = {
    30:  (1e-4, 1e-3),
    44:  (8e-5, 4e-4),
    70:  (5e-5, 2e-4),
    100: (5e-5, 3e-4),
    143: (5e-5, 3e-4),
    217: (2e-4, 8e-4),
    353: (1e-3, 6e-3),
    545: (6e-3, 4e-2),
    857: (8e-1, 4e0),
}

nrow, ncol = 3, 3
fig1 = plt.figure(figsize=[18, 12])
fig2 = plt.figure(figsize=[18, 12])
fig3 = plt.figure(figsize=[18, 12])

for iplot, (freq, threshhold) in enumerate(threshholds.items()):
    if freq < 70:
        nside = 128
    elif freq < 100:
        nside = 256
    elif freq < 143:
        nside = 512
    elif freq < 353:
        nside = 1024
    else:
        nside = 2048

    print(freq)
    cmb = hp.ud_grade(cmb_hires, nside)
    psmask = hp.ud_grade(
        hp.read_map(hfi_pipe + 'psmask_{:03}.fits'.format(freq)), nside) < .9
    fname = '/global/cscratch1/sd/keskital/npipe_maps/' \
            '{}/{}_{:03}_map.fits'.format(ver, ver, freq)
    if not os.path.isfile(fname):
        fname = '/global/cscratch1/sd/keskital/npipe_maps/' \
                '{}/{}_{:03}-1_map.fits'.format(ver, ver, freq)
    freqmap = hp.ud_grade(hp.read_map(fname), nside)
    freqmap -= cmb
    freqmap = hp.remove_monopole(freqmap, gal_cut=70)
    low, high = threshhold
    mask_low = np.ones(freqmap.size, dtype=np.float32)
    mask_high = np.ones(freqmap.size, dtype=np.float32)
    weighted_low = freqmap > low
    weighted_high = freqmap > high
    mask_low[weighted_low] = (low / freqmap[weighted_low])
    mask_high[weighted_high] = (high / freqmap[weighted_high])
    mask_low[psmask] *= 1e-10
    mask_high[psmask] *= 1e-10
    mask_madam = mask_high > 0.9
    plt.figure(1)
    hp.mollview(mask_low, xsize=1200, title='{}GHz'.format(freq),
                sub=[nrow, ncol, iplot+1])
    fname_out = hfi_pipe + 'destriping_mask_{:03}.fits'.format(freq)
    hp.write_map(
        fname_out, hp.reorder(mask_low, r2n=True), nest=True, coord='G',
        overwrite=True, dtype=np.float32)
    print('Mask written to', fname_out)
    plt.figure(2)
    hp.mollview(mask_high, xsize=1200, title='{}GHz'.format(freq),
                sub=[nrow, ncol, iplot+1])
    fname_out = hfi_pipe + 'bandpass_mask_{:03}.fits'.format(freq)
    hp.write_map(
        fname_out, hp.reorder(mask_high, r2n=True), nest=True, coord='G',
        overwrite=True, dtype=np.float32)
    print('Mask written to', fname_out)
    plt.figure(3)
    hp.mollview(mask_madam, xsize=1200, title='{}GHz'.format(freq),
                sub=[nrow, ncol, iplot+1])
    fname_out = hfi_pipe + 'madam_mask_{:03}.fits'.format(freq)
    hp.write_map(
        fname_out, hp.reorder(mask_madam, r2n=True), nest=True, coord='G',
        overwrite=True, dtype=np.float32)
    print('Mask written to', fname_out)
plt.figure(1)
plt.savefig('destriping_masks.png')
plt.figure(2)
plt.savefig('bandpass_masks.png')
plt.figure(3)
plt.savefig('madam_masks.png')
