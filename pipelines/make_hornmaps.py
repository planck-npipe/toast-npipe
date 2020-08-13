#!/usr/bin/env python

import healpy as hp
import numpy as np
from glob import glob
import astropy.io.fits as pf
import os
import sys

if len(sys.argv) == 1:
    indir = '.'
else:
    indir = sys.argv[1]

overwrite = False

fn_noise = '/global/cscratch1/sd/keskital/hfi_pipe/white_noise_levels.txt'
noise = {}
with open(fn_noise) as noisefile:
    for line in noisefile:
        det, sigma = line.split()
        noise[det] = np.float(sigma)

# average LFI

m_maps = sorted(glob(indir + '/*LFI??M_bmap.fits'))
for m_map in m_maps:
    horn_map = m_map.replace('M_bmap.fits', '_bmap.fits')
    #noise_map = m_map.replace('M_bmap.fits', '_wcov.fits')
    #hit_map = m_map.replace('bmap.fits', 'hmap.fits')
    s_map = m_map.replace('M_bmap.fits', 'S_bmap.fits')
    if os.path.isfile(horn_map) and not overwrite:
        print(horn_map, 'exists, skipping')
    else:        
        print('Reading', m_map)
        m = hp.read_map(m_map, verbose=False)
        print('Reading', s_map)
        s = hp.read_map(s_map, verbose=False)
        bad = np.logical_or(m == hp.UNSEEN, s == hp.UNSEEN)
        print('There are {} missing pixels'.format(np.sum(bad)))
        ave = 0.5 * (m + s)
        ave[bad] = hp.UNSEEN
        hp.write_map(horn_map, ave, dtype=np.float32, overwrite=True)
        print('Horn map saved in', horn_map)
    """
    if os.path.isfile(noise_map):
        print(noise_map, 'exists, skipping')
    else:        
        print('Reading', hit_map)
        hits = hp.read_map(hit_map, verbose=False)
        wcov = np.zeros(hits.size, dtype=np.float32) + hp.UNSEEN
        good = hits > 0
        m_sigma, s_sigma = None, None
        for det, sigma in noise.items():
            if det in m_map:
                m_sigma = sigma
            elif det in s_map:
                s_sigma = sigma
        wcov[good] = np.sqrt((.5*m_sigma**2 + .5*s_sigma**2) / hits[good] / 2)
        hp.write_map(noise_map, wcov, dtype=np.float32, overwrite=True)
        print('Noise map saved in', noise_map)
    """

# average HFI

rimo = pf.open('/global/cscratch1/sd/keskital/hfi_pipe/RIMO_HFI_npipe5v16.fits')

a_maps = sorted(glob(indir + '/*-?a_bmap.fits'))
for a_map in a_maps:
    a_det = a_map[-16:-10]
    a_ind = np.argwhere(rimo[1].data.field('detector') == a_det).ravel()[0]
    a_eps = rimo[1].data.field('epsilon')[a_ind]
    a_eta = (1 - a_eps) / (1 + a_eps)    
    horn_map = a_map.replace('a_bmap.fits', '_bmap.fits')
    if os.path.isfile(horn_map) and not overwrite:
        print(horn_map, 'exists, skipping')
        continue
    b_map = a_map.replace('a_bmap.fits', 'b_bmap.fits')
    b_det = b_map[-16:-10]
    b_ind = np.argwhere(rimo[1].data.field('detector') == b_det).ravel()[0]
    b_eps = rimo[1].data.field('epsilon')[b_ind]
    b_eta = (1 - b_eps) / (1 + b_eps)
    a_weight = 1 / a_eta
    b_weight = 1 / b_eta
    norm = 1 / (a_weight + b_weight)
    a_weight *= norm
    b_weight *= norm
    print('Reading', a_map, 'weight =', a_weight)
    a = hp.read_map(a_map, verbose=False)
    print('Reading', b_map, 'weight =', b_weight)
    b = hp.read_map(b_map, verbose=False)
    bad = np.logical_or(a == hp.UNSEEN, b == hp.UNSEEN)
    print('There are {} missing pixels'.format(np.sum(bad)))
    ave = a * a_weight + b * b_weight
    ave[bad] = hp.UNSEEN
    hp.write_map(horn_map, ave, dtype=np.float32, overwrite=True)
    print('Horn map saved in', horn_map)
