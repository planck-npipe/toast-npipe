# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script measures the 4-pi convolution factors for relativistic dipole.
"""

from toast_planck.utilities import list_planck, det2freq

from scipy.constants import degree

import astropy.io.fits as pf
import healpy as hp
import numpy as np
import pickle
import os

f = open('npipe_s_factors.csv', 'w')
f.write('fh,polarization,centralFreq,Int000,Int001,S100,S010,'
        'S001,S200,S110,S101,S020,S011,S002\n')

beamdir = '/global/cscratch1/sd/keskital/hfi_pipe/beams'

for det in list_planck('Planck'):
    freq = det2freq(det)
    if freq < 100:
        horn = int(det[3:5])
        arm = det[-1].lower()
    else:
        horn = int(det[4:5])
        arm = det[5:6]
    fname = os.path.join(beamdir, 'total_beam_{}.pck'.format(det))
    det, (mb_nside, mb_pixels, mb_values), (sl_nside, sl_values) = pickle.load(
        open(fname, 'rb'))

    mb_pixarea = hp.nside2pixarea(mb_nside)
    sl_pixarea = hp.nside2pixarea(sl_nside)

    mb_x, mb_y, mb_z = hp.pix2vec(mb_nside, mb_pixels)

    sl_pixels = np.arange(12 * sl_nside ** 2)
    sl_x, sl_y, sl_z = hp.pix2vec(sl_nside, sl_pixels)

    mb_norm = np.sum(mb_values) * mb_pixarea
    sl_norm = np.sum(sl_values) * sl_pixarea
    norm = mb_norm + sl_norm
    efficiency = mb_norm / norm
    norm = 1 / norm
    print('norm = {}'.format(norm))

    s100 = norm * (np.sum(mb_values * mb_x) * mb_pixarea +
                   np.sum(sl_values * sl_x) * sl_pixarea)

    s010 = norm * (np.sum(mb_values * mb_y) * mb_pixarea +
                   np.sum(sl_values * sl_y) * sl_pixarea)

    s001 = norm * (np.sum(mb_values * mb_z) * mb_pixarea +
                   np.sum(sl_values * sl_z) * sl_pixarea)

    s200 = norm * (np.sum(mb_values * mb_x * mb_x) * mb_pixarea +
                   np.sum(sl_values * sl_x * sl_x) * sl_pixarea)

    s110 = norm * (np.sum(mb_values * mb_x * mb_y) * mb_pixarea +
                   np.sum(sl_values * sl_x * sl_y) * sl_pixarea)

    s101 = norm * (np.sum(mb_values * mb_x * mb_z) * mb_pixarea +
                   np.sum(sl_values * sl_x * sl_z) * sl_pixarea)

    s020 = norm * (np.sum(mb_values * mb_y * mb_y) * mb_pixarea +
                   np.sum(sl_values * sl_y * sl_y) * sl_pixarea)

    s011 = norm * (np.sum(mb_values * mb_y * mb_z) * mb_pixarea +
                   np.sum(sl_values * sl_y * sl_z) * sl_pixarea)

    s002 = norm * (np.sum(mb_values * mb_z * mb_z) * mb_pixarea +
                   np.sum(sl_values * sl_z * sl_z) * sl_pixarea)

    sx, sy, sz = s100, s010, s001
    s = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)

    print('{}{} : sx = {}, sy = {}, sz = {}, s = {}, norm = {}'.format(
        horn, arm, sx, sy, sz, s, 1 / norm), flush=True)

    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
        horn, arm.upper(), freq * 1000, s, efficiency, s100, s010, s001, s200, s110,
        s101, s020, s011, s002))

f.close()
