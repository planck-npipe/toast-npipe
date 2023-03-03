#! /usr/bin/env python

if True:
    from toast.mpi import MPI
    comm = MPI.COMM_WORLD
    ntask = comm.size
    rank = comm.rank
else:
    comm = None
    ntask = 1
    rank = 0
prefix = '{:04} :'.format(rank)

"""

qprun_npipe.py and qp2fits_npipe.py are adapted from QuickPol. See 
    E. Hivon, S. Mottet and N. Ponthieu,
    "QuickPol: Fast calculation of effective beam matrices for CMB polarization",
    A&A (2017) 598 A25

"""

"""
 misc
     parse_detname
     get_all_masks

 beam stuff
     get_blm_det
     fill_beam_dict
     bmat

 C(l) matrix stuff
     adjoint
     dots
     interpolate_matrix
     deconv_planet

 Hit matrix stuff
     count_pix
     apply_hit_masks
     get_wsmap_det
     invert_hit_sub
     invert_hit
     make_hit_vectors
     make_hit_matrix

 main routines
     product_pre2
     program1

  main:
    +--program1
         +--get_all_masks
         +--qp_file
         +--parse_detname
         +--fill_beam_dict
              +--my_create_bololist_w8
              +--efh.get_angles
              +--get_blm_det
              +--get_gblm_det
         +--make_hit_matrix
              +--proc_angle
              +--my_create_bololist_w8
              +--make_hit_vectors
                  +--my_create_bololist_w8
                  +--count_pix
                  +--get_wsmap_det
                       +--efh.get_angles
                       +--count_pix
                  +--invert_hit
                       +--invert_hit_sub
                  +--apply_hit_masks
                       +--count_pix
         +--product_pre2
             +--adjoint
             +--bmat
             +--dots
             +--interpolate_matrix
         +--print_matrix
"""

import copy
import getopt
import os
import sys
import time

import numpy as np
import numpy.linalg as LA
import scipy.interpolate as inter

import astropy.io.fits as pyfits
import healpy as hp

from toast_planck.utilities import (load_RIMO, list_planck, detector_weights,
                                    qp_file)

auxdir = '/project/projectdirs/planck/data/npipe/aux'
fn_rimo_lfi = os.path.join(auxdir, 'RIMO_LFI_npipe5_symmetrized.fits')
fn_rimo_hfi = os.path.join(auxdir, 'RIMO_HFI_npipe5v16_symmetrized.fits')

hitgrpfull = os.path.join(auxdir, 'polmoments')
blmdir = os.path.join(auxdir, 'beams')
outdir = './quickpol_output'
#blmfile_lfi = blmdir + 'mb_lfi_{}_dx12_smear.alm'
#blmfile_hfi = blmdir + 'BS_HBM_DX11v67_I5_HIGHRES_POLAR_{}_xp_alm.fits'
blmfile = blmdir + '/blm_{}.fits'
blm_ref = 'Dxx'
spin_ref = 'Pxx'
release = 'npipe6v20'

smax = 6
#planet = 'Saturn'
planet = ''
rhobeam = 'IMO'
rhohit = 'IMO'
#test = False
test = True  # Only sample a small fraction of the pixels
conserve_memory = False

NO_COLOR = '\x1b[0m'
RED_COLOR = '\x1b[31;01m'
GREEN_COLOR = '\x1b[32;11m'
YELLOW_COLOR = '\x1b[33;11m'
BLUE_COLOR = '\x1b[34;11m'
MAGENTA_COLOR  ='\x1b[35;11m'
CYAN_COLOR = '\x1b[36;11m'


# ==============================================================================


def print_matrix(matrix, l=None, ctype=''):
    """ print_matrix()
    prints real or complex matrix
    """
    if ctype != '':
        print(prefix, ctype)
    if l is not None:
        print(prefix, ('l=%s'%(str(l))))
    for i in range(3):
        if isinstance(matrix[0,0], np.complex):
            v = ['%+.2e %+.2ei' % (n.real, n.imag) for n in (matrix[i])]
        else:
            v = ['%+.2e' % (n) for n in (matrix[i])]
        vv = ' %s   %s   %s ' % (v[0], v[1], v[2])
        print(prefix, vv)
    print(flush=True)


#-------------------------------------------------------------------------------

def proc_angle(angle):
    """ proc_angle(angle)
    0 -> '000', 90 -> '+90', -90 -> '-90', 180 -> '180'
    """
    st = '%+03d' % (angle)
    st = st.replace('+180', '180')
    st = st.replace('-180', '180')
    st = st.replace('+00', '000')
    return st


def get_angles(RIMO, shorts, ref='Dxx'):
    angles = np.zeros(len(shorts))
    for idet, det in enumerate(shorts):
        if ref == 'Dxx':
            angles[idet] = RIMO[det].psi_uv + RIMO[det].psi_pol
        elif ref == 'Pxx':
            if 'LFI' in det:
                angles[idet] = RIMO[det].psi_pol
            else:
                angles[idet] = 0
        else:
            raise RuntimeError('Unknown referential: {}'.format(ref))
    angles = np.radians(angles)
    return angles


def my_create_bololist_w8(detset, RIMO):
    """ shorts, w8, psb = my_create_bololist_w8(detset, IMO) 
    """
    dets = list_planck(detset)
    ndet = len(dets)
    w8 = np.zeros(ndet, dtype=float)
    psb = np.zeros(ndet, dtype=bool)
    rho = np.zeros(ndet, dtype=float)
    for idet, det in enumerate(dets):
        if det[-1] in 'abMS':
            horn = det[:-1]
            is_psb = True
        else:
            horn = det
            is_psb = False
        weight = detector_weights[horn]
        w8[idet] = weight
        psb[idet] = is_psb
        eps = RIMO[det].epsilon
        rho[idet] = (1 - eps) / (1 + eps)
    return dets, w8, psb, rho


def parse_detname(detname):
    """ parse_detname(detname)
        '100'    -> hitgrpfull, '100' ,''
    """
    short = detname
    hitgrp = hitgrpfull
    return hitgrp, short

# -----------------------------------------------------------

def get_all_masks(release, dets, lmax=None):

    masks_names = [None, None]
    masks = [None, None]
    w_cutsky = np.ones((3, 3, lmax+1), dtype=np.float64)
    masks_means = np.ones((3, 3), dtype=np.float64)

    return masks, masks_names, w_cutsky, masks_means

# -----------------------------------------------------------

def get_blm_det(blmfile, det, lmax=None, mmax=None):
    # ---------
    # read blm
    # ---------

    isbalm = True
    renorm = True
    fitsfile = blmfile.format(det)
    data = pyfits.getdata(fitsfile)
    Tix = data.field(0)
    Tre = data.field(1)
    Tim = data.field(2)
    polbeam_in = False
    ndb = 1

    try:
        dataG = pyfits.getdata(fitsfile, 2)
        Gre = dataG.field(1)
        Gim = dataG.field(2)
        dataC = pyfits.getdata(fitsfile, 3)
        Cre = dataC.field(1)
        Cim = dataC.field(2)
        print(prefix, GREEN_COLOR, 'Read polarized Blm from %s' % (fitsfile),
              NO_COLOR, flush=True)
        polbeam_in = True
        ndb = 3
    except:
        print(prefix, RED_COLOR, 'WARNING: Polarized Blm not found in %s'
              '' % (fitsfile), NO_COLOR, flush=True)

    ls = np.array(np.floor(np.sqrt(Tix - 1)), dtype=int)
    ms = Tix - ls*ls - ls - 1

    if lmax is None:
        lmax = np.max(ls)
    if mmax is None:
        mmax = np.max(ms)

    idxs = (ls <= lmax) * (ms <= mmax)

    ret = np.zeros((lmax + 1, mmax + 1, ndb), dtype=np.complex)
    ret[ls[idxs], ms[idxs], 0] = Tre[idxs] + 1j * Tim[idxs]
    if polbeam_in:
        ret[ls[idxs], ms[idxs], 1] = -((Gre[idxs] - Cim[idxs]) +
                                       1j * (Gim[idxs] + Cre[idxs]))
        ret[ls[idxs], ms[idxs], 2] = -((Gre[idxs] + Cim[idxs]) +
                                       1j * (Gim[idxs] - Cre[idxs]))

    if isbalm == True:
        # file is balm, so renormalize and scale.
        tfacb0 = ret[0, 0, 0]
        print(prefix, fitsfile)
        print(prefix, 'Beam normalisation: ', det, tfacb0*np.sqrt(4*np.pi), ndb,
              flush=True)
        if not renorm:
            print(prefix, 'Beam NOT renormalized', flush=True)
            tfacb0 = 1.
        for l in range(lmax + 1):
            for kb in range(ndb):
                ret[l, :, kb] *= 1 / (tfacb0 * np.sqrt((2*l+1)))

    return ret


def get_gblm_det(RIMO, det, lmax=4000, mmax=0):
    # ---------
    # generate Gaussian blm
    # ---------
    print(prefix, 'In gblm')
    fwhm_am = RIMO[det].fwhm
    fwhm_rad = np.radians(fwhm_am / 60)

    print(prefix, 'Gaussian circular beam: ', det, fwhm_am, fwhm_rad, flush=True)
    ret = np.zeros((lmax + 1, mmax + 1), dtype=np.complex)
    l = np.arange(0, lmax + 1)
    ret[:, 0] = hp.gauss_beam(fwhm_rad, lmax)

    return ret


def fill_beam_dict(RIMO, blmfile, lmax, mmax, detset, blm_ref, angle_sdeg=0.):
    # ----------------------
    # create blm dictionary
    # ----------------------
    short, w8, psb, rho = my_create_bololist_w8(detset, RIMO)
    # angle from input blm (Pxx or Dxx) to hat_blm (Pxx)
    # in Radians (differ by -90 deg from
    #             quickring.orient.get_hfi_det_orient_angles)
    angles = get_angles(RIMO, short, ref=blm_ref)
    angle_srad = np.radians(angle_sdeg)  # overall shift in Radians
    print(prefix, 'detset=', detset)
    print(prefix, BLUE_COLOR, short, w8, angles, NO_COLOR, flush=True)
    nd = len(w8)
    dd = []
    for idet, det in enumerate(short):
        if blmfile != '':
            # get blm
            blm = get_blm_det(blmfile, det, lmax=lmax, mmax=mmax)
        else:
            # Gaussian, IMO based
            blm = get_gblm_det(RIMO, det, lmax=lmax, mmax=mmax)
        # store them in a dictionary
        d1 = {'det': det, 'angle': angles[idet],
              'psb': psb[idet], 'w8': w8[idet], 'lmax': lmax, 'mmax': mmax,
              'blm': blm, 'rho': rho[idet], 'ndb': np.size(blm, 2),
              'angle_shift': angle_srad}
        print(prefix, blmfile, det, w8[idet], psb[idet], rho[idet], flush=True)
        dd.append(d1)

    return dd


def bmat(bdict, l, s, rhobeam=None, verbose=False):
    # ----------------------------------------
    # write beam matrix (E.4) from blm dictionary
    # ----------------------------------------
    mmax = bdict['mmax']
    ndb = bdict['ndb']
    sgn = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])[0:mmax + 1]
    n = mmax
    b = np.zeros((2 * n + 1, ndb), dtype=np.complex)
    # According to qrun_Nersc.py, mphi is identical for temperature
    # and polarization so we collapse the extra dimension
    mphi = np.arange(n + 1, dtype=float)  # m
    mphi *= (bdict['angle'] + bdict['angle_shift'])  # m * (phi + phi0)
    phase = np.cos(mphi) + 1j * np.sin(mphi)  # e^{i m phi}
    blm = bdict['blm'][l, 0:n + 1, 0] * phase  # b_lm e^{i m mphi}
    b[n:2 * n + 1, 0] = blm[0:n + 1]
    b[n-1::-1, 0] = np.conj((blm*sgn)[1:n + 1])
    if ndb == 3:
        blm1 = bdict['blm'][l, 0:n + 1, 1] * phase
        blm2 = bdict['blm'][l, 0:n + 1, 2] * phase
        b[n:2*n + 1, 1] = blm1[0:n + 1]
        b[n:2*n + 1, 2] = blm2[0:n + 1]
        b[n-1::-1, 1] = np.conj((blm2*sgn)[1:n + 1])
        b[n-1::-1, 2] = np.conj((blm1*sgn)[1:n + 1])

    if rhobeam == 'Ideal':
        rho = float(bdict['psb'])  # 1 for PSB, 0 for SWB
    elif rhobeam == 'IMO':
        rho = float(bdict['rho'])
    elif isinstance(rhobeam, (int, float, complex)):
        rho = rhobeam * 1.
    else:
        raise RuntimeError('Unknown rhobeam type: {}'.format(rhobeam))

    rhoc = rho.conjugate()

    if verbose:
        print(prefix, 'det =', bdict['det'], ', l =', l, ', s =', s, ', rho =',
              rho, flush=True)

    if ndb == 1:
        b = b[:, 0]
        bmat = np.array([
            [     b[n+s  ],      b[n+s-2],      b[n+s+2]],
            [rho *b[n+s+2], rho *b[n+s  ], rho *b[n+s+4]],
            [rhoc*b[n+s-2], rhoc*b[n+s-4], rhoc*b[n+s]]])
    elif ndb == 3:
        bmat = np.array([
            [b[n+s, 0], b[n+s-2, 0], b[n+s+2, 0]],
            [b[n+s, 1], b[n+s-2, 1], b[n+s+2, 1]],
            [b[n+s, 2], b[n+s-2, 2], b[n+s+2, 2]]])
    else:
        raise RuntimeError(
            'Invalid value of ndb (polarized blm): {}, expected 1 or 3'
            ''.format(ndb))
    bmat *= bdict['w8']

    return bmat


# ------------------ 3x3 matrix related ----------------------------------------

def adjoint(matrix):
    return np.transpose(np.conjugate(matrix))


def dots(m1, m2, m3=None, m4=None, m5=None, m6=None, m7=None, m8=None, m9=None,
         m10=None):
    m = np.dot(m1, m2)
    if m3 is not None:
        m = np.dot(m, m3)
    if m4 is not None:
        m = np.dot(m, m4)
    if m5 is not None:
        m = np.dot(m, m5)
    if m6 is not None:
        m = np.dot(m, m6)
    if m7 is not None:
        m = np.dot(m, m7)
    if m8 is not None:
        m = np.dot(m, m8)
    if m9 is not None:
        m = np.dot(m, m9)
    if m10 is not None:
        m = np.dot(m, m10)
    return m


def interpolate_matrix(matrix, lmax, lb):
    n1 = np.size(matrix, 1)
    n2 = np.size(matrix, 2)
    kind = 'cubic' # for interp1d
    do_cplx = isinstance(matrix[0, 0, 0], np.complex)
    matout = np.zeros((lmax+1, n1, n2), dtype=type(matrix[0, 0, 0]))
    l = np.arange(lmax+1)
    for i in range(n1):
        for j in range(n2):
            yb = matrix[lb, i, j]
            if do_cplx:
                yb_i = yb.imag
                yb = yb.real
                f_i = inter.interp1d(lb, yb_i, kind=kind)
                #f_i = inter.UnivariateSpline(lb, yb_i, k=3)
                matout[l, i, j] += f_i(l) * 1.j
            f_r = inter.interp1d(lb, yb, kind=kind)
            #f_r = inter.UnivariateSpline(lb, yb, k=3)
            matout[l, i, j] += f_r(l)
    return matout


def deconv_planet(matrix, planet='', w_cutsky=None, masks_means=None):
    lmax = np.size(matrix, 0) - 1
    n1 = np.size(matrix, 1)
    n2 = np.size(matrix, 2)
    factor = 1.17741 / 60 # turns radius in arcsec into FWHM in arcmin
    am2rad = np.radians(1 / 60) # turns arcmin into radians
    if planet == 'Saturn':
        #  9.50''
        #wl_planet = qb.spectra.blm_cl(qb.blm_gauss(9.50*factor, lmax))
        wl_planet = (hp.gauss_beam(9.50*factor*am2rad, lmax=lmax))**2
    else:
        wl_planet = np.ones((lmax+1),dtype=type(1.))
    print(prefix, planet + ' (2000) ', wl_planet[2000].flatten(), flush=True)
    for i in range(n1):
        for j in range(n2):
            matrix[:, i, j] /= wl_planet

    if w_cutsky is not None:
        print(prefix, 'W  cut sky (2000) ', w_cutsky[:, :, 2000].flatten(), flush=True)
        for i in range(n1):
            for j in range(n2):
                matrix[:, i, j] *= w_cutsky[i, j, :]

    if masks_means is not None:
        print(prefix, 'masks means', masks_means.flatten(), flush=True)
        for i in range(n1):
            for j in range(n2):
                matrix[:, i, j] /= masks_means[i, j]

    return matrix

#==============================================================
# ----- Hit matrix related ---------
#==============================================================

def count_pix(pixels):
    if pixels is None or len(pixels) < 2 or len(pixels) > 3:
        raise RuntimeError('Invalid pixels: {}'.format(pixels))
    nhigh = pixels[1] - pixels[0] + 1
    if len(pixels) == 3:
        skip = pixels[2]
    else:
        skip = 1
    sample = (skip > 1)
    nlow = (nhigh + skip - 1) // skip
    
    return nlow, nhigh, skip, sample


def apply_hit_masks(ih, masks=None, pixels=None):
    if masks is not None:
        nlow_, nhigh_, skip, sample_ = count_pix(pixels)
        lm = len(masks)
        if lm == 1:
            mask_t = masks[0][pixels[0]:pixels[1] + 1:skip]
            mask_p = mask_t
        elif lm == 2:
            mask_t = masks[0][pixels[0]:pixels[1] + 1:skip]
            mask_p = masks[1][pixels[0]:pixels[1] + 1:skip]
        else:
            raise RuntimeError('Invalid mask dimension: {}'.format(lm))

        for k in range(3):
            ih[:, 0, k] *= mask_t
            ih[:, 1, k] *= mask_p
            ih[:, 2, k] *= mask_p

    return ih


# ----------------------- reading hit matrix ---------------------------


def get_wsmap_det(RIMO, nside, grp, det, smax, spin_ref, pixels=None,
                  detset=None):
    """  w(s,p,d) = sum_{t in p}  exp(i |s| psi_t_d)
    """
    if detset in ['30A', '30B', '44A', '44B']:
        # These frequencies could not be split by detector
        subset = detset[-1]
        fitsfile1 = os.path.join(
            grp, 'polmoments_{}_hits.{}.fits'.format(det, subset))
        fitsfile2 = os.path.join(
            grp, 'polmoments_{}.{}.fits'.format(det, subset))
    else:
        fitsfile1 = os.path.join(grp, 'polmoments_{}_hits.fits'.format(det))
        fitsfile2 = os.path.join(grp, 'polmoments_{}.fits'.format(det))
    # myangle is the angle between the reference frame used for the
    # spin moments and the detector polarization-sensitive direction.
    # Ideally, if
    #    spin_ref == 'Dxx' then myangle = psi_uv + psi_pol
    #    spin_ref == 'Pxx' then myangle = psi_pol
    if spin_ref == 'Pxx':
        myangle = 0
    else:
        myangle = get_angles(RIMO, [det], ref=spin_ref)[0]
    print(prefix, MAGENTA_COLOR, fitsfile2, det, myangle, NO_COLOR)

    nlow, nhigh, skip, sample = count_pix(pixels)
    print(prefix, 'nlow =', nlow, ', nhigh =', nhigh, ', skip =', skip,
          ', sample =', sample, flush=True)

    wsmp = np.zeros((smax+1, nlow), dtype=np.complex)

    t1 = time.time()
    # Use healpy read_map so map is always in RING ordering
    #hit = pyfits.getdata(fitsfile1).field(0).flatten()
    #spins = pyfits.getdata(fitsfile2)
    hit = hp.read_map(fitsfile1, verbose=False)
    spins = hp.read_map(fitsfile2, None, verbose=False)
    nside_hit = hp.get_nside(hit)
    if nside != nside_hit:
        hit = hp.ud_grade(hit, nside, power=-2)
        spins = hp.ud_grade(spins, nside, power=-2)
    t2 = time.time()
    print(prefix, 'Loaded hits and spin moments in {:.2f}s'.format(t2 - t1),
          flush=True)

    for s in range(smax + 1):
        if s == 0:
            buf = hit
        else:
            #buf = spins.field(2*s - 2).flatten() + \
            #    1.j * spins.field(2*s - 1).flatten()
            buf = spins[2 * s - 2] + 1j * spins[2 * s - 1]
            if myangle != 0:
                buf *= np.cos(s * myangle) + 1j * np.sin(s * myangle)
        buf = buf[pixels[0]:pixels[1] + 1]
        if sample:
            wsmp[s] = buf[::skip]
        else:
            wsmp[s] = buf
    return wsmp


def invert_hit_sub(matrix, thr=1.e-3, polar=True):
    n = np.size(matrix, 0)
    nt = np.size(matrix)
    out = np.zeros((n, 3, 3), dtype=np.complex)
    if nt == 9 * n:
        # input: nx3x3 array
        z2 = matrix[:, 0, 2]
        x = matrix[:, 1, 1].real
        z4 = matrix[:, 1, 2]
    else:
        # input: nx3 array
        x = matrix[:, 0].real
        z2 = matrix[:, 1]
        z4 = matrix[:, 2]

    Rho2 = np.conjugate(z2) * z2
    Rho4 = np.conjugate(z4) * z4
    xm = z2 * np.conjugate(z4)
    t1 = xm - x * np.conjugate(z2)
    t1c = np.conjugate(t1)
    t2 = z2*z2 - z4
    t2c = np.conjugate(t2)
    #det = np.real(1 - Rho4 + 2*((z2 * xm).real - Rho2))
    det = np.real(x**2 - Rho4 + 2*((z2 * xm).real - x * Rho2))

    if polar:
        out[:, 0, 0] = (x ** 2 - Rho4) / det
        out[:, 0, 1] = t1 / det
        out[:, 0, 2] = t1c / det
        out[:, 1, 0] = t1c / det
        out[:, 1, 1] = (x - Rho2) / det
        out[:, 1, 2] = t2 / det
        out[:, 2, 0] = t1 / det
        out[:, 2, 1] = t2c / det
        out[:, 2, 2] = out[:, 1, 1]
        bad = np.where((np.abs(det) < thr) + (np.isnan(det)))
        out[bad, :, :] = 0.
        nbad = np.size(bad)
    else:
        # unpolarized map
        out[:, :, :] = np.identity(3)
        nbad = 0

    return out, nbad


def invert_hit(matrix, thr=None, polar=None):
    n = np.size(matrix, 0)
    nbad = 0
    out = np.zeros((n, 3, 3), dtype=np.complex)
    step = 1024 * 16
    for first in range(0, n, step):
        last = first + step
        if (last > n):
            last = n
        out[first:last], nbad1 = invert_hit_sub(matrix[first:last], thr=thr,
                                                polar=polar)
        nbad += nbad1
    return out, nbad


def make_hit_vectors(
        RIMO, nside, hgrp, detset, smax, pixels, spin_ref, thr=None,
        rhohit=None, masks=None):
    t00 = time.time()
    det, w8, psb, rho_IMO = my_create_bololist_w8(detset, RIMO)
    nd = len(w8)
    # number of sampled pixels
    npq, nhigh_, skip_, sample_ = count_pix(pixels)
    hit = np.zeros(npq, dtype=float)
    hs = np.zeros((npq, 3), dtype=np.complex)
    hv = np.zeros((nd, smax+1, npq), dtype=np.complex)
    # do polarized hit matrix only if at least 3 PSBs
    polar = np.sum(psb) > 2
    # build hit count and spin vectors
    if rhohit == 'Ideal':
        rho = np.array(psb) * 1. # 1  for PSB, 0 for SWB
    elif rhohit == 'IMO':
        rho = rho_IMO
    else:
        raise RuntimeError('Invalid rhohit: %s' % (rhohit))
    rho1w8 = rho * w8  # rho * w8
    rho2w8 = rho * rho1w8  # rho**2 * w8
    t0 = time.time()
    print(prefix, rho, rho1w8, rho2w8, polar)
    print(prefix, '.. prepare:', t0-t00, flush=True)
    for i, d in enumerate(det):
        # hit info for sampled pixels
        hv[i] = get_wsmap_det(RIMO, nside, hgrp, d, smax, spin_ref,
                              pixels=pixels, detset=detset)
        hit += w8[i] * hv[i, 0].real
        if polar:
            hs[:, 0] += rho2w8[i] * hv[i, 0]
            hs[:, 1] += rho1w8[i] * hv[i, 2]
            hs[:, 2] += rho2w8[i] * hv[i, 4]
    if not polar:
        badT = np.where(hit == 0)
    hit = np.maximum(hit, 1.) # put 1 in empty pixels
    t1 = time.time()
    print(prefix, '..     get:', t1-t0, flush=True)
    # inverse hit matrix from non-trivial elements of hit matrix
    for s in [0, 1, 2]:
        hs[:, s] /= hit
    ih, nbad = invert_hit(hs, thr=thr, polar=polar)
    if not polar:
        nbad = np.size(badT)
    del hs
    t2 = time.time()
    print(prefix, '..  invert:', t2-t1, '    nbad:', nbad,
          nbad*1./ npq, '  thr=', thr, polar, flush=True)
    # apply masks
    ih = apply_hit_masks(ih, masks, pixels=pixels)
    t2b = time.time()
    print(prefix, '.. masks:', t2b-t2, flush=True)
    # hit vectors
    for i, d, in enumerate(det):
        for s in range(smax + 1):
            hv[i, s] /= hit
    del hit
    t3 = time.time()
    print(prefix, '..  vector:', t3 - t2b, flush=True)
    # apply inverse hit matrix
    hf = np.zeros((nd, smax+1, 3, npq), dtype=np.complex)
    step = 1024 * 4
    for i, d in enumerate(det):
        for s in range(smax + 1):
            for first in range(0, npq, step):
                last = first + step
                if (last > npq):
                    last = npq
                hs0 = hv[i, s, first:last]
                if s <= smax - 2:
                    hsp2 = hv[i, s+2, first:last]
                else:
                    hsp2 = 0  # bug correction 2016-06-16
                if s <= smax-4:
                    hsp4 = hv[i, s+4, first:last]
                else:
                    hsp4 = 0  # bug correction 2016-06-16
                if s >= 2:
                    hsm2 = hv[i, s-2, first:last]
                else:
                    hsm2 = np.conjugate(hv[i, np.abs(s-2), first:last])
                if s >= 4:
                    hsm4 = hv[i, s-4, first:last]
                else:
                    hsm4 = np.conjugate(hv[i, np.abs(s-4), first:last])
                #  \tilde\omega_s[0]
                hf[i, s, 0, first:last] = \
                    ih[first:last, 0, 0] * hs0  + \
                    rho[i]*(ih[first:last, 0, 1] * hsp2 +
                            ih[first:last, 0, 2] * hsm2)

                #  \rho\tilde\omega_s[2]
                hf[i, s, 1, first:last] = \
                    ih[first:last, 1, 0] * hsm2 + \
                    rho[i] * (ih[first:last, 1, 1] * hs0 +
                              ih[first:last, 1, 2] * hsm4)
                #  \rho\tilde\omega_s[-2]
                hf[i,s,2,first:last] = \
                    ih[first:last, 2, 0] * hsp2 + \
                    rho[i] * (ih[first:last, 2, 1] * hsp4 +
                              ih[first:last, 2, 2] * hs0)

    df = np.zeros((nd, 3, 2, npq), dtype=np.complex)
    d2 = np.zeros((nd, 3, npq), dtype=np.complex)
    t4 = time.time()
    print(prefix, '..Inv*vect:', t4-t3, flush=True)
    del hv
    #
    return hf, df, d2, nbad


def make_hit_matrix(
        RIMO, nside, hgrp1, detset1, hgrp2, detset2, smax, spin_ref,
        test=False, thr=None, angle_shift=0., savefile='', force_det=None,
        release=None, rhohit=None, masks=[None, None], conserve_memory=True):
    success = False
    if savefile != '':
        sf2 = savefile
        sc = '_A' + proc_angle(angle_shift) + '_'
        s0 = sc.replace(proc_angle(angle_shift), proc_angle(0))
        sf2 = sf2.replace(sc, s0)
        sf2 = sf2.replace('_cmbfast_','_old_')
        #sf2 = sf2.replace('_cmbfast_', '_cmbfast_PRELIM_')
        if force_det is not None:
            sf2 = sf2.replace('_FD%s'%(force_det), '')
        print(prefix, 'Try reading hit matrix from %s ...'%(sf2), flush=True)
        try:
            oldjunk = np.load(sf2)
            hitmat = oldjunk['hit_mat']
            varmat = oldjunk['var_matrix']
            v2mat = oldjunk['v2_matrix']
            skip = oldjunk['skip']
            nbad_1 = oldjunk['nbad1']
            nbad_2 = oldjunk['nbad2']
            print(prefix, "     ... Success!", flush=True)
            success = True
        except:
            print(prefix, "     ... Failure!", flush=True)
            success = False
    if not success:
        print(prefix, hgrp1, detset1, smax)
        print(prefix, hgrp2, detset2, smax, flush=True)
        det1, w1, psb1, rho1 = my_create_bololist_w8(detset1, RIMO)
        nd1 = len(w1)
        det2, w2, psb2, rho2 = my_create_bololist_w8(detset2, RIMO)
        nd2 = len(w2)
        nd = max(nd1, nd2)
        npix = 12 * nside ** 2
        # nq = 12*2 if (nd ==4) else 12*8
        if conserve_memory:
            if nd <= 4:
                nq = 12
            else:
                nq = 12 * 2
        else:
            nq = 1

        npt = 0
        hitmat = np.zeros((nd1, nd2, smax + 1, 3, 3), dtype=np.complex)
        varmat = np.zeros((3, 3, 2, 2), dtype=np.complex)
        v2mat = np.zeros((3), dtype=np.complex)
        v2mean = np.zeros((3, 4), dtype=np.complex)
        print(prefix, 'hit matrix: ', np.min(hitmat.real), np.max(hitmat.real),
              flush=True)
        #qmax = 2 if (test) else nq
        qmax = nq
        if test:
            skip = 64
        else:
            skip = 1
        npq = npix // nq
        nbad_1 = 0
        nbad_2 = 0
        for iq in range(qmax):
            t0 = time.time()
            npt += npq
            pixels = [iq*npq, (iq+1)*npq-1, skip]
            print(prefix, BLUE_COLOR, pixels, npt, str(100*npt/npix) + ' %',
                  NO_COLOR)
            print(prefix, detset1, hgrp1, flush=True)
            hv1, hd1, h21, nbad1 = make_hit_vectors(
                RIMO, nside, hgrp1, detset1, smax, pixels, spin_ref,
                thr=thr, rhohit=rhohit, masks=masks[0])
            nbad_1 += nbad1
            #if (detset2 != detset1): # corrected 2016-01-18
            if (detset2 != detset1 or hgrp1 != hgrp2):
                print(prefix, detset2, hgrp2, flush=True)
                hv2, hd2, h22, nbad2 = make_hit_vectors(
                    RIMO, nside, hgrp2, detset2, smax, pixels, spin_ref,
                    thr=thr, rhohit=rhohit, masks=masks[1])
            else:
                print(prefix, RED_COLOR, 'skip calculations for ', detset2, hgrp2,
                      NO_COLOR, flush=True)
                hv2 = hv1
                hd2 = hd1
                h22 = h21
                nbad2 = nbad1
            nbad_2 += nbad2
            t1 = time.time()
            print(prefix, '. generation: {:.2f} s'.format(t1 - t0), flush=True)
            step = 1024 * 4
            # make_hit_vectors returns some empty arrays
            do_varmat = np.any(hd2 != 0) and np.any(hd1 != 0)
            do_v2mat = ((np.any(h22 != 0) and np.any(hv1 != 0)) or
                        (np.any(hv2 != 0) and np.any(h21 != 0)))
            for first in range(0, npq, step):
                last = min(first + step, npq)
                for i1 in range(nd1):
                    for i2 in range(nd2):
                        for s in range(smax+1):
                            for u1 in range(3):
                                for u2 in range(3):
                                    # = sum_p h1 . conj(h2)
                                    hitmat[i1, i2, s, u1, u2] += np.vdot(
                                        hv2[i2, s, u2, first:last],
                                        hv1[i1, s, u1, first:last])

                        # apply weights w1 and w2 to subpixel
                        # contributions to mimic proper scalar
                        # product with w*b
                        if do_varmat:
                            for k1_1 in range(3):
                                for k1_2 in range(3):
                                    for k2_1 in range(2):
                                        for k2_2 in range(2):
                                            varmat[k1_1, k1_2, k2_1, k2_2] += \
                                                np.vdot(
                                                    hd2[i2, k1_2, k2_2, first:last],
                                                    hd1[i1, k1_1, k2_1, first:last]
                                                ) * (w2[i2]*w1[i1])
                        if do_v2mat:
                            for k,v in enumerate([0, 2, -2]):
                                v2mat[k] += (
                                    np.vdot(h22[i2, k, first:last],
                                            hv1[i1, 0, v, first:last]) +
                                    np.vdot(hv2[i2, 0, v, first:last],
                                            h21[i1, k, first:last])
                                ) * (w2[i2] * w1[i1])
                for k, v in enumerate([0, 2, -2]):
                    for i1 in range(nd1):
                        v2mean[k, 0] += np.sum(hv1[i1, 0, v, first:last]) * w1[i1]
                        v2mean[k, 1] += np.sum(h21[i1, k, first:last]) * w1[i1]
                    for i2 in range(nd2):
                        v2mean[k, 2] += np.sum(hv2[i2, 0, v,first:last]) * w2[i2]
                        v2mean[k, 3] += np.sum(h22[i2, k, first:last]) * w2[i2]
            t2 = time.time()
            print(prefix, '. compression: {:.2f} s, npq = {}, step = {}'
                  ''.format(t2-t1, npq, step), flush=True)
        #------
        hitmat /= (npt / skip) # divide hit matrix by number of (sampled) pixels
        varmat /= (npt / skip) # divide var matrix by number of (sampled) pixels,
        v2mean /= (npt / skip)
        v2mat /= (npt / skip) # divide var matrix by number of (sampled) pixels,
        for k in range(3):
            v2mat[k] -= np.conjugate(v2mean[k, 3]) * v2mean[k, 0] + \
                        np.conjugate(v2mean[k, 2]) * v2mean[k, 1]
        varmat *= (4 * np.pi / npix)  # multiply var matrix by pixel area
        v2mat *= (4 * np.pi / npix)  # multiply var matrix by pixel area

    return hitmat, varmat, v2mat, nbad_1, nbad_2, skip

# ------------------------------------------------------------------------------

def product_pre2(
        bdict1, bdict2, hit_matrix, lmax, smax, intypes, lstep=1, planet='',
        pconv='', rhobeam=None, w_cutsky=None, masks_means=None):

    # constant matrices
    diag = np.array([
        [1, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5]])
    idiag = np.array([
        [1, 0, 0],
        [0, 2.0, 0],
        [0, 0, 2.0]])
    if pconv == 'cmbfast':
        # to be consistent with CMBFAST and Healpix on TE and TB
        rot = np.array([
            [1, 0, 0],
            [0, -1, -1j],
            [0, -1, 1j]])
    elif pconv == 'old':
        rot = np.array([
            [1, 0, 0],
            [0, 1, 1j],
            [0, 1, -1j]])
    arot = adjoint(rot)
    irot = LA.inv(rot)
    airot = adjoint(irot)
    swap12 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]])

    # input C(l) selector
    mcl_in = np.array([
        ['TT', 'TE', 'TB'],
        ['TE', 'EE', 'EB'],
        ['TB', 'EB', 'BB']])
    nc = len(intypes)
    cpp = np.zeros((nc, 3, 3), dtype=np.complex)
    for ic, intype in enumerate(intypes):
        mcl = (mcl_in == intype)*1.
        cpp[ic] = dots(rot, mcl, arot)

    cout = np.zeros((nc, lmax+1, 3, 3),dtype=np.complex)
    lb = np.arange(0, lmax+1, lstep)
    lb[-1] = lmax  # Cannot extrapolate
    nd1 = np.size(bdict1)
    nd2 = np.size(bdict2)
    time_bmat = 0.
    time_other = 0.
    my_count = 0
    time_00 = time.time()
    # precompute
    nl = np.size(lb)
    bm1_full = np.zeros((2 * smax + 1, nd1, nl, 3, 3), dtype=np.complex)
    bm2_full = np.zeros((2 * smax + 1, nd2, nl, 3, 3), dtype=np.complex)
    for i1 in range(nd1):
        for s in np.arange(-smax, smax + 1):
            for l in lb:
                bm1_full[s + smax, i1, l // lstep] = bmat(
                    bdict1[i1], l, s, rhobeam=rhobeam,
                    verbose=(l + abs(s) < 1))
    for i2 in range(nd2):
        for s in np.arange(-smax, smax + 1):
            for l in lb:
                bm2_full[s + smax, i2, l // lstep] = bmat(
                    bdict2[i2], l, s, rhobeam=rhobeam,
                    verbose=(l + abs(s) < 1))
    time_0p = time.time()
    time_pre = time_0p - time_00
    for s in np.arange(-smax, smax + 1):
        for i1 in range(nd1):
            for i2 in range(nd2):
                hmat = hit_matrix[i1, i2, np.abs(s)]
                if (s < 0):
                    hmat = dots(swap12, hmat.conjugate(), swap12)

                for l in lb:
                    # beam related information
                    time_01 = time.time()
                    bm1 = bm1_full[s + smax, i1, l // lstep]
                    bm2 = bm2_full[s + smax, i2, l // lstep]
                    abm1 = adjoint(bm1)
                    time_02 = time.time()
                    time_bmat += time_02 - time_01

                    # product  B.C.B
                    m1 = dots(idiag, abm1,  diag)
                    m2 = dots(diag, bm2, idiag)
                    for ic in range(nc):
                        m = dots(m1, cpp[ic], m2)

                        # multiply by hit matrix
                        mh = m * hmat

                        # spin to stokes
                        mf = dots(irot, mh, airot)

                        # store
                        cout[ic, l] += mf

                    time_other += time.time() - time_02
                    my_count += 1
    time_final = time.time()
    time_tot = time_pre + time_bmat + time_other
    print(prefix, 'Product_pre2 routine')
    print(prefix, '   Total:  {:.2f} s. count = {}'.format(
        time_final - time_00, my_count))
    for name, elapsed in [('Pre', time_pre), ('Bmat', time_bmat),
                          ('Other', time_other), ('Total', time_tot)]:
        print(prefix, '   {}: {:.2f} s'.format(name, elapsed))
    print(prefix, '   Frac: {:.2f} s'.format(time_tot / (time_final - time_00)),
          flush=True)

    if lstep > 1:
        for ic in range(nc):
            cout[ic] = interpolate_matrix(cout[ic], lmax, lb)
        #cout[ic] = deconv_planet(cout[ic], planet=planet,
        #    w_cutsky=w_cutsky, masks_means=masks_means)

    return cout


# ==============================================================================


def detset2nside(detset):
    if detset.startswith('0') or detset.startswith('LFI'):
        nside = 1024
    else:
        nside = 2048
    return nside


def program1(
        RIMO, mytype, blmfile, outdir, smax, spin_ref, blm_ref,
        angle_shift=0, force_det=None, release=None, rhobeam=None,
        rhohit=None, test=False, planet='Saturn', conserve_memory=True,
        overwrite=False):

    lstep = 10
    mmax = 10
    #thr = 1.e-3  # initial value
    #thr = 1.e-2 # test
    thr = 3.e-3
    pconv = 'cmbfast'

    #    angle_shift = 90 # 0, +90, -90, 180

    if isinstance(mytype, str):
        raise RuntimeError('mytype cannot be a string: "{}"'.format(mytype))
    else:
        ldets = mytype
        if isinstance(ldets[0], str):
            ldets = [ldets]

    print(prefix, 'rhobeam, rhohit: ', rhobeam, rhohit)
    print(prefix, 'detectors: ', ldets, flush=True)

    for ds in ldets:
        detset1 = ds[0]
        detset2 = ds[1]

        nside = min(detset2nside(detset1), detset2nside(detset2))
        lmax = 4 * nside

        t0 = time.time()
        masks, masks_names, w_cutsky, masks_means = get_all_masks(release, ds,
                                                                  lmax=lmax)
        savefile = qp_file(
            outdir, ds, lmax=lmax, smax=smax, angle_shift=angle_shift,
            full=(not test), pconv=pconv, force_det=force_det, release=release,
            rhobeam=rhobeam, rhohit=rhohit)

        if os.path.isfile(savefile) and not overwrite:
            print(prefix, '{} exists, skipping...'.format(savefile))
            continue

        dirname = os.path.dirname(savefile)
        if not os.path.exists(dirname):
            print(prefix, 'Creating %s' % (dirname))
            os.makedirs(dirname)
        print(prefix, 'Writing into %s' % (savefile), flush=True)

        # find out relevant scanning information
        hitgrp1, ds1 = parse_detname(detset1)
        hitgrp2, ds2 = parse_detname(detset2)

        bdict1 = fill_beam_dict(
            RIMO, blmfile, lmax, mmax, ds1, blm_ref, angle_sdeg=angle_shift)
        if ds2 != ds1:
            bdict2 = fill_beam_dict(
                RIMO, blmfile, lmax, mmax, ds2, blm_ref, angle_sdeg=angle_shift)
        else:
            bdict2 = copy.deepcopy(bdict1)

        t1 = time.time()
        print(prefix, 'prelim:      {:.2f} s'.format(t1-t0), flush=True)

        # get hit_matrix
        hit_matrix, var_matrix, v2_matrix, nbad1, nbad2, skip = make_hit_matrix(
            RIMO, nside, hitgrp1, ds1, hitgrp2, ds2, smax, spin_ref,
            test=test, thr=thr, angle_shift=angle_shift, savefile=savefile,
            force_det=force_det, release=release, rhohit=rhohit, masks=masks,
            conserve_memory=conserve_memory)

        t2 = time.time()
        print(prefix, 'hit matrix: ', t2-t1, ', bad pixels: ',nbad1, nbad2, skip,
              flush=True)

        print(prefix, 'sub pixel 1 Var matrix')
        # D_00++ + D_00--
        print(prefix, var_matrix[0, 0, 0, 0] + var_matrix[0, 0, 1, 1])
        # D_22++ + D_22--
        print(prefix, var_matrix[1, 1, 0, 0] + var_matrix[1, 1, 1, 1])
        # D_-2-2++ + D_-2-2--
        print(prefix, var_matrix[2, 2, 0, 0] + var_matrix[2, 2, 1, 1])

        print(prefix, 'sub pixel 2 Var matrix')
        print(prefix, v2_matrix[0]) #,0],v2_matrix[0,1],v2_matrix[0,2]
        print(prefix, v2_matrix[1]) #,0],v2_matrix[1,1],v2_matrix[1,2]
        print(prefix, v2_matrix[2]) #,0],v2_matrix[2,1],v2_matrix[2,2]

        lshow = 1000
        print(prefix, lmax, smax, lstep, lshow, flush=True)

        beam_mat = dict()
        #ctypes_in = ['TT']
        ctypes_in = ['TT', 'TE', 'EE', 'BB', 'TB', 'EB' ]
        wxx = product_pre2(
            bdict1, bdict2, hit_matrix, lmax, smax, ctypes_in,
            lstep=lstep, planet=planet,pconv=pconv, rhobeam=rhobeam,
            w_cutsky=w_cutsky, masks_means=masks_means)
        for ic, ctype_in in enumerate(ctypes_in):
            beam_mat[ctype_in] = wxx[ic].real
            print_matrix(wxx[ic, 1], l=1, ctype=ctype_in)
            print_matrix(wxx[ic, lshow], l=lshow, ctype=ctype_in)

        np.savez(
            savefile,
            IMO=RIMO, blmfile=blmfile,
            hitgrp1=hitgrp1, hitgrp2=hitgrp2, lmax=lmax, mmax=mmax, lstep=lstep,
            detset1=detset1, detset2=detset2, smax=smax, test=test,
            planet=planet, ctypes=ctypes_in, beam_mat=beam_mat,
            hit_mat=hit_matrix, thr=thr, angle_shift=angle_shift, pconv=pconv,
            release=release, rhobeam=rhobeam, rhohit=rhohit,
            masks_names=masks_names, nbad1=nbad1,
            nbad2=nbad2, blm_ref=blm_ref, skip=skip, var_matrix=var_matrix,
            v2_matrix=v2_matrix, ndb1=bdict1[0]['ndb'], ndb2=bdict2[0]['ndb'])

        print(prefix, 'Results saved in {}'.format(savefile))

        tend = time.time()
        print(prefix, 'computations: {:.2f}'.format(tend-t2))
        print(prefix, 'TOTAL: {:.2f} s'.format(tend-t0), flush=True)


#==============================================================================
#==============================================================================
#==============================================================================


if __name__ == '__main__':
    routine = 'qprun.py'

    usage = 'usage: ' + routine + '  [arguments] '

    #opts, args = getopt.getopt(sys.argv[1:], '')
    #if len(args) == 0:
    #    args = ['']
    #if len(args) >= 2:
    #    period = int(args[1])
    #else:
    #    period = 6
    #print(args[0], period)

    # Load and merge LFI and HFI RIMOs
    print(prefix, 'Loading RIMO from', fn_rimo_lfi, flush=True)
    LFIRIMO = load_RIMO(fn_rimo_lfi)
    print(prefix, 'Loading RIMO from', fn_rimo_hfi, flush=True)
    HFIRIMO = load_RIMO(fn_rimo_hfi)
    RIMO = {**LFIRIMO, **HFIRIMO}

    freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]

    detsets = []
    for suffix in ['GHz', 'A', 'B']:
        for freq in freqs:
            detset = '{:03}{}'.format(freq, suffix)
            detsets.append(detset)

    detsetpairs = []

    # Full frequency and detector set auto and cross spectra

    for idetset1, detset1 in enumerate(detsets):
        for idetset2, detset2 in enumerate(detsets):
            #if idetset2 < idetset1:
            #    continue
            # No cross spectra between full frequency and
            # detsets
            if detset1.endswith('GHz') and detset2[-1] in 'AB':
                continue
            if detset2.endswith('GHz') and detset1[-1] in 'AB':
                continue
            detsetpairs.append((detset1, detset2))

    # Single detector and single horn auto spectra

    for det in list_planck('Planck'):
        # Single detector
        detsetpairs.append((det, det))
        if det[-1] in 'aM':
            # Single horn
            horn = det[:-1]
            detsetpairs.append((horn, horn))

    for ipair, detsetpair in enumerate(detsetpairs):
        if ipair % ntask != rank:
            continue
        program1(RIMO, detsetpair, blmfile, outdir, smax, spin_ref, blm_ref,
                 release=release, rhobeam=rhobeam, rhohit=rhobeam,
                 test=test, planet=planet, conserve_memory=conserve_memory,
                 overwrite=False)
