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
    main
      +--q2f
           +--clobber
           +--my_mwrfits
"""

import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.optimize

import astropy.io.fits as pyfits
import healpy as hp

from toast_planck.utilities import qp_file, list_planck

outdir = './quickpol_output'
indir = './quickpol_output'
smax = 6
docross = True
blfile = True
wlfile = True
blTEBfile = True
overwrite = False
release = 'npipe6v20'
full = False  # False : Only sample a small fraction of the pixels
do_plot = False

NO_COLOR = '\x1b[0m'
GREEN_COLOR = '\x1b[32;11m'
RED_COLOR = '\x1b[31;01m'
BLUE_COLOR = '\x1b[34;11m'
BOLD = '\x1b[1;01m'

t1 = np.array([ # sym
    ['TT', 'TE', 'TB'],
    ['TE', 'EE', 'EB'],
    ['TB', 'EB', 'BB']])
t2 = np.array([ # non-sym
    ['TT', 'TE', 'TB'],
    ['ET', 'EE', 'EB'],
    ['BT', 'BE', 'BB']])
kk = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2], [1, 0], [2, 0], [2, 1]]
t3 = [t2[k1, k2] for k1, k2 in kk]


# ==============================================================================


def fit_gauss(bl):
    """
    Fit a Gaussian beam to the provided beam window function
    """
    ell = np.arange(bl.size)
    def gaussbeam(ell, sigma):
        beam = np.exp(-.5 * ell * (ell + 1) * sigma ** 2)
        return beam
    def resid(p, ell, bl):
        sigma = p[0]
        return bl - gaussbeam(ell, sigma)
    p0 = [np.radians(.5)]
    result = scipy.optimize.least_squares(
        resid, p0, method='lm', args=(ell, bl), max_nfev=10000)
    if not result.success:
        raise RuntimeError(
            'Gaussian fitting failed: {}'.format(result.message))
    sigma = result.x[0]
    return gaussbeam(ell, sigma), sigma


#-------------------------------------------------------------------------------

def clobber(filename, overwrite):
    write = True
    if os.path.exists(filename):
        if overwrite:
            print(prefix, '%sOverwriting %s%s' % (RED_COLOR, filename, NO_COLOR),
                  flush=True)
        else:
            print(prefix, '%s%s already exists. Skip%s' % (BLUE_COLOR, filename,
                                                   NO_COLOR), flush=True)
            write = False
    return write

#-------------------------------------------------------------------------------

def my_mwrfits(
        filename, data, colnames=None,keys=None, bintable=False, ftype=None,
        extnames=None, origin=None, dets=None):
    """Write columns to a fits file in a table extension.

    Parameters
    ----------
    filename : str
      The fits file name
    data : list of 1D arrays
      A list of 1D arrays to write in the table
    colnames : list of str
      The column names
    keys : dict-like
      A dictionary with keywords to write in the header
    """

    hline = '----------------------------------------------------------------'
    if ftype == 'B':
        # name = 'WINDOW FUNCTION'
        comments = [
            'Beam Window Function B(l)',
            'Compatible with Healpix (synfast, smoothing, ...) and PolSpice',
            'To be squared before applying to power spectrum',
            '  C_map(l) = C_sky(l) * B(l)^2 ']
    if ftype == 'B_TEB':
        # name = 'WINDOW FUNCTIONS'
        comments = [
            'Beam Window Functions B(l), for T, E and B',
            'Compatible with Healpix (synfast, smoothing, ...) and PolSpice',
            'To be squared before applying to power spectrum',
            '  C_TT_map(l) = C_TT_sky(l) * B_T(l)^2 ',
            '  C_EE_map(l) = C_EE_sky(l) * B_E(l)^2 ',
            '  C_BB_map(l) = C_BB_sky(l) * B_B(l)^2 ']
    if ftype == 'W':
        # name = 'WINDOW FUNCTIONS'
        comments = [
            'Beam Window Functions W(l) = B(l)^2',
            'Applies directly to power spectrum              ' ,
            '  C_map(l) = C_sky(l) * W(l) ',
            'Includes cross-talk terms   ']

    # ---- primary header -----
    hdu = pyfits.PrimaryHDU(None)
    #hdu.name = name
    hhu = hdu.header.set
    #hhb = hdu.header.add_blank
    hhc = hdu.header.add_comment
    #hhh = hdu.header.add_history
    fdate = datetime.datetime.now().strftime('%Y-%m-%d')
    hhu('DATE', fdate, comment=' Creation date (CCYY-MM-DD) of FITS header')
    if extnames is not None:
        nx = len(extnames)
        hhu('NUMEXT', nx, 'Number of extensions')
        for xt in range(nx):
            hhu('XTNAME%d'%(xt+1), extnames[xt],
                'Name of extension #%d' % (xt+1))

    hhc(hline)
    for mycom in comments:
        hhc(mycom)
    if origin is not None:
        for myor in origin:
            hhc(myor)
        hhc(hline)
    if dets is not None:
        for id, det in enumerate(dets):
            hhu('DET%d'%(id+1), det, 'Detector (set)')

    hdulist = pyfits.HDUList([hdu])

    # ---- other HDUs : tables ----
    getformat = hp.fitsfunc.getformat

    for xt in range(len(data)):
        cols = []
        for line in range(len(data[xt])):
            namei = colnames[xt][line]
            array = data[xt][line]
            if bintable:
                nt = len(array)  # total length
                repeat = nt  # length / cell
                fmt = str(repeat) + getformat(array)
                array = np.reshape(array, (nt // repeat, repeat))
            else:
                fmt = getformat(array)
            cols.append(pyfits.Column(name=namei,
                                      format=fmt,
                                      array=array))
            if bintable:
                tbhdu = pyfits.BinTableHDU.from_columns(cols)
            else:
                tbhdu = pyfits.TableHDU.from_columns(cols)

        if extnames is not None:
            tbhdu.name = extnames[xt]

        ncols = len(cols)
        tbhdu.header['MAX-LPOL'] = (len(data[xt][0]) - 1, 'Maximum L multipole')
        tbhdu.header['POLAR'] = ((ncols > 1))
        tbhdu.header['BCROSS'] = ((ncols > 4))
        tbhdu.header['ASYMCL'] = ((ncols > 6))

        tbhdu.header.add_comment(hline)
        for mycom in comments:
            tbhdu.header.add_comment(mycom)
        for myor in origin:
            tbhdu.header.add_comment(myor)
        tbhdu.header.add_comment(hline)

        if type(keys) is dict:
            for k, v in list(keys.items()):
                tbhdu.header[k] = (v)

        hdulist.append(tbhdu)

    # write the file
    hdulist.writeto(filename, overwrite=True)

    # checking out the file
    #
    try:
        # pyfits.info(filename)
        p1 = pyfits.getdata(filename)
        junk = hp.mrdfits(filename)
        print(prefix, '%s checking out %s%s' % (GREEN_COLOR, filename, NO_COLOR),
              flush=True)
    except:
        raise RuntimeError('Failed to load {}'.format(filename))

        
#---------------------------------------------------------------------------------


def detset2lmax(detset):
    if detset.startswith('0') or detset.startswith('LFI'):
        lmax = 4 * 1024
    else:
        lmax = 4 * 2048
    return lmax

def detset2pol(detset):
    if '545' in detset or '857' in detset or 'LFI' in detset or '-' in detset:
        pol = False
    else:
        pol = True
    return pol



def q2f(indir, outdir, dets, smax, release=None, full=True, blfile=True,
        blTEBfile=True, wlfile=True, overwrite=True, do_plot=False):
    pconv = 'cmbfast'
    angle_shift = 0
    force_det = None
    rhobeam = 'IMO'
    rhohit = 'IMO'

    lmax = min(detset2lmax(dets[0]), detset2lmax(dets[1]))
    pol = detset2pol(dets[0]) and detset2pol(dets[1])

    fz = qp_file(indir, dets, lmax=lmax, smax=smax, angle_shift=angle_shift,
                 full=full, force_det=force_det, release=release,
                 rhobeam=rhobeam, rhohit=rhohit)
    print(prefix, '--------------------')
    print(prefix, fz, flush=True)
    try:
        dz1 = np.load(fz)
    except:
        print(prefix, '%s not found' % fz, flush=True)
        return

    f32 = np.float32
    bm1 = dz1['beam_mat'].tolist()
    TT = f32(bm1['TT'])
    renorm = TT[0, 0, 0]
    TT /= renorm
    EE = f32(bm1['EE']) / renorm
    BB = f32(bm1['BB']) / renorm
    TE = f32(bm1['TE']) / renorm
    print(prefix, '%s Renorm-1 = %s %s' % (BLUE_COLOR, str(renorm-1), NO_COLOR),
          flush=True)
    wtt = TT[0:lmax + 1, 0, 0]
    bl = np.sqrt(np.abs(wtt)) * np.sign(wtt)
    imin = np.argmin(bl)
    imax = np.argmax(bl)
    wee = EE[0:lmax + 1, 1, 1]
    wbb = BB[0:lmax + 1, 2, 2]
    bl_E = np.sqrt(np.abs(wee)) * np.sign(wee)
    bl_B = np.sqrt(np.abs(wbb)) * np.sign(wbb)

    ineg = np.where(bl < 0)[0]
    print(prefix, 'Max = ', bl[imax], imax)
    if len(ineg) > 0:
        print(prefix, '%s Neg = %s %s %s' % (RED_COLOR, str(ineg[0]), str(ineg[-1]),
                                     NO_COLOR), flush=True)
    print(prefix, 'Min = ', bl[imin], imin, flush=True)

    fitsfile_T = os.path.join(
        outdir, 'Bl_%s_%sx%s.fits' % (release, dets[0], dets[1]))
    fitsfile_TEB = os.path.join(
        outdir, 'Bl_TEB_%s_%sx%s.fits' % (release, dets[0], dets[1]))
    fitsfile_W = os.path.join(
        outdir, 'Wl_%s_%sx%s.fits' % (release, dets[0], dets[1]))
    #print(prefix, fitsfile_T)
    #print(prefix, fitsfile_W)
    #print(prefix, len(bl))
    #print(prefix, np.size(bl))
    fdate = datetime.datetime.now().strftime('%Y-%m-%d')
    origin = ['Adapted from', fz, 'by %s on %s' % (__file__, fdate)]

    gaussbeam, sigma = fit_gauss(bl)
    fwhm = np.abs(np.degrees(sigma) * 60 * np.sqrt(8. * np.log(2.)))

    # T B(l)
    if (blfile and clobber(fitsfile_T, overwrite)):
        extnames = ['WINDOW FUNCTION']
        my_mwrfits(fitsfile_T, [[bl]], colnames=[['TEMPERATURE']],
                   bintable=False, ftype='B', extnames=extnames,
                   origin=origin, dets=dets)
        if do_plot:
            # Make a simple plot of the window function
            hdulist = pyfits.open(fitsfile_T)
            plt.figure()
            plt.gca().set_title('{} {} x {}'.format(release, dets[0], dets[1]))
            plt.semilogy(hdulist[1].data.field(0), label='T')
            ylim = [1e-8, 2] # plt.gca().get_ylim()
            plt.plot(gaussbeam, label='{:.2f}\' FWHM'.format(fwhm))
            plt.gca().set_ylim(ylim)
            plt.legend(loc='best')
            fn_plot = fitsfile_T.replace('.fits', '.png')
            plt.savefig(fn_plot)
            print(prefix, 'Plot saved in', fn_plot, flush=True)
            plt.close()
            hdulist.close()

    # T, E, B B(l)
    if (blTEBfile and clobber(fitsfile_TEB, overwrite) and pol):
        extnames = ['WINDOW FUNCTIONS']
        my_mwrfits(fitsfile_TEB, [[bl, bl_E, bl_B]], colnames=[['T', 'E', 'B']],
                   bintable=False, ftype='B_TEB', extnames=extnames,
                   origin=origin, dets=dets)
        if do_plot:
            # Make a simple plot of the window function
            hdulist = pyfits.open(fitsfile_TEB)
            plt.figure()
            plt.gca().set_title('{} {} x {}'.format(release, dets[0], dets[1]))
            for i in range(3):
                plt.semilogy(hdulist[1].data.field(i), label='TEB'[i])
            ylim = [1e-8, 2] # plt.gca().get_ylim()
            plt.plot(gaussbeam, label='{:.2f}\' FWHM'.format(fwhm))
            plt.gca().set_ylim(ylim)
            plt.legend(loc='best')
            fn_plot = fitsfile_TEB.replace('.fits', '.png')
            plt.savefig(fn_plot)
            print(prefix, 'Plot saved in', fn_plot, flush=True)
            plt.close()
            hdulist.close()

    # W(l)
    if (wlfile and clobber(fitsfile_W, overwrite) and pol):
        extnames = ['TT', 'EE', 'BB', 'TE']
        data = [
            [TT[0:, k1, k2] for k1,k2 in kk],
            [EE[0:, k1, k2] for k1,k2 in kk],
            [BB[0:, k1, k2] for k1,k2 in kk],
            [TE[0:, k1, k2] for k1,k2 in kk]]
        colnames = [
            [extnames[0] + '_2_' + c for c in t3],
            [extnames[1] + '_2_' + c for c in t3],
            [extnames[2] + '_2_' + c for c in t3],
            [extnames[3] + '_2_' + c for c in t3]]
        my_mwrfits(fitsfile_W, data, colnames=colnames, bintable=True,
                   ftype='W', extnames=extnames, origin=origin, dets=dets)
        if do_plot:
            # Make a simple plot of the window function
            hdulist = pyfits.open(fitsfile_W)
            plt.figure(figsize=[18, 12])
            plt.gca().set_title('{} {} x {}'.format(release, dets[0], dets[1]))
            for ifield, field in enumerate(['TT_2_TE', 'TT_2_EE', 'TT_2_BB']):
                plt.subplot(2, 2, 1 + ifield)
                plt.plot(hdulist[1].data.field(field).flatten(), label=field)
                plt.legend(loc='best')
                plt.gca().axhline(0, color='k')
            for ifield, field in enumerate(['EE_2_BB']):
                plt.subplot(2, 2, 4 + ifield)
                plt.plot(hdulist[2].data.field(field).flatten(), label=field)
                plt.legend(loc='best')
                plt.gca().axhline(0, color='k')
            fn_plot = fitsfile_W.replace('.fits', '.png')
            plt.savefig(fn_plot)
            print(prefix, 'Plot saved in', fn_plot, flush=True)
            plt.close()
            hdulist.close()

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

if __name__ == '__main__':

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
        q2f(outdir, indir, detsetpair, smax, release=release, full=full,
            blfile=blfile, blTEBfile=blTEBfile, wlfile=wlfile,
            overwrite=overwrite, do_plot=do_plot)
