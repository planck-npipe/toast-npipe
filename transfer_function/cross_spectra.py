import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from spice import ispice
import healpy as hp

from planck_util import log_bin

lmax = 2048
nbin = 300
mapdir = '/Users/reijo/Work/npipe6/'
mapdirdx12 = '/Users/reijo/data/dx12/'
fgfwhm = np.radians(1)
fglmax = 512
nside = 512

def get_cleaned_map(fname, freq):
    fname_cleaned = 'fgcleaned_' + os.path.basename(fname)
    dx12 = 'dx12' in fname
    if not os.path.isfile(fname_cleaned):
        m = hp.ud_grade(
            hp.read_map(fname, range(3), verbose=False, nest=True), nside,
            order_in='NEST', order_out='RING')
        if dx12:
            dipo = 0
        else:
            fname_dipo = '/Users/reijo/data/hfi_pipe/dipole_nside{:04}.fits' \
            ''.format(nside)
            print('Loading', fname_dipo)
            dipo = hp.read_map(fname_dipo, verbose=False)
        sfreq = '{:03}'.format(freq)
        fgmaps = []
        npix = 12 * nside ** 2
        good = np.zeros(npix, dtype=np.bool)
        bad = np.ones(npix, dtype=np.bool)
        if freq != 217:
            fgfreqs = [30, 217, 353]
        else:
            fgfreqs = [30, 353]
        for fgfreq in fgfreqs:
            fgfname = fname.replace(sfreq, '{:03}'.format(fgfreq))
            print('Loading ', fgfname)
            fgmap = hp.ud_grade(
                hp.read_map(fgfname, range(3), verbose=False, nest=True), nside,
                order_in='NEST', order_out='RING')
            print('Smoothing')
            fgmap = hp.smoothing(fgmap, fwhm=fgfwhm, lmax=fglmax, iter=0,
                                 verbose=False)
            fgi = fgmap[0] - dipo
            fgp = np.sqrt(fgmap[1]**2 + fgmap[2]**2)
            print('Sorting')
            fgi_sorted = np.sort(fgi)
            fgp_sorted = np.sort(fgp)
            good[fgp > fgp_sorted[int(0.70 * npix)]] = True
            bad[fgi > fgi_sorted[int(0.90 * npix)]] = False
            fgmaps.append(fgmap)
        mask = good * bad
        templates = []
        for fgmap in fgmaps:
            templates.append(
                np.hstack([fgmap[1][mask], fgmap[2][mask]]))
        templates = np.vstack(templates)
        target = np.hstack([m[1][mask], m[2][mask]])
        invcov = np.dot(templates, templates.T)
        cov = np.linalg.inv(invcov)
        proj = np.dot(templates, target)
        coeff = np.dot(cov, proj)
        for cc, fgmap in zip(coeff, fgmaps):
            m[1] -= cc * fgmap[1]
            m[2] -= cc * fgmap[2]
        norm = 1 - np.sum(coeff)
        m[1] /= norm
        m[2] /= norm
        hp.write_map(fname_cleaned, m, coord='G')
        print('Wrote cleaned map to {}. Coeff = {}'.format(fname_cleaned,
                                                           coeff))
    return fname_cleaned

#for fsky in 90, 82, 74, 66, 59, 52, 44, 37, 31, 25:
for fsky in 90, 52, 25:
    fig1 = plt.figure(figsize=[18, 12])
    fig2 = plt.figure(figsize=[12, 8])
    axes1 = []
    for i in range(4):
        axes1.append(fig1.add_subplot(2, 2, 1+i))
    ax2 = fig2.add_subplot(1, 1, 1)

    for freq1, subset1, freq2, subset2 in [
            #(70, 'A', 70, 'B'),
            #(100, 'A', 100, 'B'),
            #(143, 'A', 143, 'B'),
            #(217, 'A', 217, 'B'),
            (70, '', 100, ''),
            (100, '', 143, ''),
            (100, '', 217, ''),
            (143, '', 217, ''),
            (70, 'dx12', 100, 'dx12'),
            (100, 'dx12', 143, 'dx12'),
            (100, 'dx12', 217, 'dx12'),
            (143, 'dx12', 217, 'dx12'),
            ]:
        if 'dx12' in subset1:
            fname1 = mapdirdx12 + 'dx12_{:03}_map.fits'''.format(freq1)
        else:
            fname1 = mapdir + 'npipe6v19{}_{:03}_map.fits'.format(subset1, freq1)
        if 'dx12' in subset2:
            fname2 = mapdirdx12 + 'dx12_{:03}_map.fits'''.format(freq2)
        else:
            fname2 = mapdir + 'npipe6v19{}_{:03}_map.fits'.format(subset2, freq2)

        label = '{}{}x{}{}'.format(freq1, subset1, freq2, subset2)

        fname1_clean = get_cleaned_map(fname1, freq1)
        fname2_clean = get_cleaned_map(fname2, freq2)

        if freq1 < 100:
            nside1 = 1024
        else:
            nside1 = 2048

        if freq2 < 100:
            nside2 = 1024
        else:
            nside2 = 2048
        #mask1 = '/Users/reijo/data/hfi_pipe/clmask_30pc_ns{:04}.fits'.format(nside1)
        #mask2 = '/Users/reijo/data/hfi_pipe/clmask_30pc_ns{:04}.fits'.format(nside2)
        mask1 = 'clmask_{:02}fsky_nside{:04}.fits'.format(fsky, nside1)
        mask2 = 'clmask_{:02}fsky_nside{:04}.fits'.format(fsky, nside2)
        #fgmask1 = '/Users/reijo/data/hfi_pipe/clmask_30pc_ns{:04}.fits'.format(nside)
        #fgmask2 = '/Users/reijo/data/hfi_pipe/clmask_30pc_ns{:04}.fits'.format(nside)
        fgmask1 = 'clmask_{:02}fsky_nside{:04}.fits'.format(fsky, nside)
        fgmask2 = 'clmask_{:02}fsky_nside{:04}.fits'.format(fsky, nside)

        nums = {30: 1, 44: 2, 70: 3, 100: 4, 143: 5, 217: 6, 545: 7, 545: 8, 857: 9}
        num1 = nums[freq1]
        num2 = nums[freq2]
        if subset1 == 'dx12':
            subset1 = ''
        fname_beam1 = '/Users/reijo/Work/npipe6/npipe6v19{}_beam_windows.fits[{}]' \
            ''.format(subset1, num1)
        if subset2 == 'dx12':
            subset2 = ''
        fname_beam2 = '/Users/reijo/Work/npipe6/npipe6v19{}_beam_windows.fits[{}]' \
            ''.format(subset2, num2)

        fname_cl = 'cl_{}_{:02}fsky.fits'.format(label, fsky)
        fname_cl_clean = 'fgcleaned_cl_{}_{:02}fsky.fits'.format(label, fsky)

        if not os.path.isfile(fname_cl):
            ispice(fname1, fname_cl, nlmax=lmax,
                   beam_file1=fname_beam1, beam_file2=fname_beam2,
                   mapfile2=fname2, maskfile1=mask1, maskfile2=mask2,
                   polarization='YES', subav='YES', subdipole='YES',
                   symmetric_cl='YES')

        if not os.path.isfile(fname_cl_clean):
            ispice(fname1_clean, fname_cl_clean, nlmax=lmax,
                   beam_file1=fname_beam1, beam_file2=fname_beam2,
                   mapfile2=fname2_clean, maskfile1=fgmask1, maskfile2=fgmask2,
                   polarization='YES', subav='YES', subdipole='YES',
                   symmetric_cl='YES')

        cl = hp.read_cl(fname_cl)
        clclean = hp.read_cl(fname_cl_clean)
        lmaxclean = clclean[0].size - 1
        ell = np.arange(lmax+1)
        ellclean = np.arange(lmaxclean+1)
        ellbin, hits = log_bin(ell, nbin=nbin)
        ellbinclean, hits = log_bin(ellclean, nbin=nbin)
        norm = ell * (ell + 1) / 2 / np.pi * 1e12
        normclean = ellclean * (ellclean + 1) / 2 / np.pi * 1e12

        for i in range(4):
            ax = axes1[i]
            #plt.plot(ell[2:-1], (norm * cl[i])[2:-1])
            clbin, hits = log_bin(norm * cl[i], nbin=nbin)
            ax.plot(ellbin[2:-1], clbin[2:-1], zorder=100, label=label)
            clbin, hits = log_bin(normclean * clclean[i], nbin=nbin)
            #ax.plot(ellbin[2:-1], clbin[2:-1], zorder=100,
            #        label=label + '-fgclean')
            if i == 1:
                ax2.plot(ellbinclean[2:-1], clbin[2:-1], zorder=100,
                        label=label + '-fgclean')


    #cl0 = np.genfromtxt('/Users/reijo/Software/camb/test/ffp10_scalCls.dat').T
    cl9 = np.genfromtxt(
        '/Users/reijo/Software/camb/test/ffp9/'
        'base_plikHM_TT_lowTEB_lensing_lensedCls.dat').T
    cl10 = np.genfromtxt(
        '/Users/reijo/Software/camb/test/ffp10/ffp10_lensedCls.dat').T
    for i in range(4):
        ax = axes1[i]
        axes = [ax]
        if i == 1:
            axes.append(ax2)
        for ax in axes:
            ax.plot(cl9[0], cl9[1+i], ls='--', color='k', lw=4,
                    label=r'$\tau=0.066$') # FFP9
            ax.plot(cl10[0], cl10[1+i], ls='-', color='k', lw=4,
                    label=r'$\tau=0.060$') # FFP10
            ax.axhline(0, color='k')
            ax.set_xscale('log')
            if i in [1, 2]:
                ax2.set_xlim([2, 50])
                ax2.set_ylim([-.25, .25])
            ax.set_xlabel(r'Multipole, $\ell$')
            ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{{{}}}/2\pi$'.format(
                ['TT','EE','BB','TE'][i]))
    axes[-1].legend(loc='best')
    ax2.legend(loc='best')
    for fig in [fig1, fig2]:
        fig.suptitle('fsky = {}%'.format(fsky))
        
    fname = 'cl_all_{:02}fsky.png'.format(fsky)
    fig1.savefig(fname)
    fname = 'cl_lowEE_{:02}fsky.png'.format(fsky)
    fig2.savefig(fname)

    plt.show()
