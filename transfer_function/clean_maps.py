#
# THIS SCRIPT IS NOT FINISHED
#
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import pickle

import healpy as hp
import astropy.io.fits as pf
from spice import ispice
from planck_util import log_bin

def freq_to_fwhm(freq):
    ifreq = int(freq)
    fwhms = {30: 33, 44: 24, 70: 14,
             100: 10, 143: 7.1, 217: 5.5,
             353: 5, 545: 5, 857: 5}
    if ifreq not in fwhms:
        raise RuntimeError('Unknown frequency: {}'.format(freq))
    return fwhms[ifreq]

do_plot = False
cachedir = 'maps_and_cl'
fwhm_deg = 3
fwhm_arcmin = fwhm_deg * 60
fwhm_rad = np.radians(fwhm_deg)
lmax = 256
nside = 256
nbin = 100
nell = 40
freqs = 30, 44, 70, 100, 143, 217, 353
#freqs = int(sys.argv[1]),
norm = 1e15
#npipedir = '/project/projectdirs/planck/data/npipe'
npipedir = '/global/cscratch1/sd/keskital/npipe_maps'
#mcstart, mcstop, ver, quickpol = 0, 100, 'npipe6v19', False
mcstart, mcstop, ver, quickpol, cachedir = (
    100, 155, 'npipe6v20', True, 'maps_and_cl_npipe6v20')
#mcstart, mcstop, ver, quickpol = 156, 165, 'npipe6v20', True

fsky = 60 # 27, 33, 39, 46, 53, 60, 67, 74, 82, 90

for subset in '', 'A', 'B':
    if subset == '':
        sname = 'GHz'
    else:
        sname = subset
    factors = {'EE': {}, 'BB': {}, 'TE': {}}
    errors = {'EE': {}, 'BB': {}, 'TE': {}}
    simdir = '{}/{}{}_sim'.format(npipedir, ver, subset)
    for freq in freqs:
        if freq < 100:
            nside_in = 1024
        else:
            nside_in = 2048
        wbeam = freq_to_fwhm(freq)

        fn_mask = 'clmask_{:02}fsky_nside{:04}.fits'.format(fsky, nside)

        fn_dipo = '/global/cscratch1/sd/keskital/hfi_pipe/dipole_nside{:04}.fits' \
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


        def clean_map(m, fgmaps, dipo, cmb):
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
            for cc, fgmap in zip(coeff, fgmaps):
                m[1] -= cc * fgmap[1]
                m[2] -= cc * fgmap[2]
            norm = 1 - np.sum(coeff)
            return m

        cl_in = []
        cl_out = []

        lmax_out = 50

        for mc in range(mcstart, mcstop):
            fn_cl_clean = '{}/clcross_{:04}_{:03}{}_x_cmb_cleaned_{:02}fsky.fits' \
                          ''.format(cachedir, mc, freq, subset, fsky)
            fn_cl_cmb = '{}/clcross_{:04}_{:03}{}_cmb_{:02}fsky.fits' \
                          ''.format(cachedir, mc, freq, subset, fsky)
            if not os.path.isfile(fn_cl_clean) or not os.path.isfile(fn_cl_cmb):
                fn_cmb = '{}/smoothed_cmb_{:04}_{:03}.fits'.format(cachedir, mc, freq)
                if not os.path.isfile(fn_cmb):
                    if quickpol:
                        fn_in = '{}/{:04}/input/' \
                                'ffp9_cmb_scl_{:03}_alm_mc_{:04}_nside{:04}_' \
                                'quickpol.fits' \
                                ''.format(simdir, mc, freq, mc, nside_in)
                    else:
                        fn_in = '{}/{:04}/input/' \
                                'ffp9_cmb_scl_{:03}_alm_mc_{:04}_nside{:04}_' \
                                'fwhm{:.1f}.fits' \
                                ''.format(simdir, mc, freq, mc, nside_in, wbeam)
                    cmb = load_map(fn_in, fn_cmb, nside, fwhm_rad, lmax)
                else:
                    cmb = hp.read_map(fn_cmb, None)
                fn_clean = '{}/cleaned_{:04}_{:03}{}.fits'.format(
                    cachedir, mc, freq, subset)
                if not os.path.isfile(fn_clean):
                    fgmaps = []
                    for name in ['sky_model', 'sky_model_deriv']:
                        if quickpol:
                            fn_in = '{}/skymodel_cache/' \
                                    '{}_{:03}GHz_nside{}_quickpol_cfreq_zodi.fits' \
                                    ''.format(simdir, name, freq, nside_in)
                            fn_out = '{}/smoothed_{}_{:03}_quickpol.fits'.format(
                                cachedir, name, freq)
                        else:
                            fn_in = '{}/skymodel_cache/' \
                                    '{}_{:03}GHz_nside{}_fwhm{:.1f}.fits' \
                                    ''.format(simdir, name, freq, nside_in, wbeam)
                            fn_out = '{}/smoothed_{}_{:03}.fits'.format(
                                cachedir, name, freq)
                        fgmap = load_map(fn_in, fn_out, nside, fwhm_rad, lmax)
                        fgmaps.append(fgmap)

                    fn_in = '{}/{:04}/{}{}_{:03}_map.fits'.format(
                        simdir, mc, ver, subset, freq)
                    fn_out = '{}/smoothed_{:04}_{:03}{}.fits'.format(
                        cachedir, mc, freq, subset)
                    m = load_map(fn_in, fn_out, nside, fwhm_rad, lmax)
                    cleaned = clean_map(m, fgmaps, dipo, cmb)
                    print('Writing', fn_clean)
                    hp.write_map(fn_clean, cleaned, coord='G', overwrite=True)
                ispice(fn_cmb, fn_cl_clean, nlmax=lmax,
                       beam1=fwhm_arcmin, beam2=fwhm_arcmin,
                       mapfile2=fn_clean, weightfile1=fn_mask, weightfile2=fn_mask,
                       polarization='YES', subav='YES', subdipole='YES',
                       symmetric_cl='YES')
                ispice(fn_cmb, fn_cl_cmb, nlmax=lmax,
                       beam1=fwhm_arcmin, beam2=fwhm_arcmin,
                       mapfile2=fn_cmb, weightfile1=fn_mask, weightfile2=fn_mask,
                       polarization='YES', subav='YES', subdipole='YES',
                       symmetric_cl='YES')

            print('Loading', fn_cl_clean)
            cl_cmb = hp.read_cl(fn_cl_cmb)

            print('Loading', fn_cl_cmb)
            cl_clean = hp.read_cl(fn_cl_clean)

            cl_in.append(cl_cmb[1:4][:lmax_out + 1])
            cl_out.append(cl_clean[1:4][:lmax_out + 1])

        cl_in = np.array(cl_in)
        cl_out = np.array(cl_out)

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

        nrow = 4
        ncol = 4
        for imode, mode in enumerate(['EE', 'BB', 'TE']):
            if do_plot:
                plt.figure(figsize=[4*ncol, 3*nrow])
                plt.suptitle('{}{} {} fsky = {}%'.format(
                    freq, sname, mode, fsky))
            factors[mode][freq] = np.zeros(nell + 2)
            errors[mode][freq] = np.zeros(nell + 2)
            for ell in range(2, nell + 2):
                x = cl_in[:, imode, ell] * norm
                y = cl_out[:, imode, ell] * norm
                if False:
                    c, cerr = get_corr_and_var(x, y)
                else:
                    # Remove 10% of the largest input values
                    xsorted = np.sort(x)
                    lim = xsorted[int(x.size*.9)]
                    good = x < lim
                    c, cerr = get_corr_and_var(x[good], y[good])
                factors[mode][freq][ell] = c
                errors[mode][freq][ell] = cerr

                if ell - 1 > nrow * ncol or not do_plot:
                    continue

                plt.subplot(nrow, ncol, ell - 1)
                ax = plt.gca()
                ax.set_title('$\ell$ = {}'.format(ell))

                plt.plot(x, y, '.')

                vmin = min(0, np.amin(x))
                vmax = np.amax(x)
                xx = np.array([vmin, vmax])
                plt.plot(xx, xx, color='k')

                if (ell - 2) // ncol == nrow - 1:
                    ax.set_xlabel('Input C$_\ell$ x 1e15')
                if (ell - 2) % ncol == 0:
                    ax.set_ylabel('Output C$_\ell$ x 1e15')

                plt.plot(xx, c * xx,
                         label='k = {:.3f} $\pm$ {:.3f}'.format(c, cerr))
                plt.legend(loc='best')

            if do_plot:
                plt.subplots_adjust(hspace=0.25, wspace=0.2)
                plt.savefig('{}_bias_{:03}{}.png'.format(mode, freq, subset))
                plt.close()

    pickle.dump([factors, errors],
                open('suppression_factors_{:02}fsky{}.pck'.format(fsky, subset),
                     'wb'),
                protocol=2)

    # Plot the transfer functions

    if do_plot:
        for imode, mode in enumerate(['EE', 'BB', 'TE']):
            plt.figure(figsize=[18, 12])
            plt.suptitle('{} bias fsky = {} {}'.format(mode, fsky, subset))
            ell = np.arange(nell + 2)
            for ifreq, freq in enumerate(freqs[2:-1]):
                c = factors[mode][freq]
                err = errors[mode][freq]
                plt.errorbar(ell[2:] + (ifreq - 2) / 20, c[2:], err[2:],
                             label='{}GHz'.format(freq))
            ax = plt.gca()
            ax.set_xscale('log')
            ax.set_ylim([0, 2])
            ax.axhline(1, color='k')
            plt.legend(loc='best')
            plt.savefig('{}_bias_{:02}fsky{}.png'.format(mode, fsky, subset))
            plt.show()

    # Save the transfer functions

    for ifreq, freq in enumerate(freqs):
        fname_bl_in = '/project/projectdirs/planck/data/npipe/npipe6v20{}/'\
                      'quickpol/' \
                      'Bl_TEB_npipe6v20_{:03}{}x{:03}{}.fits'.format(
                          subset, freq, sname, freq, sname)
        hdulist = pf.open(fname_bl_in, 'readonly')
        # Augment header
        hdulist[1].header['biasfsky'] = (fsky, 'fsky used to evaluate E mode bias')
        # Make sure all columns are loaded
        hdulist[1].data.field('T')[:] *= 1
        hdulist[1].data.field('E')[:] *= 1
        hdulist[1].data.field('B')[:] *= 1
        for w in [False]:
            if w:
                fname_bl = 'Bl_TEB_{}_{:03}{}x{:03}{}_with_E_tf_{:02}fsky.fits'.format(
                    ver, freq, sname, freq, sname, fsky)
            else:
                fname_bl = 'Bl_TEB_{}_{:03}{}x{:03}{}_only_E_tf_{:02}fsky.fits'.format(
                    ver, freq, sname, freq, sname, fsky)
                # null the original TF
                hdulist[1].data.field('T')[:] = 1
                hdulist[1].data.field('E')[:] = 1
                hdulist[1].data.field('B')[:] = 1
            for imode, mode in enumerate(['E']):
                c = factors[mode+mode][freq]
                err = errors[mode+mode][freq]
                # Apply the transfer function between ell=2 and ell=10
                # and only when it is less than 1.
                n = c.size
                use_c = np.zeros(n, dtype=np.bool)
                use_c[2:11] = True
                use_c[c > 1] = False
                if mode == 'E':
                    #use_c[err > .1] = False
                    use_c[err > err[2]] = False
                elif mode == 'B':
                    use_c[err > .2] = False
                tf = np.ones(n)
                tf[use_c] = c[use_c]
                # Combine the transfer functions
                hdulist[1].data.field(mode)[:n] *= tf
            hdulist.writeto(fname_bl, overwrite=True)
            print('Wrote', fname_bl)
