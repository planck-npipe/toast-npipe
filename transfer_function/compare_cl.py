import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os
import sys

from planck_util import log_bin

npipegain = 1.002
npipefwhm = np.radians(.5 / 60)

nbin = 300
fsky = 52 # 90, 52, 25
fig = plt.figure(figsize=[18, 12])
plt.suptitle('fsky = {}%'.format(fsky))
axes = [fig.add_subplot(2, 2, 1+i) for i in range(4)]
for freq1, freq2 in [(70, 100), (100, 143), (100, 217), (143, 217)]:
    name0 = '{:03}x{:03} Legacy'.format(freq1, freq2)
    cl0 = hp.read_cl(
        'cl_{}dx12x{}dx12_{:02}fsky.fits'.format(freq1, freq2, fsky))
    name1 = '{:03}x{:03} NPIPE'.format(freq1, freq2)
    cl1 = hp.read_cl(
        'cl_{}x{}_{:02}fsky.fits'.format(freq1, freq2, fsky))
    for freq in [freq1, freq2]:
        if freq > 70:
            cl1 *= npipegain

    lmax = cl0[0].size - 1
    ell = np.arange(lmax + 1)
    ellbin, hits = log_bin(ell, nbin=nbin)
    norm = ell * (ell + 1) / 2 / np.pi * 1e12
    npipebeam = hp.gauss_beam(npipefwhm, lmax=lmax)

    for i in range(2):
        cl0bin, hits = log_bin(norm * cl0[i], nbin=nbin)
        cl1bin, hits = log_bin(norm * cl1[i], nbin=nbin)
        ax = axes[i]
        comp = ['TT', 'EE', 'BB'][i]
        ax.set_title(comp)
        ax.plot(ellbin[2:], cl0bin[2:], label=name0)
        ax.plot(ellbin[2:], cl1bin[2:], label=name1)
        if i == 0:
            ax.set_ylim([-100, 6100])
        elif i == 1:
            ax.set_ylim([-10, 50])
        ax = axes[2 + i]
        ax.set_title(comp + ' ratio')
        ax.plot(ellbin[2:], cl1bin[2:] / cl0bin[2:],
                label='{} / {}'.format(name1, name0))
        ax.set_ylim([.99, 1.01])
        ax.axhline(1, color='k')
        ax.plot(ell, npipebeam ** 2, color='k', lw=2)
    axes[1].legend(loc='best')
    axes[3].legend(loc='best')
    plt.show()
plt.savefig('clross_comparison_fsky{:02}.png'.format(fsky))
