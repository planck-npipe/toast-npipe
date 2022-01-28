import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
from planck_util import list_planck

ver = 'npipe6'
do_bootstrap = False
do_alias = True
if do_alias:
    aliases = ['_low', '', '_high']
else:
    aliases = ['']
scale = 1e6
unit = '$\mu$K'
amp = 1.25e-5 * scale

def bootstrap(x, niter=10):
    """ Measure bootstrap mean and error of x.
    """
    nx = x.size
    means = np.zeros(niter)
    for iiter in range(niter):
        ind = (np.random.rand(nx)*nx).astype(int)
        means[iiter] = np.mean(x[ind])
    meanmean = np.mean(means)
    meanerr = np.std(means)
    return meanmean, meanerr

for freq in 30, 44, 70,:
#for freq in 70,:
    #amp = {30:6e-6, 44:35e-6, 70:6e-6}[freq] * scale
    #amp2 = {30:20e-6, 44:120e-6, 70:70e-6}[freq] * scale
    amp = {30:25e-6, 44:250e-6, 70:300e-6}[freq] * scale
    amp2 = {30:75e-6, 44:800e-6, 70:800e-6}[freq] * scale
    fsample = {30:4096/126, 44:4096/88, 70:4096/52}[freq]
    if do_alias:
        frac = fsample - int(fsample)
        offsets = [-frac, 0, frac]
    else:
        offsets = [0]
    linefreq = []
    fline = 1.
    while fline < fsample/2:
        if do_alias:
            linefreq.append(fline-frac)
        linefreq.append(fline)
        if do_alias:
            linefreq.append(fline+frac)
        fline += 1
    linefreq = np.array(linefreq)
    nline = linefreq.size
    dets = list_planck(freq)
    """
    if freq < 70:
        detsets = [dets]
        detsetnames = ['']
    else:
        detsets = [
            dets[0:2]+dets[-2:], dets[2:4]+dets[-4:-2], dets[4:8]]
        detsetnames = ['detset1', 'detset2', 'detset3']
    """
    detsets = []
    detsetnames = []
    for det in dets:
        detsets.append([det])
        detsetnames.append(det)

    for detset, detsetname in zip(detsets, detsetnames):
        plt.figure(figsize=[18, 12])
        plt.suptitle('{}GHz {}'.format(freq, detsetname))
        for det in detset:
            fn = 'output_{}_{}_preproc/ring_info_{}_iter01.pck'.format(
                ver, freq, det)
            if not os.path.isfile(fn):
                print('File not found:', fn)
                continue
            info = pickle.load(open(fn, 'rb'))
            good = info.failed + info.outlier == 0
            #good[info.ring_number < 22000] = False
            ngood = np.sum(good)
            ring = info.ring_number[good]

            cosamps = np.zeros(nline)
            sinamps = np.zeros(nline)
            coserrs = np.zeros(nline)
            sinerrs = np.zeros(nline)
            nt = 1000
            t = np.arange(nt+1) / nt
            templates = np.zeros([5, t.size])
            for icol, col in enumerate(['0', '1', '2', '3', '']):
                template = np.zeros_like(t)
                iline = 0
                for fcenter in np.arange(1, int(fsample//2+1)):
                    for offset, alias in zip(offsets, aliases):
                        line = '{:02}Hz{}{}'.format(fcenter, alias, col)
                        fline = fcenter + offset
                        try:
                            cosamp = info['cos_ampl_' + line][good]
                            sinamp = info['sin_ampl_' + line][good]
                        except:
                            continue
                        if do_bootstrap:
                            cosmean, coserr = bootstrap(cosamp)
                            sinmean, sinerr = bootstrap(sinamp)
                        else:
                            cosmean = np.mean(cosamp)
                            sinmean = np.mean(sinamp)
                        coserr = np.std(cosamp) / np.sqrt(ngood)
                        sinerr = np.std(sinamp) / np.sqrt(ngood)
                        cosamps[iline] = cosmean
                        sinamps[iline] = sinmean
                        coserrs[iline] = coserr
                        sinerrs[iline] = sinerr
                        ffac = 2 * np.pi * fline * t
                        template += cosmean*np.cos(ffac) + sinmean*np.sin(ffac)
                        iline += 1

                ind = np.argsort(linefreq)

                if col == '':
                    lw = 5
                else:
                    lw = 2

                plt.subplot(3, 1, 1)
                plt.errorbar(linefreq[ind], cosamps[ind]*scale, coserrs[ind]*scale,
                             fmt='-o', label=det+col, lw=lw)
                plt.subplot(3, 1, 2)
                plt.errorbar(linefreq[ind], sinamps[ind]*scale, sinerrs[ind]*scale,
                             fmt='-o', label=det+col, lw=lw)
                plt.subplot(3, 1, 3)
                plt.plot(t, template*scale, label=det+col, lw=lw)

                templates[icol] = template
            fname_out = '1Hz_template_{}.txt'.format(det)
            fout = open(fname_out, 'w')
            for i, t in enumerate(t):
                fout.write('{} {} {} {} {} {}\n'.format(
                    t, templates[0, i], templates[1, i], templates[2, i],
                    templates[3, i], templates[4, i]))
            fout.close()
            print('Templates saved in {}'.format(fname_out))

        plt.subplot(3, 1, 1)
        ax = plt.gca()
        ax.set_title('Cosine')
        ax.axhline(0, color='k')
        ax.set_ylim([-amp, amp])
        ax.set_xlim([0, 40])
        ax.set_xlabel('Freq [Hz]')
        ax.set_ylabel('Amp [{}]'.format(unit))

        plt.subplot(3, 1, 2)
        ax = plt.gca()
        ax.set_title('Sine')
        ax.axhline(0, color='k')
        ax.set_ylim([-amp, amp])
        ax.set_xlim([0, 40])
        ax.set_xlabel('Freq [Hz]')
        ax.set_ylabel('Amp [{}]'.format(unit))

        plt.subplot(3, 1, 3)
        ax = plt.gca()
        ax.set_title('Spike template')
        ax.axhline(0, color='k')
        ax.set_ylim([-amp2, amp2])
        ax.set_xlim([-.01, 1.01])
        ax.set_xlabel('t [s]')
        ax.set_ylabel('Amp [{}]'.format(unit))

        plt.legend(loc='lower right')

        plt.subplots_adjust(hspace=0.3, bottom=0.05, left=0.05,
                            right=0.95, top=0.93)

        plt.savefig(
            'line_amplitudes_{}_{}GHz{}.png'.format(ver, freq, detsetname))
#plt.show()
