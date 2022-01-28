import sys

from scipy.constants import degree, arcmin, arcsec
import scipy.optimize
import scipy.signal

import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np


def center_phase(phase):
    return np.degrees(phase - np.pi) * 60.


def gaussian(amp, pos, sigma, phase):
    return amp * np.exp(-0.5 * (phase - pos) ** 2 / sigma ** 2)


def fit_gaussian(phase, signal):

    def residuals(p, phase, signal):
        amp, pos, sigma = p
        model = gaussian(amp, pos, sigma, phase)
        return signal - model

    amp0 = np.amax(signal)
    pos0 = np.pi
    sigma0 = np.radians(10 / 60)
    p0 = [amp0, pos0, sigma0]  # amp, pos, sigma
    p, _, _, mesg, ierr = scipy.optimize.leastsq(
        residuals, p0, args=(phase, signal), maxfev=1000, full_output=True)
    if ierr not in range(1, 5):
        raise RuntimeError(
            'leastsq failed: {}'.format(mesg))
    amp, pos, sigma = p
    return amp, pos, sigma


def modgaussian(p, phase):
    pos, amp, sigma, alpha = p
    g = amp * np.exp(-0.5 * np.abs((phase - pos) / sigma) ** (2 + alpha))
    return g


def fit_modgaussian(phase, signal):

    def residuals(p, phase, signal):
        model = modgaussian(p, phase)
        return signal - model

    amp0 = np.amax(signal)
    pos0 = np.pi
    sigma0 = np.radians(10 / 60)
    p0 = [pos0, amp0, sigma0, 0]
    popt, _, _, mesg, ierr = scipy.optimize.leastsq(
        residuals, p0, args=(phase, signal), maxfev=1000000, full_output=True)
    if ierr not in range(1, 5):
        raise RuntimeError(
            'leastsq failed: {}'.format(mesg))
    return popt


def measure_correction(phase, signal, model, fwhm):
    """ Measure the correction from binned signal
    """
    adaptive_phase, adaptive_signal = bin_adaptively(
        phase, signal, model, fwhm)

    correction = np.interp(phase, adaptive_phase, adaptive_signal)

    # Suppress the correction in the main beam
    beta = 30  # Fix the slope of the suppression to be steep
    r0 = 1e-3  # at -30dB ...
    w0 = .1  # ... suppress the correction by .1
    alpha = np.log(1 - w0 ** (1 / beta)) / np.log(r0)
    lower_limit = np.abs(model)
    asignal = np.abs(signal)
    replace = asignal < lower_limit
    lower_limit[replace] = asignal[replace]
    lower_limit /= np.amax(lower_limit)
    correction *= (1 - (lower_limit) ** alpha) ** beta

    return correction


def measure_transfer_function(phase, signal, correction, model, fwhm):
    """ Measure the desired transfer function as a ratio between
    corrected and uncorrected Fourier transforms.

    """
    tsample = 60 / phase.size
    fsignal = np.fft.rfft(signal)
    fmodel = np.fft.rfft(model)
    fcorrected = np.fft.rfft(signal - correction)
    freq = np.fft.rfftfreq(phase.size, tsample)
    # fcorrected = tf * fsignal ==>  tf = fcorrected / fsignal
    tf = fcorrected / fsignal

    binfreq, bintf = bin_transfer_function(freq, tf)
    tf_interp = np.interp(freq, binfreq, bintf)
    tf_interp[0] = 1

    # Measure roughly where to set the cut-off frequency
    # p0 = np.sum(np.abs(fsignal)**2)
    p0 = np.sum(np.abs(fmodel) ** 2)
    fcut = int(1 / (fwhm / (2 * np.pi) * 60))
    print('Starting fcut fit at {:.2f} Hz'.format(fcut))
    while fcut < 100:
        lp = lowpass(freq, fcut)
        p1 = np.sum(np.abs(fmodel * lp) ** 2)
        # break when the filter wipes only 1e-4 of the total power
        # Use the model instead of the real data not to hit the noise floor.
        if 1 - p1 / p0 < 1e-4:
            break
        fcut += .1
    print('Power threshold met at {:.2f} Hz'.format(fcut))
    while fcut < 100:
        lp = lowpass(freq, fcut)
        lpmodel = np.fft.irfft(fmodel * lp)
        ppos = np.sum(np.abs(lpmodel[lpmodel > 0]) ** 2)
        pneg = np.sum(np.abs(lpmodel[lpmodel < 0]) ** 2)
        # Break when there is negligible negative power in the
        # lowpassed model
        if pneg < 1e-6 * ppos and np.amin(lpmodel) > -1e-3:
        #if pneg < 1e-6 * ppos:
            break
        fcut += .1
    print('Ringing threshold met at {:.2f} Hz'.format(fcut))
    fcut *= 1.2  # Add a some margin
    print('Adding a low pass filter at fcut = {:.2f}'.format(fcut))

    lp = lowpass(freq, fcut)
    tf_interp *= lp

    return freq, tf, binfreq, bintf, tf_interp, fsignal, fcut


def bin_transfer_function(freq, tf):
    """ Bin the transfer function estimate
    """
    binfreq = []
    bintf = []
    istart = 1
    wbin = 1  # 1 Hz bins -- WIDE BINS WILL NOT WORK
    # wbin = .25
    while istart < tf.size:
        # if freq[istart] < wbin:
        #    istop = istart + 1
        # else:
        istop = istart + 1
        while ((istop < freq.size) and (freq[istop] - freq[istart] < wbin)
               and (freq[istop] - freq[istart] < freq[istart])):
            istop += 1
        ind = slice(istart, istop)
        binfreq.append(np.mean(freq[ind]))
        bintf.append(np.mean(tf[ind]))
        # istart = istop
        istart += int((istop - istart) // 10 + 1)
    binfreq = np.array(binfreq)
    bintf = np.array(bintf)
    # Extrapolate the lowest bin to zero frequency and normalize the
    # transfer function
    binfreq[0] = 0
    bintf /= bintf[0]
    """
    # No correction below .2Hz
    fcut = .01
    sigma = fcut
    suppress = 1 - np.exp(-.5 * (binfreq - fcut) ** 2 / sigma ** 2)
    suppress[binfreq < fcut] = 0
    bintf.real = (bintf.real - 1) * suppress + 1
    bintf.imag = bintf.imag * suppress
    """
    # First two bins should have no correction not to affect the
    # low frequency transfer function
    # bintf[1] = 1 + 0j
    return binfreq, bintf


def lowpass(freq, fcut, beta=None):
    """ Fourier transform of the low pass filter
    """
    lp = np.ones_like(freq)
    good = freq != 0
    if beta is None:
        beta = -30
    lp[freq != 0] = freq[good] ** beta / (freq[good] ** beta + fcut ** beta)
    return lp


def bin_adaptively(phase, signal, model, fwhm):
    """ Create an adaptively binned version of the signal
    """
    wbin = 2 * np.pi / phase.size
    nbin_fwhm = fwhm / wbin
    win = scipy.signal.hann(int(4 * nbin_fwhm / 2) * 2 + 1)
    win /= np.sum(win)
    smooth_signal = scipy.signal.fftconvolve(
        signal - model, win, mode='same') + model
    win = scipy.signal.hann(int(1 * nbin_fwhm / 2) * 2 + 1)
    win /= np.sum(win)
    less_smooth_signal = scipy.signal.fftconvolve(
        signal - model, win, mode='same') + model
    # combine the corrections
    beta = 30  # Fix the slope of the suppression to be steep
    alpha = np.log(1 - .1 ** (1 / beta)) / np.log(1e-6)
    w = (1 - (model / np.amax(model)) ** alpha) ** beta
    smooth_signal = (1 - w) * less_smooth_signal + w * smooth_signal
    # Unsmoothed signal down to -10dB
    alpha = np.log(1 - .1 ** (1 / beta)) / np.log(1e-1)
    w = (1 - (model / np.amax(model)) ** alpha) ** beta
    smooth_signal = (1 - w) * signal + w * smooth_signal
    # Smooth out the binned signal completely at a distance
    smooth_signal *= np.exp(-.5 * (phase - np.pi) ** 2 / (10 * fwhm) ** 2)
    return phase, smooth_signal


def plot_results(
        det, phase, signal, gaussian_model, model, correction,
        tf_interp, freq, tf, binfreq, bintf, fsignal, fwhm, fcut):
    """ Final plotting for all results
    """
    nbin = signal.size
    wbin = 2 * np.pi / nbin
    # Plot the lowpass filter
    plt.figure(figsize=[18, 12])
    fmodel = np.fft.rfft(model)
    fgaussian = np.fft.rfft(gaussian_model)
    fcorrected = fsignal * tf_interp
    lp = lowpass(freq, fcut)
    norm = 1 / np.amax(np.abs(fmodel) ** 2)
    plt.plot(freq, norm * np.abs(fsignal) ** 2, '.', label='signal')
    plt.plot(freq, norm * np.abs(fcorrected) ** 2, '.', label='filtered signal')
    plt.plot(freq, norm * np.abs(fmodel) ** 2, '.', label='model')
    plt.plot(freq, norm * np.abs(fgaussian) ** 2, '.', label='gaussian')
    plt.plot(freq, lp ** 2, '-', label='low-pass')
    plt.plot(freq, np.abs(tf_interp) ** 2, '-', label='full filter')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    ax.set_xlim([0, 100])
    ax.set_ylim([1e-10, 3])
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.axvline(fcut)
    ax.set_title(
        '{} : nbin = {}, wbin = {:.3f}". fcut = {:.2f} Hz. FWHM = {:.3f}"'
        ''.format(det, nbin, wbin / arcmin, fcut, fwhm / arcmin))
    plt.savefig('filters_{}.png'.format(det))

    plt.figure(figsize=[18, 12])
    plt.suptitle(det)
    # plt.plot(binned_phase_lowress, binned_signal_lowress, 'o')
    plt.plot(center_phase(phase), gaussian_model, lw=4, label='Gaussian fit')
    plt.plot(center_phase(phase), model, lw=4, label='Modified gaussian fit')
    # plt.plot(phase, model, label='Gaussian fit')

    plt.plot(center_phase(phase), correction, lw=4, label='Correction')

    signal2 = np.fft.irfft(tf_interp * fsignal, nbin)
    plt.plot(center_phase(phase), signal, 'o', label='phase-binned signal')
    plt.plot(center_phase(phase), signal2, 'o', label='filtered signal')
    adaptive_phase, adaptive_signal = bin_adaptively(
        phase, signal, model, fwhm)
    adaptive_phase2, adaptive_signal2 = bin_adaptively(
        phase, signal2, model, fwhm)
    plt.plot(center_phase(adaptive_phase), adaptive_signal, '-',
             label='binned signal')
    plt.plot(center_phase(adaptive_phase2), adaptive_signal2, '-',
             label='filtered & binned signal')
    ax = plt.gca()
    ax.axhline(0, color='k')
    ax.set_xlabel('Phase [arc min]')
    ax.set_ylabel('Signal')
    amp = 10 * fwhm / arcmin; ax.set_xlim([-amp, amp])
    amp = 1e-3; ax.set_ylim([-amp, amp])
    plt.legend(loc='best')
    plt.savefig('signal_{}.png'.format(det))

    """
    plt.figure()
    plt.suptitle(det)
    plt.subplot(2, 1, 1)
    plt.plot(freq, tf.real, '.', label='Full TF')
    # plt.plot(freq, tf2.real, '.')
    plt.plot(binfreq, bintf.real, '-o', label='binned TF')
    plt.plot(freq, tf_interp.real, '-', label='lowpassed TF')
    plt.gca().set_xlim([0, 200])
    plt.gca().set_ylim([-5, 5])
    plt.gca().axhline(0, color='k')
    plt.gca().axhline(1, color='k', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.plot(freq, tf.imag, '.', label='Full TF')
    # plt.plot(freq, tf2.imag, '.')
    plt.plot(binfreq, bintf.imag, '-o', label='binned TF')
    plt.plot(freq, tf_interp.imag, '-', label='lowpassed TF')
    plt.gca().set_xlim([0, 200])
    plt.gca().set_ylim([-5, 5])
    plt.gca().axhline(0, color='k')
    plt.legend(loc='best')
    """

    #plt.show()
    return


def process_hdu(hdu):
    """ Process the planet observations in the HDU into a filter function,
    including a possible lowpass filter.

    """
    det = hdu.header['extname']
    print('Processing', det)
    all_time = hdu.data.field('time').ravel()
    all_signal = hdu.data.field('signal').ravel()
    all_phase = hdu.data.field('phase').ravel()
    ring = hdu.data.field('ring').ravel()
    all_dtheta = hdu.data.field('dtheta').ravel()
    all_dphi = hdu.data.field('dphi').ravel()
    all_pntflag = hdu.data.field('pntflag').ravel()
    ring_lims = np.argwhere(np.diff(ring) != 0).ravel() + 1

    phasevec = []
    signalvec = []
    weightvec = []
    fwhmvec = []
    istart = 0
    for iring in range(ring_lims.size + 1):
        istart = process_ring(
            iring, all_time, all_signal, all_phase, ring, all_dtheta, all_dphi,
            all_pntflag, ring_lims, istart, phasevec, signalvec, weightvec,
            fwhmvec)

    # Choose sufficient bin size
    fwhm = np.median(fwhmvec)
    nbin = 2 ** 10
    while True:
        wbin = 2 * np.pi / nbin
        if wbin < fwhm / 20:
            break
        nbin *= 2

    print('Will use {} bins of {:.3f}" width'.format(nbin, wbin / arcmin))

    # average the scaled and accumulated signal

    phasevec = np.hstack(phasevec)
    signalvec = np.hstack(signalvec)
    weightvec = np.hstack(weightvec)
    phase, signal = bin_signal(phasevec, signalvec, weightvec, nbin, wbin)

    # Fit the data with a gaussian model

    subtract_dipo(phase, signal)
    amp, pos, sigma = fit_gaussian(phase, signal)
    gaussian_model = gaussian(amp, pos, sigma, phase)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    print('amp : {:8.4f}, pos : {:8.4f}\', FWHM : {:8.4f}"'
          ''.format(amp * 1e3, np.degrees(pos - np.pi) * 3600,
                    np.degrees(fwhm) * 60))

    # Create a more precise model using a modified Gaussian and actual
    # bin values in the beam center

    ind_fit = np.abs(phase - np.pi) < np.radians(1)
    intense = np.abs(signal) > .1
    ind_fit[intense] = False
    popt = fit_modgaussian(phase[ind_fit], signal[ind_fit])
    model = modgaussian(popt, phase)
    model[intense] = signal[intense]

    # Create the desired correction as a difference between the smooth,
    # symmetric model and the adaptively binned signal

    correction = measure_correction(phase, signal, model, fwhm)

    (freq, tf, binfreq, bintf, tf_interp, fsignal,
     fcut) = measure_transfer_function(phase, signal, correction, model, fwhm)

    save_filter(det, freq, tf_interp, fcut)

    plot_results(det, phase, signal, gaussian_model, model, correction,
                 tf_interp, freq, tf, binfreq, bintf, fsignal, fwhm, fcut)
    return


def save_filter(det, freq, tf, fcut):
    """
    """
    filterfile = 'total_filter_{}.txt'.format(det)
    with open(filterfile, 'w') as fout:
        for i in range(freq.size):
            if freq[i] > 100:
                break
            fout.write('{} {} {}\n'.format(freq[i], tf[i].real, tf[i].imag))
    print('Filter saved in {}'.format(filterfile))

    filterfile = 'lowpass_filter_{}.txt'.format(det)
    lp = lowpass(freq, fcut)
    with open(filterfile, 'w') as fout:
        for i in range(freq.size):
            if freq[i] > 100:
                break
            fout.write('{} {} {}\n'.format(freq[i], lp[i], 0.0))
    print('Filter saved in {}'.format(filterfile))
    return


def bin_signal(phase, signal, weight, nbin, wbin):
    print('sorting phases')
    ind = np.argsort(phase)
    phase[:] = phase[ind]
    signal[:] = signal[ind]
    weight[:] = weight[ind]

    print('binning and deglitching signal')
    nbad_tot = 0
    binned_phase = np.zeros(nbin)
    binned_signal = np.zeros(nbin)
    binned_weight = np.zeros(nbin)
    istart = 0
    binlims = np.searchsorted(phase, np.arange(nbin) * wbin)
    binlims[0] = 0
    for ibin in range(nbin):
        istart = binlims[ibin]
        if ibin + 1 < nbin:
            istop = binlims[ibin + 1]
        else:
            istop = signal.size
        if istop == istart:
            continue
        ind = slice(istart, istop)
        binphase = phase[ind]
        binsignal = signal[ind]
        binweight = weight[ind]
        med = np.median(binsignal)
        chisq = (binsignal - med) ** 2 * binweight
        # are we in a signal-dominated bin?
        if np.median(chisq) > 10:
            binrms = np.std(binsignal - med)
            chisq = (binsignal - med) ** 2 / binrms ** 2
            good = np.sqrt(chisq) < 3
        else:
            good = np.sqrt(chisq) < 2
        ngood = np.sum(good)
        if ngood == 0:
            raise RuntimeError()
        nbad_tot += istop - istart - ngood
        binned_weight[ibin] = np.sum(binweight[good])
        binned_phase[ibin] = (np.sum(binphase[good] * binweight[good]) /
                              binned_weight[ibin])
        binned_signal[ibin] = (np.sum(binsignal[good] * binweight[good]) /
                               binned_weight[ibin])
    print('Discarded {} ({:.2f} %) outliers'.format(
        nbad_tot, nbad_tot * 100 / signal.size))
    return binned_phase, binned_signal


def subtract_dipo(phase, signal):
    """ Rough fit for an offset and a dipole excluding the samples
    near the planet.
    """
    good = np.abs(phase - np.pi) > np.radians(1)
    rms = np.std(signal[good])
    ngood = np.sum(good)
    templates = np.vstack([np.ones(ngood), np.cos(phase[good]),
                           np.sin(phase[good])])
    invcov = np.dot(templates, templates.T)
    cov = np.linalg.inv(invcov)
    proj = np.dot(templates, signal[good])
    coeff = np.dot(cov, proj)
    print('offset : {:8.4} uK, cos : {:8.4f} uK, sin : {:8.4f} uK'.format(
        coeff[0] * 1e6, coeff[1] * 1e6, coeff[2] * 1e6))
    signal -= coeff[0] + coeff[1] * np.cos(phase) + coeff[2] * np.sin(phase)
    return rms


def process_ring(
        iring, all_time, all_signal, all_phase, ring, all_dtheta, all_dphi,
        all_pntflag, ring_lims, istart, phasevec, signalvec, weightvec,
        fwhmvec):
    """ Scale and translate one ring of data.
    """
    ring_number = ring[istart]
    print('Processing ring # {} ( {} / {} )'.format(
        ring_number, iring + 1, ring_lims.size + 1))
    if iring < ring_lims.size:
        istop = ring_lims[iring]
    else:
        istop = all_time.size
    ind = slice(istart, istop)
    good = all_pntflag[ind] == 0
    # time = all_time[ind][good]
    signal = all_signal[ind][good]
    phase = all_phase[ind][good]
    dtheta = all_dtheta[ind][good]
    dphi = all_dphi[ind][good]

    i0 = np.argmin(dtheta ** 2 + dphi ** 2)
    phase0 = phase[i0]
    phase += np.pi - phase0
    phase[phase < 0] += 2 * np.pi
    phase[phase > 2 * np.pi] -= 2 * np.pi

    rms = subtract_dipo(phase, signal)

    amp, pos, sigma = fit_gaussian(phase, signal)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    print('amp : {:8.4f} mK, pos : {:8.4f}\', FWHM : {:8.4f}"'
          ''.format(amp * 1e3, np.degrees(pos - np.pi) * 3600, fwhm / arcmin))

    phasevec.append(phase - (pos - np.pi))
    signalvec.append(signal / amp)
    weightvec.append(np.ones(signal.size, dtype=np.float64) * (amp / rms) ** 2)
    fwhmvec.append(fwhm)

    return istop


def main():
    fname = sys.argv[1]

    hdulist = pf.open(fname)
    for hdu in hdulist[1:]:
        process_hdu(hdu)


if __name__ == "__main__":


    main()
