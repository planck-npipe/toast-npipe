# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
# from memory_profiler import profile

import os
import pickle
from toast_planck.preproc_modules.filters import flagged_running_average
from toast_planck.reproc_modules.destripe_tools import fast_hit_binning

from scipy.constants import degree, arcmin
from scipy.signal import medfilt, convolve

from toast import qarray as qa
import toast
from toast._libtoast import filter_polynomial as polyfilter
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.utils import Logger, Environment

from toast_planck.imo import IMO
from toast_planck.preproc_modules import (
    Transf1, GainCorrector, MapSampler, TauDeconvolver, GapFiller,
    JumpCorrector, Differencer, LineRemover, LineRemoverLFI, Despiker,
    InputEstimator, Dipoler, PlanetFlagger, RingMasker, LFINLCorrector,
    SignalEstimator, GlitchRemoverLFI, GlitchFlagger)
from toast_planck.utilities import (
    bolo_to_pnt, read_gains, ADU2Volt, ADU2Volt_post953,
    diode_gains, to_diodes)

import numpy as np
import signal as signals
import toast.tod as tt


# import warnings
# warnings.filterwarnings('error')
class DespikerError(Exception):
    pass


class TimeOut(Exception):
    pass


def timeouthandler(signum, frame):
    raise TimeOut('Preproc TimeOut')


class OpticalRingJob(object):
    """ A simple container for all the information required to process
    one optical ring.
    """

    def __init__(
            self, adc_iteration, ringinfo, ring_number, ring_start, ring_stop,
            ind_from, ind_to, iring, startsample, stopsample, det, fwhm, tme,
            sig, flg, fsample, ringflg, pntflg, ssoflg, phse, q, v, iquw,
            drk1, drk2, drkflg1, drkflg2, phse4k, prty, gains,
            weight0, weight1, lowpass0, lowpass1, previously_failed_rings,
            spike_params):
        self.adc_iteration = adc_iteration
        self.ringinfo = ringinfo
        self.ring_number = ring_number
        self.ring_start = ring_start
        self.ring_stop = ring_stop
        self.ind_from = ind_from
        self.ind_to = ind_to
        self.iring = iring
        self.startsample = startsample
        self.stopsample = stopsample
        self.det = det
        self.fwhm = fwhm
        self.tme = tme
        self.sig = sig
        self.flg = flg
        self.fsample = fsample
        self.ringflg = ringflg
        self.pntflg = pntflg
        self.ssoflg = ssoflg
        self.phse = phse
        self.q = q
        self.v = v
        self.iquw = iquw
        self.drk1 = drk1
        self.drk2 = drk2
        self.drkflg1 = drkflg1
        self.drkflg2 = drkflg2
        self.phse4k = phse4k
        self.prty = prty
        self.gains = gains
        self.weight0 = weight0
        self.weight1 = weight1
        self.lowpass0 = lowpass0
        self.lowpass1 = lowpass1
        self.previously_failed_rings = previously_failed_rings
        self.spike_params = spike_params


class OpticalRingJobOutput(object):
    """ A simple container for all information produced by optical
    ring processing
    """

    def __init__(
            self, nlcorrected, sig, flg, ssoflg, maskflg, error, iring,
            ind_from, ind_to, ringinfo, ring_start, ring_stop, ring_number):
        self.nlcorrected = nlcorrected
        self.sig = sig
        self.flg = flg
        self.ssoflg = ssoflg
        self.maskflg = maskflg
        self.error = error
        self.iring = iring
        self.ind_from = ind_from
        self.ind_to = ind_to
        self.ringinfo = ringinfo
        self.ring_start = ring_start
        self.ring_stop = ring_stop
        self.ring_number = ring_number


class OpPreproc(toast.Operator):
    """
    Operator for timestream pre-processing.

    Args:
        input (str): if None, read input TOD, otherwise the name of the
            cached data.
    """

    def __init__(
            self, imofile, freq,
            bg_map_path=None, bg_pol=False, bg_has_dipole=False, bg_nside=1024,
            detmask=None, pntmask=2, ssomask=None, maskfile=None,
            maskfile_adc=None, nbin_phase=3000, jump_filter_len=40000,
            preproc_dark=True, preproc_common=True, calfile=None,
            measure_ADC=False, nadc_iter=1, adc_correction=None,
            margin=0, effdir_out=None, flag_planets=True,
            planet_flag_radius=2.0, bad_rings=None, out='.',
            jump_threshold=4.0,
            timeout=120.0, timeout_intermediate=60.,
            nphase4k=18, deltaADU=0.1, single_diode=None,
            shdet_mode=False, global_phase_shift=0., g0=None, v0=None,
            tabulated_transfer_function=None, verbose=True,
            intense_threshold=0.01,
            async_time=1000., async_time_intermediate=800):

        self.async_time = async_time
        self.async_time_intermediate = async_time_intermediate
        self.intense_threshold = intense_threshold
        self._imo = IMO(imofile)
        self._freq = freq
        self._lfi_mode = np.int(freq) < 100
        self._bg_map_path = bg_map_path
        self._bg_pol = bg_pol
        self._bg_has_dipole = bg_has_dipole
        self._bg_nside = bg_nside
        self._detmask = detmask
        self._pntmask = pntmask
        self._ssomask = ssomask
        self._maskfile = maskfile
        self._maskfile_adc = maskfile_adc
        self._nbin_phase = nbin_phase
        self._jump_filter_len = jump_filter_len
        self._preproc_dark = preproc_dark
        self._preproc_common = preproc_common
        self._calfile = calfile
        self._tabulated_transfer_function = tabulated_transfer_function
        self._global_phase_shift = global_phase_shift
        if self._lfi_mode:
            self._measure_ADC = False
            self._nadc_iteration = 1
            self._adc_correction = None
        else:
            self._measure_ADC = measure_ADC
            self._nadc_iteration = nadc_iter
            self._adc_correction = adc_correction
            if self._nadc_iteration > 1:
                self._measure_ADC = True
        self._margin = margin
        self._effdir_out = effdir_out
        self._flag_planets = flag_planets
        self._planet_flag_radius = planet_flag_radius  # in FWHM
        self._bad_rings = bad_rings
        self._out = out
        self._jump_threshold = jump_threshold
        self._timeout = timeout
        self._timeout_intermediate = min(timeout, timeout_intermediate)
        self._nphase4k = nphase4k
        self._deltaADU = deltaADU
        self._single_diode = single_diode
        self._verbose = verbose
        self._shdet_mode = shdet_mode
        self._g0 = g0
        self._v0 = v0
        self.local_starts = None
        self.local_stops = None
        self.nring = None
        self.ring_offset = None
        self.nsamp = None
        self.comm = None
        self.rank = None
        self.ntask = None
        # For _test_despiker_gaps:
        self.last_ring_number = -1
        self.last_sorted = None
        self.last_tol = None
        self.last_dxmax0 = None
        super().__init__()

    def remove_glitches(
            self, det, sig, flg, ssoflg, pntflg, bg, dipo, phse,
            adc_iteration, zeroflg, ring_number, maskflg, ind_from, fwhm):
        """Apply despike to the signal

        """
        # Use the glitch flagger to build a signal estimate.
        timer = Timer()
        timer.start()
        flagger = GlitchFlagger(fwhm=fwhm, twice=True, threshold=3.0,
                                wkernel=5)
        sig0 = sig.copy()
        flg0, flg_intense, signal_estimate0 = flagger.flag_glitches(
            sig, zeroflg, phase=phse, pntflag=pntflg, dark=False)
        good = (zeroflg + pntflg) == 0
        frac = np.sum(flg0[good] != 0) / np.sum(good)
        intense_frac = np.sum(flg_intense) / flg_intense.size
        timer.stop()
        rms = np.std((sig - signal_estimate0)[zeroflg + pntflg + flg0 == 0])
        timer.report(
            '{:4} : Ring {:5} glitch flagging.'
            ' frac = {:.3f}, intense_frac = {}, RMS = {}'.format(
                self.rank, ring_number, frac, intense_frac, rms)
        )

        # Then run despike to estimate the glitch tails

        bolo_id = bolo_to_pnt(det)
        factth = {100: 1.0, 143: 1.0, 217: 1.0,
                  353: 1.0, 545: 5.0, 857: 15.0}[int(self._freq)]

        critcut = 4.0

        despiker = Despiker(
            self._imo, det, bolo_id, critcut=critcut, verbose=False,
            factth=factth)

        sig0 -= signal_estimate0
        timer = Timer()
        timer.start()
        sig1, flg1 = despiker.despike(ring_number, sig0, zeroflg + flg_intense)
        timer.stop()
        if sig1 is None:
            raise DespikerError(
                'Despiker failed in {:.1f}s.'.format(timer.seconds()))
        glitch_estimate = sig0 - sig1
        sig0 += signal_estimate0
        sig1 += signal_estimate0
        flg1 = flg1 != 0
        good = zeroflg + pntflg == 0
        frac = np.sum(flg1[good] != 0) / np.sum(good)
        rms = np.std((sig1 - signal_estimate0)[zeroflg + pntflg + flg1 == 0])
        timer.report(
            '{:4} : Ring {:5} despike'
            ' frac = {:.3f}. RMS = {}'.format(
                self.rank, ring_number, frac, rms),
        )
        
        """
        # DEBUG begin
        import matplotlib.pyplot as plt
        import pdb
        t = np.arange(sig.size) / 180.3737
        good = pntflg + zeroflg == 0
        offset = np.median(signal_estimate0[good]) - np.median(sig[good])
        plt.plot(t[good], (sig0 - signal_estimate0)[good] + offset, 'o',
                 label='all data', ms=6)
        good = pntflg + zeroflg + flg1 == 0
        plt.plot(t[good], (sig1 - signal_estimate0)[good] + offset, 'o',
                 label='unflagged (despike)', ms=4)
        good = flg0 == 0
        plt.plot(t[good], (sig0 - signal_estimate0)[good] + offset, 'o',
                 label='unflagged (glitch flagger)', ms=2)
        plt.plot(t, glitch_estimate, '-', label='glitch estimate', lw=2)
        plt.plot(t, signal_estimate0, '-', label='signal estimate', lw=2)
        plt.legend(loc='best')
        pdb.set_trace()
        # DEBUG end
        """

        sig = sig1
        flg = flg1

        # For intense signals, glitch flagging works better than despike

        sig[flg_intense] = sig0[flg_intense]
        flg[flg_intense] = flg0[flg_intense]

        # Build a signal estimate for subsequent modules

        signal_estimate = bg + dipo
        not_intense = flg_intense + pntflg == 0
        signal_estimate0 -= np.median(signal_estimate0[not_intense])
        signal_estimate0 += np.median(signal_estimate[not_intense])
        signal_estimate[flg_intense] = signal_estimate0[flg_intense]

        gapflg = self._test_despiker_gaps(
            flg, pntflg, ssoflg, phse, 0.5 * arcmin, ring_number)
        if gapflg is not None:
            sig[gapflg] = sig0[gapflg]
            flg[gapflg] = flg0[gapflg]
            flg_intense[gapflg] = True

        """
        # DEBUG begin
        import matplotlib.pyplot as plt
        import pdb
        for x in [np.arange(sig.size), phse]:
            plt.figure()
            good = pntflg + zeroflg == 0
            plt.plot(x[good], sig0[good], 'o', ms=6, label='all data')
            good = flg + pntflg + zeroflg == 0
            plt.plot(x[good], sig[good], 'o', ms=4,
                     label='unflagged (despike)')
            if gapflg is not None:
                good = pntflg + zeroflg + flg == 0
                good[gapflg == 0] = 0
                plt.plot(x[good], sig[good], 'o', ms=4,
                         label='gapflg', color='purple')
            offset = np.median((signal_estimate0 - sig)[good])
            plt.plot(x, signal_estimate0 - offset, 'ko', ms=2,
                     label='signal_estimate')
            plt.legend(loc='best')
        pdb.set_trace()
        # DEBUG end
        """

        frac = np.sum(flg[ind_from]) / len(flg[ind_from])

        flg[zeroflg] = True

        if maskflg is not None:
            flg_intense[maskflg] = True
        if ssoflg is not None:
            flg_intense[ssoflg] = True

        return (sig, flg, frac, glitch_estimate, flg_intense, signal_estimate,
                signal_estimate0)

    def process_dark_bolo(self, det, darktod, darkflag, tod, fsample,
                          timestamps, pntflag, phase, ringmasker, parity):
        """Process dark bolometer

        Apply all relevant preprocessing steps to a dark bolometer

        """
        self.comm.barrier()
        if self.rank == 0:
            print('Setting up processing for {}'.format(det),
                  flush=True)
        loop_timer = Timer()
        loop_timer.start()

        dtype = [
            ('ring_number', np.int), ('start_time', np.float),
            ('start_index', np.int),
            ('glitch_frac', np.float), ('njump', np.int),
            ('total_rms', np.float), ('signal_rms', np.float),
            ('noise_rms', np.float), ('noise_offsetND', np.float),
            ('gain0', np.float), ('gain1', np.float),
            ('rms0', np.float), ('rms1', np.float),
            ('total_frac', np.float), ('sso_frac', np.float),
            ('pnt_frac', np.float), ('mask_frac', np.float),
            ('bad_frac', np.float), ('minval', np.float), ('maxval', np.float),
            ('mean', np.float), ('median', np.float), ('rms', np.float),
            ('failed', np.bool), ('outlier', np.bool),
            ('despikererror', np.bool), ('timeout', np.bool)]
        for linefreq in [10, 20, 30, 40, 50, 60, 70, 80,
                         17, 16, 25, 43, 46, 48, 57]:
            dtype += [('cos_ampl_{:02}Hz'.format(linefreq), np.float),
                      ('sin_ampl_{:02}Hz'.format(linefreq), np.float)]

        ringinfos = np.recarray([self.nring, ], dtype=dtype)

        local_ring_ranges = {}

        ringinfos.fill(0)

        bolo_id = bolo_to_pnt(det)

        transf1 = Transf1()
        gaincorrector = GainCorrector(self._imo, bolo_id)
        gapfiller = GapFiller()
        # jumpcorrector = JumpCorrector(
        #    self._jump_filter_len, nbin=self._nbin_phase,
        #    threshold=self._jump_threshold)
        lineremover = LineRemover(fsample=fsample)

        self.despiker = Despiker(self._imo, det, bolo_id, verbose=False)

        self.comm.barrier()
        if self.rank == 0:
            print('Processing {}'.format(det), flush=True)

        darktod_out = np.zeros(self.nsamp + 2 * self._margin, dtype=np.float64)
        darkflag_out = np.zeros(self.nsamp + 2 * self._margin, dtype=np.uint8)

        # Initialize to sentinel values
        darktod_out[:] = np.nan
        darkflag_out[:] = 1

        ring_number = self.ring_offset - 1

        signals.signal(signals.SIGALRM, timeouthandler)

        for iring, (ring_start,
                    ring_stop) in enumerate(zip(self.local_starts,
                                                self.local_stops)):
            ring_timer = Timer()
            ring_timer.start()
            # Set a timeout to be raised.  We'll cancel it at
            # the end of the loop.
            signals.alarm(int(self._timeout))
            ring_number += 1

            if self._verbose:
                print('{:4} : Processing ring {:4} ({} / {})'
                      ''.format(self.rank, ring_number, iring + 1,
                                self.nring), flush=True)

            ringinfo = ringinfos[iring]
            ringinfo.ring_number = ring_number

            startsample = (ring_start + tod.globalfirst +
                           tod.local_samples[0] - self._margin)
            stopsample = (startsample + (ring_stop - ring_start) +
                          2 * self._margin)

            ringinfo.start_index = startsample + self._margin

            # Slice that includes margins
            ind = slice(ring_start, ring_stop + 2 * self._margin)
            # Slices without margins
            # ring buffer slice
            ind_from = slice(self._margin,
                             ring_stop - ring_start + self._margin)
            # total array slice
            ind_to = slice(ring_start + self._margin, ring_stop + self._margin)
            local_ring_ranges[ring_number] = ind_to

            try:
                tme = timestamps[ind]
                ringinfo.start_time = tme[self._margin]
                drk = darktod[ind].copy()
                flg = darkflag[ind] != 0  # casting to a boolean flag

                zeroflg = drk == 0
                nbad = np.sum(zeroflg)
                if nbad != 0:
                    print('{:4} : Ring {:5} WARNING: found {} raw, '
                          'zero-valued samples.'.format(
                              self.rank, ring_number, nbad))
                    flg[zeroflg] = True

                pntflg = np.zeros_like(pntflag[ind])
                pntflg[:self._margin] = True
                pntflg[-self._margin:] = True

                if ringmasker is not None:
                    flg += ringmasker.get_mask(tme, det)

                drk = transf1.convert(drk, tme, det.replace('-', ''))

                prty = parity[ind]

                baselineND, flag_base = self._get_baseline(drk, flg)

                drk = (drk - baselineND) * (1. - 2. * prty)

                flg[flag_base] = True

                ringinfo.noise_offsetND = np.median(baselineND)

                drk = gaincorrector.correct(drk, flg)

                timer = Timer()
                timer.start()
                drk, flg = self.despiker.despike(ring_number, drk, zeroflg)
                if drk is None:
                    raise DespikerError('Despiker failed.')
                """
                flagger = GlitchFlagger()
                flg, _ = flagger.flag_glitches(drk, zeroflg, dark=True)
                """
                good = zeroflg == 0
                frac = np.sum(flg[good] != 0) / np.sum(good)
                timer.stop()
                timer.report('{:4} : Ring {:5} Dark despike'
                      ' frac = {:.3f}'.format(
                          self.rank, ring_number, np.sum(flg != 0) / flg.size))
                """
                # DEBUG begin
                import matplotlib.pyplot as plt
                import pdb
                t = np.arange(drk.size) / 180.3737
                plt.plot(t, drk, '.')
                plt.plot(t[flg == 0], drk[flg == 0])
                pdb.set_trace()
                # DEBUG end
                """

                # The first sample before a detected glitch is often
                # compromised.
                # flg = flg != 0
                # flg[np.roll(flg, -1)] = True
                # flg[zeroflg] = True
                # frac = np.sum(flg) / len(flg)
                ringinfo.glitch_frac = frac

                drk = gapfiller.fill(
                    drk, None, flg, det, ring_number, dark=True,
                    signal_estimate=None, pntflag=pntflg)

                # Jump correction can leave offsets at the ring boundary
                # drk, flg, njump = jumpcorrector.correct(drk, flg, None,
                #                                        dark=True)
                njump = 0
                ringinfo.njump = njump

                timer = Timer()
                timer.start()
                drk, flg, _, frequencies, amplitudes = lineremover.remove(
                    startsample, stopsample, drk, flg, None, dark=True,
                    pntflag=pntflg)
                timer.stop()
                if self._verbose:
                    timer.report('{:4} : Ring {:5} line-remove'
                          ''.format(self.rank, ring_number))
                for ii, fff in enumerate(frequencies):
                    ringinfo['cos_ampl_{}Hz'.format(fff)] = amplitudes[ii][0]
                    ringinfo['sin_ampl_{}Hz'.format(fff)] = amplitudes[ii][1]

                # Collect last metadata

                good = flg == 0
                good[:self._margin] = False
                good[-self._margin:] = False
                if np.sum(good) > 0:
                    ringinfo.total_rms = np.std(drk[good])
                if np.sum(good) > 0:
                    cleaned = drk.copy()
                    polyfilter(
                        5,
                        flg.astype(np.uint8),
                        [cleaned],
                        np.array([0]),
                        np.array([drk.size]),
                    )
                    ringinfo.noise_rms = np.std(cleaned[good])
                    del cleaned
                    ringinfo.minval = np.amin(drk[good])
                    ringinfo.maxval = np.amax(drk[good])
                    ringinfo.mean = np.mean(drk[good])
                    ringinfo.median = np.median(drk[good])
                    ringinfo.rms = np.std(drk[good])

                ringinfo.total_frac = np.sum(flg[ind_from] != 0,
                                             dtype=np.float64) \
                    / len(flg[ind_from])
                if pntflg is not None:
                    ringinfo.pnt_frac = np.sum(
                        pntflg[ind_from] != 0, dtype=np.float64) \
                        / len(flg[ind_from])

                darktod_out[ind_to] = drk[ind_from]
                # Casting to uint8
                darkflag_out[ind_to] = np.uint8(1) * flg[ind_from]

                # Cancel the timeout
                signals.alarm(0)
            except Exception as e:
                # raise # DEBUG
                if isinstance(e, DespikerError):
                    ringinfo.despikererror = True
                if isinstance(e, TimeOut):
                    ringinfo.timeout = True
                else:
                    # Cancel the timeout
                    signals.alarm(0)
                # The preprocessing failed. Flag the entire ring
                print('{:4} : Ring {:5} of {} preprocessing failed: "{}" '
                      'Flagging the entire ring.'.format(
                          self.rank, ring_number, det, e), flush=True)
                darktod_out[ind_to] = np.nan
                darkflag_out[ind_to] = np.uint8(1)
                ringinfo.failed = True

            ring_timer.stop()
            if self._verbose:
                ring_timer.report('{:4} : Ring {:5}'.format(
                    self.rank, ring_number))

        darktod[:] = darktod_out
        darkflag[:] = darkflag_out
        del darktod_out
        del darkflag_out

        # Process the ring metadata and add flags accordingly

        all_ringinfos = self.comm.allgather(ringinfos)
        all_ringinfos = np.hstack(all_ringinfos).view(np.recarray)

        outliers = self._get_outliers(
            all_ringinfos, ['glitch_frac', 'noise_rms', 'rms0', 'gain0'],
            jumpmax=3, tol=5., wmean=1001, werr=1001)

        for outlier in outliers:
            if outlier in local_ring_ranges:
                # Flag the entire ring
                ind = local_ring_ranges[outlier]
                darkflag[ind] |= np.uint8(8)

        if self.rank == 0:
            print('Flagged {} outlier rings for {}.'.format(len(outliers), det))
            fn_out = os.path.join(self._out, 'ring_info_{}.pck'.format(det))
            pickle.dump(all_ringinfos, open(fn_out, 'wb'), protocol=2)

        loop_timer.stop()
        if self.rank == 0:
            loop_timer.report('Process {}'.format(det))

        return

    def check_spinrate(self, time, phase):
        """
        Ensure that the spin rate is constant.

        Return flags that highlight times when the spinrate drifts.

        """
        timestep = np.diff(time)
        spinstep = np.diff(phase)
        target = 2 * np.pi / 60.
        tol = 2 * np.pi * .01
        spinstep[spinstep > 2 * np.pi - tol] -= 2 * np.pi
        spinstep[spinstep < -2 * np.pi + tol] += 2 * np.pi
        spinrate = np.abs(spinstep)
        good = timestep != 0
        spinrate[good] /= timestep[good]
        ngood = np.sum(good)
        if ngood == 0:
            mean_spinrate = 0
        else:
            mean_spinrate = np.mean(spinrate[good])
        drift = spinrate / target
        rtol = .01
        good[np.abs(drift - 1) > rtol] = False
        ngood = np.sum(good)
        if ngood == 0:
            mean_good_spinrate = 0
        else:
            mean_good_spinrate = np.mean(spinrate[good])
        bad = np.logical_not(good)
        spinflag = np.zeros(phase.size, dtype=np.bool)
        spinflag[:-1][bad] = True
        spinflag[1:][bad] = True
        return spinflag, mean_spinrate, mean_good_spinrate

    def send_optical_ring(self, helper, job):
        """ This may be too much data to serialize on one go...
        """
        self.comm.send(job, dest=helper, tag=self.rank)
        return

    def recv_optical_ring(self, helper):
        """ This may be too much data to serialize on one go...
        """
        jobout = self.comm.recv(source=helper, tag=helper)
        return jobout

    def process_remote_optical_ring(self, helpee):
        job = self.comm.recv(source=helpee, tag=helpee)
        print('{:4} : Ring {:5} received from {}'
              ''.format(self.rank, job.ring_number, helpee), flush=True)
        jobout = self.process_optical_ring(job)
        self.comm.send(jobout, dest=helpee, tag=self.rank)
        return

    def add_optical_ring(self, jobout, signal, signal_out, flags_out, ringinfos,
                         previously_failed_rings):
        if jobout.error is None:
            if jobout.nlcorrected is not None:
                if jobout.iring == self.nring - 1:
                    ntail = 2 * self._margin
                else:
                    ntail = 0
                ind_in = slice(
                    0, jobout.ring_stop - jobout.ring_start + ntail)
                ind_out = slice(jobout.ring_start,
                                jobout.ring_stop + ntail)
                signal[ind_out] = jobout.nlcorrected[ind_in]
            signal_out[jobout.ind_to] = jobout.sig[jobout.ind_from]
            flags_out[jobout.ind_to] = 0
            flags_out[jobout.ind_to] |= np.uint8(1) \
                * jobout.flg[jobout.ind_from]
            if jobout.ssoflg is not None:
                flags_out[jobout.ind_to] |= np.uint8(2) \
                    * jobout.ssoflg[jobout.ind_from]
            if jobout.maskflg is not None:
                flags_out[jobout.ind_to] |= np.uint8(4) \
                    * jobout.maskflg[jobout.ind_from]
        else:
            previously_failed_rings.add(jobout.ring_number)
            signal_out[jobout.ind_to] = np.nan
            flags_out[jobout.ind_to] = np.uint8(1)
        ringinfos[jobout.iring] = jobout.ringinfo
        return

    def assign_helpers(self, nring_left):
        """ Assign processes that have no rings left to help processes
        that have more than one ring left.
        """
        nring_left_all = np.array(self.comm.allgather(nring_left))
        if np.sum(nring_left_all) == 0:
            # Special signal: all rings processed
            return None, None
        rank_all = np.arange(self.comm.size, dtype=np.int)
        available = rank_all[nring_left_all == 0]
        need_help = rank_all[nring_left_all > 1]
        nring_left_tot = np.sum(nring_left_all)
        if self.rank == 0:
            print('Assigning helpers. {} / {} helpers available. {} need help.'
                  ' {} / {} rings not processed'.format(
                      len(available), rank_all.size, len(need_help),
                      nring_left_tot, self.nring_tot), flush=True)
        if available.size == 0:
            return [], None
        if need_help.size == 0:
            return [], None
        if available.size >= nring_left_tot - need_help.size:
            # Just assign all of the available processes
            helpers_per_task = 1000000
        else:
            # Try to balance the number of helpers per task
            helpers_per_task = np.int(np.ceil(available.size / need_help.size))
        last = 0
        for helpee in need_help:
            nassign = min(nring_left_all[helpee] - 1, helpers_per_task)
            helpers_available = available[last:last + nassign]
            last += nassign
            if self.rank == helpee:
                return list(helpers_available), None
            elif self.rank in helpers_available:
                return [], helpee
            if last > available.size:
                break
        return [], None

    def process_optical_ring(self, job):
        """ Apply all preprocessing steps to one ring of detector data.

        """
        ring_timer = Timer()
        ring_timer.start()
        # Set a timeout to be raised.  We'll cancel it at
        # the end of the loop.
        if job.adc_iteration < self._nadc_iteration - 1:
            signals.alarm(int(self._timeout_intermediate))
        else:
            signals.alarm(int(self._timeout))

        ringinfo = job.ringinfo.copy()

        ringinfo.start_index = job.startsample + self._margin
        ringinfo.ring_number = job.ring_number

        sigADU = None
        maskflg = None

        try:
            ringinfo.start_time = job.tme[self._margin]

            zeroflg = job.sig == 0
            nbad = np.sum(zeroflg)
            if nbad != 0:
                print('{:4} : Ring {:5} WARNING: found {} raw, '
                      'zero-valued samples.'.format(
                          self.rank, job.ring_number, nbad), flush=True)
                job.flg[zeroflg] = True

            if job.ringflg is not None:
                frac = np.sum(job.ringflg[job.ind_from] != 0,
                              dtype=np.float64) / len(job.ringflg[job.ind_from])
                ringinfo.bad_frac = frac

            # Require at least 10% of the signal to be unflagged to
            # even attempt processing.

            if np.sum(job.flg[job.ind_from
                              ] == 0) < 0.1 * len(job.flg[job.ind_from]):
                raise Exception('All samples are flagged.')

            if job.drkflg1 is not None and job.drkflg2 is not None:
                if np.sum(job.drkflg1[job.ind_from] == 0) \
                   < 0.1 * len(job.drkflg1[job.ind_from]) \
                   or np.sum(job.drkflg2[job.ind_from] == 0) \
                   < 0.1 * len(job.drkflg2[job.ind_from]):
                    raise Exception('All dark samples are flagged.')

            if self._margin > 0:
                # This will exclude the margins from fits
                job.pntflg[:self._margin] = True
                job.pntflg[-self._margin:] = True

            flag_intense = None

            (spinflg, mean_spinrate, mean_good_spinrate) = self.check_spinrate(
                job.tme, job.phse)
            if np.any(spinflg):
                print('{:4} : Ring {:5} spinrate drifts. '
                      'Mean = {:.3f} deg/s, mean(good) = {:.3f} deg/s,'
                      ' flagging = {:8.2f} %'.format(
                          self.rank, job.ring_number,
                          np.degrees(mean_spinrate),
                          np.degrees(mean_good_spinrate),
                          np.sum(spinflg) * 100 / spinflg.size),
                      flush=True)
            job.pntflg[spinflg] = True
            job.flg[spinflg] = True
            ringinfo.spinrate = mean_spinrate
            ringinfo.spinrate_good = mean_good_spinrate

            if np.sum(job.pntflg == 0) / job.pntflg.size < 0.1:
                raise Exception('Pointing is unstable')

            # Get the dipole

            dipo = self.dipoler.dipole(job.q, velocity=job.v, det=job.det)
            if self._bg_has_dipole:
                dipo -= self.dipoler.dipole(job.q, det=job.det)

            if not self._lfi_mode:
                dipo, _ = self.taudeconvolver.convolve(dipo, None)

            # Convert pointing to angles

            theta, phi = qa.to_position(job.q)

            # Optionally, derive our own planet flags

            if self._flag_planets:
                job.ssoflg[:] = self.planetflagger.flag(
                    theta, phi, job.tme, job.fwhm * self._planet_flag_radius)

            # Sample the (polarized) background map

            if self.mapsampler is not None:
                if self._bg_pol:
                    bg = self.mapsampler.atpol(theta, phi, job.iquw)
                else:
                    bg = self.mapsampler.at(theta, phi)

                if not self._lfi_mode:
                    bg, _ = self.taudeconvolver.convolve(bg, None)
            else:
                bg = 0

            # Sample the processing mask(s)

            if self.masksampler is not None:
                maskflg = self.masksampler.at(theta, phi) < 0.5

            if self.masksampler_adc is not None:
                maskflg_adc = self.masksampler_adc.at(theta, phi) < 0.5
            else:
                maskflg_adc = maskflg

            if self._lfi_mode and len(np.shape(job.sig)) == 2:
                # Convert the ADU signal to Volts
                diodes = to_diodes(job.det)
                for i, diode in enumerate(diodes):
                    pre953 = job.tme < 1703263248.3820317
                    if np.any(pre953) > 0:
                        gain, offset = ADU2Volt[diode]
                        job.sig[pre953, 2 * i:2 * i + 2] *= gain
                        job.sig[pre953, 2 * i:2 * i + 2] += offset
                    post953 = job.tme >= 1703263248.3820317
                    if np.any(post953) > 0:
                        gain, offset = ADU2Volt_post953[diode]
                        job.sig[post953, 2 * i:2 * i + 2] *= gain
                        job.sig[post953, 2 * i:2 * i + 2] += offset

                # Correct for ADC nonlinearity
                if self.nlcorrector is not None:
                    job.sig = self.nlcorrector.correct(job.sig, job.tme)

                # Apply relative calibration to improve signal
                # subtraction
                for i, diode in enumerate(diodes):
                    if diode in diode_gains:
                        gain = diode_gains[diode]
                        job.sig[:, 2 * i:2 * i + 2] *= gain
            else:
                if (job.adc_iteration > 0 or (self._adc_correction
                                              is not None)):
                    sig0 = job.sig.copy()
                    # Correct for ADC nonlinearity using iterative,
                    # DSP-valued profile
                    job.sig = self.inputestimator.apply_correction(job.sig,
                                                                   job.phse4k)
                    if self._verbose:
                        print('{:4} : Ring {:5} : RMS before = {}, '
                              'RMS after = {}'.format(
                                  self.rank, job.ring_number, np.std(sig0),
                                  np.std(job.sig)), flush=True)
                    del sig0

                if self._measure_ADC or self._nadc_iteration > 1:
                    sigADU = job.sig.copy()

                # From DSP to Volts
                job.sig = self.transf1.convert(job.sig, job.tme, job.det)

                if self._measure_ADC:
                    sigND = job.sig.copy()

                baselineND, flag_base = self._get_baseline(job.sig, job.flg)

                job.flg[flag_base] = True

                ringinfo.noise_offsetND = np.median(baselineND)

                # Demodulate
                job.sig = (job.sig - baselineND) * (1. - 2. * job.prty)
                if self._measure_ADC:
                    sigDemod = job.sig.copy()

                # Correct for gain nonlinearity
                job.sig = self.gaincorrector.correct(job.sig, job.flg)

            # Calibrate the TOI
            if len(np.shape(job.sig)) == 2:
                # LFI raw signal has 4 columns (2 x Sky, 2 x Ref)
                for i in range(np.shape(job.sig)[1]):
                    job.sig[:, i] = tt.calibrate(job.tme, job.sig[:, i],
                                                 *job.gains)
            else:
                job.sig = tt.calibrate(job.tme, job.sig, *job.gains)
                job.drk1 = tt.calibrate(job.tme, job.drk1, *job.gains)
                job.drk2 = tt.calibrate(job.tme, job.drk2, *job.gains)

            flag_intense = None
            if self._lfi_mode:
                timer = Timer()
                timer.start()
                nflag0 = np.sum(job.flg[job.ind_from] != 0)
                job.sig, job.flg = self.glitchremover.remove(job.sig, job.flg)
                nflag1 = np.sum(job.flg[job.ind_from] != 0)
                ringinfo.glitch_frac = (nflag1 - nflag0) \
                    / len(job.flg[job.ind_from])
                timer.stop()
                if self._verbose:
                    timer.report('{:4} : Ring {:5} glitch-remove'
                          ''.format(self.rank, job.ring_number))

                timer.clear()
                timer.start()
                job.sig, job.flg, spikepar = self.lineremover.remove(
                    job.sig, job.flg, job.fsample, job.tme, bg,
                    pntflag=job.pntflg, ssoflag=job.ssoflg, maskflag=maskflg,
                    spike_params=job.spike_params)
                timer.stop()
                if self._verbose:
                    timer.report('{:4} : Ring {:5} line-remove'
                          ''.format(self.rank, job.ring_number))
                for ipar, par in enumerate(spikepar):
                    for iline, line in enumerate(
                            np.arange(1, int(job.fsample // 2 + 1))):
                        for ialias, alias in enumerate(['_low', '',
                                                        '_high']):
                            if len(par) <= 3 * iline + ialias:
                                break
                            (a, b) = par[3 * iline + ialias]
                            key = 'cos_ampl_{:02}Hz{}{}'.format(
                                line, alias, ipar)
                            ringinfo[key] = a
                            key = 'sin_ampl_{:02}Hz{}{}'.format(
                                line, alias, ipar)
                            ringinfo[key] = b

                if len(np.shape(job.sig)) == 2:
                    # Raw LFI data with sky and reference for two diodes
                    coadd = self._single_diode is None
                    timer.clear()
                    timer.start()
                    (job.sig, job.flg, gain0, gain1, rms0, rms1, filter_params,
                     w0, w1) = self.differencer.difference_lfi(
                         job.sig[:, 0:2].T, job.flg, job.sig[:, 2:4].T, job.flg,
                         job.fsample, pntflag=job.pntflg,
                         weight0=job.weight0, weight1=job.weight1,
                         lowpass0=job.lowpass0, lowpass1=job.lowpass1,
                         ssoflag=job.ssoflg, maskflag=maskflg, bg=bg,
                         dipole=dipo, coadd_diodes=coadd)
                    timer.stop()
                    if self._verbose:
                        timer.report('{:4} : Ring {:5} difference'
                              ''.format(self.rank, job.ring_number))
                    if not coadd:
                        # Pick only one of the diodes
                        job.sig = job.sig[self._single_diode]
                    """"
                    # DEBUG begin
                    import matplotlib.pyplot as plt
                    import pdb
                    psd = np.abs(np.fft.rfft(job.sig - bg - dipo)) ** 2
                    freq = np.fft.rfftfreq(job.sig.size, d=1 / job.fsample)
                    plt.loglog(freq, psd)
                    pdb.set_trace()
                    # DEBUG end
                    """

                    ringinfo.gain0 = gain0
                    ringinfo.gain1 = gain1
                    ringinfo.rms0 = rms0
                    ringinfo.rms1 = rms1
                    ringinfo.weight0 = w0
                    ringinfo.weight1 = w1
                    for diode in [0, 1]:
                        for i, name in enumerate(
                                ['R', 'Rsigma', 'Ralpha']):
                            key = '{}{}'.format(name, diode)
                            ringinfo[key] = filter_params[diode][i]
                signal_estimate = bg + dipo
                job.sig, job.flg = self.glitchremover.remove(
                    job.sig, job.flg, signal_estimate, job.ssoflg + maskflg)
                phasebin_estimate = signal_estimate
            elif not self._shdet_mode:
                if job.adc_iteration < self._nadc_iteration - 1 \
                   and job.ring_number in job.previously_failed_rings:
                    # This ring failed to despike. Don't try again
                    # as the bad rings take the longest time and
                    # are almost certain to fail every time.
                    raise DespikerError('Despiker failed.')
                try:
                    (job.sig, job.flg, frac, glitch_estimate, flag_intense,
                     signal_estimate, phasebin_estimate
                     ) = self.remove_glitches(
                         job.det, job.sig, job.flg, job.ssoflg,
                         job.pntflg, bg, dipo, job.phse, job.adc_iteration,
                         zeroflg, job.ring_number, maskflg, job.ind_from,
                         job.fwhm)
                except DespikerError:
                    job.previously_failed_rings.add(job.ring_number)
                    raise

                job.flg[spinflg] = True

                ringinfo.glitch_frac = frac
                if self._verbose:
                    print('{:4} : Ring {:5} glitch frac = {:8.4f}'
                          ''.format(self.rank, job.ring_number, frac),
                          flush=True)
            else:
                # HFI run in shdet mode
                signal_estimate = np.zeros_like(job.sig)
                glitch_estimate = None
                if bg is not None:
                    signal_estimate += bg
                if dipo is not None:
                    signal_estimate += dipo

            # Fill gaps
            if not self._shdet_mode:
                job.sig = self.gapfiller.fill(
                    job.sig, job.phse, job.flg, job.det, job.ring_number,
                    dark=False, signal_estimate=phasebin_estimate,
                    pntflag=job.pntflg)

            if not self._lfi_mode and not self._shdet_mode:
                # Remove thermal correlation
                if self._measure_ADC:
                    sig_temp = job.sig.copy()
                # sig0 = sig.copy() # DEBUG
                timer = Timer()
                timer.start()
                (job.sig, job.flg, gain, filter_params,
                 w1, w2) = self.differencer.difference_hfi(
                    job.sig, job.flg, job.drk1, job.drkflg1, job.drk2,
                    job.drkflg2, job.phse, job.fsample, pntflag=job.pntflg,
                    ssoflag=job.ssoflg, maskflag=maskflg + flag_intense,
                    signal_estimate=phasebin_estimate, fast=True)
                timer.stop()
                if self._verbose:
                    timer.report('{:4} : Ring {:5} difference'
                          ''.format(self.rank, job.ring_number))

                ringinfo.weight1 = w1
                ringinfo.weight2 = w2
                ringinfo['R'] = filter_params[0]
                ringinfo['Rfcut'] = np.exp(filter_params[1])
                # DEBUG begin, remember to uncomment sig0 = sig.copy()
                """
                import pdb
                import matplotlib.pyplot as plt
                decorr_template = sig0 - sig
                noise = sig0 - signal_estimate
                noise2 = sig - signal_estimate
                good = flg + pntflg + ssoflg + drkflg1 + drkflg2 == 0
                time = tme - tme[0]
                plt.figure()
                plt.plot(time[good], noise[good], '.', label='noise')
                plt.plot(time[good], decorr_template[good], '.',
                         label='thermal')
                plt.legend(loc='best')
                ax = plt.gca();ax.set_xlabel('time [s]')
                ax.set_ylabel('K_CMB')
                plt.savefig('thermal1.png')
                plt.figure()
                plt.plot(time[good], noise2[good], '.', label='cleaned')
                plt.legend(loc='best')
                ax = plt.gca();ax.set_xlabel('time [s]')
                ax.set_ylabel('K_CMB')
                plt.savefig('thermal2.png')
                plt.figure()
                plt.plot(time[good], drk1[good], '.', label='dark1')
                plt.legend(loc='best')
                ax = plt.gca();ax.set_xlabel('time [s]')
                ax.set_ylabel('Volts')
                plt.savefig('thermal3.png')
                plt.figure()
                plt.plot(time[good], drk2[good], '.', label='dark2')
                plt.legend(loc='best')
                ax = plt.gca();ax.set_xlabel('time [s]')
                ax.set_ylabel('Volts')
                plt.savefig('thermal4.png')
                plt.figure()
                fthermal = np.abs(np.fft.rfft(decorr_template[good]))
                fnoise = np.abs(np.fft.rfft(noise[good]))
                fnoise2 = np.abs(np.fft.rfft(noise2[good]))
                n = np.sum(good)
                freq = np.fft.rfftfreq(n, 1 / 180.3737)
                plt.loglog(freq, fthermal, label='Thermal template')
                plt.loglog(freq, fnoise, label='Before cleaning')
                plt.loglog(freq, fnoise2, label='After cleaning')
                plt.legend(loc='best')
                ax = plt.gca();ax.set_xlabel('[Hz]')
                ax.set_ylabel('K_CMB')
                plt.savefig('thermal5.png')
                pdb.set_trace()
                sys.exit()
                """
                # DEBUG end
                if self._measure_ADC:
                    decorr_template = sig_temp - job.sig
                    del sig_temp
                if gain is not None:
                    ringinfo.gain0 = gain
                # Fill gaps again for jump correction.
                job.sig = self.gapfiller.fill(
                    job.sig, job.phse, job.flg, job.det, job.ring_number,
                    dark=False, signal_estimate=signal_estimate,
                    pntflag=job.pntflg)
            elif self._shdet_mode:
                if signal_estimate is None:
                    gain = 1.0
                else:
                    # Perform basic calibration using the signal
                    # estimate
                    good = np.logical_not(job.flg)
                    if job.pntflg is not None:
                        good[job.pntflg] = False
                    if job.ssoflg is not None:
                        good[job.ssoflg] = False
                    if maskflg is not None:
                        good[maskflg] = False
                    ngood = np.sum(good)
                    templates = np.vstack([np.ones(ngood),
                                           signal_estimate[good]])
                    invcov = np.dot(templates, templates.T)
                    cov = np.linalg.inv(invcov)
                    proj = np.dot(templates, job.sig[good])
                    coeff = np.dot(cov, proj)
                    gain = 1 / coeff[1]
                    offset = coeff[0]
                    job.sig -= offset
                    if self._verbose:
                        print('{:4} : Ring {:5} gain = {} offset = {}.'
                              ''.format(self.rank, job.ring_number, gain,
                                        offset), flush=True)
                ringinfo.gain0 = gain
                decorr_template = None

            if not self._shdet_mode:
                # Detect jumps
                if self._measure_ADC:
                    sig_temp = job.sig.copy()
                job.sig, job.flg, njump = self.jumpcorrector.correct(
                    job.sig, job.flg, job.phse, signal_estimate=signal_estimate,
                    signal_estimate_is_binned=False)

                if self._measure_ADC:
                    if njump == 0:
                        jump_estimate = None
                    else:
                        jump_estimate = sig_temp - job.sig
                    del sig_temp
            else:
                jump_estimate = None
                njump = 0

            ringinfo.njump = njump

            if not self._lfi_mode and not self._shdet_mode:
                # Remove 4K lines
                timer = Timer()
                timer.start()
                (job.sig, job.flg, line_estimate, frequencies,
                 amplitudes) = self.lineremover.remove(
                     job.startsample, job.stopsample, job.sig, job.flg,
                     job.phse, dark=False, maskflag=maskflg + flag_intense,
                     signal_estimate=signal_estimate,
                     return_line_estimate=True)
                timer.stop()
                if self._verbose:
                    timer.report(
                        '{:4} : Ring {:5} line-remove'
                        ''.format(self.rank, job.ring_number))
                for ii, ff in enumerate(frequencies):
                    ringinfo[
                        'cos_ampl_{}Hz'.format(ff)] = amplitudes[ii][0]
                    ringinfo[
                        'sin_ampl_{}Hz'.format(ff)] = amplitudes[ii][1]
            else:
                line_estimate = None

            # Construct a low frequency noise estimate and highpass
            # the signal with it
            noise_estimate = flagged_running_average(
                job.sig - signal_estimate, job.flg, np.int(1000 * job.fsample))
            # Disable the highpass for now
            # job.sig -= noise_estimate

            if not self._shdet_mode:
                # Last gapfill after jump correction and line removal.
                job.sig = self.gapfiller.fill(
                    job.sig, job.phse, job.flg, job.det, job.ring_number,
                    dark=False, signal_estimate=phasebin_estimate,
                    pntflag=job.pntflg)

            # Cancel the timeout
            signals.alarm(0)

            # Produce the input signal estimate for measuring
            # the ADC nonlinearity
            if self._measure_ADC \
               and job.adc_iteration < self._nadc_iteration - 1:
                raw_flag = job.flg.copy()
                if job.ssoflg is not None:
                    raw_flag[job.ssoflg] = True
                if flag_intense is not None:
                    raw_flag[flag_intense] = True
                if maskflg is not None:
                    raw_flag[maskflg] = True
                if maskflg_adc is not None:
                    raw_flag[maskflg_adc] = True

                """
                # EXPERIMENTAL: build the signal estimate from the
                # phase-binned estimate INSTEAD of the map-based one.
                # We build a new phase-binned signal estimate
                # to avoid line and jump contamination.
                flagger = GlitchFlagger(
                    fwhm=job.fwhm, twice=False, threshold=3.0, wkernel=3)
                _, _, phasebin_estimate = flagger.flag_glitches(
                    job.sig, job.flg, phase=job.phse, pntflag=job.pntflg,
                    dark=False)
                # Calibrate the phasebin estimate against the global estimate
                good = raw_flag == 0
                ngood = np.sum(good)
                templates = np.vstack(
                    [np.ones(ngood), phasebin_estimate[good]])
                invcov = np.dot(templates, templates.T)
                cov = np.linalg.inv(invcov)
                proj = np.dot(templates, signal_estimate[good])
                coeff = np.dot(cov, proj)
                signal_estimate = coeff[0] + coeff[1] * phasebin_estimate
                """

                self.inputestimator.estimate(
                    job.ring_number, timestamps=job.tme,
                    signal_estimate=signal_estimate,
                    noise_estimate=noise_estimate, line_estimate=line_estimate,
                    decorr_template=decorr_template,
                    glitch_estimate=glitch_estimate,
                    jump_estimate=jump_estimate, signal_demod=sigDemod,
                    flags=raw_flag, baselineND=baselineND,
                    parity=job.prty, sigND=sigND, sigADU=sigADU,
                    gains=job.gains, phase4k=job.phse4k)

            if not self._lfi_mode:
                # Deconvolve bolometric transfer function
                job.sig, job.flg = self.taudeconvolver.deconvolve(job.sig,
                                                                  job.flg)

            # Collect last metadata

            good = job.flg == 0
            good[:self._margin] = False
            good[-self._margin:] = False
            if np.sum(good) > 0:
                ringinfo.total_rms = np.std(job.sig[good])
                ringinfo.signal_rms = np.std(signal_estimate[good])
            if job.pntflg is not None:
                good[job.pntflg] = False
            if job.ssoflg is not None:
                good[job.ssoflg] = False
            if maskflg is not None:
                good[maskflg] = False
            if flag_intense is not None:
                good[flag_intense] = False
            if np.sum(good) > 0:
                cleaned = job.sig - signal_estimate
                polyfilter(
                    5,
                    job.flg.astype(np.uint8),
                    [cleaned],
                    np.array([0]),
                    np.array([job.sig.size]),
                )
                ringinfo.noise_rms = np.std(cleaned[good])
                del cleaned
                ringinfo.minval = np.amin(job.sig[good])
                ringinfo.maxval = np.amax(job.sig[good])
                ringinfo.mean = np.mean(job.sig[good])
                ringinfo.median = np.median(job.sig[good])
                ringinfo.rms = np.std(job.sig[good])

            ringinfo.total_frac = np.sum(
                job.flg[job.ind_from] != 0, dtype=np.float64) \
                / len(job.flg[job.ind_from])
            if job.ssoflg is not None:
                ringinfo.sso_frac = np.sum(
                    job.ssoflg[job.ind_from] != 0, dtype=np.float64) \
                    / len(job.flg[job.ind_from])
            if job.pntflg is not None:
                ringinfo.pnt_frac = np.sum(
                    job.pntflg[job.ind_from] != 0, dtype=np.float64) \
                    / len(job.flg[job.ind_from])
            if maskflg is not None:
                ringinfo.mask_frac = np.sum(
                    maskflg[job.ind_from] != 0, dtype=np.float64) \
                    / len(job.flg[job.ind_from])

            ringinfo.failed = False
            ringinfo.despikererror = False
            ringinfo.timeout = False
            error = None
        except Exception as e:
            error = e
            #raise  # DEBUG
            if isinstance(error, DespikerError):
                ringinfo.despikererror = True
            if isinstance(error, TimeOut):
                ringinfo.timeout = True
            else:
                # Cancel the timeout
                signals.alarm(0)
            # The preprocessing failed. Flag the entire ring
            print('{:4} : Ring {:5} of {} preprocessing failed: "{}" '
                  'Flagging the entire ring.'.format(
                      self.rank, job.ring_number, job.det, error), flush=True)
            ringinfo.failed = True

        jobout = OpticalRingJobOutput(
            sigADU, job.sig, job.flg, job.ssoflg, maskflg, error, job.iring,
            job.ind_from, job.ind_to, ringinfo, job.ring_start,
            job.ring_stop, job.ring_number)

        ring_timer.stop()
        if self._verbose:
            ring_timer.report('{:4} : Ring {:5}'.format(
                self.rank, job.ring_number))
        return jobout

    def collect_ringinfos(self, ringinfos, flags_out, det, adc_iteration):
        """ Process the ring metadata and add flags accordingly
        """
        all_ringinfos = self.comm.gather(ringinfos)
        if self.rank == 0:
            all_ringinfos = np.hstack(all_ringinfos).view(np.recarray)
        all_ringinfos = self.comm.bcast(all_ringinfos)
        if self._lfi_mode:
            outliers = self._get_outliers(
                all_ringinfos, ['noise_rms', 'rms0', 'rms1', 'gain0', 'gain1'],
                jumpmax=3, tol=5., wmean=101, werr=301)
        else:
            outliers = self._get_outliers(
                all_ringinfos, ['glitch_frac', 'noise_rms', 'rms0', 'gain0'],
                jumpmax=3, tol=5., wmean=1001, werr=1001)
        for outlier in outliers:
            if outlier in self.local_ring_ranges:
                # Flag the entire ring
                ind = self.local_ring_ranges[outlier]
                flags_out[ind] |= np.uint8(8)
        if self.rank == 0:
            print('Flagged {} outlier rings for {}.'
                  ''.format(len(outliers), det))
            fn_out = os.path.join(
                self._out,
                'ring_info_{}_iter{:02}.pck'.format(det, adc_iteration))
            pickle.dump(all_ringinfos, open(fn_out, 'wb'), protocol=2)
        return all_ringinfos, outliers

    def process_optical_detector(
            self, det, tod, fsample, timestamps, commonflags, ringmasker,
            dark1, darkflag1, dark2, darkflag2,
            pntflag, phase, velocity, parity, phase4k):
        """Process optical detector

        Apply all relevant preprocessing steps to an optical bolometer
        or radiometer

        """
        if self.rank == 0:
            if self._nadc_iteration == 1:
                print('Loading data for {}'.format(det), flush=True)

        # if self._lfi_mode:
        self.dipoler = Dipoler(full4pi='npipe', comm=self.comm, RIMO=tod.RIMO)
        # else:
        #    dipoler = Dipoler(freq=int(self._freq))

        if self._bg_map_path is not None \
           and 'DETECTOR' not in self._bg_map_path:
            # All detectors share the same template map
            self.mapsampler = MapSampler(
                self._bg_map_path, pol=self._bg_pol,
                nside=self._bg_nside, comm=self.comm, cache=tod.cache)
        else:
            self.mapsampler = None

        if self._maskfile:
            self.masksampler = MapSampler(self._maskfile, comm=self.comm,
                                          cache=tod.cache)
        else:
            self.masksampler = None

        if self._maskfile_adc is None \
           or self._maskfile_adc == self._maskfile:
            self.masksampler_adc = None
        else:
            self.masksampler_adc = MapSampler(
                self._maskfile_adc, comm=self.comm, cache=tod.cache)

        self.comm.barrier()
        if self.rank == 0:
            print('Setting up processing for {}'.format(det), flush=True)

        if self._lfi_mode:
            bolo_id = None
        else:
            bolo_id = bolo_to_pnt(det)

        fwhm = tod.RIMO[det].fwhm

        dtype = [
            ('ring_number', np.int), ('start_time', np.float),
            ('start_index', np.int),
            ('glitch_frac', np.float), ('njump', np.int),
            ('total_rms', np.float), ('signal_rms', np.float),
            ('noise_rms', np.float), ('noise_offsetND', np.float),
            ('gain0', np.float), ('gain1', np.float),
            ('rms0', np.float), ('rms1', np.float),
            ('total_frac', np.float), ('sso_frac', np.float),
            ('pnt_frac', np.float), ('mask_frac', np.float),
            ('bad_frac', np.float), ('minval', np.float), ('maxval', np.float),
            ('mean', np.float), ('median', np.float), ('rms', np.float),
            ('spinrate', np.float), ('spinrate_good', np.float),
            ('failed', np.bool), ('outlier', np.bool),
            ('despikererror', np.bool), ('timeout', np.bool)]

        if self._lfi_mode:
            for diode in [0, 1]:
                for filterpar in ['weight', 'R', 'Rsigma', 'Ralpha']:
                    dtype.append(('{}{}'.format(filterpar, diode), np.float))
            for linefreq in np.arange(1, int(fsample // 2 + 1)):
                for alias in ['_low', '', '_high']:
                    for col in ['', 0, 1, 2, 3]:
                        dtype += [
                            ('cos_ampl_{:02}Hz{}{}'.format(
                                linefreq, alias, col), np.float),
                            ('sin_ampl_{:02}Hz{}{}'.format(
                                linefreq, alias, col), np.float)]
        else:
            dtype.append(('weight1', np.float))
            dtype.append(('weight2', np.float))
            for filterpar in ['R', 'Rfcut']:
                dtype.append((filterpar, np.float))
            for linefreq in [10, 20, 30, 40, 50, 60, 70, 80,
                             17, 16, 25, 43, 46, 48, 57]:
                dtype += [('cos_ampl_{:02}Hz'.format(linefreq), np.float),
                          ('sin_ampl_{:02}Hz'.format(linefreq), np.float)]

        ringinfos = np.recarray([self.nring, ], dtype=dtype)

        self.local_ring_ranges = {}

        ringinfos.fill(0)

        if self._lfi_mode:
            self.transf1 = None
            self.gaincorrector = None
            self.inputestimator = None
            self.taudeconvolver = None
            self.nlcorrector = LFINLCorrector(det, self.comm)
            self.lineremover = LineRemoverLFI()
            self.glitchremover = GlitchRemoverLFI()
        else:
            self.transf1 = Transf1()
            self.gaincorrector = GainCorrector(self._imo, bolo_id, g0=self._g0,
                                               v0=self._v0)
            if self._measure_ADC or self._adc_correction \
               or self._nadc_iteration > 1:
                self.inputestimator = InputEstimator(
                    det, self.transf1, self.gaincorrector, self._nphase4k,
                    self._deltaADU, self.comm, out=self._out,
                    input_correction=self._adc_correction)
            else:
                self.inputestimator = None
            self.lineremover = LineRemover(fsample=fsample)
            if self._tabulated_transfer_function is not None:
                if type(self._global_phase_shift) is dict:
                    self.taudeconvolver = TauDeconvolver(
                        '', '', tabulated_tf=(
                            self._tabulated_transfer_function[0][det],
                            self._tabulated_transfer_function[1][det],
                            self._tabulated_transfer_function[2][det]),
                        extra_global_offset=self._global_phase_shift[det])
                else:
                    self.taudeconvolver = TauDeconvolver(
                        '', '', tabulated_tf=(
                            self._tabulated_transfer_function[0][det],
                            self._tabulated_transfer_function[1][det],
                            self._tabulated_transfer_function[2][det]))
                if self._global_phase_shift == 0:
                    self.taudeconvolver.fit_absolute_offset()
            else:
                self.taudeconvolver = TauDeconvolver(
                    bolo_id, self._imo,
                    extra_global_offset=self._global_phase_shift,
                    comm=self.comm)
                """
                # DEBUG begin
                print('WARNING: DEBUG TAUDECONVOLVER', flush=True)
                self.taudeconvolver = TauDeconvolver(
                    bolo_id, self._imo,
                    extra_global_offset=self._global_phase_shift,
                    comm=self.comm, filterlen=2 ** 12)
                # DEBUG end
                """

            # Old Volt-valued corrections permanently disabled.
            self.nlcorrector = None

        self.gapfiller = GapFiller(nbin=self._nbin_phase)
        self.jumpcorrector = JumpCorrector(
            self._jump_filter_len, nbin=self._nbin_phase,
            threshold=self._jump_threshold)
        self.differencer = Differencer(nbin=self._nbin_phase)

        if self._bg_map_path is not None and 'DETECTOR' in self._bg_map_path:
            # All detectors share the same template map
            self.mapsampler = MapSampler(
                self._bg_map_path.replace('DETECTOR', det), pol=self._bg_pol,
                nside=self._bg_nside, comm=self.comm, cache=tod.cache)

        # Read all of the data for this process and process interval by interval

        signal = tod.local_signal(det, margin=self._margin)
        if len(signal) != self.nsamp + 2 * self._margin:
            raise Exception(
                'Cached signal does not include margins.')

        flags = tod.local_flags(det, margin=self._margin)
        if len(flags) != self.nsamp + 2 * self._margin:
            raise Exception('Cached flags do not include margins.')

        quat = tod.local_pointing(det, margin=self._margin)
        if len(quat) != self.nsamp + 2 * self._margin:
            raise Exception('Cached quats do not include margins.')

        cachename = '{}_{}'.format(tod.WEIGHT_NAME, det)
        iquweights = tod.cache.reference(cachename)
        if len(iquweights) != self.nsamp + 2 * self._margin:
            raise Exception('Cached IQU weights do not include margins.')

        if self._calfile is not None:
            gains = read_gains(self._calfile, det, timestamps[0],
                               timestamps[-1])
        else:
            gains = None

        # Cast the flags into boolean vectors

        if self._detmask:
            detflag = (flags & self._detmask) != 0
        else:
            detflag = flags != 0

        commonmask = 255
        if self._pntmask is not None:
            commonmask -= self._pntmask
        detflag[(commonflags & commonmask) != 0] = True

        if self._ssomask is not None:
            ssoflag = (flags & self._ssomask) != 0
        else:
            if self._flag_planets:
                ssoflag = np.zeros(self.nsamp, dtype=np.bool)
            else:
                ssoflag = None

        # Add ring flags

        if ringmasker is not None:
            ringflag = ringmasker.get_mask(timestamps, det)
            detflag[ringflag] = True
        else:
            ringflag = None

        # Process

        previously_failed_rings = set()  # Don't deglitch bad rings repeatedly

        weight0, weight1 = None, None
        lowpass0, lowpass1 = None, None
        spike_params = None
        if self._lfi_mode:
            self._nadc_iteration = 2

        for adc_iteration in range(self._nadc_iteration):
            self.comm.barrier()
            if self.rank == 0:
                if self._nadc_iteration == 1:
                    print('Processing {}'.format(det), flush=True)
                else:
                    print('Processing {}, ADC iteration = {}'
                          ''.format(det, adc_iteration), flush=True)

            if self.inputestimator is not None:
                self.inputestimator.reset_cache()

            signal_out = np.zeros(self.nsamp + 2 * self._margin,
                                  dtype=np.float64)
            flags_out = np.zeros(self.nsamp + 2 * self._margin, dtype=np.uint8)

            # Initialize to sentinel values
            signal_out[:] = np.nan
            flags_out[:] = 1

            ring_number = self.ring_offset - 1
            timer_rings = Timer()
            timer_rings.start()

            signals.signal(signals.SIGALRM, timeouthandler)
            helpers_waiting = []
            helpers_working = []
            helpee = None

            for iring, (ring_start,
                        ring_stop) in enumerate(zip(self.local_starts,
                                                    self.local_stops)):
                ringinfo = ringinfos[iring]
                ring_number += 1

                if self._verbose:
                    print('{:4} : Processing ring {:4} ({} / {})'
                          ''.format(self.rank, ring_number, iring + 1,
                                    self.nring), flush=True)

                startsample = (ring_start + tod.globalfirst +
                               tod.local_samples[0] - self._margin)
                stopsample = (startsample + (ring_stop - ring_start) +
                              2 * self._margin)

                # Slice that includes margins
                ind = slice(ring_start, ring_stop + 2 * self._margin)
                # Slices without margins
                # ring buffer slice
                ind_from = slice(self._margin,
                                 ring_stop - ring_start + self._margin)
                # total array slice
                ind_to = slice(ring_start + self._margin,
                               ring_stop + self._margin)
                self.local_ring_ranges[ring_number] = ind_to

                tme = timestamps[ind]
                sig = signal[ind].copy()
                flg = detflag[ind].copy()
                ringflg = None
                if ringflag is not None:
                    ringflg = ringflag[ind].copy()
                drk1 = None
                drk2 = None
                drkflg1 = None
                drkflg2 = None
                if not self._lfi_mode and not self._shdet_mode:
                    drk1 = dark1[ind]
                    drk2 = dark2[ind]
                    # Only apply the dark quality flags.
                    drkflg1 = darkflag1[ind] & 1
                    drkflg2 = darkflag2[ind] & 1
                if pntflag is not None:
                    pntflg = pntflag[ind].copy()
                else:
                    pntflg = np.zeros_like(flg)
                ssoflg = None
                if ssoflag is not None:
                    ssoflg = ssoflag[ind]
                q = quat[ind]
                phse = phase[ind]
                iquw = None
                if self._bg_pol:
                    iquw = iquweights[ind]
                v = velocity[ind]
                phse4k = None
                prty = None
                if not self._lfi_mode:
                    # SHDet always produces a signal starting
                    # with parity 0
                    if self._shdet_mode:
                        prty = np.arange(np.size(prty)) % 2
                    else:
                        prty = parity[ind]
                    phse4k = phase4k[ind]

                job = OpticalRingJob(
                    adc_iteration, ringinfo, ring_number, ring_start, ring_stop,
                    ind_from, ind_to, iring, startsample, stopsample, det, fwhm,
                    tme, sig, flg, fsample, ringflg, pntflg, ssoflg, phse, q, v,
                    iquw, drk1, drk2, drkflg1, drkflg2, phse4k, prty, gains,
                    weight0, weight1, lowpass0, lowpass1,
                    previously_failed_rings, spike_params)

                if len(helpers_waiting) > 0:
                    helper = helpers_waiting.pop(0)
                    print('{:4} : Ring {:5} sending to {}'
                          ''.format(self.rank, ring_number, helper),
                          flush=True)
                    self.send_optical_ring(helper, job)
                    helpers_working.append(helper)
                else:
                    jobout = self.process_optical_ring(job)
                    self.add_optical_ring(
                        jobout, signal, signal_out, flags_out, ringinfos,
                        previously_failed_rings)
                    while helpers_working:
                        helper = helpers_working.pop(0)
                        jobout = self.recv_optical_ring(helper)
                        self.add_optical_ring(
                            jobout, signal, signal_out, flags_out, ringinfos,
                            previously_failed_rings)

                    nring_left = self.nring - iring - 1
                    elapsed = timer_rings.elapsed_seconds()
                    if adc_iteration < self._nadc_iteration - 1:
                        balance_load = elapsed > self.async_time_intermediate
                    else:
                        balance_load = elapsed > self.async_time
                    if nring_left > 0 and balance_load:
                        helpers_waiting, _ = self.assign_helpers(nring_left)
                        if helpers_waiting:
                            print('{:4} : Will expect help from tasks # {}'
                                  ''.format(self.rank, helpers_waiting),
                                  flush=True)
                        helpers_working = []

            timer_rings.stop()
            if self._verbose:
                timer_rings.report(
                    '{:4} : All {:4} rings'.format(self.rank, self.nring))

            timer_rings.clear()
            timer_rings.start()
            nring_helped = 0

            while True:
                helpers, helpee = self.assign_helpers(0)
                if helpers is None:
                    # All rings are processed
                    break
                if helpee is not None:
                    print('{:4} : Will help task # {}'
                          ''.format(self.rank, helpee), flush=True)
                    self.process_remote_optical_ring(helpee)
                    nring_helped += 1

            if nring_helped > 0 and self._verbose:
                timer_rings.stop()
                timer_rings.report('{:4} : All {:4} helped rings'
                      ''.format(self.rank, nring_helped))

            all_ringinfos, outliers = self.collect_ringinfos(
                ringinfos, flags_out, det, adc_iteration)

            # Measure the ADC correction, except on the last iteration

            if not self._lfi_mode and adc_iteration == 0:
                # Disabled: HFI low pass filter parameters vary so much
                # that it is not useful to fix them.
                """
                # Fit for the optimal dark bolometer transfer function
                good = (all_ringinfos.failed + all_ringinfos.outlier) == 0
                lowpass = []
                for i, name in enumerate(['R', 'Rfcut']):
                    lowpass.append(np.median(all_ringinfos[name][good]))
                weight1 = np.median(all_ringinfos.weight1[good])
                weight2 = 1 - weight1
                """
                pass

            if (self._measure_ADC and (adc_iteration
                                       < self._nadc_iteration - 1)):
                # Check if the signal estimate should be recalibrated
                # to match the detector. We don't wish to change the
                # overall gain by the ADC NL correction
                if adc_iteration == 0:
                    good = (all_ringinfos.failed + all_ringinfos.outlier) == 0
                    gain = np.mean(all_ringinfos.gain0[good])

                    # Correct the input gains for future iterations
                    if gain != 0:
                        gains[1] *= gain

                    if self.rank == 0:
                        print('Will correct the gain by {} before measuring '
                              'the ADC NL.'.format(gain), flush=True)
                else:
                    gain = None

                self.inputestimator.remove_outliers(outliers)

                self.inputestimator.measure_correction(
                    fn='ADC_NL_{}_iter{:02}.fits'.format(det, adc_iteration),
                    gain=gain)
            elif self._lfi_mode and adc_iteration == 0:
                # Fit for the optimal diode weights and low-pass filters.
                good = (all_ringinfos.failed + all_ringinfos.outlier) == 0
                lowpasses = []
                for diode in [0, 1]:
                    lowpass = []
                    for name in ['R', 'Rsigma', 'Ralpha']:
                        key = '{}{}'.format(name, diode)
                        lowpass.append(np.median(all_ringinfos[key][good]))
                    lowpasses.append(lowpass)
                lowpass0, lowpass1 = lowpasses
                weight0 = np.median(all_ringinfos.weight0[good])
                weight1 = 1 - weight0
                # Also get the average line amplitudes, essentially
                # fixing the spike template
                spike_params = []
                for ipar in range(4):
                    par = []
                    for line in range(1, int(fsample // 2 + 1)):
                        for alias in ['_low', '', '_high']:
                            try:
                                key = 'cos_ampl_{:02}Hz{}{}'.format(
                                    line, alias, ipar)
                                a = np.median(all_ringinfos[key][good])
                                key = 'sin_ampl_{:02}Hz{}{}'.format(
                                    line, alias, ipar)
                                b = np.median(all_ringinfos[key][good])
                            except Exception:
                                break
                            par.append((a, b))
                    spike_params.append(par)

        return signal_out, flags_out

    # @profile
    def exec(self, data):
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        self.comm = comm.comm_world

        self.rank = self.comm.Get_rank()
        self.ntask = self.comm.Get_size()

        if len(data.obs) != 1:
            raise RuntimeError('Preproc assumes only one observation')

        obs = data.obs[0]
        tod = obs['tod']
        self.nsamp = tod.local_samples[1]

        if 'intervals' not in obs:
            raise RuntimeError('observation must specify intervals')
        # Get local intervals for statistics.  This will cache the timestamps.
        intervals = tod.local_intervals(obs['intervals'])
        self.local_starts = [ival.first for ival in intervals]
        # choose local stops so that we process the signal continuously
        self.local_stops = [ival.first for ival in intervals[1:]]
        self.local_stops.append(intervals[-1].last + 1)
        # Extend the first and last interval to cover the entire span
        self.local_starts[0] = 0
        self.local_stops[-1] = self.nsamp
        if self._margin != 0:
            # Clear the cached timestamps so that we may read them
            # again with the margins
            tod.cache.destroy(tod.TIMESTAMP_NAME)

        self.nring = len(self.local_starts)
        self.nring_tot = self.comm.allreduce(self.nring)
        self.ring_offset = tod.globalfirst_ring
        for interval in obs['intervals']:
            if interval.last < tod.local_samples[0]:
                self.ring_offset += 1

        if self._effdir_out is not None:
            tod.cache_metadata(self._effdir_out, comm=self.comm)

        timestamps = tod.local_times(margin=self._margin)
        if len(timestamps) != self.nsamp + 2 * self._margin:
            raise Exception('Cached time stamps do not include margins.')

        commonflags = tod.local_common_flags(margin=self._margin)
        if len(commonflags) != self.nsamp + 2 * self._margin:
            raise Exception('Cached common flags do not include margins.')

        phase = tod.local_phase(margin=self._margin)
        if len(phase) != self.nsamp + 2 * self._margin:
            raise Exception('Cached phases do not include margins.')
        # Make sure phase is in [0, 2pi[.  Otherwise signal
        # estimation will fail binning
        phase %= 2 * np.pi

        velocity = tod.local_velocity(margin=self._margin)
        if len(velocity) != self.nsamp + 2 * self._margin:
            raise Exception('Cached velocities do not include margins.')

        if self._pntmask is not None:
            pntflag = (commonflags & self._pntmask) != 0
        else:
            pntflag = None

        fsample = 1. / np.median(np.diff(timestamps))

        if self._lfi_mode:
            parity = None
            phase4k = None
            dark1 = None
            dark2 = None
            darkflag1 = None
            darkflag2 = None
        else:
            parity = (
                np.arange(self.nsamp + 2 * self._margin) +
                tod.globalfirst + 1 +
                tod.local_samples[0] - self._margin) % 2
            phase4k = (
                np.arange(self.nsamp + 2 * self._margin) +
                tod.globalfirst + 1 +
                tod.local_samples[0] - self._margin) % self._nphase4k

            if not self._shdet_mode:
                dark1 = tod.local_dark('Dark-1', margin=self._margin)
                dark2 = tod.local_dark('Dark-2', margin=self._margin)
                darkflag1 = tod.local_flags('Dark-1', margin=self._margin)
                darkflag2 = tod.local_flags('Dark-2', margin=self._margin)
                if len(dark1) != self.nsamp + 2 * self._margin:
                    raise Exception('Cached darks do not include margins.')
                # Purge the cached, detector-specific EFF data
                tod.purge_eff_cache(keep_common=True)

        if self._bad_rings is not None:
            ringmasker = RingMasker(self._bad_rings)
        else:
            ringmasker = None

        if not self._lfi_mode and self._preproc_dark and not self._shdet_mode:
            # Light version of the pipeline to process the dark bolometers
            for det, darktod, darkflag in [
                    ('Dark-1', dark1, darkflag1),
                    ('Dark-2', dark2, darkflag2)]:
                self.process_dark_bolo(
                    det, darktod, darkflag, tod, fsample, timestamps,
                    pntflag, phase, ringmasker, parity)

            if self._effdir_out is not None:
                if self.rank == 0:
                    print('Saving dark data to {}'.format(
                        self._effdir_out), flush=True)
                nwrite = 10
                for iwrite in range(nwrite):
                    self.comm.barrier()
                    timer = Timer()
                    timer.start()
                    if self.rank == 0:
                        print('Writing dark TOI to {} {} / {}'.format(
                            self._effdir_out, iwrite + 1, nwrite),
                            flush=True)
                    if self.rank % nwrite == iwrite:
                        ind = slice(self._margin, len(dark1) - self._margin)
                        tod.write_dark(
                            dark1=dark1[ind], darkflag1=darkflag1[ind],
                            dark2=dark2[ind], darkflag2=darkflag2[ind],
                            effdir_out=self._effdir_out)
                    self.comm.barrier()
                    timer.stop()
                    if self.rank == 0:
                        timer.report('Write dark TOI to {} : {} / {}'
                              ''.format(self._effdir_out, iwrite + 1, nwrite))

                if self.rank == 0:
                    print('Saved dark TOI to file', flush=True)

        # Now the optical channels

        if self._flag_planets:
            self.planetflagger = PlanetFlagger()
        else:
            self.planetflagger = None

        for det in tod.local_dets:
            if self._preproc_dark and self._margin != 0:
                if self.rank == 0:
                    print('WARNING: Cannot process optical detectors after '
                          'dark ones. The margins are not populated.',
                          flush=True)
                return
            loop_timer = Timer()
            loop_timer.start()
            signal_out, flags_out = self.process_optical_detector(
                det, tod, fsample, timestamps, commonflags, ringmasker,
                dark1, darkflag1, dark2, darkflag2, pntflag, phase,
                velocity, parity, phase4k)

            # Write detector data
            ind = slice(self._margin, len(signal_out) - self._margin)
            cachename = '{}_{}'.format(tod.SIGNAL_NAME, det)
            if self.rank == 0:
                print('Caching detector data to {}'.format(
                    cachename), flush=True)
            tod.cache.put(cachename, signal_out[ind], replace=True)

            cachename = '{}_{}'.format(tod.FLAG_NAME, det)
            tod.cache.put(cachename, flags_out[ind], replace=True)

            if self._effdir_out is not None:
                if self.rank == 0:
                    print('Saving detector data to {}'.format(
                        self._effdir_out), flush=True)
                nwrite = 10
                for iwrite in range(nwrite):
                    self.comm.barrier()
                    timer = Timer()
                    timer.start()
                    if self.rank == 0:
                        print('Writing {} TOI to {} : {} / {}'.format(
                            det, self._effdir_out, iwrite + 1, nwrite),
                            flush=True)
                    if self.rank % nwrite == iwrite:
                        tod.write_tod_and_flags(
                            detector=det, data=signal_out[ind],
                            flags=flags_out[ind],
                            effdir_out=self._effdir_out)
                    self.comm.barrier()
                    timer.stop()
                    if self.rank == 0:
                        timer.report('Write {} TOI to {} : {} / {}'.format(
                            det, self._effdir_out, iwrite + 1, nwrite))

            # Purge the cached, detector-specific EFF data

            tod.purge_eff_cache(keep_common=True)

            loop_timer.stop()
            if self.rank == 0:
                loop_timer.report('Process {}'.format(det))

        # Purge the cached EFF data

        tod.purge_eff_cache()

        # Write common data

        ind = slice(self._margin, len(signal_out) - self._margin)

        time_out = timestamps[ind].copy()
        del timestamps
        tod.cache.put(tod.TIMESTAMP_NAME, time_out, replace=True)
        del time_out

        phase_out = phase[ind].copy()
        del phase
        tod.cache.put(tod.PHASE_NAME, phase_out, replace=True)
        del phase_out

        vel_out = velocity[ind].copy()
        del velocity
        tod.cache.put(tod.VELOCITY_NAME, vel_out, replace=True)
        del vel_out

        cachename = None
        commonflags_out = np.zeros_like(commonflags)
        commonflags_out |= np.uint8(1) * (
            (commonflags & (255 - self._pntmask)) != 0)
        if pntflag is not None:
            commonflags_out |= np.uint8(2) * pntflag

        del commonflags
        tod.cache.put(tod.COMMON_FLAG_NAME, commonflags_out[ind], replace=True)

        if self._preproc_common and self._effdir_out is not None:
            if self.rank == 0:
                print('Saving common flags to {}'.format(
                    self._effdir_out), flush=True)
            nwrite = 10
            for iwrite in range(nwrite):
                self.comm.barrier()
                if self.rank == 0:
                    print('Writing common flags to {} {} / {}'.format(
                        self._effdir_out, iwrite + 1, nwrite),
                        flush=True)
                if self.rank % nwrite == iwrite:
                    tod.write_common_flags(flags=commonflags_out[ind],
                                           effdir_out=self._effdir_out)
                self.comm.barrier()
                if self.rank == 0:
                    print('Wrote common flags to {} : {} / {}'
                          ''.format(
                              self._effdir_out, iwrite + 1,
                              nwrite), flush=True)
        del commonflags_out

        if not self._lfi_mode and not self._shdet_mode:
            # Cache dark bolometers
            dark1 = dark1.copy()
            dark2 = dark2.copy()
            darkflag1 = darkflag1.copy()
            darkflag2 = darkflag2.copy()
            for det, dark_out, darkflags_out in zip(
                    ['Dark-1', 'Dark-2'], [dark1, dark2],
                    [darkflag1, darkflag2]):
                cachename = '{}_{}'.format(tod.SIGNAL_NAME, det)
                tod.cache.put(cachename, dark_out[ind], replace=True)

                cachename = '{}_{}'.format(tod.FLAG_NAME, det)
                tod.cache.put(cachename, darkflags_out[ind], replace=True)
            del dark1
            del dark2
            del darkflag1
            del darkflag2

        # Trim the margins out from the cached objects

        ind = slice(self._margin, self.nsamp + self._margin)
        for key in tod.cache.keys():
            nsamp_tot = len(tod.cache.reference(key))
            if nsamp_tot == self.nsamp + 2 * self._margin:
                buf = tod.cache.reference(key)[ind].copy()
                tod.cache.put(key, buf, replace=True)
                del buf

        return

    def _get_baseline(self, signal, flag, wkernel=10822, threshold=3.5):

        # Convolve with the 3-point kernel

        good = np.logical_not(flag)

        offset = np.mean(signal[good])

        result = np.convolve(signal - offset, [0.25, 0.5, 0.25], mode='same')
        flag_out = np.convolve(flag, [1, 1, 1], mode='same') != 0

        # Apply crude deglitching, re-applying the 3-pt filter if needed

        for _ in range(10):
            good = np.logical_not(flag_out)
            rms = np.std(result[good])
            if np.isnan(rms):
                raise Exception('RMS is NaN')
            outliers = np.abs(result) > threshold * rms
            outliers[flag_out] = False

            if np.sum(outliers) == 0:
                break

            flag_out[outliers] = True
            good = np.logical_not(flag_out)

            if np.sum(good) == 0:
                raise Exception('No good samples left.')

            offset = np.mean(signal[good])
            result = np.convolve(signal - offset, [0.25, 0.5, 0.25],
                                 mode='same')
            flag_out = np.convolve(flag_out, [1, 1, 1], mode='same') != 0

        # Now compute a masked running mean

        baseline = flagged_running_average(result, flag_out, wkernel)

        baseline += offset

        return baseline, flag_out

    def _get_outliers(self, ringinfo, fields, tol=5.0, jumpmax=2,
                      wmean=1001, werr=1001):

        outliers = np.zeros(len(ringinfo), dtype=np.bool)

        outliers[ringinfo.failed] = True
        outliers[ringinfo.pnt_frac > .2] = True

        for field in fields:
            x = ringinfo[field]
            if np.all(x == 0):
                # This field was not used
                continue
            outliers[np.logical_not(np.isfinite(x))] = True
            for i in range(10):
                good = np.logical_not(outliers)
                # make sure we only raise flags for previously unflagged rings
                xx = np.zeros_like(x)
                if i == 0 or len(ringinfo) < 100:
                    # very rough first pass
                    if np.sum(good) > 300:
                        median = medfilt(x[good], 101)
                    else:
                        median = np.median(x[good])
                    xx[good] = x[good] - median
                    rms = np.std(xx[good])
                    outliers[np.abs(xx) > tol * rms * 2] = True
                else:
                    wmean = min(wmean, len(x) // 10)
                    werr = min(werr, len(x) // 10)
                    median = flagged_running_average(x, outliers, wmean)
                    xx[good] = x[good] - median[good]
                    rms = np.sqrt(flagged_running_average(
                        (x - median) ** 2, outliers, werr))
                    nout1 = np.sum(outliers)
                    outliers[np.abs(xx) > tol * rms] = True
                    nout2 = np.sum(outliers)
                    if nout1 == nout2:
                        break

        outliers[ringinfo.njump > jumpmax] = True

        outliers[ringinfo.failed] = False
        ringinfo.outlier[outliers] = True

        return ringinfo.ring_number[outliers]

    def _test_despiker_gaps(self, flg, pntflg, ssoflg, phse, tol, ring_number):

        # Raise the flag for stray unflagged samples
        n = 5
        flg_test = convolve(flg != 0, np.ones(n) / n, mode='same') > (n - 2) / n

        # First determine the unflagged gap length to handle resonant rings

        if self.last_ring_number == ring_number:
            ii = self.last_sorted
            min_tol = self.last_tol
            dxmax0 = self.last_dxmax0
        else:
            ii = np.argsort(phse)
            self.last_ring_number = ring_number
            self.last_sorted = ii
            # Test for resonance -- implies larger gaps
            good = pntflg == 0
            x = phse[ii][good[ii]]
            dx = np.diff(np.append(x, x[0] + 2 * np.pi))
            dxmax0 = np.amax(dx)
            if dxmax0 > tol:
                print('{:4} : Ring {:5} is resonant -- increasing gap '
                      'tolerance from {} to {}'.format(
                          self.rank, ring_number, tol / arcmin,
                          dxmax0 / arcmin * 1.1))
                min_tol = dxmax0 * 1.1
            else:
                min_tol = 0
            self.last_tol = min_tol
            self.last_dxmax0 = dxmax0

        tol = max(tol, min_tol)
        good = flg_test == 0
        good[pntflg] = False
        if ssoflg is not None:
            good[ssoflg] = True
        x = phse[ii][good[ii]]
        dx = np.diff(np.append(x, x[0] + 2 * np.pi))
        isgap = dx > tol
        ngap = np.sum(isgap)

        if ngap > 0:
            gapflag = np.zeros_like(flg)
            for istart, istep in enumerate(dx):
                if istep > tol:
                    gapstart = x[istart]
                    if istart + 1 < x.size:
                        gapstop = x[istart + 1]
                        gapflag[np.logical_and(phse >= gapstart,
                                               phse <= gapstop)] = True
                    else:
                        gapstop = x[0]
                        gapflag[np.logical_and(phse <= gapstop,
                                               phse >= gapstart)] = True
            gapflag[pntflg] = False
            dxmax = np.amax(dx)
            dxtot = np.sum(dx[isgap])
            print('{:4} : Ring {:5} Flagging produced {} gaps larger than '
                  '{:.2f} arc min. Gaps total {:.3f} degrees. Largest gap is '
                  '{:.3f} arcmin. Largest unflagged gap is {:.3f} arcmin.'
                  ''.format(self.rank, ring_number, ngap, tol / arcmin,
                            dxtot / degree, dxmax / arcmin, dxmax0 / arcmin))
            return gapflag

        # Finding full gaps is insufficient. We can still have segments
        # of the spin phase that have substantially more flagged samples

        phase_step = np.radians(1)  # use degree-wide bins
        nbin = int(2 * np.pi / phase_step)
        # Adjust the bin width so that the last bin has the same width
        phase_step = 2 * np.pi / nbin
        phase_bin = (x // phase_step).astype(np.int32)
        nbin = int(np.ceil(2 * np.pi / phase_step))
        hits = np.zeros(nbin, dtype=np.int32)
        fast_hit_binning(phase_bin, hits)
        """
        bin_start = 0
        i = 0
        bin_hits = []
        while bin_start < 2 * np.pi:
            bin_stop = bin_start + phase_step
            hits = 0
            while i < len(x) and x[i] < bin_stop:
                hits += 1
                i += 1
            bin_hits.append(hits)
            if i == len(x):
                break
            bin_start = bin_stop
        hits = np.array(bin_hits)
        """
        hitmean = np.mean(hits)
        hitstd = np.std(hits)
        err = (hitmean - np.amin(hits)) / hitstd
        if err > 6:
            # 6 sigma outlier in hits!
            print('{:4} : Ring {:5} Mean hits/deg = {:.2f}. RMS = {:.2f}. '
                  'minimum is = {:.2f} sigma, {} hits'.format(
                      self.rank, ring_number, hitmean, hitstd,
                      err, np.amin(hits)), flush=True)
            return None

        return None
