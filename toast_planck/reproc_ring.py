# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""
reproc_ring provides one class: OpReprocRing -- the NPIPE reprocessing
operator.
"""
# from memory_profiler import profile
# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)

from collections import OrderedDict
import copy
import glob
import os
import pickle

import astropy.io.fits as pf
import healpy as hp
import numpy as np
from toast import qarray as qa
import toast
from toast.todmap import OpMadam
from toast.mpi import MPI
import toast.timing as timing
import toast.tod as tt
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers

from .utilities import freq_to_fwhm, read_gains, get_pair, plug_holes
from .preproc_modules import Dipoler, MapSampler, RingMasker
from .preproc_modules.filters import flagged_running_average
from .reproc_modules import Differentiator, SkyModel, UltimateDestriper, Zodier
from .reproc_modules.destripe_tools import (
    bin_ring,
    bin_ring_extra,
    get_glob2loc,
    fast_scanning,
    fast_scanning32,
)

XAXIS, YAXIS, ZAXIS = np.eye(3)

cached_mapsamplers = {}


class RingTemplate(object):
    def __init__(self, template, offset):
        self.template = template
        self.offset = offset


@function_timer
def get_skymodel(
    freq,
    nside,
    sync,
    sync_pol,
    freefree,
    ame,
    dust,
    dust_pol,
    comm,
    fwhm,
    cachedir,
    cache,
    bpcorrect,
    bpcorrect2,
):
    t1 = MPI.Wtime()
    if comm.rank == 0:
        os.makedirs(cachedir, exist_ok=True)
    fn_skymodel = os.path.join(
        cachedir,
        "sky_model_{:03}GHz_nside{:04}_fwhm{:.1f}.fits".format(freq, nside, fwhm),
    )
    fn_skymodel_deriv = os.path.join(
        cachedir,
        "sky_model_deriv_{:03}GHz_nside{:04}_fwhm{:.1f}.fits".format(freq, nside, fwhm),
    )
    fn_skymodel_deriv2 = os.path.join(
        cachedir,
        "sky_model_deriv2_{:03}GHz_nside{:04}_fwhm{:.1f}.fits".format(
            freq, nside, fwhm
        ),
    )
    mapsampler_fg = None
    mapsampler_fg_deriv = None
    mapsampler_fg_deriv2 = None
    try:
        fg_iqu = None
        if comm.rank == 0 and os.path.isfile(fn_skymodel):
            print("Loading cached sky model from " + fn_skymodel, flush=True)
            fg_iqu = hp.read_map(fn_skymodel, None, nest=True)
        fg_iqu = comm.bcast(fg_iqu)
        if fg_iqu is None:
            raise RuntimeError("Failed to load " + fn_skymodel)
        mapsampler_fg = MapSampler(
            None, pol=True, comm=comm, preloaded_map=fg_iqu, nest=True, nside=nside
        )
        if bpcorrect:
            fg_deriv_iqu = None
            if comm.rank == 0 and os.path.isfile(fn_skymodel_deriv):
                print("Loading cached sky model from " + fn_skymodel_deriv, flush=True)
                fg_deriv_iqu = hp.read_map(fn_skymodel_deriv, None, nest=True)
            fg_deriv_iqu = comm.bcast(fg_deriv_iqu)
            if fg_deriv_iqu is None:
                raise RuntimeError("Failed to load " + fn_skymodel_deriv)
            mapsampler_fg_deriv = MapSampler(
                None,
                pol=True,
                comm=comm,
                preloaded_map=fg_deriv_iqu,
                nest=True,
                nside=nside,
            )
        if bpcorrect2:
            fg_deriv2_iqu = None
            if comm.rank == 0 and os.path.isfile(fn_skymodel_deriv2):
                print("Loading cached sky model from " + fn_skymodel_deriv2, flush=True)
                fg_deriv2_iqu = hp.read_map(fn_skymodel_deriv2, None, nest=True)
            fg_deriv2_iqu = comm.bcast(fg_deriv2_iqu)
            if fg_deriv2_iqu is None:
                raise RuntimeError("Failed to load " + fn_skymodel_deriv2)
            mapsampler_fg_deriv2 = MapSampler(
                None,
                pol=True,
                comm=comm,
                preloaded_map=fg_deriv2_iqu,
                nest=True,
                nside=nside,
            )
    except Exception as e:
        if comm.rank == 0:
            print(
                "Failed to load cached sky model from {} ({}). "
                "Caching a new one.".format(cachedir, e),
                flush=True,
            )
        skymodel = SkyModel(
            nside, sync, sync_pol, freefree, ame, dust, dust_pol, comm, fwhm=fwhm
        )
        fg_iqu = np.array(skymodel.eval(freq))
        fg_iqu = hp.reorder(fg_iqu, r2n=True)
        if comm.rank == 0:
            print("Caching sky model to {}".format(fn_skymodel), flush=True)
            hp.write_map(fn_skymodel, fg_iqu, nest=True, coord="G", overwrite=True)
        mapsampler_fg = MapSampler(
            "fg",
            pol=True,
            nside=nside,
            comm=comm,
            cache=cache,
            preloaded_map=fg_iqu,
            nest=True,
        )
        if bpcorrect:
            # Measure frequency derivative of the foreground model
            step = 0.01  # in GHz
            fg2 = np.array(skymodel.eval(freq + step))
            fg2 = hp.reorder(fg2, r2n=True)
            fg_deriv_iqu = (fg2 - fg_iqu) / step
            if comm.rank == 0:
                print("Caching sky model to {}".format(fn_skymodel_deriv), flush=True)
                hp.write_map(
                    fn_skymodel_deriv,
                    fg_deriv_iqu,
                    nest=True,
                    coord="G",
                    overwrite=True,
                )
            # bandpass mismatch template
            mapsampler_fg_deriv = MapSampler(
                "fg_deriv",
                pol=True,
                nside=nside,
                comm=comm,
                cache=cache,
                preloaded_map=fg_deriv_iqu,
                nest=True,
            )
        if bpcorrect2:
            # Measure the second derivative for another template
            step = 0.01
            fg3 = np.array(skymodel.eval(freq + step + step))
            fg3 = hp.reorder(fg3, r2n=True)
            fg_deriv_iqu2 = (fg3 - fg2) / step
            fg_deriv2_iqu = (fg_deriv_iqu2 - fg_deriv_iqu) / step
            del fg3, fg_deriv_iqu2
            if comm.rank == 0:
                print("Caching sky model to {}".format(fn_skymodel_deriv2), flush=True)
                hp.write_map(fn_skymodel_deriv2, fg_deriv2_iqu, nest=True, coord="G")
            mapsampler_fg_deriv2 = MapSampler(
                "fg_deriv2",
                pol=True,
                nside=nside,
                comm=comm,
                cache=cache,
                preloaded_map=fg_deriv2_iqu,
                nest=True,
                overwrite=True,
            )
    t2 = MPI.Wtime()
    if comm.rank == 0:
        print("Initialized the sky model in {:.2f}s".format(t2 - t1), flush=True)
    return mapsampler_fg, mapsampler_fg_deriv, mapsampler_fg_deriv2


class OpReprocRing(toast.Operator):
    """
    Operator for timestream re-processing in ring domain.
    """

    @function_timer
    def __init__(
        self,
        freq,
        pntflags="pointing_flags",
        nharm=20,
        do_bands=False,
        do_derivs=True,
        detmask=1,
        commonmask=1,
        pntmask=2,
        ssomask=2,
        maskfile=None,
        maskfile_bp=None,
        maskfile_project=None,
        polmap=None,
        polmap2=None,
        polmap3=None,
        force_polmaps=False,
        pol_fwhm=60,
        pol_lmax=512,
        pol_nside=256,
        pix_nside=None,
        do_zodi=True,
        differential_zodi=True,
        independent_zodi=False,
        zodi_cache="./zodi_cache",
        skymodel_cache="./skymodel_cache",
        do_dipo=True,
        nside=1024,
        destriper_pixlim=1e-2,
        do_fsl=True,
        split_fsl=False,
        cmb=None,
        co=None,
        co2=None,
        co3=None,
        dust=None,
        dust_pol=None,
        sync=None,
        sync_pol=None,
        ame=None,
        freefree=None,
        recalibrate=True,
        bandpass_nside=256,
        bandpass_fwhm=60,
        effdir_out=None,
        save_tod=True,
        map_dir="./",
        niter=1,
        calibrate=True,
        nlcalibrate=False,
        fit_distortion=False,
        save_maps=True,
        save_survey_maps=True,
        save_template_maps=False,
        save_single_maps=True,
        single_nside=None,
        out=".",
        forcepol=False,
        forcefsl=False,
        fslnames=None,
        asymmetric_fsl=False,
        pscorrect=False,
        psradius=30,
        bpcorrect=False,
        bpcorrect2=False,
        fgdipole=False,
        madampars=None,
        bad_rings=None,
        temperature_only=False,
        temperature_only_destripe=False,
        temperature_only_intermediate=False,
        calfile=None,
        calfile_iter=None,
        effective_amp_limit=0.02,
        gain_step_mode=None,
        min_step_length=10,
        max_step_length=100,
        outlier_threshold=10.0,
        restore_dipole=False,
        symmetrize=False,
        ssofraclim=0.01,
        mcmode=False,
        cache_dipole=True,
        quss_correct=False,
        # Precomputed sky model mapsamplers
        fg=None,
        fg_deriv=None,
        cmb_mc=None,
        force_conserve_memory=False,
        polparammode=False,
        save_destriper_data=False,
    ):
        """
        Args:
            input (str):  if None, read input TOD, otherwise the name
                of the cached data.
            output (str):  if None, write TOD, otherwise the name to use
                in the cache.

        """
        self.fg = fg
        self.fg_deriv = fg_deriv
        self.cmb_mc = cmb_mc
        self.ssofraclim = ssofraclim
        self.symmetrize = symmetrize
        self.effective_amp_limit = effective_amp_limit
        if gain_step_mode not in [None, 'mission', 'years', 'survey']:
            raise RuntimeError(
                'Invalid gain step mode: {}'.format(gain_step_mode))
        self.gain_step_mode = gain_step_mode
        self.min_step_length = min_step_length
        self.max_step_length = max_step_length
        self.outlier_threshold = outlier_threshold
        self.freq = int(freq)
        self.fwhm = freq_to_fwhm(self.freq)
        self.pntflags = pntflags
        self.nharm = nharm
        self.do_bands = do_bands
        self.do_derivs = do_derivs
        self.detmask = detmask  # Default value (1) assumes preprocessing
        self.commonmask = commonmask  # Default value (1) assumes preprocessing
        self.pntmask = pntmask  # Default value (2) assumes preprocessing
        self.ssomask = ssomask  # Default value (2) assumes preprocessing
        self.maskfile = maskfile
        self.maskfile_bp = maskfile_bp
        self.maskfile_project = maskfile_project
        self.mask = None
        self.mask_bp = None
        self.polmap = polmap
        self.polmap2 = polmap2
        self.polmap3 = polmap3
        self.force_polmaps = force_polmaps
        self.pol_fwhm = pol_fwhm
        self.pol_lmax = pol_lmax
        self.pix_nside = pix_nside
        self.pol_nside = min(pol_nside, pix_nside)
        self.single_nside = single_nside
        self.nside = min(nside, pix_nside)
        self.destriper_pixlim = destriper_pixlim
        self.npix = hp.nside2npix(nside)
        self.ndegrade = (self.pix_nside // self.nside) ** 2
        self.do_zodi = do_zodi
        self.differential_zodi = differential_zodi
        self.independent_zodi = independent_zodi
        self.zodi_cache = zodi_cache
        self.skymodel_cache = skymodel_cache
        self.restore_dipole = restore_dipole
        self.do_dipo = do_dipo
        self.do_fsl = do_fsl
        self.split_fsl = split_fsl
        self.cmb = cmb
        self.co1 = co
        self.co2 = co2
        self.co3 = co3
        self.dust = dust
        self.dust_pol = dust_pol
        self.sync = sync
        self.sync_pol = sync_pol
        self.ame = ame
        self.freefree = freefree
        self.recalibrate = recalibrate
        self.fgdipole = fgdipole
        self.bandpass_nside = min(bandpass_nside, pix_nside)
        self.bandpass_fwhm = max(bandpass_fwhm, self.fwhm)
        self.effdir_out = effdir_out
        self.save_tod = save_tod
        self.map_dir = map_dir
        self.save_maps = save_maps
        self.save_survey_maps = save_survey_maps
        self.save_template_maps = save_template_maps
        self.save_single_maps = save_single_maps
        self.calibrate = calibrate
        self.nlcalibrate = nlcalibrate
        self.fit_distortion = fit_distortion
        self.nllowpass = 1
        if niter is not None:
            if not self.calibrate and niter > 1:
                raise ValueError("Cannot iterate without calibrating")
            self.niter = niter
        else:
            self.niter = 1
        if out is None:
            self.out = "./"
        else:
            self.out = out
        self.forcepol = forcepol
        self.forcefsl = forcefsl
        self.fslnames = fslnames
        self.asymmetric_fsl = asymmetric_fsl
        self.pscorrect = pscorrect
        self.psradius = psradius
        self.bpcorrect = bpcorrect or quss_correct
        self.bpcorrect2 = bpcorrect2
        # Monte Carlo mode does not support bpcorrect2
        if self.bpcorrect2 and self.fg_deriv is not None:
            self.bpcorrect2 = False
        self.madampars = madampars
        self.bad_rings = bad_rings
        self.calfile = calfile
        self.calfile_iter = calfile_iter
        self.last_gainestimators = {}
        self.last_gains = {}
        self.best_fit_amplitudes = {}
        self.pol_amplitudes = {}
        self.rough_gains = None
        self.old_gains = None

        self.quss_correct = quss_correct
        if self.quss_correct:
            self.symmetrize = True
        self.temperature_only = temperature_only
        self.temperature_only_intermediate = temperature_only_intermediate
        self.temperature_only_destripe = temperature_only_destripe
        self.mcmode = mcmode
        self.polparammode = polparammode
        self.save_destriper_data = save_destriper_data
        self.cache_dipole = cache_dipole

        self.mapsampler_freq = None
        self.mapsamplers = {}
        self.freq_symmetric = ["pol0", "pol1", "pol2", "dipo_foreground"]
        if not self.independent_zodi:
            self.freq_symmetric += [
                "zodi band1",
                "zodi band2",
                "zodi band3",
                "zodi blob",
                "zodi cloud",
                "zodi ring",
            ]
        self.horn_symmetric = ["dipo_orbital"]
        if not self.asymmetric_fsl:
            self.horn_symmetric.append("fsl")
        self.comm = None
        self.rank = None
        self.ntask = None
        self.rimo = None
        self.dets = None
        self.ndet = None
        self.fsample = None
        self.nring = None
        self.nring_tot = None
        self.gain_ring = None
        self.local_dipo_amp = None
        self.local_calib_rms = None
        self.local_starts = None
        self.local_stops = None
        self.local_start_times = None
        self.local_stop_times = None
        self.local_bad_rings = None
        self.data = None
        self.zodinames = None
        self.zodier = None
        self.namplitude_tot = None
        self.nsamp = None
        self.sample_offset = None
        self.siter = None
        self.iiter = None
        self.cache = None
        self.dipoler = None
        self.tod = None
        self.ring_offset = None
        self.differentiator = None
        self.detweights = None
        self.ring_numbers = None
        self.template_offsets = None
        self.ngain = None
        self.gain_rms = {}
        self.gain_mean = {}
        self.template_names = None
        self.single_detector_mode = False
        self.force_conserve_memory = force_conserve_memory

        super().__init__()

    @function_timer
    def get_header(self, det):
        """
        Return a list of FITS header entries for best fit amplitudes
        """
        header = []
        for key in sorted(self.best_fit_amplitudes[det]):
            value = self.best_fit_amplitudes[det][key]
            if len(key) > 8:
                key = "HIERARCH " + key
            header.append((key, value))
        # Polarization template amplitudes are in a different dictionary,
        # unless they were subtracted from the TOD
        for key in sorted(self.pol_amplitudes[det]):
            value = self.pol_amplitudes[det][key]
            if len(key) > 8:
                key = "HIERARCH " + key
            header.append((key, value))
        return header

    @function_timer
    def symmetrize_pointing_and_flags(self):
        """
        Symmetrize flags (and pixels) between detectors sharing a horn
        """
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("Symmetrizing flags and pointing", flush=True)

        for det1 in self.dets:
            det2 = self.get_pair(det1)
            if det2 is None or det2[-1] not in "bS":
                continue
            if det2 not in self.dets:
                continue
            flags1 = self.tod.local_flags(det1)
            flags2 = self.tod.local_flags(det2)
            flags1 |= flags2
            flagsum = self.comm.reduce(np.sum(flags1 != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after symmetrize"
                    "".format(frac, det1),
                    flush=True,
                )
            del flags1, flags2
            for name in [
                self.tod.FLAG_NAME,
                self.tod.PIXEL_NAME,
                self.tod.POINTING_NAME,
            ]:
                cachename = "{}_{}".format(name, det1)
                paircachename = "{}_{}".format(name, det2)
                self.cache.destroy(paircachename)
                self.cache.add_alias(paircachename, cachename)

        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "Flags and pointing symmetrized in {:.2f} s" "".format(stop - start),
                flush=True,
            )
        return

    @function_timer
    def symmetrize_pixels(self):
        """ OpMadam will replace aliases with copies.  This method
        returns the pixel number aliases.
        """
        if not self.symmetrize:
            return
        self.comm.Barrier()
        for det1 in self.dets:
            det2 = self.get_pair(det1)
            if det2 is None or det2[-1] not in "bS":
                continue
            if det2 not in self.dets:
                continue
            for name in [self.tod.PIXEL_NAME]:
                cachename = "{}_{}".format(name, det1)
                paircachename = "{}_{}".format(name, det2)
                if self.cache.exists(paircachename):
                    self.cache.destroy(paircachename)
                self.cache.add_alias(paircachename, cachename)
        return

    @function_timer
    def load_tod(self, data):
        """
        Load and cache pointing and signal
        """
        nobs = len(data.obs)
        if nobs != 1:
            raise RuntimeError(
                "reprocessing assumes there is exactly one "
                "observation on every process, not {}".format(nobs)
            )

        self.comm.Barrier()
        start0 = MPI.Wtime()
        if self.rank == 0:
            print("Loading data", flush=True)

        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Loading common data", flush=True)

        obs = data.obs[0]
        tod = obs["tod"]

        self.nsamp = tod.local_samples[1]
        self.nsamp_tot = self.comm.allreduce(self.nsamp)
        self.sample_offset = tod.globalfirst + tod.local_samples[0]
        self.cache = tod.cache
        self.tod = tod
        self.rimo = tod.RIMO

        self.detweights = {}
        for det in tod.local_dets:
            net = tod.rimo[det].net
            fsample = tod.rimo[det].fsample
            self.detweights[det] = 1.0 / (fsample * net ** 2)

        if self.effdir_out is not None:
            tod.cache_metadata(self.effdir_out, comm=self.comm)

        timestamps = tod.local_times()

        self.fsample = 1.0 / np.median(np.diff(timestamps))

        commonflags = tod.local_common_flags()

        if tod.cache.exists(self.pntflags):
            pntflags = tod.cache.reference(self.pntflags)
        else:
            pntflags = (commonflags & self.pntmask) != 0
            pntflags = tod.cache.put(self.pntflags, pntflags.astype(np.uint8))

        commonflags = (commonflags & self.commonmask) != 0

        commonsum = self.comm.reduce(np.sum(commonflags))
        pntsum = self.comm.reduce(np.sum(pntflags))
        if self.rank == 0:
            commonfrac = commonsum * 100 / self.nsamp_tot
            pntfrac = pntsum * 100 / self.nsamp_tot
            print(
                "{:8.3f} % of the TOD is flagged with common flags\n"
                "{:8.3f} % of the TOD is flagged with pointing flags"
                "".format(commonfrac, pntfrac),
                flush=True,
            )

        if self.do_zodi:
            position = tod.local_position()
        else:
            position = None

        # Cache phase
        tod.local_phase()

        velocity = tod.local_velocity()

        intervals = tod.local_intervals(obs["intervals"])
        starts = [ival.first for ival in intervals]
        stops = [ival.last - 1 for ival in intervals]
        self.local_starts = np.array(starts)
        self.local_stops = np.array(stops)

        self.local_start_times = timestamps[self.local_starts]
        self.local_stop_times = timestamps[self.local_stops - 1]
        self.local_bad_rings = np.zeros(len(intervals), dtype=bool)

        # Establish a ring index relative to the first ring on the first
        # process.

        self.ring_offset = 0
        for interval in obs["intervals"]:
            if interval.last < tod.local_samples[0]:
                self.ring_offset += 1

        self.nring = len(self.local_starts)
        self.nring_tot = self.comm.allreduce(self.nring)

        # These are absolute ring numbers, not relative to the first process

        self.ring_numbers = (
            np.arange(self.nring) + self.ring_offset + tod.globalfirst_ring
        )

        if self.bad_rings is not None:
            ringmasker = RingMasker(self.bad_rings)
        else:
            ringmasker = None

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print("    Read common TOD in {:.2f} s".format(stop - start), flush=True)
            # if (self.mcmode and stop - start > 30):
            #    print('ERROR: MC loading should only take a few seconds. '
            #          'Terminating.', flush=True)
            #    self.comm.Abort(-1)

        self.dets = tod.local_dets
        self.ndet = len(self.dets)
        for det in self.dets:
            self.best_fit_amplitudes[det] = {}
            self.pol_amplitudes[det] = {}

        for det in self.dets:
            pairdet = self.get_pair(det)
            start = MPI.Wtime()
            if self.rank == 0:
                print("    Loading detector {}".format(det), flush=True)

            # Get signal
            signal = tod.local_signal(det)
            if self.calfile is not None:
                if self.rank == 0:
                    print(
                        "    Calibrating signal using {}".format(self.calfile),
                        flush=True,
                    )
                gains = read_gains(self.calfile, det, timestamps[0], timestamps[-1])
                tt.calibrate(timestamps, signal, *gains, inplace=True)

            # Get flags
            flags = tod.local_flags(det)

            flags[:] = flags & (self.detmask | self.ssomask)

            flagsum = self.comm.reduce(np.sum(flags != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged with detector flags"
                    "".format(frac, det),
                    flush=True,
                )

            for istart, istop in zip(self.local_starts, self.local_stops):
                ind = slice(istart, istop)
                ssoflags = flags[ind] & self.ssomask != 0
                ssofrac = np.sum(ssoflags) / ssoflags.size
                if ssofrac > self.ssofraclim and not self.mcmode:
                    flags[ind] |= 1

            flagsum = self.comm.reduce(np.sum(flags != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after excessive SSO flags"
                    "".format(frac, det),
                    flush=True,
                )

            flags[commonflags] |= 255

            flagsum = self.comm.reduce(np.sum(flags != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after applying common flags"
                    "".format(frac, det),
                    flush=True,
                )

            nan = np.isnan(signal)
            flags[nan] |= 1
            signal[nan] = 0

            flagsum = self.comm.reduce(np.sum(flags != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after flagging NaNs"
                    "".format(frac, det),
                    flush=True,
                )

            # Optionally apply ring flags
            if ringmasker is not None:
                ringmask = ringmasker.get_mask(timestamps, det)
                flags[ringmask] |= 255

            flagsum = self.comm.reduce(np.sum(flags != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after applying bad rings"
                    "".format(frac, det),
                    flush=True,
                )

            if self.restore_dipole:
                dipoler = Dipoler(comm=self.comm, RIMO=self.rimo)
                quat = self.tod.local_pointing()
                dipo = dipoler.dipole(quat, velocity=velocity, det=det)
                signal += dipo

            # Get IQU weights
            cachename = "{}_{}".format(self.tod.WEIGHT_NAME, det)
            if not tod.cache.exists(cachename):
                raise RuntimeError(
                    "reprocessign assumes that pointing weights are cached."
                )

            # Optionally get Far Side Lobe signal
            if self.do_fsl:
                fsl = tod.local_fsl(det)
                nan = np.isnan(fsl)
                nnan = np.sum(nan)
                nan[fsl == 0] = True
                nzero = np.sum(nan) - nnan
                if (nnan + nzero) != 0:
                    print(
                        "{:4} : WARNING : Flagging {} NaNs and {} zeroes in FSL TOD."
                        "".format(self.rank, nnan, nzero),
                        flush=True,
                    )
                    flags[nan] |= 1
                    fsl[nan] = 0
                # Average the FSL template in each horn
                if pairdet is not None:
                    cachename = "{}_{}".format(tod.FSL_NAME, det)
                    paircachename = "{}_{}".format(tod.FSL_NAME, pairdet)
                    if self.cache.exists(paircachename):
                        pairflags = tod.local_flags(pairdet)
                        pairfsl = tod.local_fsl(pairdet)
                        pairfsl = self.cache.reference(paircachename)
                        if fsl is pairfsl:
                            # Not already averaged
                            pairnan = pairfsl == 0
                            pairnnan = np.sum(pairnan)
                            # Symmetrize the FSL and associated flags
                            if nnan != 0:
                                pairflags[nan] |= 1
                                pairfsl[nan] = 0
                            if pairnnan != 0:
                                flags[nan] |= 1
                                fsl[nan] = 0
                            fsl[:] = 0.5 * (fsl + pairfsl)
                            del pairfsl, pairflags
                            self.cache.destroy(paircachename)
                            self.cache.add_alias(paircachename, cachename)
                del fsl

            flagsum = self.comm.reduce(np.sum(flags != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after applying FSL flags"
                    "".format(frac, det),
                    flush=True,
                )

            self.comm.Barrier()
            stop = MPI.Wtime()
            if self.rank == 0:
                print(
                    "    Read {} TOD in {:.2f} s".format(det, stop - start), flush=True
                )

        if self.do_fsl and self.forcefsl:
            # Remove FSL assuming unit amplitude
            for det in self.dets:
                signal = tod.local_signal(det)
                fsl = tod.local_fsl(det)
                signal -= fsl
                self.best_fit_amplitudes[det]["fsl"] = 1.0
                del signal
                del fsl

        # Use single precision where it makes sense

        for det in self.dets:
            quats = tod.local_pointing(det).astype(np.float32)
            self.cache.put("{}_{}".format(tod.POINTING_NAME, det), quats, replace=True)
            del quats
            weights = self.tod.local_weights(det).astype(np.float32)
            self.cache.put("{}_{}".format(tod.WEIGHT_NAME, det), weights, replace=True)
            del weights
        velocity = tod.local_velocity().astype(np.float32)
        self.cache.put(tod.VELOCITY_NAME, velocity, replace=True)
        del velocity
        if tod.cache.exists(tod.POSITION_NAME):
            position = tod.local_position().astype(np.float32)
            self.cache.put(tod.POSITION_NAME, position, replace=True)
            del position
        phase = tod.local_phase().astype(np.float32)
        self.cache.put(tod.PHASE_NAME, phase, replace=True)
        del phase

        self.comm.Barrier()
        stop0 = MPI.Wtime()
        if self.rank == 0:
            print("Read all data in {:.2f} s".format(stop0 - start0), flush=True)

        return

    @function_timer
    def compress_tod(self, rings, update=False):
        """
        Make rings and cache them.
        """
        self.comm.Barrier()
        if self.rank == 0:
            print("Compressing TOD", flush=True)

        if self.maskfile is not None and not update:
            # The processing mask rejects the galaxy for fitting
            # fainter templates.
            self.comm.Barrier()
            start = MPI.Wtime()
            if self.rank == 0:
                print("    Loading mask", flush=True)
            self.mask = MapSampler(
                self.maskfile,
                nside=self.nside,
                comm=self.comm,
                cache=self.cache,
                nest=True,
            )
            self.comm.Barrier()
            stop = MPI.Wtime()
            if self.rank == 0:
                print(
                    "    Loaded mask from {} in {:.2f} s"
                    "".format(self.maskfile, stop - start),
                    flush=True,
                )

        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Binning rings", flush=True)

        nbytes = 0

        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            ind = slice(istart, istop)
            pntflags = self.cache.reference(self.pntflags)[ind]
            phase = self.tod.local_phase()[ind]
            velocity = self.tod.local_velocity()[ind]
            if self.do_zodi:
                position = self.tod.local_position()[ind]
            if not update:
                rings[iring] = {}
                rings[iring]["velocity"] = np.mean(velocity, 0)
                if self.do_zodi:
                    rings[iring]["position"] = np.mean(position, 0)
                rings[iring]["ind"] = (istart, istop)
            ngood = 0
            for det in self.dets:
                if update:
                    if det not in rings[iring]:
                        continue
                signal = self.tod.local_signal(det)[ind]
                detflags = self.tod.local_flags(det)[ind]
                quat = self.tod.local_pointing(det)[ind]
                iquweights = self.tod.local_weights(det)[ind]
                pixels = self.tod.local_pixels(det)[ind]

                if update:
                    # Check that a new and old compressed TOD conform
                    # so the templates need not be redone.
                    old_ring = rings[iring][det]
                    new_ring = bin_ring(
                        pixels // self.ndegrade,
                        signal.astype(np.float64),
                        iquweights.astype(np.float64),
                        quat.astype(np.float64),
                        phase.astype(np.float64),
                        detflags | pntflags,
                        self.mask.Map[:],
                    )
                    if new_ring is None or new_ring.signal.size < 10:
                        del rings[iring][det]
                        continue
                    if old_ring.signal.size != new_ring.signal.size or np.sum(
                        old_ring.hits
                    ) != np.sum(new_ring.hits):
                        print(
                            "WARNING: {} : Mismatch between new and old rings "
                            "on ring {} ({}) {}. "
                            "old size = {}, new size = {}, "
                            "old ind = {}, new ind = {}, "
                            "old nhit = {}, new nhit = {}, "
                            "old rms = {}, new rms = {}".format(
                                self.rank,
                                self.ring_offset + iring,
                                iring,
                                det,
                                old_ring.signal.size,
                                new_ring.signal.size,
                                rings[iring]["ind"],
                                (istart, istop),
                                np.sum(old_ring.hits),
                                np.sum(new_ring.hits),
                                np.std(old_ring.signal),
                                np.std(new_ring.signal),
                            ),
                            flush=True,
                        )
                        del rings[iring][det]
                        continue
                    rings[iring][det] = new_ring
                    nbytes += new_ring.nbytes
                else:
                    ring = bin_ring(
                        pixels // self.ndegrade,
                        signal.astype(np.float64),
                        iquweights.astype(np.float64),
                        quat.astype(np.float64),
                        phase.astype(np.float64),
                        detflags | pntflags,
                        self.mask.Map[:],
                    )
                    if ring is None or ring.signal.size < 10:
                        continue
                    rings[iring][det] = ring
                    nbytes += ring.nbytes
                ngood += 1

            if ngood == 0:
                # None of the detectors have this ring.  Flag it as bad
                # so it is not included in calibration steps.
                self.local_bad_rings[iring] = True

        nbytes_all = self.comm.gather(nbytes)

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            nbytes_all = np.array(nbytes_all) / 2 ** 20
            nbytes_min = np.amin(nbytes_all)
            nbytes_max = np.amax(nbytes_all)
            nbytes_mean = np.mean(nbytes_all)
            nbytes_std = np.std(nbytes_all)
            print(
                "    Binned rings in {:.2f} s. Compressed size = "
                "{:.2f} +- {:.2f} MB (Min = {:.2f} MB, max = {:.2f} MB)"
                "".format(
                    stop - start, nbytes_mean, nbytes_std, nbytes_min, nbytes_max
                ),
                flush=True,
            )

        return rings

    @function_timer
    def _sample_map(
        self,
        mapsampler,
        pixels,
        iquweights,
        pntflags,
        detflags,
        ring,
        glob2loc,
        ring_theta,
        ring_phi,
        ipix,
        iweights,
    ):
        # Sample the full frequency map
        if mapsampler.nside > self.nside:
            ipix, iweights = None, None
            pix = pixels // (self.pix_nside // mapsampler.nside) ** 2
            gain_template = self.mapsampler_freq.Map[pix]
            gain_template = bin_ring_extra(
                pixels // self.ndegrade,
                gain_template.astype(np.float64),
                pntflags | detflags,
                ring.pixels,
                glob2loc,
            )
            """
            if not self.temperature_only:
                pol_template = (
                    iquweights[:, 1] * mapsampler.Map_Q[pix] +
                    iquweights[:, 2] * mapsampler.Map_U[pix])
                pol_template = bin_ring_extra(
                    pixels // self.ndegrade,
                    pol_template.astype(np.float64),
                    pntflags | detflags, ring.pixels, glob2loc)
                gain_template += pol_template
            """
        else:
            if ipix is None or iweights is None:
                ipix, iweights = hp.get_interp_weights(
                    self.bandpass_nside, ring_theta, ring_phi, nest=True
                )
            gain_template = self.mapsampler_freq.atpol(
                ring_theta,
                ring_phi,
                ring.weights,
                interp_pix=ipix,
                interp_weights=iweights,
                pol=False, # self.temperature_only_intermediate,  # Uses a smoothed polarization template
            ).astype(np.float64)
            """
            if not self.temperature_only:
                pol_template = self.mapsampler_freq.atpol(
                    ring_theta, ring_phi, ring.weights,
                    interp_pix=ipix, interp_weights=iweights,
                    pol=True, onlypol=True).astype(np.float64)
                gain_template += pol_template
            """
        return gain_template, ipix, iweights

    @function_timer
    def update_templates(self, rings, templates):
        """
        Update the gain template using the latest full frequency map and
        the latest bandpass corrections.
        """
        if not self.calibrate:
            return
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Updating templates", flush=True)

        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            ind = slice(istart, istop)
            pntflags = self.cache.reference(self.pntflags)[ind]
            for det in self.dets:
                if det not in rings[iring]:
                    if det in templates[iring]:
                        del templates[iring][det]
                    continue
                for name in "gain", "distortion":
                    if name not in self.template_offsets:
                        if name in templates[iring][det]:
                            del templates[iring][det][name]
                        continue
                if det[-1] in "bS" and self.symmetrize:
                    continue
                ring = rings[iring][det]
                glob2loc = get_glob2loc(ring.pixels)
                detflags = self.tod.local_flags(det)[ind]
                pixels = self.tod.local_pixels(det)[ind]
                iquweights = self.tod.local_weights(det)[ind]
                pairdet = self.get_pair(det)
                ring_theta, ring_phi = qa.to_position(ring.quat)
                ipix, iweights = None, None
                gain_template, ipix, iweights = self._sample_map(
                    self.mapsampler_freq,
                    pixels,
                    iquweights,
                    pntflags,
                    detflags,
                    ring,
                    glob2loc,
                    ring_theta,
                    ring_phi,
                    ipix,
                    iweights,
                )
                # Add all templates that we have fitted.  The closer we
                # make the gain template look like the gain fluctuations,
                # the better.
                # The orbital dipole is particularly important.
                for name, template in templates[iring][det].items():
                    if template.offset is None:
                        # The overall calibration is degenerate so we
                        # must not include components in the gain
                        # template that are not also independently
                        # fitted.
                        continue
                    if name not in self.best_fit_amplitudes[det]:
                        continue
                    amp = self.best_fit_amplitudes[det][name]
                    gain_template += amp * template.template
                # Replace the gain template
                offset = templates[iring][det]["gain"].offset
                templates[iring][det]["gain"] = RingTemplate(gain_template, offset)
                if self.fit_distortion:
                    offset = templates[iring][det]["distortion"].offset
                    dtemplate = gain_template.copy()
                    mid = np.median(dtemplate)
                    dtemplate[dtemplate > mid] = 0
                    templates[iring][det]["distortion"] = RingTemplate(
                        dtemplate, offset
                    )
                if pairdet is not None:
                    iquweights = self.tod.local_weights(pairdet)[ind]
                    gain_template, ipix, iweights = self._sample_map(
                        self.mapsampler_freq,
                        pixels,
                        iquweights,
                        pntflags,
                        detflags,
                        ring,
                        glob2loc,
                        ring_theta,
                        ring_phi,
                        ipix,
                        iweights,
                    )
                    for name, template in templates[iring][pairdet].items():
                        if template.offset is None:
                            continue
                        if name not in self.best_fit_amplitudes[pairdet]:
                            continue
                        if name in self.horn_symmetric:
                            # We *have* to use the same amplitude for
                            # both detectors in a horn.  Otherwise
                            # subtracting the gain mean messes up the
                            # amplitudes.
                            amp = self.best_fit_amplitudes[det][name]
                        else:
                            amp = self.best_fit_amplitudes[pairdet][name]
                        gain_template += amp * template.template
                    offset = templates[iring][pairdet]["gain"].offset
                    templates[iring][pairdet]["gain"] = RingTemplate(
                        gain_template, offset
                    )
                    if self.fit_distortion:
                        offset = templates[iring][pairdet]["distortion"].offset
                        dtemplate = gain_template.copy()
                        mid = np.median(dtemplate)
                        dtemplate[dtemplate > mid] = 0
                        templates[iring][pairdet]["distortion"] = RingTemplate(
                            dtemplate, offset
                        )

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "    Updated gain template in {:.2f} s".format(stop - start), flush=True
            )

        return

    @function_timer
    def _set_up_zodi(self):
        if self.do_zodi:
            emissivities = np.ones(6, dtype=np.float64)
            self.zodinames = [
                "zodi cloud",
                "zodi band1",
                "zodi band2",
                "zodi band3",
                "zodi ring",
                "zodi blob",
            ]
            self.zodier = Zodier(int(self.freq), emissivities=emissivities)
        else:
            self.zodinames = None
            self.zodier = None
        return

    @function_timer
    def _save_fgmap(self, mapsampler_fg):
        if self.iiter == 0 and not self.mcmode and self.rank == 0:
            fname = os.path.join(self.out, "gain_template_iter00.fits")
            map_out = [
                mapsampler_fg.Map[:],
                mapsampler_fg.Map_Q[:],
                mapsampler_fg.Map_U[:],
            ]
            try:
                hp.write_map(fname, map_out, dtype=np.float32, nest=True)
            except Exception:
                hp.write_map(
                    fname, map_out, dtype=np.float32, overwrite=True, nest=True
                )
            print("        Gain target saved in {}".format(fname), flush=True)
        return

    @function_timer
    def _set_up_dipoler(self):
        if self.mcmode:
            full4pi = False
        else:
            full4pi = "npipe"
        self.dipoler = Dipoler(
            full4pi=full4pi, comm=self.comm, RIMO=self.rimo, symmetrize_4pi=True
        )
        return

    @function_timer
    def _get_pair_templates(self, templates, iring, det, pairdet):
        # Shortcut for detector pairs
        for name, template in templates[iring][pairdet].items():
            if name == "offset":
                templates[iring][det][name] = RingTemplate(
                    template.template, self.ring_offset + iring
                )
            if name in self.mapsamplers.keys() or name in ["pol0", "pol1", "pol2"]:
                # These templates are sampled separately to allow for
                # polarization
                continue
            else:
                # gain offsets are set later and the rest of
                # the templates are full mission
                templates[iring][det][name] = RingTemplate(
                    template.template, template.offset
                )
        return

    @function_timer
    def _set_up_differentiator(self):
        if self.nharm > 1:
            self.differentiator = Differentiator(nharm=self.nharm, fsample=self.fsample)
        else:
            self.differentiator = None
        return

    @function_timer
    def _set_up_foreground(self):
        start = MPI.Wtime()
        mapsampler_fg = self.fg
        if self.bpcorrect:
            self.mapsamplers["fg_deriv"] = self.fg_deriv
            if self.bpcorrect2:
                self.mapsamplers["fg_deriv2"] = self.fg_deriv2
            for name, path in [("CO", self.co1), ("CO2", self.co2), ("CO3", self.co3)]:
                if name in self.mapsamplers:
                    # We already loaded this map in a previous iteration
                    continue
                if path is not None:
                    if path in cached_mapsamplers:
                        self.mapsamplers[name] = cached_mapsamplers[path]
                    else:
                        self.mapsamplers[name] = MapSampler(
                            path,
                            pol=False,
                            nside=self.bandpass_nside,
                            comm=self.comm,
                            cache=self.cache,
                            nest=True,
                            pscorrect=self.pscorrect,
                            psradius=self.psradius,
                        )
                        if self.mcmode:
                            cached_mapsamplers[path] = self.mapsamplers[name]
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "       Sky model initialized in {:.2f} s" "".format(stop - start),
                flush=True,
            )
        return mapsampler_fg

    @function_timer
    def _set_up_polarization(self, mapsampler_fg):
        start = MPI.Wtime()
        for ipol, polmap in enumerate([self.polmap, self.polmap2, self.polmap3]):
            if polmap:
                key = "pol{}".format(ipol)
                if key not in self.mapsamplers:
                    if self.rank == 0:
                        print(
                            "        Loading polarization template from {}"
                            ", fwhm = {}, lmax = {}, nside = {}".format(
                                polmap, self.pol_fwhm, self.pol_lmax, self.pol_nside
                            ),
                            flush=True,
                        )
                    polsampler = MapSampler(
                        polmap,
                        pol=True,
                        nside=self.pol_nside,
                        comm=self.comm,
                        cache=self.cache,
                        nest=True,
                    )
                    if self.pol_fwhm:
                        polsampler.smooth(fwhm=self.pol_fwhm, lmax=self.pol_lmax)
                    self.mapsamplers[key] = polsampler
        """
        else:
            # Use a polarization target from the skymodel
            if self.rank == 0:
                print('        Using polarization template from the skymodel'
                      ''.format(self.polmap), flush=True)
            self.mapsamplers['pol'] = mapsampler_fg
        """
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "        Polarization map(s) initialized in {:.2f} s"
                "".format(stop - start),
                flush=True,
            )
        return

    @function_timer
    def _set_up_cmb(self, mapsampler_fg):
        if self.cmb is None:
            return
        start = MPI.Wtime()
        # Add the CMB map to the foreground map for a complete
        # calibration target
        if self.cmb_mc is None:
            mapsampler_cmb = MapSampler(
                self.cmb,
                pol=True,
                nside=self.bandpass_nside,
                comm=self.comm,
                cache=self.cache,
                nest=True,
            )
        else:
            mapsampler_cmb = self.cmb_mc
        mapsampler_fg += mapsampler_cmb
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "        CMB initialized in {:.2f} s" "".format(stop - start),
                flush=True,
            )
        del mapsampler_cmb
        return

    @function_timer
    def _add_offset_template(self, iring, nbin, templates, det, namplitude):
        offset = self.ring_offset + iring
        template = np.ones(nbin)
        templates[iring][det]["offset"] = RingTemplate(template, offset)
        if "offset" not in namplitude:
            namplitude["offset"] = self.nring_tot
        return

    def _add_dipole_template(
        self,
        ring,
        iring,
        templates,
        namplitude,
        velocity,
        nbin,
        det,
        phase,
        pixels,
        pntflags,
        detflags,
        glob2loc,
    ):
        # Create the dipole templates by
        #  1) sampling at the pixel centroids
        #  2) interpolating the sampled values to full TOD
        #  3) phase-binning the interpolated signal into a ring template
        # This approach is substantially cheaper than evaluating the directly
        # to every detector time stamp and then phase-binning it.
        if self.fgdipole:
            ring_dipo_total, ring_dipo_fg = self.dipoler.dipole(
                ring.quat, velocity=np.tile(velocity, [nbin, 1]), det=det,
                fg_deriv=self.fg_deriv, obs_freq=self.freq,
            )
        else:
            ring_dipo_total = self.dipoler.dipole(
                ring.quat, velocity=np.tile(velocity, [nbin, 1]), det=det
            )
        dipo_total = self._interpolate_ring_template(ring, ring_dipo_total, phase)
        ring_dipo_total = bin_ring_extra(
            pixels // self.ndegrade,
            dipo_total.astype(np.float64),
            pntflags | detflags,
            ring.pixels,
            glob2loc,
        )
        if self.fgdipole:
            dipo_fg = self._interpolate_ring_template(ring, ring_dipo_fg, phase)
            ring_dipo_fg = bin_ring_extra(
                pixels // self.ndegrade,
                dipo_fg.astype(np.float64),
                pntflags | detflags,
                ring.pixels,
                glob2loc,
            )
        else:
            ring_dipo_fg = None
        if self.do_dipo:
            ring_dipo_solsys = self.dipoler.dipole(ring.quat, det=det)
            dipo_solsys = self._interpolate_ring_template(ring, ring_dipo_solsys, phase)
            ring_dipo_solsys = bin_ring_extra(
                pixels // self.ndegrade,
                dipo_solsys.astype(np.float64),
                pntflags | detflags,
                ring.pixels,
                glob2loc,
            )
            ring_dipo_orbital = ring_dipo_total - ring_dipo_solsys
        else:
            ring_dipo_orbital = None
        dipo_amp = ring_dipo_total.ptp()
        if self.local_dipo_amp[iring] == 0 or (self.local_dipo_amp[iring] < dipo_amp):
            self.local_dipo_amp[iring] = dipo_amp
        # Add the total dipole as a template for interpolation.
        # We will not fit for it, as is indicated by the None offset.
        templates[iring][det]["dipo_total"] = RingTemplate(ring_dipo_total, None)
        if self.do_dipo:
            # We will fit for the orbital dipole
            templates[iring][det]["dipo_orbital"] = RingTemplate(ring_dipo_orbital, 0)
            if "dipo_orbital" not in namplitude:
                namplitude["dipo_orbital"] = 1
            if self.fgdipole:
                templates[iring][det]["dipo_foreground"] = RingTemplate(ring_dipo_fg, 0)
                if "dipo_foreground" not in namplitude:
                    namplitude["dipo_foreground"] = 1
        return ring_dipo_total, ring_dipo_orbital, ring_dipo_fg

    @function_timer
    def _add_fsl_template(
        self,
        det,
        pixels,
        ind,
        pntflags,
        detflags,
        templates,
        ring,
        iring,
        namplitude,
        glob2loc,
    ):
        if self.do_fsl:
            # FSL read from disk
            fsl = self.tod.local_fsl(det)
            ring_fsl = bin_ring_extra(
                pixels // self.ndegrade,
                fsl[ind],
                pntflags | detflags,
                ring.pixels,
                glob2loc,
            )
            if self.forcefsl:
                offset = None
            else:
                offset = 0
            templates[iring][det]["fsl"] = RingTemplate(ring_fsl, offset)
            if not self.forcefsl and "fsl" not in namplitude:
                namplitude["fsl"] = 1
            del fsl, ring_fsl
        if self.fslnames is not None:
            # FSL(s) convolved on the fly
            for fslname in self.fslnames:
                cachename = f"{fslname}_{det}"
                fsl = self.tod.cache.reference(cachename)
                # DEBUG begin
                if np.any(np.isnan(fsl)):
                    import pdb
                    pdb.set_trace()
                # DEBUG end
                ring_fsl = bin_ring_extra(
                    pixels // self.ndegrade,
                    fsl[ind],
                    pntflags | detflags,
                    ring.pixels,
                    glob2loc,
                )
                # DEBUG begin
                if np.any(np.isnan(ring_fsl)):
                    import pdb
                    pdb.set_trace()
                # DEBUG end
                templates[iring][det][fslname] = RingTemplate(ring_fsl, 0)
                if fslname not in namplitude:
                    namplitude[fslname] = 1
                del fsl, ring_fsl
        return

    @function_timer
    def _add_bandpass_templates(
        self,
        ring,
        iring,
        ring_interp_pix,
        ring_interp_weights,
        templates,
        ring_theta,
        ring_phi,
        det,
        namplitude,
        pixels,
        pntflags,
        detflags,
        glob2loc,
        iquweights,
    ):
        for name, mapsampler in self.mapsamplers.items():
            if mapsampler.nside == self.bandpass_nside:
                ipix = ring_interp_pix
                iweights = ring_interp_weights
            else:
                ipix = None
                iweights = None
            if name in ["pol0", "pol1", "pol2"]:
                continue
            if mapsampler.nside > self.nside:
                # We are downgrading so it is enough to sample the pixel
                # values without interpolation
                pix = pixels // (self.pix_nside // mapsampler.nside) ** 2
                if not mapsampler.pol or (
                    self.temperature_only or self.temperature_only_destripe
                ):
                    fg_toi = mapsampler.Map[pix]
                else:
                    fg_toi = (
                        iquweights[:, 0] * mapsampler.Map[pix]
                        + iquweights[:, 1] * mapsampler.Map_Q[pix]
                        + iquweights[:, 2] * mapsampler.Map_U[pix]
                    )
                fg_toi = bin_ring_extra(
                    pixels // self.ndegrade,
                    fg_toi.astype(np.float64),
                    pntflags | detflags,
                    ring.pixels,
                    glob2loc,
                )
            else:
                # FIXME: just like above, polarization in this branch should be
                # conditional:
                #    pol = not (self.temperature_only or self.temperature_only_destripe)
                fg_toi = mapsampler.atpol(
                    ring_theta,
                    ring_phi,
                    ring.weights,
                    interp_pix=ipix,
                    interp_weights=iweights,
                    pol=True,
                    onlypol=False,
                ).astype(np.float64)
            if fg_toi is None:
                continue
            templates[iring][det][name] = RingTemplate(fg_toi, 0)
            if name not in namplitude:
                namplitude[name] = 1
            del fg_toi
        return

    @function_timer
    def _add_polarization_templates(
        self,
        ring,
        iring,
        ring_interp_pix,
        ring_interp_weights,
        templates,
        ring_theta,
        ring_phi,
        det,
        namplitude,
        pixels,
        pntflags,
        detflags,
        glob2loc,
        iquweights,
    ):
        for name in ["pol0", "pol1", "pol2"]:
            if name not in self.mapsamplers:
                continue
            mapsampler = self.mapsamplers[name]
            if mapsampler.nside == self.bandpass_nside:
                ipix = ring_interp_pix
                iweights = ring_interp_weights
            else:
                ipix = None
                iweights = None
            if mapsampler.nside > self.nside:
                # We are downgrading so it is enough to sample the pixel
                # values without interpolation
                pix = pixels // (self.pix_nside // mapsampler.nside) ** 2
                fg_toi = (
                    iquweights[:, 1] * mapsampler.Map_Q[pix]
                    + iquweights[:, 2] * mapsampler.Map_U[pix]
                )
                fg_toi = bin_ring_extra(
                    pixels // self.ndegrade,
                    fg_toi.astype(np.float64),
                    pntflags | detflags,
                    ring.pixels,
                    glob2loc,
                )
            else:
                fg_toi = mapsampler.atpol(
                    ring_theta,
                    ring_phi,
                    ring.weights,
                    interp_pix=ipix,
                    interp_weights=iweights,
                    pol=True,
                    onlypol=True,
                ).astype(np.float64)
            templates[iring][det][name] = RingTemplate(fg_toi, 0)
            if name not in namplitude:
                namplitude[name] = 1
            del fg_toi
        return

    @function_timer
    def _add_zodi_templates(
        self, ring, iring, position, nbin, det, namplitude, templates, iquweights
    ):
        if self.do_zodi:
            zodidir = "{}/{}/{:04}".format(self.zodi_cache, det, self.nside)
            if self.rank == 0 and not os.path.isdir(zodidir):
                os.makedirs(zodidir)
            ring_number = iring + self.ring_offset
            zodifile = "zodi_{}_{:04}_{:05}.pck".format(det, self.nside, ring_number)
            zodifile = os.path.join(zodidir, zodifile)
            ring_zodis = None
            if os.path.isfile(zodifile):
                ring_zodis = np.fromfile(zodifile, dtype=np.float32).reshape([6, -1])
                if len(ring_zodis[0]) != len(ring.quat):
                    zodidir_old = zodidir
                    zodidir = "{}_NEW/{}/{:04}".format(self.zodi_cache, det, self.nside)
                    print(
                        "WARNING: Cached zodi in {} is incompatible. "
                        "Will compute a new one and cache under {}.".format(
                            zodidir_old, zodidir
                        ),
                        flush=True,
                    )
                    ring_zodis = None
            if ring_zodis is None:
                ring_zodis = self.zodier.zodi(ring.quat, np.tile(position, [nbin, 1]))
                if self.differential_zodi:
                    # Then measure the zodiacal emission in the same
                    # direction 6 months later and remove the common mode.
                    # We only want to remove the seasonally varying part of
                    # zodi emission.
                    ring_zodis2 = self.zodier.zodi(ring.quat, np.tile(-position, [nbin, 1]))
                    ring_zodis -= 0.5 * (ring_zodis + ring_zodis2)
                    del ring_zodis2
                if not os.path.isdir(zodidir):
                    try:
                        os.makedirs(zodidir)
                    except Exception:
                        pass
                ring_zodis = ring_zodis.astype(np.float32)
                ring_zodis.tofile(zodifile)
            for zodiname, ring_zodi in zip(self.zodinames, ring_zodis):
                # FIXME : this is where we would add polarized Zodi templates
                templates[iring][det][zodiname] = RingTemplate(ring_zodi, 0)
                if zodiname not in namplitude:
                    namplitude[zodiname] = 1
        return

    @function_timer
    def _add_calibration_template(
        self,
        ring_dipo_orbital,
        ring_dipo_total,
        ring_interp_pix,
        ring_interp_weights,
        ring_theta,
        ring_phi,
        ring,
        iring,
        templates,
        det,
        pixels,
        pntflags,
        detflags,
        glob2loc,
    ):
        if not self.calibrate:
            return
        sampler = self.mapsampler_freq
        if self.mapsampler_freq_has_dipole:
            dipo = ring_dipo_orbital
        else:
            dipo = ring_dipo_total
        if sampler.nside == self.bandpass_nside:
            ipix = ring_interp_pix
            iweights = ring_interp_weights
        else:
            ipix = None
            iweights = None
        if sampler.nside > self.nside:
            # We are downgrading so it is enough to sample the pixel
            # values without interpolation
            pix = pixels // (self.pix_nside // sampler.nside) ** 2
            gain_template = sampler.Map[pix]
            gain_template = bin_ring_extra(
                pixels // self.ndegrade,
                gain_template.astype(np.float64),
                pntflags | detflags,
                ring.pixels,
                glob2loc,
            )
        else:
            gain_template = sampler.atpol(
                ring_theta,
                ring_phi,
                ring.weights,
                interp_pix=ipix,
                interp_weights=iweights,
                pol=False,
            ).astype(np.float64)
        gain_template += dipo
        calib_rms = np.std(gain_template * ring.mask)
        if self.local_calib_rms[iring] < calib_rms:
            self.local_calib_rms[iring] = calib_rms
        templates[iring][det]["gain"] = RingTemplate(gain_template, 0)
        if self.fit_distortion:
            # First order signal distortion is a different gain applied
            # to opposite extremes of the signal.
            dtemplate = gain_template.copy()
            mid = np.median(dtemplate)
            dtemplate[dtemplate > mid] = 0
            templates[iring][det]["distortion"] = RingTemplate(dtemplate, 0)
        del gain_template
        return

    @function_timer
    def _get_signal_estimate(
        self, theta, phi, iquweights, det, templates, ring, iring, phase
    ):
        """ Derive a signal-only estimate for every sample

        """
        # Use the most recent iteration
        signal_estimate = self.mapsampler_freq.atpol(
            theta, phi, iquweights, pol=False
        ).astype(np.float64)
        if self.mapsampler_freq_has_dipole:
            # Add the orbital dipole
            dipole = self._interpolate_ring_template(
                ring, templates[iring][det]["dipo_orbital"].template, phase
            )
        else:
            # Add the total dipole
            dipole = self._interpolate_ring_template(
                ring, templates[iring][det]["dipo_total"].template, phase
            )
        signal_estimate += dipole
        return signal_estimate

    @function_timer
    def _add_nl_template(
        self,
        signal_estimate,
        pixels,
        pntflags,
        detflags,
        ring,
        iring,
        glob2loc,
        templates,
        namplitude,
        det,
    ):
        """ Derive a nonlinear gain template
        """
        if not self.nlcalibrate:
            return
        # Save the dipole-free mapsampler for applying the correction
        self.mapsampler_fg = self.mapsampler_freq
        lowpassed_signal = flagged_running_average(
            signal_estimate,
            np.isnan(signal_estimate),
            int(self.nllowpass * self.fsample),
        )
        lowpassed_signal -= np.median(lowpassed_signal)
        nltemplate = signal_estimate * lowpassed_signal
        ring_nltemplate = bin_ring_extra(
            pixels // self.ndegrade,
            nltemplate.astype(np.float64),
            pntflags | detflags,
            ring.pixels,
            glob2loc,
        )
        templates[iring][det]["nlgain"] = RingTemplate(ring_nltemplate, 0)
        if "nlgain" not in namplitude:
            namplitude["nlgain"] = 1
        return

    @function_timer
    def _add_band_templates(
        self,
        signal_estimate,
        pixels,
        pntflags,
        detflags,
        ring,
        iring,
        glob2loc,
        templates,
        det,
        namplitude,
    ):
        """ Add transfer function residual templates
        """
        if self.differentiator is None:
            return
        bands, derivs = self.differentiator.differentiate(
            signal_estimate, do_bands=self.do_bands, do_derivs=self.do_derivs
        )
        for i, band in enumerate(bands):
            # Omit the calibration for the first harmonic
            # to avoid degeneracy
            if i == 0:
                continue
            name = "spin_harmonic_{:02}_gain".format(i)
            ring_band = bin_ring_extra(
                pixels // self.ndegrade,
                band.astype(np.float64),
                pntflags | detflags,
                ring.pixels,
                glob2loc,
            )
            templates[iring][det][name] = RingTemplate(ring_band, 0)
            if name not in namplitude:
                namplitude[name] = 1
        for i, deriv in enumerate(derivs):
            name = "spin_harmonic_{:02}_deriv".format(i)
            ring_deriv = bin_ring_extra(
                pixels // self.ndegrade,
                deriv.astype(np.float64),
                pntflags | detflags,
                ring.pixels,
                glob2loc,
            )
            templates[iring][det][name] = RingTemplate(ring_deriv, 0)
            if name not in namplitude:
                namplitude[name] = 1
        return

    @function_timer
    def build_templates(self, rings):
        """
        Build ring templates for
          - gain
          - bandpass mismatch (fg_deriv, fg_deriv2 and CO)
          - FSL (far side lobes)
          - zodiacal light
          - transfer function residuals
          - orbital dipole

        If we are iterating, only update the gain template using the
        most recent full frequency map and bandpass corrections.
        """
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Building templates", flush=True)

        self._set_up_zodi()
        mapsampler_fg = self._set_up_foreground()
        self._set_up_polarization(mapsampler_fg)
        self._set_up_cmb(mapsampler_fg)
        self._save_fgmap(mapsampler_fg)
        self._set_up_dipoler()
        self._set_up_differentiator()

        memreport("after template set-up", self.comm)

        if self.mapsampler_freq is None:
            self.mapsampler_freq = mapsampler_fg
            self.mapsampler_freq_has_dipole = False

        self.local_dipo_amp = np.zeros(self.nring)
        self.local_calib_rms = np.zeros(self.nring)

        namplitude = OrderedDict()
        templates = OrderedDict()
        ngood_ring = 0

        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            templates[iring] = OrderedDict()
            ind = slice(istart, istop)
            velocity = rings[iring]["velocity"]
            if self.do_zodi:
                position = rings[iring]["position"]
            else:
                position = None
            pntflags = self.cache.reference(self.pntflags)[ind]
            phase = self.tod.local_phase()[ind]
            for det in self.dets:
                templates[iring][det] = OrderedDict()
                if det not in rings[iring]:
                    # This detector and ring were completely flagged
                    continue

                detflags = self.tod.local_flags(det)[ind]
                pixels = self.tod.local_pixels(det)[ind]
                iquweights = self.tod.local_weights(det)[ind]

                ngood_ring += 1

                ring = rings[iring][det]
                nbin = ring.pixels.size

                ring_theta, ring_phi = qa.to_position(ring.quat)

                ring_interp_pix, ring_interp_weights = hp.get_interp_weights(
                    self.bandpass_nside, ring_theta, ring_phi, nest=True
                )

                glob2loc = get_glob2loc(ring.pixels)

                # bandpass templates are sampled separately, even for
                # detectors in the same horn (to allow for polarized
                # bandpass mismatch).
                self._add_bandpass_templates(
                    ring,
                    iring,
                    ring_interp_pix,
                    ring_interp_weights,
                    templates,
                    ring_theta,
                    ring_phi,
                    det,
                    namplitude,
                    pixels,
                    pntflags,
                    detflags,
                    glob2loc,
                    iquweights,
                )

                if not self.skip_polmaps:
                    self._add_polarization_templates(
                        ring,
                        iring,
                        ring_interp_pix,
                        ring_interp_weights,
                        templates,
                        ring_theta,
                        ring_phi,
                        det,
                        namplitude,
                        pixels,
                        pntflags,
                        detflags,
                        glob2loc,
                        iquweights,
                    )

                pairdet = self.get_pair(det)
                if pairdet is not None and pairdet in templates[iring]:
                    self._get_pair_templates(templates, iring, det, pairdet)
                    continue

                self._add_offset_template(iring, nbin, templates, det, namplitude)

                (
                    ring_dipo_total, ring_dipo_orbital, ring_dipo_fg
                ) = self._add_dipole_template(
                    ring,
                    iring,
                    templates,
                    namplitude,
                    velocity,
                    nbin,
                    det,
                    phase,
                    pixels,
                    pntflags,
                    detflags,
                    glob2loc,
                )

                self._add_fsl_template(
                    det,
                    pixels,
                    ind,
                    pntflags,
                    detflags,
                    templates,
                    ring,
                    iring,
                    namplitude,
                    glob2loc,
                )

                self._add_zodi_templates(
                    ring, iring, position, nbin, det, namplitude, templates, iquweights
                )

                self._add_calibration_template(
                    ring_dipo_orbital,
                    ring_dipo_total,
                    ring_interp_pix,
                    ring_interp_weights,
                    ring_theta,
                    ring_phi,
                    ring,
                    iring,
                    templates,
                    det,
                    pixels,
                    pntflags,
                    detflags,
                    glob2loc,
                )

                del ring_dipo_total, ring_dipo_orbital, ring_dipo_fg

                if not self.nlcalibrate and self.differentiator is None:
                    continue

                iquweights = self.tod.local_weights(det)[ind]
                quat = self.tod.local_pointing(det)[ind]
                if self.mcmode:
                    theta = self.cache.reference("theta_{}".format(det))[ind]
                    phi = self.cache.reference("phi_{}".format(det))[ind]
                else:
                    theta, phi = qa.to_position(quat)

                signal_estimate = self._get_signal_estimate(
                    theta, phi, iquweights, det, templates, ring, iring, phase
                )

                self._add_nl_template(
                    signal_estimate,
                    pixels,
                    pntflags,
                    detflags,
                    ring,
                    iring,
                    glob2loc,
                    templates,
                    namplitude,
                    det,
                )

                self._add_band_templates(
                    signal_estimate,
                    pixels,
                    pntflags,
                    detflags,
                    ring,
                    iring,
                    glob2loc,
                    templates,
                    det,
                    namplitude,
                )

        if self.cmb_mc:
            # Remove CMB from the foreground mapsampler to prepare for
            # the next iteration of Monte Carlo
            mapsampler_fg -= self.cmb_mc
        del mapsampler_fg
        # We will build FSL and dipole templates by interpolating
        # the phase-binned templates.  Discard the full resolution FSL.
        self.cache.clear("fsl.*")

        memreport("after adding templates", self.comm)

        if self.calibrate:
            # Assign gains to intervals based on the calibrator strength
            self.comm.Barrier()
            self.get_gain_schedule(self.gain_step_mode)
            self.ngain = len(self.gain_ring)
            namplitude["gain"] = self.ngain
            if self.fit_distortion:
                namplitude["distortion"] = self.ngain

        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            for det in self.dets:
                if "gain" not in templates[iring][det]:
                    continue
                offset = self.gain_offset(iring)
                templates[iring][det]["gain"].offset = offset
                if self.fit_distortion:
                    templates[iring][det]["distortion"].offset = offset

        # Processes that have no valid rings have an incomplete namplitude
        # dictionary.  Loop over processes until we find one that has
        # valid data and broadcast the namplitude dict.

        for rank in range(self.ntask):
            ngood = self.comm.bcast(ngood_ring, root=rank)
            if ngood != 0:
                namplitude = self.comm.bcast(namplitude, root=rank)
                break

        # Translate the amplitude offsets to point to a single amplitude
        # vector

        self.template_names = sorted(namplitude.keys())

        offsets = OrderedDict()
        offset = 0
        namplitude_tot = 0
        for name in self.template_names:
            offsets[name] = [offset, namplitude[name]]
            offset += namplitude[name]
            namplitude_tot += namplitude[name]

        self.namplitude_tot = namplitude_tot
        self.template_offsets = offsets

        # First set the offsets for every detector ...

        for ring in sorted(rings.keys()):
            for det in self.dets:
                for name in templates[ring][det]:
                    if templates[ring][det][name].offset is None:
                        continue
                    offset = offsets[name]
                    templates[ring][det][name].offset += offset[0]

        # ... and then relative to the beginning of the 1D amplitude vector

        for ring in sorted(rings.keys()):
            for idet, det in enumerate(self.dets):
                detoffset = idet * self.namplitude_tot
                for name in templates[ring][det]:
                    if name in self.freq_symmetric:
                        # This template is shared across all detectors
                        continue
                    if templates[ring][det][name].offset is None:
                        continue
                    templates[ring][det][name].offset += detoffset

        # Finally apply special rules to some of the templates to bind
        # their amplitudes

        for ring in sorted(rings.keys()):
            for det in self.dets:
                if det not in templates[ring] or det[-1] not in "bS":
                    continue
                pairdet = get_pair(det)
                for name in templates[ring][det]:
                    if name in self.horn_symmetric and name in templates[ring][pairdet]:
                        # Use the same template amplitude for both
                        # detectors in a horn.
                        if templates[ring][det][name].offset is None:
                            continue
                        templates[ring][det][name].offset = templates[ring][pairdet][
                            name
                        ].offset

        # Run a quick check to make sure no two templates are bound
        # to a same amplitude

        all_good = True
        for iring in rings.keys():
            for idetname in self.dets:
                if idetname not in rings[iring]:
                    continue
                for iname in templates[iring][idetname]:
                    ioffset = templates[iring][idetname][iname].offset
                    if ioffset is None:
                        continue
                    for jring in rings.keys():
                        for jdetname in self.dets:
                            if jdetname not in rings[jring]:
                                continue
                            for jname in templates[jring][jdetname]:
                                joffset = templates[jring][jdetname][jname].offset
                                if joffset is None:
                                    continue
                                if iname != jname and ioffset == joffset:
                                    print(
                                        "ERROR: template offset conflict: "
                                        "{}:{}:{} = {} {}:{}:{} = {}"
                                        "".format(
                                            idetname,
                                            iring,
                                            iname,
                                            ioffset,
                                            jdetname,
                                            jring,
                                            jname,
                                            joffset,
                                        ),
                                        flush=True,
                                    )
                                    all_good = False
        if not all_good:
            raise Exception("Template offset conflict")

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print("    Built templates in {:.2f} s. ".format(stop - start), flush=True)

        return templates, namplitude_tot

    @function_timer
    def project_offsets(self, rings, templates, do_signal=False):
        """
        Project ring offsets out from the templates to make the
        template covariance matrix more diagonal
        """
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Projecting offsets from templates", flush=True)

        for iring, ring in rings.items():
            for det in self.dets:
                if det not in ring:
                    continue
                hit = ring[det].hits * ring[det].mask
                nhit = np.sum(hit)
                if nhit == 0:
                    continue
                if do_signal:
                    ring_offset = np.sum(ring[det].signal * hit) / nhit
                    ring[det].signal[:] -= ring_offset
                for name, template in templates[iring][det].items():
                    if name == "offset" or template.offset is None:
                        continue
                    ring_offset = np.sum(template.template * hit) / nhit
                    template.template[:] -= ring_offset

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print("    Offsets projected in {:.2f} s:".format(stop - start), flush=True)
        return

    @function_timer
    def rough_cal(self, rings, templates):
        """
        0th order calibration by doing a linear regression of the
        gain template on the signal.
        """
        if not self.recalibrate or self.iiter != 0:
            return

        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Measuring rough calibration", flush=True)

        # We'll measure the rough calibration only across the dipole
        # extrema.  Start with a normalized velocity vector.

        vel = self.dipoler.baryvel.copy()
        vel /= np.sqrt(np.dot(vel, vel))

        ntemplate = 2
        self.rough_gains = OrderedDict()

        for det in self.dets:
            invcov = np.zeros([ntemplate, ntemplate], dtype=np.float64)
            proj = np.zeros(ntemplate, dtype=np.float64)
            for iring, ring in rings.items():
                if det not in ring:
                    continue
                hits = ring[det].hits * ring[det].mask
                signal = ring[det].signal.copy()
                # Only use dipole extrema
                vec = hp.pix2vec(self.nside, ring[det].pixels)
                dipo = np.dot(vel, vec)
                bad = np.abs(dipo) < 0.8
                hits[bad] = 0
                nhit = np.sum(hits)
                if nhit == 0:
                    continue
                template1 = templates[iring][det]["gain"].template.copy()
                template2 = templates[iring][det]["dipo_orbital"].template.copy()
                # Project out the offsets considering this mask
                for x in [signal, template1, template2]:
                    offset = np.sum(x * hits) / nhit
                    x -= offset
                arr = np.vstack([template1, template2])
                invcov += np.dot(arr, (arr * hits).T)
                proj += np.dot(arr, signal * hits)
            self.comm.Allreduce(MPI.IN_PLACE, invcov)
            self.comm.Allreduce(MPI.IN_PLACE, proj)
            if np.all(proj) == 0:
                if self.rank == 0:
                    print(
                        "WARNING: no data to measure rough calibration for {}"
                        "".format(det)
                    )
                self.rough_gains[det] = 1
                continue
            cov = np.linalg.inv(invcov)
            coeff = np.dot(cov, proj)
            gain = 1 / coeff[0]
            if self.rank == 0:
                print(
                    "Rough absolute gain for {} is {}." "".format(det, 1 / coeff[0]),
                    flush=True,
                )
                print(
                    "Auxiliary orbital dipole amplitude for {} is {}"
                    "".format(det, coeff[1]),
                    flush=True,
                )
            self.rough_gains[det] = gain
            # Apply the rough gains to compressed TOD
            for iring, ring in rings.items():
                if det not in ring:
                    continue
                ring[det].signal[:] *= gain

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "    Rough calibration done in {:.2f} s:".format(stop - start),
                flush=True,
            )
        return

    @function_timer
    def detect_outliers(self, rings, templates, threshold=5):
        """
        Use linear regression at the ring level to find anomalous
        template amplitudes.
        """

        if self.nring_tot < 100 or self.temperature_only or self.mcmode:
            # Not enough rings to detect outliers
            return

        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Detecting outliers", flush=True)

        for det in self.dets:
            pairdet = self.get_pair(det)
            fits, covs, ncrop = OrderedDict(), [], 0
            flags = self.tod.local_flags(det)
            if pairdet is not None:
                pairflags = self.tod.local_flags(pairdet)
            else:
                pairflags = None
            for iring in rings:
                if det not in rings[iring]:
                    continue
                signal = rings[iring][det].signal
                hits = rings[iring][det].hits * rings[iring][det].mask
                arr, names = [], []
                for name in templates[iring][det]:
                    if "zodi" in name or "harmonic" in name or "offset" in name:
                        # Only use a subset of the available templates
                        continue
                    template = templates[iring][det][name]
                    if template.offset is None:
                        continue
                    names.append(name)
                    arr.append(template.template)
                arr = np.array(arr)
                invcov = np.dot(arr, (arr * hits).T)
                proj = np.dot(arr, signal * hits)
                try:
                    cov = np.linalg.inv(invcov)
                    coeffs = np.dot(cov, proj)
                except Exception:
                    print(
                        "WARNING: failed to invert template covariance "
                        "det = {}, ring = {}, invcov = {}, names = {}".format(
                            det, iring + self.ring_offset, invcov, names
                        )
                    )
                    continue
                for name, coeff in zip(names, coeffs):
                    if name not in fits:
                        fits[name] = []
                    fits[name].append((iring + self.ring_offset, coeff))
                covs.append((iring + self.ring_offset, [names, invcov, proj, cov]))
            all_fits = OrderedDict()
            all_covs = self.comm.allgather(covs)
            all_covs = [entry for cov in all_covs for entry in cov]
            for name in self.template_names:
                if name in fits:
                    vals = np.array(fits[name])
                else:
                    vals = np.zeros([0, 2])
                all_vals = np.vstack(self.comm.allgather(vals))
                if len(all_vals) == 0:
                    continue
                all_vals = all_vals.T
                nval = len(all_vals[1])
                good = np.isfinite(all_vals[1])
                wmean = min(100, nval // 10)
                for _ in range(10):
                    # Smooth
                    smooth = flagged_running_average(
                        all_vals[1], np.logical_not(good), wmean
                    )
                    sigma = np.sqrt(
                        flagged_running_average(
                            (all_vals[1] - smooth) ** 2, np.logical_not(good), wmean
                        )
                    )
                    # Detect outliers
                    ngood1 = np.sum(good)
                    good[np.abs(all_vals[1] - smooth) > threshold * sigma] = False
                    ngood2 = np.sum(good)
                    if ngood1 == ngood2:
                        break
                good[:wmean] = True
                good[-wmean:] = True
                all_fits[name] = [
                    (all_vals[0] + 1e-6).astype(int),
                    all_vals[1],
                    good,
                ]
                bad = np.logical_not(good)
                nbad = np.sum(bad)
                if self.rank == 0 and nbad != 0:
                    print(
                        "Cropped {} outlier fits for {} {}".format(nbad, det, name),
                        flush=True,
                    )
                for bad_ring in all_vals[0][bad]:
                    iring = int(bad_ring + 1e-6) - self.ring_offset
                    if iring < 0 or iring >= self.nring:
                        continue
                    istart = self.local_starts[iring]
                    istop = self.local_stops[iring]
                    ind = slice(istart, istop)
                    if det in rings[iring]:
                        ncrop += 1
                        flags[ind] |= 255
                        del rings[iring][det]
                    if det in templates[iring]:
                        del templates[iring][det]
                    if pairdet in rings[iring]:
                        pairflags[ind] |= 255
                        del rings[iring][pairdet]
                    if pairdet in templates[iring]:
                        del templates[iring][pairdet]
            ncrop = self.comm.allreduce(ncrop)
            flagsum = self.comm.reduce(np.sum(flags != 0))
            if pairdet is not None:
                pairflagsum = self.comm.reduce(np.sum(pairflags != 0))
            if self.rank == 0:
                print("Cropped {} outlier fits for {}".format(ncrop, det), flush=True)
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after outlier detection"
                    "".format(frac, det),
                    flush=True,
                )
                if pairdet is not None:
                    frac = pairflagsum * 100 / self.nsamp_tot
                    print(
                        "{:8.3f} % of {} TOD is flagged after pair outlier "
                        "detection".format(frac, pairdet),
                        flush=True,
                    )
                fname = os.path.join(
                    self.out, "ring_fits{}_{}.pck".format(self.siter, det)
                )
                with open(fname, "wb") as fout:
                    pickle.dump([all_fits, all_covs], fout, protocol=2)
                print("Ring fits saved in {}".format(fname), flush=True)

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print("    Outliers detected in {:.2f} s".format(stop - start), flush=True)

        return

    @function_timer
    def destripe(self, rings, templates, namplitude, iiter):
        """
        Destripe the rings with the ring templates.
        """
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Destriping/fitting templates", flush=True)

        if self.temperature_only or self.temperature_only_destripe:
            nnz = 1
        else:
            nnz = 3

        if self.force_polmaps:
            # Forcefully de-polarize the signal assuming that the
            # polarization templates already have the correct amplitudes
            if self.rank == 0:
                print("        Subtracting polarization prior", flush=True)
            for name in ["pol0", "pol1", "pol2"]:
                for iring in rings.keys():
                    for idet, det in enumerate(self.dets):
                        if det in templates[iring] and name in templates[iring][det]:
                            rings[iring][det].signal[:] -= \
                                templates[iring][det][name].template

        if not (self.temperature_only or self.temperature_only_destripe) \
           or self.force_polmaps:
            # Disable fitting the polarization templates
            for name in ["pol0", "pol1", "pol2"]:
                for iring in rings.keys():
                    for idet, det in enumerate(self.dets):
                        if det in templates[iring] and name in templates[iring][det]:
                            templates[iring][det][name].offset = None
                if name in self.template_offsets:
                    del self.template_offsets[name]

        # Stationary templates will have the average correction
        # across detectors removed due to a degeneracy.
        # orbital dipole, foreground dipole, zodi and the polarization template
        # are not stationary
        stationary = [
            "offset",
            "gain",
            "nlgain",
            "fg_deriv",
            "fg_deriv2",
            "CO",
            "CO2",
            "CO3",
            "distortion",
        ]
        """
        if self.do_bands:
            # The frequency band calibration seems to be degenerate.
            # Not fixing the frequency average to zero will suppress
            # or enhance small scales.
            for i in range(1, self.nharm):
                stationary.append('spin_harmonic_{:02}_gain'.format(i))
        """

        destriper = UltimateDestriper(
            self.npix,
            nnz,
            self.comm,
            threshold=self.destriper_pixlim,
            itermax=1000,
            cglimit=1e-12,
            ndegrade=1,
            dir_out=self.out,
            precond=True,
            stationary_templates=stationary,
        )

        memreport("after initializing destriper", self.comm)

        # Identify outlier rings by approximate noise RMS

        namplitude_tot = self.ndet * namplitude

        if self.iiter == 0 and self.nring_tot > 100 and not self.mcmode:
            threshold = self.outlier_threshold
            # if self.iiter > 0:
            #    # The RMS is considerably lower after the first iteration
            #    threshold *= 2
            destriper.flag_outliers(
                rings,
                self.dets,
                templates,
                namplitude,
                namplitude_tot,
                self.template_offsets,
                verbose=True,
                siter=self.siter,
                save_maps=True,
                threshold=threshold,
            )

            memreport("after flagging outliers", self.comm)

        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            ind = slice(istart, istop)
            for det in self.dets:
                if det not in rings[iring]:
                    # The ring was removed as an outlier.
                    # Update flags and templates.
                    if det in templates[iring]:
                        del templates[iring][det]
                    flags = self.tod.local_flags(det)
                    flags[ind] |= 255
                    pairdet = self.get_pair(det)
                    # For paired detectors, apply the treatment symmetrically
                    if pairdet is None:
                        continue
                    if pairdet in rings[iring]:
                        del rings[iring][pairdet]
                    if pairdet in templates[iring]:
                        del templates[iring][pairdet]
                    pairflags = self.tod.local_flags(pairdet)[ind]
                    pairflags[ind] |= 255

        for det in self.dets:
            flags = self.tod.local_flags(det)
            flagsum = self.comm.reduce(np.sum(flags != 0))
            if self.rank == 0:
                frac = flagsum * 100 / self.nsamp_tot
                print(
                    "{:8.3f} % of {} TOD is flagged after destriper outlier "
                    "detection".format(frac, det),
                    flush=True,
                )

        # Destripe the rest of the rings

        best_fits = copy.deepcopy(self.best_fit_amplitudes)
        if iiter == 0:
            # Orbital dipole templates are present in
            # the gain template at unit amplitude.
            for det in self.dets:
                best_fits[det]["dipo_orbital"] = 1

        destriped_toi, baselines = destriper.destripe(
            rings,
            self.dets,
            templates,
            namplitude,
            namplitude_tot,
            self.template_offsets,
            verbose=True,
            in_place=True,
            siter=self.siter,
            save_maps=(not self.mcmode),
            return_baselines=True,
            best_fits=best_fits,
        )
        del destriped_toi
        del destriper

        # DEBUG begin
        if self.save_destriper_data:
            fname = os.path.join(
                self.out,
                "destriper_data{}_{:05}.pck".format(self.siter, self.rank))
            with open(fname, "wb") as fout:
                pickle.dump([
                    rings, self.dets, templates, namplitude, namplitude_tot,
                    self.template_offsets, best_fits, self.nside
                ], fout, protocol=3)
            if self.rank == 0:
                print("destriper data saved in {}".format(fname), flush=True)
        # DEBUG end

        memreport("after destriping", self.comm)

        if self.rank == 0:
            for idet, det in enumerate(self.dets):
                print("    {:8}".format(det), end="")
                # Mission-long templates
                names = [
                    "dipo_orbital",
                    "dipo_foreground",
                    "fg_deriv",
                    "fg_deriv2",
                    "CO",
                    "CO2",
                    "CO3",
                    "fsl",
                    "pol0",
                    "pol1",
                    "pol2",
                    "nlgain",
                ]
                if self.fslnames is not None:
                    names += self.fslnames
                for name in names:
                    if name not in self.template_offsets:
                        continue
                    offset = self.template_offsets[name][0]
                    if name not in self.freq_symmetric:
                        offset += idet * self.namplitude_tot
                        if name in self.horn_symmetric and det[-1] in "bS":
                            offset -= self.namplitude_tot
                    amp = baselines[offset]
                    print(" {} = {:14.10f}".format(name, amp), end="")
                # Segmented templates
                for name in ["gain", "offset", "distortion"]:
                    if name not in self.template_offsets:
                        continue
                    offset, namp = self.template_offsets[name]
                    offset += idet * self.namplitude_tot
                    if name in self.horn_symmetric and det[-1] in "bS":
                        offset -= self.namplitude_tot
                    amp = baselines[offset : offset + namp]
                    print(
                        " {} = {:14.10f} +- {:14.10f}".format(
                            name, np.mean(amp), np.std(amp)
                        ),
                        end="",
                    )
                print("", flush=True)

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print("    Destriping done in {:.2f} s".format(stop - start), flush=True)

        return baselines

    @function_timer
    def get_gain_schedule(self, mode=None, mult=1):
        """
        Determine how many gain amplitudes to solve and where to apply
        them.
        """
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("        Building gain schedule", flush=True)

        # List of known gain discontinuities. Must be ordered
        gain_breaks = [
            1637476793.3234963,  # Irregular gains on OD 191..193
            1643185991.833288,  # RF on OD 257 (more accurate)
            # 1660333260.7358425, # SCS switchover covered by year breaks
            1674605048.2714763,  # Jump in LFI gains
            # 1690572682.7412982, # Jump in LFI gains, OD 806
            # 1690642692.9991145, # Jump in LFI gains, OD 806
            # 1700121545.4119861, # Jump in LFI gains, OD 916
            1702824487.2,  # Spin-up campaign OD 943
            # DPU reboot following a major solar flare OD 985
            1706068061.8980036,
            # 1707161626.7680182, # Jump in LFI gains, OD 998
            1716342737.1179709,  # Jump in LFI gains
            # 1735995002.8799901, # Jump in LFI gains, OD 1331
            1754012979.5404563,  # LFI Stand-by mode OD 1540
            # 1758412202.0367115, # Jump in LFI gains, OD 1591
        ]
        years = [
            1660332428.40196,  # OD  456, year 2, survey 3
            1690652161.40216,  # OD  807, year 3, survey 5
            1722706147.90196,  # OD 1178, year 4, survey 7
            1754234276.65201,  # OD 1543, year 5, survey 9
        ]
        surveys = [
            1644353466.40193,  # OD  270, survey 2
            1675889738.40202,  # OD  636, survey 4
            1706619075.65199,  # OD  992, survey 6
            1738355319.65196,  # OD 1359, survey 8
        ]

        if mode is None:
            breaks = sorted(gain_breaks + years + surveys)
            amp_limit = self.effective_amp_limit * mult
        elif mode == "mission":
            breaks = []
            amp_limit = 1e10
        elif mode == "years":
            breaks = list(years)
            amp_limit = 1e10
        elif mode == "surveys":
            breaks = sorted(years + surveys)
            amp_limit = 1e10
        else:
            raise Exception("Unknown gain schedule: {}".format(mode))
        breaks.append(1e30)  # for convenience

        times = np.hstack(self.comm.allgather(self.local_start_times))
        bad_rings = np.hstack(self.comm.allgather(self.local_bad_rings))
        dipo_amp = np.hstack(self.comm.allgather(self.local_dipo_amp))
        calib_rms = np.hstack(self.comm.allgather(self.local_calib_rms))

        if self.rank == 0 and not self.mcmode:
            fname = os.path.join(self.out, "calibrator_strength.pck")
            with open(fname, "wb") as fout:
                pickle.dump([times, bad_rings, dipo_amp, calib_rms], fout, protocol=2)
            print("Calibrator strength saved in {}".format(fname), flush=True)

        ibreak = 0
        istart = 0
        gain_starts = []
        do_not_merge = False
        while istart < self.nring_tot:
            istop = istart + 1
            # effective_amp = dipo_amp[istart]
            effective_amp = calib_rms[istart]
            while times[istart] >= breaks[ibreak]:
                ibreak += 1
            # Extend the gain step until
            #    1) step length is in specified range AND
            #    2) step S/N is above the desired threshold
            #  OR
            #    3) we encounter a hard break
            while (
                istop < self.nring_tot
                and (
                    (istop - istart < self.min_step_length)
                    or (effective_amp / np.sqrt(istop - istart) <= amp_limit)
                )
                and times[istop] < breaks[ibreak]
                and istop - istart < self.max_step_length
            ):
                if not bad_rings[istop]:
                    # effective_amp += dipo_amp[istop]
                    effective_amp += calib_rms[istop]
                istop += 1
            effective_amp /= np.sqrt(istop - istart)
            if effective_amp <= amp_limit and not do_not_merge:
                # This interval is truncated by break.
                # Append it to the previous one, except if there is no
                # previous one
                if not gain_starts:
                    gain_starts.append(0)
            else:
                gain_starts.append(istart)
            if effective_amp <= amp_limit:
                # This step was truncated because of a break
                # or maximum step length.
                do_not_merge = True
            else:
                do_not_merge = False
            istart = istop

        gain_start_times = times[gain_starts]

        # Total number of gain steps

        if self.rank == 0:
            print(
                "        There are {} gain steps. amp limit = {}".format(
                    len(gain_starts), amp_limit
                ),
                flush=True,
            )

        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "        Built gain schedule {:.2f} s".format(stop - start), flush=True
            )

        self.gain_ring = np.array(gain_starts)
        self.gain_time = np.array(gain_start_times)

        return

    @function_timer
    def gain_offset(self, ring, ring_offset=None):
        """
        Translate a local (zero-based) ring number into global gain step.
        The global step lengths are stored in self.gain_ring
        """
        # IF gain_ring = [0, 3] THEN
        # ring numbers  0 1 2 3 4 5
        #        yield  0 0 0 1 1 1

        if ring_offset is None:
            offset = (
                np.searchsorted(self.gain_ring, ring + self.ring_offset, side="right")
                - 1
            )
        else:
            offset = (
                np.searchsorted(self.gain_ring, ring + ring_offset, side="right") - 1
            )

        return offset

    @function_timer
    def get_amp(self, det, offset, baselines, name):
        """Return the template amplitude.

        """
        return baselines[offset]

    @function_timer
    def get_gains(self, baselines, dipo_subtracted=False):
        """
        Translate gain template amplitudes into gains.  Possibly smooth
        the resulting gain solution.
        """

        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Expanding gains", flush=True)

        # Full frequency orbital dipole gain correction

        orbital_gains = []
        if self.recalibrate:
            for idet, det in enumerate(self.dets):
                if "dipo_orbital" in self.template_offsets:
                    offset = self.template_offsets["dipo_orbital"][0]
                    offset += idet * self.namplitude_tot
                    if "dipo_orbital" in self.horn_symmetric and det[-1] in "bS":
                        offset -= self.namplitude_tot
                    amp = self.get_amp(det, offset, baselines, "dipo_orbital")
                    if dipo_subtracted:
                        orbital_gains.append(amp + 1)
                    else:
                        orbital_gains.append(amp)
                else:
                    orbital_gains.append(1.0)
        if orbital_gains:
            orbital_gain = 1 / np.mean(orbital_gains)
            if self.rank == 0:
                print(
                    "    Will adjust overall gain by {} to match orbital "
                    "dipole.  All (inverse) orbital gains: {}".format(
                        orbital_gain, orbital_gains
                    ),
                    flush=True,
                )
        else:
            orbital_gain = 1

        # Individual detector relative, time-dependent gain corrections

        if "gain" not in self.template_offsets:
            gains = np.ones([self.ndet, self.nring])
            distortions = None
        else:
            gains = []
            if self.fit_distortion:
                distortions = []
            else:
                distortions = None
            if self.calibrate:
                my_ring = np.arange(self.nring) + self.ring_offset
                ind = np.searchsorted(self.gain_ring, my_ring, side="right") - 1
                for idet, det in enumerate(self.dets):
                    if "gain" not in self.template_offsets:
                        continue
                    offset = self.template_offsets["gain"][0]
                    offset += idet * self.namplitude_tot
                    # Get all gains for this detector
                    step_gains = self.get_amp(
                        det, slice(offset, offset + self.ngain), baselines, "gain"
                    )
                    if self.fit_distortion:
                        doffset = self.template_offsets["distortion"][0]
                        doffset += idet * self.namplitude_tot
                        step_distortions = self.get_amp(
                            det,
                            slice(doffset, doffset + self.ngain),
                            baselines,
                            "distortion",
                        )
                    else:
                        step_distortions = None
                    # Invert for actual gain
                    step_gains = 1 / (1 + step_gains)
                    if self.rough_gains is not None:
                        step_gains *= self.rough_gains[det]
                    if "gain" not in self.gain_rms:
                        self.gain_rms["gain"] = {}
                        self.gain_mean["gain"] = {}
                    smooth_gains = step_gains.copy()
                    """
                    # Impose a basic continuity requirement to replace
                    # poorly sampled outlier gains from a smoothed gain.
                    if len(step_gains) > 10:
                        self.crop_outlier_gains(smooth_gains, det)
                    """
                    good = smooth_gains != 1
                    if np.sum(good) > 1:
                        self.gain_rms["gain"][det] = np.std(smooth_gains[good])
                    else:
                        self.gain_rms["gain"][det] = 1
                    if np.sum(good) > 0:
                        self.gain_mean["gain"][det] = np.mean(smooth_gains[good])
                    else:
                        self.gain_mean["gain"][det] = 1
                    if self.rank == 0:
                        fname = os.path.join(
                            self.out, "step_gains{}_{}.pck".format(self.siter, det)
                        )
                        with open(fname, "wb") as fout:
                            pickle.dump(
                                [
                                    self.gain_time,
                                    step_gains,
                                    smooth_gains,
                                    step_distortions,
                                ],
                                fout,
                                protocol=2,
                            )
                        print("Step gains saved in {}".format(fname), flush=True)
                    # Interpolate to every locally available ring
                    my_gains = smooth_gains[ind]
                    # Store for future application
                    gains.append(my_gains)
                    if self.fit_distortion:
                        my_distortions = step_distortions[ind]
                        distortions.append(my_distortions)
                gains = np.vstack(gains)
                if self.fit_distortion:
                    distortions = np.vstack(distortions)
                self.comm.Barrier()
                stop = MPI.Wtime()
                if self.rank == 0:
                    print(
                        "    Gains expanded in {:.2f} s:" "".format(stop - start),
                        flush=True,
                    )

        return gains, orbital_gain, distortions

    @function_timer
    def crop_outlier_gains(self, gains, det):
        """
        Find gross outlier gains and flag them as bad.
        """
        ngain = len(gains)
        good = gains != 1
        wmean = max(10, ngain // 10)
        for _ in range(10):
            # Smooth gain
            smooth = flagged_running_average(gains, np.logical_not(good), wmean)
            sigma = np.std((gains - smooth)[good])
            # Detect outliers
            ngood1 = np.sum(good)
            good[np.abs(gains - smooth) > 3 * sigma] = False
            ngood2 = np.sum(good)
            if ngood1 == ngood2:
                break
        # the flagged running average is not well
        # defined at the ends
        good[:wmean] = True
        good[-wmean:] = True
        bad = np.logical_not(good)
        nbad = np.sum(bad)
        gains[bad] = smooth[bad]
        if self.rank == 0 and nbad != 0:
            print("Cropped {} outlier gains for {}".format(nbad, det), flush=True)
        flags = self.tod.local_flags(det)
        pairdet = self.get_pair(det)
        if pairdet is not None:
            pairflags = self.tod.local_flags(pairdet)
        else:
            pairflags = None
        ncrop = 0
        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            start_time = self.local_start_times[iring]
            igain = np.searchsorted(self.gain_time, start_time, side="right")
            igain -= 1
            if bad[igain]:
                ind = slice(istart, istop)
                flags[ind] |= 1
                if pairflags is not None:
                    pairflags[ind] |= 1
                ncrop += 1
        ncrop = self.comm.reduce(ncrop)
        if self.rank == 0 and ncrop != 0:
            print(
                "Cropped {} rings matching outlier gains for {}" "".format(ncrop, det),
                flush=True,
            )

        return

    @function_timer
    def save_gains(self, gains, orbital_gain, distortions):
        """
        Save the step and cumulative gains to file.
        """
        if not self.calibrate:
            return

        # Save the gains
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Saving gains", flush=True)

        if self.rank == 0:
            hdulist = [pf.PrimaryHDU()]

        times = self.comm.gather(self.local_start_times)
        if self.rank == 0:
            times = np.hstack(times)
            times_out = (times * 1e9).astype(np.int64)
            # Make sure first period starts before the mission
            times_out[0] = 1.6e18
            cols = [pf.Column(name="OBT", format="K", array=times_out)]
            hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols))
            hdulist.append(hdu)

        for idet, det in enumerate(self.dets):
            all_gains = self.comm.gather(gains[idet] * orbital_gain)
            if self.fit_distortion:
                all_distortions = self.comm.gather(distortions[idet] * orbital_gain)
            if self.old_gains is not None:
                all_gains_tot = self.comm.gather(gains[idet] * self.old_gains[idet])
            else:
                all_gains_tot = None
            if self.rank == 0:
                all_gains = np.hstack(all_gains)
                cols = []
                if all_gains_tot is not None:
                    all_gains_tot = np.hstack(all_gains_tot)
                    cols.append(
                        pf.Column(name="cumulative", format="D", array=all_gains_tot)
                    )
                cols.append(pf.Column(name="step", format="D", array=all_gains))
                if self.fit_distortion:
                    all_distortions = np.hstack(all_distortions)
                    cols.append(
                        pf.Column(name="distortion", format="D", array=all_distortions)
                    )
                hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols))
                hdu.header["extname"] = det
                hdulist.append(hdu)

        if self.rank == 0:
            if self.single_detector_mode:
                filename = os.path.join(
                    self.out, "gains_{}{}.fits".format(self.dets[0], self.siter)
                )
            else:
                filename = os.path.join(
                    self.out, "gains_{:03}{}.fits".format(self.freq, self.siter)
                )
            if os.path.isfile(filename):
                os.remove(filename)
            pf.HDUList(hdulist).writeto(filename)

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "    Gains saved to {} in {:.2f} s:" "".format(filename, stop - start),
                flush=True,
            )

        return

    @function_timer
    def _apply_gains(self, rings, gains):
        if self.calibrate:
            for idet, det in enumerate(self.dets):
                for iring, (istart, istop) in enumerate(
                    zip(self.local_starts, self.local_stops)
                ):
                    if det not in rings[iring]:
                        continue
                    ind = slice(istart, istop)
                    signal = self.tod.local_signal(det)[ind]
                    gain = gains[idet][iring]
                    signal *= gain
        return

    @function_timer
    def _subtract_fsl(
        self,
        det,
        iring,
        old_fits,
        templates,
        baselines,
        gain,
        signal,
        best_fits,
        orbital_gain,
        pairdet,
        pair_old_fits,
        pairgain,
        pairsignal,
        pair_best_fits,
        phase,
        ring,
    ):
        if self.do_fsl:
            # FSL read from disk
            if "fsl" in old_fits:
                old_amp = old_fits["fsl"]
            else:
                old_amp = 0
            if self.forcefsl:
                amp = 0
            else:
                offset = templates[iring][det]["fsl"].offset
                amp = self.get_amp(det, offset, baselines, "fsl")
            fsltemplate = self._interpolate_ring_template(
                ring, templates[iring][det]["fsl"].template, phase
            )
            corr = (1 - gain) * old_amp + amp
            signal -= corr * fsltemplate
            if "fsl" not in best_fits:
                if "fsl" in old_fits:
                    best_fits["fsl"] = old_fits["fsl"]
                else:
                    best_fits["fsl"] = 0.0
                best_fits["fsl"] += amp * orbital_gain
            if pairdet is not None:
                if "fsl" in pair_old_fits:
                    old_amp = pair_old_fits["fsl"]
                else:
                    old_amp = 0
                if self.forcefsl:
                    amp = 0
                else:
                    offset = templates[iring][pairdet]["fsl"].offset
                    amp = self.get_amp(pairdet, offset, baselines, "fsl")
                corr = (1 - pairgain) * old_amp + amp
                pairsignal -= corr * fsltemplate
                if "fsl" not in pair_best_fits:
                    if "fsl" in pair_old_fits:
                        pair_best_fits["fsl"] = pair_old_fits["fsl"]
                    else:
                        pair_best_fits["fsl"] = 0.0
                    pair_best_fits["fsl"] += amp * orbital_gain
        if self.fslnames is not None:
            # FSL(s) convolved on the fly
            for fslname in self.fslnames:
                if fslname in old_fits:
                    old_amp = old_fits[fslname]
                else:
                    old_amp = 0
                offset = templates[iring][det][fslname].offset
                amp = self.get_amp(det, offset, baselines, fslname)
                fsltemplate = self._interpolate_ring_template(
                    ring, templates[iring][det][fslname].template, phase
                )
                corr = (1 - gain) * old_amp + amp
                signal -= corr * fsltemplate
                if fslname not in best_fits:
                    if fslname in old_fits:
                        best_fits[fslname] = old_fits[fslname]
                    else:
                        best_fits[fslname] = 0.0
                    best_fits[fslname] += amp * orbital_gain
        return

    @function_timer
    def _interpolate_ring_template(self, ring, template, phase):
        """ Expand a phase-binned ring template

        """
        nbin = ring.phase.size
        sorted_phase = np.zeros(nbin)
        fast_scanning(sorted_phase, ring.phaseorder, ring.phase)
        sorted_phase = np.hstack(
            [sorted_phase[-1] - 2 * np.pi, sorted_phase, sorted_phase[0] + 2 * np.pi]
        )
        sorted_template = np.zeros(nbin)
        fast_scanning(sorted_template, ring.phaseorder, template)
        sorted_template = np.hstack(
            [sorted_template[-1], sorted_template, sorted_template[0]]
        )
        interpolated = np.interp(phase, sorted_phase, sorted_template)
        return interpolated

    @function_timer
    def _subtract_dipole(
        self,
        templates,
        iring,
        det,
        old_fits,
        baselines,
        gain,
        signal,
        best_fits,
        orbital_gain,
        pairdet,
        pair_old_fits,
        pairgain,
        pairsignal,
        pair_best_fits,
        phase,
        ring,
    ):
        for dipo_name in ["dipo_orbital", "dipo_foreground"]:
            if dipo_name not in templates[iring][det]:
                continue
            if dipo_name in old_fits:
                old_amp = old_fits[dipo_name]
            else:
                old_amp = 0
            offset = templates[iring][det][dipo_name].offset
            amp = self.get_amp(det, offset, baselines, dipo_name)
            corr = (1 - gain) * old_amp + amp
            dipotemplate = self._interpolate_ring_template(
                ring, templates[iring][det][dipo_name].template, phase
            )
            signal -= corr * dipotemplate
            if dipo_name not in best_fits:
                if dipo_name in old_fits:
                    best_fits[dipo_name] = old_fits[dipo_name]
                else:
                    best_fits[dipo_name] = 0.0
                best_fits[dipo_name] += amp * orbital_gain
            if pairdet is not None:
                if dipo_name in pair_old_fits:
                    old_amp = pair_old_fits[dipo_name]
                else:
                    old_amp = 0
                offset = templates[iring][pairdet][dipo_name].offset
                amp = self.get_amp(pairdet, offset, baselines, dipo_name)
                corr = (1 - pairgain) * old_amp + amp
                pairsignal -= corr * dipotemplate
                if dipo_name not in pair_best_fits:
                    if dipo_name in pair_old_fits:
                        pair_best_fits[dipo_name] = pair_old_fits[dipo_name]
                    else:
                        pair_best_fits[dipo_name] = 0.0
                    pair_best_fits[dipo_name] += amp * orbital_gain
            del dipotemplate
        return

    @function_timer
    def _subtract_nonlinearity(
        self,
        det,
        templates,
        iring,
        baselines,
        signal,
        best_fits,
        old_fits,
        pairdet,
        pairsignal,
        pair_best_fits,
        pair_old_fits,
        phase,
        ring,
    ):
        if not self.nlcalibrate:
            return
        dipo = self._interpolate_ring_template(
            ring, templates[iring][det]["dipo_orbital"].template, phase
        )
        lowpassed_signal = flagged_running_average(
            signal + dipo, np.isnan(signal), int(self.nllowpass * self.fsample)
        )
        lowpassed_signal -= np.median(lowpassed_signal)
        offset = templates[iring][det]["nlgain"].offset
        nlgain = self.get_amp(det, offset, baselines, "nlgain")
        signal /= 1 + nlgain * lowpassed_signal
        if "nlgain" not in best_fits:
            if "nlgain" in old_fits:
                best_fits["nlgain"] = old_fits["nlgain"]
            else:
                best_fits["nlgain"] = 0.0
            best_fits["nlgain"] += nlgain
        if pairdet is not None:
            offset = templates[iring][pairdet]["nlgain"].offset
            nlgain = self.get_amp(pairdet, offset, baselines, "nlgain")
            pairsignal /= 1 + nlgain * lowpassed_signal
            if "nlgain" not in pair_best_fits:
                if "nlgain" in pair_old_fits:
                    pair_best_fits["nlgain"] = pair_old_fits["nlgain"]
                else:
                    pair_best_fits["nlgain"] = 0.0
                pair_best_fits["nlgain"] += nlgain
        return

    @function_timer
    def _subtract_zodi(
        self,
        phase,
        det,
        iring,
        templates,
        old_fits,
        baselines,
        gain,
        signal,
        best_fits,
        orbital_gain,
        pairdet,
        pair_old_fits,
        pairgain,
        pairsignal,
        pair_best_fits,
        ring,
    ):
        if self.zodier is None:
            return
        zoditemplate = np.zeros(ring.phase.size)
        if pairdet is not None:
            pairzoditemplate = np.zeros(ring.phase.size)
        for zodiname in self.zodinames:
            offset = templates[iring][det][zodiname].offset
            if zodiname in old_fits:
                old_amp = old_fits[zodiname]
            else:
                old_amp = 0
            amp = self.get_amp(det, offset, baselines, zodiname)
            corr = (1 - gain) * old_amp + amp
            zoditemplate += corr * templates[iring][det][zodiname].template
            if zodiname not in best_fits:
                if zodiname in old_fits:
                    best_fits[zodiname] = old_fits[zodiname]
                else:
                    best_fits[zodiname] = 0.0
                best_fits[zodiname] += amp * orbital_gain
            if pairdet is None:
                continue
            offset = templates[iring][pairdet][zodiname].offset
            if zodiname in pair_old_fits:
                old_amp = pair_old_fits[zodiname]
            else:
                old_amp = 0
            amp = self.get_amp(pairdet, offset, baselines, zodiname)
            corr = (1 - pairgain) * old_amp + amp
            # pairsignal -= corr * zodi
            pairzoditemplate += corr * templates[iring][det][zodiname].template
            if zodiname not in pair_best_fits:
                if zodiname in pair_old_fits:
                    pair_best_fits[zodiname] = pair_old_fits[zodiname]
                else:
                    pair_best_fits[zodiname] = 0.0
                pair_best_fits[zodiname] += amp * orbital_gain
        # interpolate the co-added zodi templates to the sample phases
        signal -= self._interpolate_ring_template(ring, zoditemplate, phase)
        if pairdet is not None:
            pairsignal -= self._interpolate_ring_template(ring, pairzoditemplate, phase)
        return

    @function_timer
    def _subtract_distortion(
        self,
        phase,
        det,
        iring,
        templates,
        old_fits,
        baselines,
        gain,
        signal,
        best_fits,
        orbital_gain,
        ring,
    ):
        if not self.fit_distortion:
            return
        offset = templates[iring][det]["distortion"].offset
        amp = self.get_amp(det, offset, baselines, "distortion") * gain
        template = amp * templates[iring][det]["distortion"].template
        if "distortion" not in best_fits:
            if "distortion" in old_fits:
                best_fits["distortion"] = old_fits["distortion"]
            else:
                best_fits["distortion"] = 0.0
            best_fits["distortion"] += amp * orbital_gain
        # interpolate the template to the sample phases
        signal -= self._interpolate_ring_template(ring, template, phase)
        return

    @function_timer
    def _subtract_map_templates(
        self,
        templates,
        iring,
        det,
        interp_pix,
        interp_weights,
        theta,
        phi,
        iquweights,
        old_fits,
        baselines,
        gain,
        signal,
        best_fits,
        orbital_gain,
    ):
        for name, mapsampler in self.mapsamplers.items():
            if name in templates[iring][det]:
                if name in ["pol0", "pol1", "pol2"]:
                    onlypol = True
                else:
                    onlypol = False
                if not self.temperature_only and onlypol:
                    # Polarization was a nuisance parameter in
                    # temperature_only destriping but we want
                    # to preserve it in the TOD.
                    # We record the best fit amplitude for use in the
                    # gain template for submm frequencies
                    offset = templates[iring][det][name].offset
                    if offset is None:
                        continue
                    amp = self.get_amp(det, offset, baselines, name)
                    self.pol_amplitudes[det][name] = amp
                    continue
                if mapsampler.nside == self.bandpass_nside:
                    ipix = interp_pix
                    iweights = interp_weights
                else:
                    ipix = None
                    iweights = None
                fg_toi = mapsampler.atpol(
                    theta,
                    phi,
                    iquweights,
                    interp_pix=ipix,
                    interp_weights=iweights,
                    pol=True,
                    onlypol=onlypol,
                ).astype(np.float64)
                if not np.all(np.isfinite(fg_toi)):
                    print(
                        "{:4} : WARNING: non-finite value in "
                        "fg_toi: {} {} {}".format(self.rank, det, iring, name),
                        flush=True,
                    )
                offset = templates[iring][det][name].offset
                if offset is None:
                    continue
                if name in old_fits:
                    old_amp = old_fits[name]
                else:
                    old_amp = 0
                amp = self.get_amp(det, offset, baselines, name)
                corr = (1 - gain) * old_amp + amp
                signal -= corr * fg_toi
                if name not in best_fits:
                    if name in old_fits:
                        best_fits[name] = old_fits[name]
                    else:
                        best_fits[name] = 0.0
                    best_fits[name] += amp * orbital_gain
        return

    @function_timer
    def _subtract_band_templates(
        self, signal, templates, iring, det, baselines, best_fits, old_fits
    ):
        if self.differentiator is None:
            return
        # While the transfer function residuals are measured
        # fitting a signal estimate, the corrections are
        # applied using the actual signal.
        #
        # This template is special regarding the gain corrections.
        # It automatically matches the gain fluctuations in the
        # signal so we need not apply the gain.
        self.differentiator.set_signal(signal)
        for harm in range(self.nharm):
            band_amp = 1
            # Omit the calibration for the first harmonic
            if self.do_bands and harm > 0:
                # to avoid degeneracy
                name = "spin_harmonic_{:02}_gain".format(harm)
                offset = templates[iring][det][name].offset
                amp = self.get_amp(det, offset, baselines, name)
                band_amp -= amp
                if name not in best_fits:
                    if name in old_fits:
                        best_fits[name] = old_fits[name]
                    else:
                        best_fits[name] = 0.0
                    best_fits[name] += amp
            if self.do_derivs:
                name = "spin_harmonic_{:02}_deriv".format(harm)
                offset = templates[iring][det][name].offset
                amp = self.get_amp(det, offset, baselines, name)
                band_amp -= 1j * amp
                if name not in best_fits:
                    if name in old_fits:
                        best_fits[name] = old_fits[name]
                    else:
                        best_fits[name] = 0.0
                    best_fits[name] += amp
            if band_amp != 1:
                self.differentiator.correct_band(band_amp, harm)
        signal[:] = self.differentiator.get_signal()
        return

    @function_timer
    def _apply_orbital_gain(self, orbital_gain, rings, templates):
        """ Finally apply orbital dipole calibration
        """
        for det in self.dets:
            if det[-1] in "bS" and self.symmetrize:
                continue
            pairdet = self.get_pair(det)
            for iring, (istart, istop) in enumerate(
                zip(self.local_starts, self.local_stops)
            ):
                if det not in rings[iring]:
                    continue
                ind = slice(istart, istop)
                signal = self.tod.local_signal(det)[ind]
                flags = self.tod.local_flags(det)[ind]
                if pairdet is not None:
                    pairsignal = self.tod.local_signal(pairdet)[ind]
                    pairflags = self.tod.local_flags(pairdet)[ind]
                if orbital_gain != 1:
                    signal *= orbital_gain
                    if pairdet is not None:
                        pairsignal *= orbital_gain

                # Paranoid check for NaNs in the cleaned TOD follows
                bad = np.logical_not(np.isfinite(signal))
                if pairdet is not None:
                    bad[np.logical_not(np.isfinite(pairsignal))] = True
                nbad = np.sum(bad)
                if nbad != 0:
                    print(
                        "{:4} : WARNING: {}/{} ring {} contains {} bad "
                        "values AFTER cleaning".format(
                            self.rank, det, pairdet, iring, nbad
                        ),
                        flush=True,
                    )
                    signal[bad] = 0
                    flags[bad] |= 255
                    if pairdet is not None:
                        pairsignal[bad] = 0
                        pairflags[bad] |= 255
                    ngood = bad.size - nbad
                    if ngood == 0:
                        del rings[iring][det]
                        if det in templates[iring]:
                            del templates[iring][det]
                        if pairdet is not None:
                            if pairdet in rings[iring]:
                                del rings[iring][pairdet]
                            if pairdet in templates[iring]:
                                del templates[iring][pairdet]
        return

    @function_timer
    def clean_tod(self, rings, templates, gains, baselines, orbital_gain):
        """
        Expand the ring templates into TOD and subtract from the signal.
        """
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Subtracting templates", flush=True)

        self._apply_gains(rings, gains)

        # Store the old best fits so we may retroactively correct
        # the variable gain on the templates.

        old_best_fits = copy.deepcopy(self.best_fit_amplitudes)
        self.best_fit_amplitudes = {}
        for det in self.dets:
            self.best_fit_amplitudes[det] = {}

        # Correct templates that are shared in a horn.

        for idet, det in enumerate(self.dets):
            if det[-1] in "bS" and self.symmetrize:
                continue
            pairdet = self.get_pair(det)
            best_fits = self.best_fit_amplitudes[det]
            old_fits = old_best_fits[det]
            if pairdet is not None:
                pair_best_fits = self.best_fit_amplitudes[pairdet]
                pair_old_fits = old_best_fits[pairdet]
                pairidet = self.dets.index(pairdet)
            else:
                pair_best_fits = None
                pair_old_fits = None
                pairidet = None
                pairgain = None
                pairsignal = None
            for iring, (istart, istop) in enumerate(
                zip(self.local_starts, self.local_stops)
            ):
                if det not in rings[iring]:
                    continue
                if self.calibrate and self.gain_rms["gain"][det] > 1e-3:
                    gain = gains[idet][iring] / self.gain_mean["gain"][det]
                else:
                    gain = 1
                if pairdet is not None:
                    if self.calibrate and self.gain_rms["gain"][pairdet] > 1e-3:
                        pairgain = (
                            gains[pairidet][iring] / self.gain_mean["gain"][pairdet]
                        )
                    else:
                        pairgain = 1
                ind = slice(istart, istop)
                phase = self.tod.local_phase()[ind]
                signal = self.tod.local_signal(det)[ind]
                if pairdet is not None:
                    pairsignal = self.tod.local_signal(pairdet)[ind]
                quat = self.tod.local_pointing(det)[ind]
                if self.nlcalibrate:
                    if self.mcmode:
                        theta = self.cache.reference("theta_{}".format(det))
                        phi = self.cache.reference("phi_{}".format(det))
                    else:
                        theta, phi = qa.to_position(quat)
                else:
                    theta, phi = None, None

                self._subtract_fsl(
                    det,
                    iring,
                    old_fits,
                    templates,
                    baselines,
                    gain,
                    signal,
                    best_fits,
                    orbital_gain,
                    pairdet,
                    pair_old_fits,
                    pairgain,
                    pairsignal,
                    pair_best_fits,
                    phase,
                    rings[iring][det],
                )

                self._subtract_dipole(
                    templates,
                    iring,
                    det,
                    old_fits,
                    baselines,
                    gain,
                    signal,
                    best_fits,
                    orbital_gain,
                    pairdet,
                    pair_old_fits,
                    pairgain,
                    pairsignal,
                    pair_best_fits,
                    phase,
                    rings[iring][det],
                )

                self._subtract_nonlinearity(
                    det,
                    templates,
                    iring,
                    baselines,
                    signal,
                    best_fits,
                    old_fits,
                    pairdet,
                    pairsignal,
                    pair_best_fits,
                    pair_old_fits,
                    phase,
                    rings[iring][det],
                )

                self._subtract_zodi(
                    phase,
                    det,
                    iring,
                    templates,
                    old_fits,
                    baselines,
                    gain,
                    signal,
                    best_fits,
                    orbital_gain,
                    pairdet,
                    pair_old_fits,
                    pairgain,
                    pairsignal,
                    pair_best_fits,
                    rings[iring][det],
                )

        # Correct the templates that are not shared within a horn

        for idet, det in enumerate(self.dets):
            best_fits = self.best_fit_amplitudes[det]
            old_fits = old_best_fits[det]
            for iring, (istart, istop) in enumerate(
                zip(self.local_starts, self.local_stops)
            ):
                if det not in rings[iring]:
                    continue
                if self.calibrate and self.gain_rms["gain"][det] > 1e-3:
                    gain = gains[idet][iring] / self.gain_mean["gain"][det]
                else:
                    gain = 1
                ind = slice(istart, istop)
                phase = self.tod.local_phase()[ind]
                signal = self.tod.local_signal(det)[ind]
                iquweights = self.tod.local_weights(det)[ind]
                quat = self.tod.local_pointing(det)[ind]

                if self.mcmode:
                    theta = self.cache.reference("theta_{}".format(det))[ind]
                    phi = self.cache.reference("phi_{}".format(det))[ind]
                else:
                    theta, phi = qa.to_position(quat)

                interp_pix, interp_weights = hp.get_interp_weights(
                    self.bandpass_nside, theta, phi, nest=True
                )

                self._subtract_map_templates(
                    templates,
                    iring,
                    det,
                    interp_pix,
                    interp_weights,
                    theta,
                    phi,
                    iquweights,
                    old_fits,
                    baselines,
                    gain,
                    signal,
                    best_fits,
                    orbital_gain,
                )

                self._subtract_band_templates(
                    signal, templates, iring, det, baselines, best_fits, old_fits
                )

                self._subtract_distortion(
                    phase,
                    det,
                    iring,
                    templates,
                    old_fits,
                    baselines,
                    gain,
                    signal,
                    best_fits,
                    orbital_gain,
                    rings[iring][det],
                )

        self._apply_orbital_gain(orbital_gain, rings, templates)

        # Copy the fit values for templates that we no longer fit for

        for det in self.dets:
            for key in old_best_fits[det]:
                if key not in self.best_fit_amplitudes[det]:
                    self.best_fit_amplitudes[det][key] = old_best_fits[det][key]

        # Processes that have no valid data for a detector will have
        # no entries in the best_fits.  Find a process with all detectors
        # and broadcast the best fit dictionary

        ngood = 0
        for det, vals in self.best_fit_amplitudes.items():
            ngood += len(vals)
        ngoods = np.array(self.comm.allgather(ngood))
        imax = np.argmax(ngoods)
        self.best_fit_amplitudes = self.comm.bcast(self.best_fit_amplitudes, root=imax)

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "    Subtracted templates in {:.2f} s".format(stop - start), flush=True
            )

        return

    @function_timer
    def make_full_map(self):
        """
        Use Madam to destripe cleaned TOD into a full frequency map.
        """
        if self.rank == 0:
            print("    Making full map", flush=True)
        pars = self.madampars.copy()
        pars["path_output"] = self.out
        pars["temperature_only"] = self.temperature_only
        # Destripe with ring offsets not to
        # alter the signature of the templates.
        if self.iiter != self.niter - 1:
            # Override the destriping parameters on all but the last iteration
            pars["write_map"] = True
            pars["write_binmap"] = False
            pars["write_matrix"] = False
            pars["write_wcov"] = False
            pars["write_hits"] = False
            pars["mode_detweight"] = 2  # Horn uniform weighting
            pars["kfirst"] = True
            pars["kfilter"] = False
            pars["base_first"] = 10000.0
            pars["good_baseline_fraction"] = 0.2
            pars["cglimit"] = 1e-12
            pars["iter_max"] = 1000
            name_out = self.tod.SIGNAL_NAME
            # Average the noise weights not to confuse the filter, even with
            # uniform weights
            wmean = 0
            for det in self.dets:
                wmean += 1.0 / self.detweights[det]
            wmean = self.ndet / wmean
            detweights = {}
            for det in self.dets:
                detweights[det] = wmean
            pars["noise_weights_from_psd"] = False
            pars["nsubchunk"] = 1
            pars["isubchunk"] = 0
            if self.temperature_only or self.temperature_only_intermediate:
                mode = "I"
                pars["force_pol"] = False
                pars["write_leakmatrix"] = False
                pars["nside_cross"] = self.nside
            else:
                # Destripe an IQU map
                mode = "IQU"
                pars["force_pol"] = True
                pars["pixlim_map"] = 1e-3
                pars["pixlim_cross"] = 1e-3
                pars["write_leakmatrix"] = False
                pars["nside_cross"] = pars["nside_map"] // 2
            if self.single_detector_mode:
                pars["file_root"] = "calibrated{}_{}".format(self.siter, self.dets[0])
                pars["base_first"] = 10000.0
            else:
                pars["file_root"] = "madam_reproc_{}_{}{}".format(
                    mode, self.freq, self.siter
                )
            pars["nside_map"] = self.pix_nside
            pars["survey"] = [
                "survey1 : 1628777577.0 - 1644353466.0",
                "survey2 : 1644353466.0 - 1660332428.0",
                "survey3 : 1660332428.0 - 1675889738.0",
                "survey4 : 1675889738.0 - 1690650790.0",
                "survey5 : 1690650790.0 - 1706765161.0",
                "survey6 : 1706765161.0 - 1722703733.0",
                "survey7 : 1722703733.0 - 1738319530.0",
                "survey8 : 1738319530.0 - 1754258019.0",
                "survey9 : 1754258019.0 - 1759526018.0",
            ]
            for key in ["detset", "detset_nopol"]:
                if key in pars:
                    del pars[key]
            pars["bin_subsets"] = self.save_survey_maps and not self.mcmode
        else:
            # On the last iteration we use the user-provided destriping
            # parameters.
            pars["write_map"] = True
            pars["write_binmap"] = False
            pars["write_wcov"] = not self.mcmode
            pars["write_matrix"] = not self.mcmode
            pars["write_hits"] = not self.mcmode
            if self.mcmode:
                # In MC mode, skip over previously written half ring maps
                nsub = int(pars["nsubchunk"])
                isubchunk = 0
                if nsub > 1:
                    if self.rank == 0:
                        for isub in range(1, nsub + 1):
                            fn_out = os.path.join(
                                pars["path_output"],
                                pars["file_root"]
                                + "_map_sub{}of{}.fits".format(isub, nsub),
                            )
                            if os.path.isfile(fn_out):
                                print(
                                    "        {} exists, skipping".format(fn_out),
                                    flush=True,
                                )
                                isubchunk = isub
                            else:
                                break
                    isubchunk = self.comm.bcast(isubchunk)
                    pars["isubchunk"] = str(isubchunk)
            name_out = None
            detweights = self.detweights

        start1 = MPI.Wtime()
        madam = OpMadam(
            name=self.tod.SIGNAL_NAME,
            name_out=name_out,
            dets=self.dets,
            pixels=self.tod.PIXEL_NAME,
            weights=self.tod.WEIGHT_NAME,
            pixels_nested=True,
            flag_name=self.tod.FLAG_NAME,
            flag_mask=self.detmask | self.ssomask,
            common_flag_mask=self.commonmask,
            params=pars,
            detweights=detweights,
            purge=False,
            mcmode=False,
            conserve_memory=self.force_conserve_memory,
            translate_timestamps=False,
        )

        # Rank # 0 checks if the output map already exists
        if self.rank == 0:
            fn_out = os.path.join(pars["path_output"], pars["file_root"] + "_map.fits")
            there = os.path.isfile(fn_out)
        else:
            there = None
        there = self.comm.bcast(there)

        if there and self.mcmode and self.iiter == self.niter - 1:
            if self.rank == 0:
                print("        {} exists, skipping Madam".format(fn_out), flush=True)
        else:
            if self.rank == 0:
                print("        Calling Madam.", flush=True)
            memreport("before Madam call, iiter = {}".format(self.iiter), self.comm)
            madam.exec(self.data)
            del madam
            self.symmetrize_pixels()
            self.comm.Barrier()
            stop1 = MPI.Wtime()
            if self.rank == 0:
                print(
                    "        Madam done in {:.2f} s".format(stop1 - start1), flush=True
                )
            memreport("after Madam call", self.comm)

        self._update_frequency_map(pars)
        return

    @function_timer
    def fit_quss(self, rings, templates, gains):
        """ Measure bandpass mismatch corrections from the QUSS
        spurious maps.
        """
        if self.rank == 0:
            npix = 12 * self.bandpass_nside ** 2

            # Load the mask
            fname = self.maskfile
            if self.maskfile_bp is not None:
                fname = self.maskfile_bp
            print("Reading {}".format(fname), flush=True)
            mask = hp.read_map(fname, nest=True, dtype=np.float32)
            mask = hp.ud_grade(
                mask,
                self.bandpass_nside,
                order_in="nested",
                order_out="nested",
                dtype=np.float32,
            )

            # Load the spurious maps
            fname = os.path.join(
                self.out,
                "madam_reproc_QUSS_{}{}_bmap.fits".format(self.freq, self.siter),
            )
            fname = self._latest_madam_map(fname)
            print("Reading {}".format(fname), flush=True)
            ss = hp.read_map(
                fname,
                range(2, self.nhorn + 2),
                nest=True,
                dtype=np.float32,
            )
            print("Downgrading SS", flush=True)
            ss = hp.ud_grade(
                ss,
                self.bandpass_nside,
                order_in="nested",
                order_out="nested",
                dtype=np.float32,
            )

            # Load the noise estimates from the right columns of the full
            # noise matrix file
            ind = []
            nrow = 2 + self.nhorn
            i = 0
            for row in range(nrow):
                for col in range(row, nrow):
                    if row == col and row > 1:
                        ind.append(i)
                    i += 1
            fname = fname.replace("_bmap", "_wcov")
            print("Reading {}".format(fname), flush=True)
            ssnoise = hp.read_map(fname, ind, nest=True, dtype=np.float32)
            print("Downgrading SS noise")
            ssnoise[ssnoise == 0] = hp.UNSEEN
            ssnoise = hp.ud_grade(
                ssnoise,
                self.bandpass_nside,
                order_in="nested",
                order_out="nested",
                power=2,
                dtype=np.float32,
            )
            for noise in ssnoise:
                bad = noise == hp.UNSEEN
                noise[:] = 1 / noise
                noise[bad] = 0
                noise *= mask

            corrections = {}
            for ihorn, horn in enumerate(self.horns):
                print("Fitting {}".format(horn[:-1]), flush=True)
                target = ss[ihorn]
                noise = ssnoise[ihorn]
                print("Adding offset as template")
                sources = [np.ones(npix)]
                sourcenames = ["offset"]
                print("Adding gain as template")
                sources.append(self.mapsampler_freq.Map[:])
                sourcenames.append("gain")
                for name, mapsampler in self.mapsamplers.items():
                    if name in ["pol0", "pol1", "pol2"]:
                        continue
                    print("Adding {} as a leakage source".format(name), flush=True)
                    sources.append(mapsampler.Map[:])
                    sourcenames.append(name)
                nsource = len(sources)
                invcov = np.zeros([nsource, nsource])
                proj = np.zeros(nsource)
                for row in range(nsource):
                    rowsource = sources[row] * noise
                    for col in range(row, nsource):
                        colsource = sources[col]
                        invcov[row, col] = np.dot(rowsource, colsource)
                        if row != col:
                            invcov[col, row] = invcov[row, col]
                        proj[row] = np.dot(rowsource, target)
                cov = np.linalg.inv(invcov)
                coeff = np.dot(cov, proj)
                corrections[horn] = {}
                for name, cc in zip(sourcenames, coeff):
                    corrections[horn][name] = cc
                    print("{:8} : {:15} {:18}".format(horn, name, cc), flush=True)
            del ss
            del ssnoise
            del target
            del sources
        else:
            corrections = None
            sourcenames = None

        corrections = self.comm.bcast(corrections)
        sourcenames = self.comm.bcast(sourcenames)

        if self.rank == 0:
            print("Applying horn corrections", flush=True)

        # Apply the QUSS corrections to the horns.
        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            ind = slice(istart, istop)
            for det1 in self.horns:
                if det1 not in rings[iring]:
                    continue
                det2 = self.get_pair(det1)
                idet1 = self.dets.index(det1)
                idet2 = self.dets.index(det2)
                if "gain" in sourcenames:
                    # Apply gain correction.  The offset correction is
                    # lost due to unknown common offset.
                    # FIXME: must save these changes to the gains.
                    gain = np.abs(1 + corrections[det1]["gain"]) ** 0.5
                    signal1 = self.tod.local_signal(det1)[ind]
                    signal2 = self.tod.local_signal(det2)[ind]
                    signal1 /= gain
                    signal2 *= gain
                    gains[idet1] /= gain
                    gains[idet2] *= gain
                    del signal1, signal2
                # Apply bandpass mismatch corrections
                quat = self.tod.local_pointing(det1)[ind]
                if self.mcmode:
                    theta = self.cache.reference("theta_{}".format(det1))[ind]
                    phi = self.cache.reference("phi_{}".format(det1))[ind]
                else:
                    theta, phi = qa.to_position(quat)
                ipix, iweights = hp.get_interp_weights(
                    self.bandpass_nside, theta, phi, nest=True
                )
                for name in sourcenames:
                    if name in self.mapsamplers:
                        mapsampler = self.mapsamplers[name]
                        amp1 = 0.5 * corrections[det1][name]
                        amp2 = -amp1
                        for det, amp in [(det1, amp1), (det2, amp2)]:
                            if det == det1 or mapsampler.pol:
                                iquweights = self.tod.local_weights(det)[ind]
                                fg_toi = mapsampler.atpol(
                                    theta,
                                    phi,
                                    iquweights,
                                    interp_pix=ipix,
                                    interp_weights=iweights,
                                ).astype(np.float64)
                                del iquweights
                            signal = self.tod.local_signal(det)[ind]
                            signal -= amp * fg_toi
                            del signal
                            best_fits = self.best_fit_amplitudes[det]
                            if name not in best_fits:
                                best_fits[name] = 0.0
                            best_fits[name] += amp

        return

    @function_timer
    def _latest_madam_map(self, fname):
        fname_out = fname
        for i in range(1, 100):
            fname_test = fname.replace(".fits", "_{:03}.fits".format(i))
            if os.path.isfile(fname_test):
                print("{} exists.".format(fname_test), flush=True)
                fname_out = fname_test
            else:
                print("{} does not exist.".format(fname_test), flush=True)
                break
        return fname_out

    @function_timer
    def make_quss_map(self):
        """
        Use Madam to bin a QUSS map for measuring polarization corrections.
        """
        if self.rank == 0:
            print("    Making QUSS map", flush=True)
        pars = self.madampars.copy()
        # pars['mode_detweight'] = 1 # Uniform weighting
        pars["mode_detweight"] = 2  # Horn uniform weighting
        pars["path_output"] = self.out
        pars["kfirst"] = False
        # Bin a QUSS map using previously destriped timelines
        mode = "QUSS"
        pars["force_pol"] = True
        pars["temperature_only"] = False
        pars["pixlim_map"] = 1e-6
        pars["kfirst"] = False
        pars["write_map"] = False
        pars["write_binmap"] = True
        pars["write_hits"] = False
        pars["write_leakmatrix"] = False
        pars["write_matrix"] = False
        pars["write_wcov"] = True
        pars["file_root"] = "madam_reproc_{}_{}{}".format(mode, self.freq, self.siter)
        pars["nside_map"] = self.pix_nside
        pars["nsubchunk"] = 1
        pars["isubchunk"] = 0
        pars["bin_subsets"] = False

        # Average the noise weights not to confuse the filter, even with
        # uniform weights

        wmean = 0
        for det in self.dets:
            wmean += 1.0 / self.detweights[det]
        wmean = self.ndet / wmean
        detweights = {}
        for det in self.dets:
            detweights[det] = wmean

        # Create differenced timelines
        self.horns = []
        for det1 in self.dets:
            det2 = self.get_pair(det1)
            if det2 is None or det2[-1] not in "bS":
                continue
            self.horns.append(det1)
            signal1 = self.tod.local_signal(det1)
            signal2 = self.tod.local_signal(det2)
            signal1 -= signal2
        del signal1, signal2

        # Create QUSS weights
        self.nhorn = len(self.horns)
        weights_name = "weights_diff"
        for ihorn, det1 in enumerate(self.horns):
            det2 = self.get_pair(det1)
            iquweights1 = self.tod.local_weights(det1)
            iquweights2 = self.tod.local_weights(det2)
            cachename = "{}_{}".format(weights_name, det1)
            qussweights = self.cache.create(
                cachename, np.float32, [self.nsamp, 2 + self.nhorn]
            )
            qussweights[:, 0] = iquweights1[:, 1] - iquweights2[:, 1]
            qussweights[:, 1] = iquweights1[:, 2] - iquweights2[:, 2]
            qussweights[:, ihorn + 2] = 1
        del iquweights1, iquweights2, qussweights

        start1 = MPI.Wtime()
        madam = OpMadam(
            name=self.tod.SIGNAL_NAME,
            name_out=None,
            dets=self.horns,
            pixels=self.tod.PIXEL_NAME,
            weights=weights_name,
            pixels_nested=True,
            flag_name=self.tod.FLAG_NAME,
            flag_mask=self.detmask | self.ssomask,
            common_flag_mask=self.commonmask,
            params=pars,
            detweights=detweights,
            purge=False,
            purge_weights=True,
            mcmode=False,
            conserve_memory=False,
            translate_timestamps=False,
        )
        if self.rank == 0:
            print("        Calling Madam.", flush=True)
        madam.exec(self.data)
        del madam
        self.symmetrize_pixels()
        self.comm.Barrier()
        stop1 = MPI.Wtime()
        if self.rank == 0:
            print("        Madam done in {:.2f} s".format(stop1 - start1), flush=True)

        self.cache.clear(weights_name + "_.*")

        # restore undifferenced timelines
        for det1 in self.horns:
            det2 = self.get_pair(det1)
            signal1 = self.tod.local_signal(det1)
            signal2 = self.tod.local_signal(det2)
            signal1 += signal2

        self.symmetrize_pixels()
        return

    @function_timer
    def _update_frequency_map(self, pars):
        """ Load the latest full frequency map to use as a gain target

        """
        self.comm.Barrier()
        start = MPI.Wtime()
        start1 = MPI.Wtime()
        if self.rank == 0:
            print("        Loading frequency map", flush=True)
        # If there already was a frequency map in place, Madam appended
        # a running index to the filename.
        path = os.path.join(pars["path_output"], pars["file_root"] + "_map.fits")
        if self.rank == 0:
            path = self._latest_madam_map(path)
            if self.single_detector_mode:
                header = self.get_header(self.dets[0])
                hdulist = pf.open(path, mode="update")
                for key, value in header:
                    hdulist[1].header[key] = value
                hdulist.flush()
                hdulist.close()
            full_map = np.array(
                hp.ud_grade(
                    hp.read_map(path, None, dtype=np.float32, nest=True),
                    self.bandpass_nside,
                    order_in="NESTED",
                    order_out="NESTED",
                )
            )
            shape = full_map.shape
        else:
            shape = None
            path = None
        shape = self.comm.bcast(shape)
        self.frequency_map_path = self.comm.bcast(path)
        if self.rank != 0:
            full_map = np.zeros(shape, dtype=np.float32)
        self.comm.Bcast(full_map)

        self.comm.Barrier()
        stop1 = MPI.Wtime()
        if self.rank == 0:
            print("        Loaded map in {:.2f} s".format(stop1 - start1), flush=True)

        # Fill missing pixels in the frequency map to allow interpolation

        self.comm.Barrier()
        start1 = MPI.Wtime()
        if self.rank == 0:
            print("        Plugging frequency map holes", flush=True)

        for i, component in enumerate(np.atleast_2d(full_map)):
            plug_holes(component, verbose=(self.rank == 0 and i == 0), nest=True)

        self.comm.Barrier()
        stop1 = MPI.Wtime()
        if self.rank == 0:
            print(
                "        Plugged holes in {:.2f} s".format(stop1 - start1), flush=True
            )

        if np.any(full_map == hp.UNSEEN):
            raise Exception("Plug_holes failed")

        # In certain cases we want to override the polarization estimate

        if self.temperature_only_intermediate and "pol0" in self.mapsamplers:
            start1 = MPI.Wtime()
            if self.rank == 0:
                det = self.dets[0]
                print(
                    "        Adding polarization to freqmap from pol templates",
                    flush=True
                )
                if len(full_map) == 3:
                    full_map = full_map[0]
                qmap = np.zeros_like(full_map)
                umap = np.zeros_like(full_map)
                nside = hp.get_nside(full_map)
                npix = full_map.size
                pix = np.arange(full_map.size, dtype=int)
                theta, phi = hp.pix2ang(nside, pix, nest=True)
                del pix
                interp_pix, interp_weights = hp.get_interp_weights(
                    self.mapsamplers["pol0"].nside,
                    theta,
                    phi,
                    nest=True,
                )
                del theta, phi
                buf = np.zeros(npix, dtype=np.float64)
                for name in ["pol0", "pol1", "pol2"]:
                    if name not in self.pol_amplitudes[det]:
                        continue
                    amp = self.pol_amplitudes[det][name]
                    # Interpolate the pol map to full resolution pixels
                    fast_scanning32(buf, interp_pix, interp_weights, self.mapsamplers[name].Map_Q[:])
                    qmap += amp * buf
                    fast_scanning32(buf, interp_pix, interp_weights, self.mapsamplers[name].Map_U[:])
                    umap += amp * buf
                full_map = np.vstack([full_map, qmap, umap])
                del qmap, umap, interp_pix, interp_weights, buf
            self.comm.Bcast(full_map)
            stop1 = MPI.Wtime()
            if self.rank == 0:
                print(
                    "        Sampled polmap in {:.2f} s".format(stop1 - start1), flush=True
                )
        else:
            if self.rank == 0:
                print(
                    f"        Polarization in the gain template based on frequency map."
                    f" temperature_only_intermediate = {self.temperature_only_intermediate},"
                    f" pol_amplitudes = {self.pol_amplitudes}", flush=True
                )

        # Update the frequency map sampler

        del self.mapsampler_freq
        pol = not self.temperature_only
        self.mapsampler_freq = MapSampler(
            "frequency_map",
            pol=pol,
            nside=self.bandpass_nside,
            comm=self.comm,
            cache=self.cache,
            preloaded_map=full_map,
            nest=True,
        )
        self.mapsampler_freq_has_dipole = True
        del full_map

        """
        Smoothing the polarization template may compromise single detector maps
        if self.pol_fwhm:
            self.mapsampler_freq.smooth(fwhm=self.pol_fwhm, lmax=self.pol_lmax)
        """

        if self.rank == 0 and not self.mcmode:
            fname = os.path.join(
                self.out, "gain_template_iter{:02}.fits".format(self.iiter + 1)
            )
            if self.temperature_only:
                map_out = self.mapsampler_freq.Map[:]
            else:
                map_out = [
                    self.mapsampler_freq.Map[:],
                    self.mapsampler_freq.Map_Q[:],
                    self.mapsampler_freq.Map_U[:],
                ]
            try:
                hp.write_map(fname, map_out, dtype=np.float32, nest=True)
            except Exception:
                hp.write_map(
                    fname, map_out, dtype=np.float32, overwrite=True, nest=True
                )
            del map_out
            print("Gain target saved in {}".format(fname), flush=True)

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print(
                "    Built full frequency map in {:.2f} s".format(stop - start),
                flush=True,
            )
        return

    @function_timer
    def _update_map_headers(self, header, madam):
        """ Add the best fit values to the map headers
        """
        for i in range(10, 0, -1):
            pattern = "{}/{}*_bmap_{:03}.fits".format(
                madam.params["path_output"], madam.params["file_root"], i
            )
            fnames = sorted(glob.glob(pattern))
            pattern = "{}/{}*_map_{:03}.fits".format(
                madam.params["path_output"], madam.params["file_root"], i
            )
            fnames += sorted(glob.glob(pattern))
            if len(fnames) != 0:
                break
        if len(fnames) == 0:
            pattern = (
                madam.params["path_output"]
                + "/"
                + madam.params["file_root"]
                + "*_bmap.fits"
            )
            fnames = sorted(glob.glob(pattern))
            pattern = (
                madam.params["path_output"]
                + "/"
                + madam.params["file_root"]
                + "*_map.fits"
            )
            fnames += sorted(glob.glob(pattern))
        for fname in fnames:
            hdulist = pf.open(fname, mode="update")
            for key, value in header:
                hdulist[1].header[key] = value
            hdulist.flush()
            hdulist.close()
        return

    @function_timer
    def _destripe_single_map(self, det, pol_signal, pars):
        """ Bin a de-polarized single detector map from the
        destriped timestream.

        """
        pars = pars.copy()
        pars["path_output"] = os.path.join(self.out, "bandpass_corrected")
        pars["file_root"] = "madam_I" + self.siter + "_" + det
        pars["kfirst"] = True
        pars["write_map"] = True
        pars["bin_subsets"] = self.save_survey_maps and not self.mcmode
        pars["write_hits"] = not self.mcmode
        pars["info"] = 2

        madam = OpMadam(
            name=self.tod.SIGNAL_NAME,
            name_out=self.tod.SIGNAL_NAME,
            dets=[det],
            pixels=self.tod.PIXEL_NAME,
            weights=self.tod.WEIGHT_NAME,
            pixels_nested=True,
            flag_name=self.tod.FLAG_NAME,
            flag_mask=self.detmask | self.ssomask,
            common_flag_mask=self.commonmask,
            params=pars,
            detweights=self.detweights,
            purge=False,
            mcmode=False,
            conserve_memory=False,
            translate_timestamps=False,
        )

        # Depolarize the signal
        self.tod.local_signal(det)[:] -= pol_signal

        madam.exec(self.data)
        # Running madam wipes out the links
        self.symmetrize_pixels()

        # Add the polarization back
        self.tod.local_signal(det)[:] += pol_signal

        if self.rank == 0:
            header = self.get_header(det)
            self._update_map_headers(header, madam)
        del madam

        self.comm.Barrier()
        return

    @function_timer
    def _bin_polarization_templates(self, det, theta, phi, pol_signal, pars):
        # Make maps of the estimated polarization signal and
        # the polarization angle derivative
        pars = pars.copy()
        pars["path_output"] = os.path.join(self.out, "pol")
        pars["kfirst"] = False
        pars["write_binmap"] = True
        pars["file_root"] = "madam_I" + self.siter + "_" + det

        madam = OpMadam(
            name=self.tod.SIGNAL_NAME,
            name_out=None,
            dets=[det],
            pixels=self.tod.PIXEL_NAME,
            weights=self.tod.WEIGHT_NAME,
            pixels_nested=True,
            flag_name=self.tod.FLAG_NAME,
            flag_mask=self.detmask | self.ssomask,
            common_flag_mask=self.commonmask,
            params=pars,
            detweights=self.detweights,
            purge=False,
            mcmode=False,
            conserve_memory=False,
            translate_timestamps=False,
        )

        # Build the templates
        iquweights = self.tod.local_weights(det)
        if self.temperature_only:
            pol_signal = np.zeros_like(self.tod.local_signal(det))
            pol_signal_deriv = np.zeros_like(self.tod.local_signal(det))
            for name in ["pol0", "pol1", "pol2"]:
                if name not in self.pol_amplitudes[det]:
                    continue
                amp = self.pol_amplitudes[det][name]
                pol_signal += (
                    self.mapsamplers[name].atpol(theta, phi, iquweights, onlypol=True)
                    * amp
                )
                pol_signal_deriv += (
                    self.mapsamplers[name].atpol(
                        theta, phi, iquweights, onlypol=True, pol_deriv=True
                    )
                    * amp
                )
        else:
            pol_signal_deriv = self.mapsampler_freq.atpol(
                theta, phi, iquweights, onlypol=True, pol_deriv=True
            )
        del iquweights

        # Bin polarization template
        self.tod.local_signal(det)[:] = pol_signal
        madam.exec(self.data)
        self.symmetrize_pixels()
        self.comm.Barrier()

        # Bin polarization derivative template
        madam.params["path_output"] = os.path.join(self.out, "pol_deriv")
        self.tod.local_signal(det)[:] = pol_signal_deriv
        madam.exec(self.data)
        self.symmetrize_pixels()

        del madam
        self.comm.Barrier()
        return

    @function_timer
    def _bin_calibrated_map(self, det, pol_signal, rings, theta, phi, templates, pars):
        """ Bin the bandpass correction template

        """
        iquweights = self.tod.local_weights(det)
        bp_template = -pol_signal
        for iring, (istart, istop) in enumerate(
            zip(self.local_starts, self.local_stops)
        ):
            if det not in rings[iring]:
                # This detector and ring were completely flagged
                continue
            ind = slice(istart, istop)
            interp_pix, interp_weights = hp.get_interp_weights(
                self.bandpass_nside, theta[ind], phi[ind], nest=True
            )

            for name, mapsampler in self.mapsamplers.items():
                if name in ["pol", "pol0", "pol1", "pol2"]:
                    # Don't add the polarization back
                    continue
                if (
                    name in templates[iring][det]
                    and name in self.best_fit_amplitudes[det]
                ):
                    if mapsampler.nside == self.bandpass_nside:
                        ipix = interp_pix
                        iweights = interp_weights
                    else:
                        ipix = None
                        iweights = None
                    fg_toi = mapsampler.atpol(
                        theta[ind],
                        phi[ind],
                        iquweights[ind],
                        interp_pix=ipix,
                        interp_weights=iweights,
                    ).astype(np.float64)
                    amp = self.best_fit_amplitudes[det][name]
                    bp_template[ind] += amp * fg_toi
                    del fg_toi
            del interp_pix, interp_weights
        del iquweights

        pars = pars.copy()
        pars["path_output"] = os.path.join(self.out, "calibrated")
        if self.mcmode and not self.polparammode:
            pars["kfirst"] = True
            pars["info"] = 2
            pars["write_map"] = True
        else:
            pars["kfirst"] = False
            pars["write_binmap"] = True
        pars["file_root"] = "madam_I" + self.siter + "_" + det

        madam = OpMadam(
            name=self.tod.SIGNAL_NAME,
            name_out=None,
            dets=[det],
            pixels=self.tod.PIXEL_NAME,
            weights=self.tod.WEIGHT_NAME,
            pixels_nested=True,
            flag_name=self.tod.FLAG_NAME,
            flag_mask=self.detmask | self.ssomask,
            common_flag_mask=self.commonmask,
            params=pars,
            detweights=self.detweights,
            purge=False,
            mcmode=False,
            conserve_memory=False,
            translate_timestamps=False,
        )

        # Undo the bandpass correction
        self.tod.local_signal(det)[:] += bp_template

        # Rank # 0 checks if the output map already exists
        if self.rank == 0:
            fn_out = os.path.join(pars["path_output"], pars["file_root"] + "_map.fits")
            there = os.path.isfile(fn_out)
        else:
            there = None
        there = self.comm.bcast(there)

        if there and self.mcmode:
            if self.rank == 0:
                print("        {} exists, skipping Madam".format(fn_out), flush=True)
        else:
            # Bin the map
            madam.exec(self.data)
            self.symmetrize_pixels()
            self.comm.Barrier()

        if self.rank == 0:
            header = self.get_header(det)
            self._update_map_headers(header, madam)
        del madam

        self.comm.Barrier()
        return

    @function_timer
    def make_single_maps(self, rings, templates):
        """
        Make de-polarized single detector maps
        """
        if self.iiter != self.niter - 1:
            # Making single detector maps is expensive.  Only output
            # them from the final iteration.
            return
        if (not self.save_single_maps) or (self.single_detector_mode):
            return

        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Building single maps", flush=True)

        pars = self.madampars.copy()

        # pars['mode_detweight'] = 1 # Uniform weighting
        pars["mode_detweight"] = 2  # Horn uniform weighting
        pars["write_map"] = False
        pars["write_binmap"] = False
        pars["write_matrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = False
        pars["write_leakmatrix"] = False
        pars["force_pol"] = False
        pars["temperature_only"] = True
        pars["nsubchunk"] = 1
        pars["isubchunk"] = 0
        pars["path_output"] = self.out
        pars["info"] = 0
        pars["survey"] = [
            "survey1 : 1628777577.0 - 1644353466.0",
            "survey2 : 1644353466.0 - 1660332428.0",
            "survey3 : 1660332428.0 - 1675889738.0",
            "survey4 : 1675889738.0 - 1690650790.0",
            "survey5 : 1690650790.0 - 1706765161.0",
            "survey6 : 1706765161.0 - 1722703733.0",
            "survey7 : 1722703733.0 - 1738319530.0",
            "survey8 : 1738319530.0 - 1754258019.0",
            "survey9 : 1754258019.0 - 1759526018.0",
        ]
        for key in ["detset", "detset_nopol"]:
            if key in pars:
                del pars[key]
        if self.single_nside is None:
            pars["nside_map"] = self.pix_nside
        else:
            pars["nside_map"] = self.single_nside

        for det in self.dets:
            if self.rank == 0:
                print("        Processing {}".format(det), flush=True)
            quat = self.tod.local_pointing(det)
            if self.mcmode:
                theta = self.cache.reference("theta_{}".format(det))
                phi = self.cache.reference("phi_{}".format(det))
            else:
                theta, phi = qa.to_position(quat)
            del quat
            # Build a polarization TOD
            iquweights = self.tod.local_weights(det)
            if self.temperature_only:
                pol_signal = np.zeros_like(self.tod.local_signal(det))
            else:
                pol_signal = self.mapsampler_freq.atpol(
                    theta, phi, iquweights, onlypol=True
                )
            del iquweights
            if self.single_nside is None or self.pix_nside == self.single_nside:
                pixels_copy = None
            else:
                pixels_copy = self.tod.local_pixels(det).copy()
                self.tod.local_pixels(det)[:] = hp.ang2pix(
                    self.single_nside, theta, phi, nest=True
                )

            # Single detector destriping is not biased by residual
            # bandpass mismatch
            if not self.mcmode or self.polparammode:
                self._destripe_single_map(det, pol_signal, pars)
            # Make a copy of the destriped signal
            signal_copy = self.tod.local_signal(det).copy()
            # Remove the bandpass mismatch template and bin the map.
            # NOTE: in MC mode we destripe the calibrated, not
            # bandpass-corrected signal.  This saves time but means
            # that the signal after reproc is NOT single-detector
            # destriped (not used *currently* for anything).
            self._bin_calibrated_map(
                det, pol_signal, rings, theta, phi, templates, pars
            )
            if not self.mcmode or self.polparammode:
                self._bin_polarization_templates(det, theta, phi, pol_signal, pars)

            # Restore the destriped signal after binning the templates
            self.tod.local_signal(det)[:] = signal_copy
            del theta, phi, pol_signal, signal_copy
            if pixels_copy is not None:
                self.tod.local_pixels(det)[:] = pixels_copy
                del pixels_copy

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print("    Built single maps in {:.2f} s".format(stop - start), flush=True)

        return

    @function_timer
    def write_tod(self):
        """
        Write the calibrated and bandpass-corrected TOD
        """
        if self.mcmode:
            return
        self.comm.Barrier()
        start = MPI.Wtime()
        if self.rank == 0:
            print("    Writing TOD", flush=True)

        self.tod.cache.clear("{}_.*".format(self.tod.FSL_NAME))

        if self.effdir_out is None:
            return

        for det in self.dets:
            signal = self.tod.local_signal(det)
            flags = self.tod.local_flags(det)

            nwrite = 10
            for iwrite in range(nwrite):
                self.comm.Barrier()
                if self.rank == 0:
                    print(
                        "        Writing {} TOI to {} {} / {}".format(
                            det, self.effdir_out, iwrite + 1, nwrite
                        ),
                        flush=True,
                    )
                start1 = MPI.Wtime()
                if self.rank % nwrite == iwrite:
                    self.tod.write_tod_and_flags(
                        detector=det,
                        data=signal,
                        flags=flags,
                        effdir_out=self.effdir_out,
                    )
                self.comm.Barrier()
                stop1 = MPI.Wtime()
                if self.rank == 0:
                    print(
                        "        Wrote {} TOI to {} in {:.2f} s {} / {}"
                        "".format(
                            det, self.effdir_out, stop1 - start1, iwrite + 1, nwrite
                        ),
                        flush=True,
                    )

        self.comm.Barrier()
        stop = MPI.Wtime()
        if self.rank == 0:
            print("    Wrote TOD in {:.2f} s".format(stop - start), flush=True)

        return

    @function_timer
    def get_pair(self, det):
        """ Return the other detector in a horn.

        Return the other detector in a horn but only if we are enforcing
        horn symmetry.

        """
        if self.symmetrize:
            return get_pair(det)
        else:
            return None

    @function_timer
    def exec(self, data):
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world

        self.rank = cworld.Get_rank()
        self.ntask = cworld.Get_size()
        self.comm = cworld
        self.data = data

        self.load_tod(data)

        memreport("after load_tod", self.comm)

        if self.fg is None:
            self.fg, self.fg_deriv, self.fg_deriv2 = get_skymodel(
                self.freq,
                self.bandpass_nside,
                self.sync,
                self.sync_pol,
                self.freefree,
                self.ame,
                self.dust,
                self.dust_pol,
                MPI.COMM_WORLD,
                self.bandpass_fwhm,
                self.skymodel_cache,
                self.cache,
                self.bpcorrect or self.fgdipole,
                self.bpcorrect2,
            )

        if len(self.dets) == 1:
            # Single detector mode
            # self.single_detector_mode = True
            self.temperature_only = True
            self.bpcorrect = False
            self.maskfile_bp = None
            self.symmetrize = False
            self.save_maps = True

        if self.symmetrize:
            self.symmetrize_pointing_and_flags()

        self.old_gains = np.ones([self.ndet, self.nring], dtype=np.float64)

        baselines = None
        orbital_gain = None
        rings = {}

        for self.iiter in range(self.niter):

            memreport("at beginning of iteration", self.comm)

            self.comm.Barrier()
            start = MPI.Wtime()
            if self.rank == 0:
                print(
                    "Starting iteration {} / {}".format(self.iiter + 1, self.niter),
                    flush=True,
                )

            self.siter = "_iter{:02}".format(self.iiter)

            self.skip_polmaps = False
            if self.iiter == 0:
                self.compress_tod(rings, update=False)
                memreport("after compress tod", self.comm)
                templates, namplitude = self.build_templates(rings)
            elif (
                self.iiter == self.niter - 1
                and self.maskfile_bp is not None
                and self.maskfile != self.maskfile_bp
                and self.bpcorrect
                and not self.quss_correct
            ):
                # For the last iteration, build new templates with a
                # much smaller mask and disable calibration and
                # transfer function corrections
                self.calibrate = False
                self.nlcalibrate = False
                self.fit_distortion = False
                self.recalibrate = False
                self.nharm = 0
                self.do_zodi = False
                self.do_dipo = False
                self.do_fsl = False
                self.zodier = None
                self.maskfile = self.maskfile_bp
                self.compress_tod(rings, update=False)
                self.temperature_only_destripe = False
                if self.temperature_only_intermediate:
                    self.skip_polmaps = True
                elif not self.temperature_only:
                    self.polmap = None
                    self.polmap2 = None
                    self.polmap3 = None
                    for name in ["pol", "pol0", "pol1", "pol2"]:
                        if name in self.mapsamplers:
                            del self.mapsamplers[name]
                    self.force_polmaps = False
                templates, namplitude = self.build_templates(rings)
                if not self.temperature_only:
                    for iring in rings.keys():
                        for det in self.dets:
                            if det in templates[iring]:
                                for name in ["pol", "pol0", "pol1", "pol2"]:
                                    if name in templates[iring][det]:
                                        templates[iring][det][name].offset = None
                for name in ["pol", "pol0", "pol1", "pol2"]:
                    if name in self.template_offsets:
                        del self.template_offsets[name]
            else:
                self.compress_tod(rings, update=True)
                memreport("after compress_tod", self.comm)
                self.update_templates(rings, templates)

            self.project_offsets(rings, templates, (self.iiter == 0))

            if self.iiter == 0:
                self.detect_outliers(rings, templates)
                self.rough_cal(rings, templates)

            memreport("after project/detect/rough_cal", self.comm)

            baselines = self.destripe(rings, templates, namplitude, self.iiter)

            memreport("after destripe", self.comm)

            gains, orbital_gain, distortions = self.get_gains(
                baselines, dipo_subtracted=(self.iiter != 0)
            )

            self.clean_tod(rings, templates, gains, baselines, orbital_gain)

            if self.iiter == self.niter - 1:
                if self.quss_correct:
                    # After the last iteration, use QUSS to further improve
                    # the bandpass mismatch correction
                    self.make_quss_map()
                    self.fit_quss(rings, templates, gains)
                # Last iteration: delete harmonic templates
                for iring in sorted(templates.keys()):
                    for det in sorted(templates[iring].keys()):
                        for name in sorted(templates[iring][det].keys()):
                            if "harmonic" in name:
                                del templates[iring][det][name]

            memreport("after clean_tod", self.comm)

            self.make_full_map()

            self.save_gains(gains, orbital_gain, distortions)

            memreport("after make_full_map", self.comm)

            self.make_single_maps(rings, templates)

            if self.calibrate:
                # Make a copy of the most recent gains
                self.old_gains *= gains * orbital_gain

            self.rough_gains = None

            self.comm.Barrier()
            stop = MPI.Wtime()
            if self.rank == 0:
                print(
                    "Completed iteration in {:.2f} s".format(stop - start), flush=True
                )

        memreport("after iterations", self.comm)

        del self.mapsampler_freq
        del self.mapsamplers
        del self.mask
        del self.mask_bp
        if self.cache.exists("mask"):
            self.cache.destroy("mask")
        if self.cache.exists("mask_bp"):
            self.cache.destroy("mask_bp")
        self.cache.clear("orbital_dipole.*")

        self.write_tod()

        memreport("after write_tod", self.comm)

        return
