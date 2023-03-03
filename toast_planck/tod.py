# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

import copy

from scipy.constants import c

import astropy.io.fits as pf
import healpy as hp
import numpy as np
import toast.cache as tc
from toast.mpi import MPI
import toast.qarray as qa
from toast.tod import TOD, Noise
from toast.tod.interval import Interval

from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers

from .utilities import (load_ringdb, count_samples, read_eff,
                        write_eff, load_RIMO, bolos_by_freq, to_diodes,
                        to_radiometer, list_files, PLANCK_DETINDX)

XAXIS, YAXIS, ZAXIS = np.eye(3, dtype=np.float64)

# Inverse light speed in km / s ( the assumed unit for velocity )
CINV = 1e3 / c

PATTERN_SEPARATOR = ':'

noisefiles = {}

# cache all searched directories
filenames_cache = {}


class Exchange(TOD):

    PHASE_NAME = 'phase'
    FSL_NAME = 'fsl'
    WEIGHT_NAME = 'weights'
    PIXEL_NAME = 'pixels'

    """Provide pointing and detector timestreams for
    Planck Exchange File Format data.

    Args:
        comm (mpi4py.MPI.Comm): MPI communicator over which the data is
            distributed.
        detectors (list): list of names to use for the detectors.
             Must match the names in the FITS HDUs.
        ringdb: Path to an SQLite database defining ring boundaries.
        effdir: directory containing the exchange files
        obt_range: data span in TAI seconds, overrides ring_range
        ring_range: data span in pointing periods, overrides od_range
        od_range: data span in operational days
        sim (bool): if True, we only read the pointing data and flags,
                     since the TOD will be simulated.
    """

    # @profile
    def __init__(self, comm=None, detectors=None, ringdb=None,
                 effdir_in=None, extra_effdirs=None,
                 effdir_in_diode0=None, effdir_in_diode1=None,
                 effdir_out=None, effdir_dark=None, effdir_pntg=None,
                 effdir_fsl=None, effdir_flags=None,
                 obt_range=None, ring_range=None, od_range=None, freq=None,
                 RIMO=None, coord='G', obtmask=0, flagmask=0, darkmask=0,
                 pntflagmask=2, sim=False, lfi_raw=False, do_eff_cache=True,
                 noisefile='RIMO', noisefile_simu='RIMO'):
        if ringdb is None:
            raise ValueError('You must provide a path to the ring database')
        if effdir_in is None:
            raise ValueError('You must provide a path to the exchange files')
        if freq is None:
            raise ValueError('You must specify the frequency to run on')
        if RIMO is None:
            raise ValueError('You must specify which RIMO to use')

        self.ringdb_path = ringdb
        self.ringdb = load_ringdb(self.ringdb_path, comm, freq)
        self.RIMO_path = RIMO
        self.RIMO = load_RIMO(self.RIMO_path, comm)
        self.sim = sim
        self.lfi_raw = lfi_raw
        self.noisefile = noisefile
        self.noisefile_simu = noisefile_simu
        if do_eff_cache:
            self.eff_cache = {}
        else:
            self.eff_cache = None
        self.freq = freq
        self.coord = coord

        self.comm = comm
        if comm is None:
            self.rank = 0
        else:
            self.rank = comm.rank

        self.globalstart = 0.0
        self.globalfirst = 0
        self.globalfirst_ring = 0
        self.allsamp = 0
        self.allrings = []
        self.allmodes = []
        self.modenames = []

        if self.rank == 0:
            (self.globalstart, self.globalfirst, self.allsamp, self.allrings,
             self.allmodes, self.modenames,
             self.globalfirst_ring) = count_samples(
                 self.ringdb, self.freq, obt_range, ring_range, od_range)

        if comm is not None:
            self.globalstart = comm.bcast(self.globalstart, root=0)
            self.globalfirst = comm.bcast(self.globalfirst, root=0)
            self.globalfirst_ring = comm.bcast(self.globalfirst_ring, root=0)
            self.allsamp = comm.bcast(self.allsamp, root=0)
            self.allrings = comm.bcast(self.allrings, root=0)
            self.allmodes = comm.bcast(self.allmodes, root=0)
            self.modenames = comm.bcast(self.modenames, root=0)

        if detectors is None:
            detectors = bolos_by_freq(self.freq)
        self.fsample = self.rimo[detectors[0]].fsample

        # for data distribution, we need the contiguous sample
        # ranges based on the ring starts.

        ringstarts = [x.first for x in self.allrings]

        self.ringsizes = [
            (y - x) for x, y in zip(ringstarts[:-1], ringstarts[1:])]
        self.ringsizes.append(
            self.allrings[-1].last - self.allrings[-1].first + 1)

        super().__init__(
            comm, detectors, self.allsamp, detindx=PLANCK_DETINDX,
            sampsizes=self.ringsizes)

        self.obtmask = obtmask
        self.flagmask = flagmask
        self.darkmask = darkmask
        self.pntflagmask = pntflagmask

        self.satobtmask = 1
        self.satquatmask = 1
        self.satvelmask = 1

        if self.coord == 'E':
            self.coordmatrix = None
            self.coordquat = None
        else:
            (self.coordmatrix,
             _, _) = hp.rotator.get_coordconv_matrix(['E', self.coord])
            self.coordquat = qa.from_rotmat(self.coordmatrix)

        self.cache_effdirs(effdir_in, effdir_in_diode0, effdir_in_diode1,
                           effdir_out, effdir_dark, effdir_pntg, effdir_fsl,
                           extra_effdirs, effdir_flags)

        if self.noisefile == 'RIMO':
            self.noise = self.get_rimo_noise()
        else:
            self.noise = self.get_noisefile_noise(self.noisefile)

        if self.noisefile == self.noisefile_simu:
            self.noise_simu = self.noise
        else:
            if self.noisefile_simu == 'RIMO':
                self.noise_simu = self.get_rimo_noise()
            else:
                self.noise_simu = self.get_noisefile_noise(self.noisefile_simu)

        self.ringsets = {}
        return

    def cache_effdirs(self, effdir_in, effdir_in_diode0, effdir_in_diode1,
                      effdir_out, effdir_dark, effdir_pntg, effdir_fsl,
                      extra_effdirs, effdir_flags):
        """ Cache the metadata so we don't need to look for files
        while reading and writing

        """
        if effdir_in is not None and PATTERN_SEPARATOR in effdir_in:
            self.effdir_in, self.effdir_in_pattern = effdir_in.split(
                PATTERN_SEPARATOR)
        else:
            self.effdir_in, self.effdir_in_pattern = effdir_in, None
        self.effdir_in_diode0 = effdir_in_diode0
        self.effdir_in_diode1 = effdir_in_diode1
        if effdir_out is not None and PATTERN_SEPARATOR in effdir_out:
            self.effdir_out, self.effdir_out_pattern = effdir_out.split(
                PATTERN_SEPARATOR)
        else:
            self.effdir_out, self.effdir_out_pattern = effdir_out, None
        self.effdir_out = effdir_out
        if effdir_dark is not None:
            self.effdir_dark = effdir_dark
        else:
            self.effdir_dark = self.effdir_in
        if effdir_pntg is not None:
            self.effdir_pntg = effdir_pntg
        else:
            self.effdir_pntg = self.effdir_in
        self.effdir_fsl = effdir_fsl
        self.extra_effdirs = extra_effdirs
        if effdir_flags is None:
            self.effdir_flags = self.effdir_in
            self.effdir_flags_pattern = self.effdir_in_pattern
        else:
            if PATTERN_SEPARATOR in effdir_flags:
                (self.effdir_flags, self.effdir_flags_pattern
                 ) = effdir_flags.split(PATTERN_SEPARATOR)
            else:
                (self.effdir_flags, self.effdir_flags_pattern
                 ) = effdir_flags, None

        if self.rank == 0:
            all_effdirs = [
                self.effdir_in, self.effdir_out, self.effdir_pntg,
                self.effdir_dark, self.effdir_fsl, self.effdir_flags,
                self.effdir_in_diode0, self.effdir_in_diode1]
            if self.extra_effdirs is not None:
                for effdir in self.extra_effdirs:
                    all_effdirs.append(effdir)

            for effdir in all_effdirs:
                if effdir is None:
                    continue
                if effdir in filenames_cache:
                    continue
                print('Building a list of files under {} ...'.format(effdir),
                      end='', flush=True)
                timer = Timer()
                timer.start()
                filenames_cache[effdir] = sorted(list_files(effdir))
                timer.stop()
                timer.report("List files")
        if self.comm is None:
            self.filenames = filenames_cache
        else:
            self.filenames = self.comm.bcast(filenames_cache, root=0)
        return

    def get_rimo_noise(self):
        """ Create a noise object from the RIMO

        """
        det = self.local_dets[0]
        if det[-1] in '01' and det[-2] != '-':
            det = to_radiometer(det)
        freqs = np.exp(
            np.linspace(np.log(1e-6), np.log(self.fsample / 2), 1000))
        fmin = 1 / 86400
        psddict = {}
        freqdict = {}
        for detector in self.local_dets:
            if detector[-1] in '01' and detector[-2] != '-':
                det = to_radiometer(detector)
            else:
                det = detector
            net = self.rimo[det].net
            fknee = self.rimo[det].fknee
            alpha = -self.rimo[det].alpha  # RIMO has the slope (alpha<0)
            if fknee <= 0:
                if self.rank == 0:
                    print('WARNING: The RIMO does not have a knee frequency '
                          'for {}. Using a white noise filter.'.format(det))
                psd = net ** 2 * np.ones_like(freqs)
            else:
                psd = (net ** 2 * (fknee ** alpha + freqs ** alpha)
                       / (fmin ** alpha + freqs ** alpha))
            freqdict[detector] = freqs
            psddict[detector] = psd
        noise = Noise(detectors=self.local_dets, freqs=freqdict,
                      psds=psddict, indices=PLANCK_DETINDX)
        return noise

    def get_noisefile_noise(self, noisefile, t0=0):
        """ Create a noise object from the most appropriate
        PSD entry in the noise file.

        """
        psddict = {}
        freqdict = {}
        mixmatrix = {}
        indices = {}
        for detector in self.local_dets:
            fn = noisefile.replace('DETECTOR', detector)
            if fn in noisefiles:
                hdulist = noisefiles[fn]
            else:
                hdulist = pf.open(fn, 'readonly')
                noisefiles[fn] = hdulist
            # Find the latest PSD that starts before t0
            t = hdulist[1].data.field(0)
            i = 0
            while i + 1 < t.size and t[i + 1] < t0:
                i += 1
            mixmatrix[detector] = {}
            if len(hdulist) == 3:
                # Noise file only has one PSD
                single_detector = True
                key = detector
            else:
                if fn != noisefile:
                    raise Exception(
                        '{} contains multiple PSDs but is detector-specific'
                        ''.format(fn))
                single_detector = False
            for hdu in hdulist[2:]:
                if not single_detector:
                    key = hdu.header['extname']
                freq = hdu.data.field(0)[0]
                # toast requires the last frequency to match Nyquist
                freq[-1] = self.fsample / 2
                freqdict[key] = freq
                psddict[key] = hdu.data.field(0)[i + 1]
                if single_detector:
                    mixmatrix[detector][key] = 1
                    indices[key] = PLANCK_DETINDX[detector]
                else:
                    # parse the entire mixing matrix from the HDU header
                    indices[key] = hdu.header['index']
                    for idet in range(100):
                        weightkey = 'weight{:02}'.format(idet)
                        if weightkey in hdu.header:
                            weight = hdu.header[weightkey]
                        else:
                            break
                        detkey = 'det{:02}'.format(idet)
                        if detkey in hdu.header:
                            det = hdu.header[detkey]
                        else:
                            raise Exception(
                                'Detector not defined: {}'.format(detkey))
                        if det not in mixmatrix:
                            mixmatrix[det] = {}
                        mixmatrix[det][key] = weight
            if not single_detector:
                break
        noise = Noise(detectors=self.local_dets, freqs=freqdict,
                      psds=psddict, mixmatrix=mixmatrix, indices=indices)
        return noise

    def from_tod(self, ring, comm, noisefile='RIMO', noisefile_simu='RIMO'):
        """ Generate a new TOD object for a specific ring from an
        existing, long TOD object.

        """
        # Do a shallow copy and only replace the parts
        # that need to be private
        tod = copy.copy(self)
        tod.globalfirst_ring += ring
        tod.cache = tc.Cache()
        tod.eff_cache = self.eff_cache
        tod.comm = comm
        tod._rank_det = 0
        tod._rank_samp = 0
        tod._mpicomm = comm
        sizes = tod._sizes
        n = self.allrings[ring].last - self.allrings[ring].first + 1
        tod._nsamp = n
        tod._sizes = [n]
        tod._dist_samples = [(0, n)]
        # Crop to the list of rings and adjust the start index
        tod.allrings = [Interval(
            start=self.allrings[ring].start, stop=self.allrings[ring].stop,
            first=0, last=n)]
        offset = np.sum(sizes[:ring], dtype=int)
        tod.globalfirst += offset
        # Crop to the list of ACMS modes and adjust the start index
        modes = []
        modenames = []
        for interval, name in zip(self.allmodes, self.modenames):
            first = interval.first - offset
            last = interval.last - offset
            if first <= n and last >= 0:
                modes.append(Interval(
                    start=interval.start, stop=interval.stop,
                    first=first, last=last))
                modenames.append(name)
        tod.allmodes = modes
        tod.modenames = modenames
        tod._dist_sizes = [(0, 1)]
        t0 = self.allrings[ring].start
        if noisefile != 'RIMO':
            tod.noise = self.get_noisefile_noise(noisefile, t0)
        if noisefile_simu != 'RIMO':
            if noisefile == noisefile_simu:
                tod.noise_simu = tod.noise
            else:
                tod.noise_simu = self.get_noisefile_noise(noisefile_simu, t0)
        return tod

    def set_effdir_out(self, effdir, pattern):
        if PATTERN_SEPARATOR in effdir:
            if pattern is not None:
                raise RuntimeError(
                    'ERROR: effdir includes a pattern (effdir="{}") but '
                    'optional pattern is also provided: "{}".'.format(
                        effdir, pattern))
            self.effdir_out, self.effdir_out_pattern = effdir.split(
                PATTERN_SEPARATOR)
        else:
            self.effdir_out, self.effdir_out_pattern = effdir, pattern

        if effdir not in self.filenames:
            if self.rank == 0:
                filenames = sorted(list_files(effdir))
            else:
                filenames = None
            if self.comm is None:
                self.filenames[effdir] = filenames
            else:
                self.filenames[effdir] = self.comm.bcast(filenames, root=0)
        return

    def cache_metadata(self, path, comm=None):
        """ Perform a walk in path can store the list of found files.

        """
        if path in self.filenames:
            return
        if comm is None:
            comm = self.comm
            rank = self.rank
        else:
            rank = comm.rank
        if rank == 0:
            files = sorted(list_files(path))
        else:
            files = None
        if comm is None:
            self.filenames[path] = files
        else:
            self.filenames[path] = comm.bcast(files, root=0)
        return

    def purge_eff_cache(self, keep_common=False):
        if self.eff_cache is None:
            return
        # Remove entries from the TOD cache
        cachenames = []
        for effdir in self.eff_cache.keys():
            for od in self.eff_cache[effdir].keys():
                for extname in self.eff_cache[effdir][od].keys():
                    cachename = ':'.join([effdir, str(od), extname])
                    if keep_common and extname in [
                            'time', 'obt', 'phase', 'attitude', 'velocity',
                            'position', 'timeflag', 'obtflag', 'phaseflag',
                            'attitudeflag', 'velocityflag', 'positionflag']:
                        pass
                    else:
                        cachenames.append(cachename)
        for cachename in cachenames:
            effdir, od, extname = cachename.split(':')
            try:
                del self.eff_cache[effdir][int(od)][extname]
            except Exception:
                pass
            try:
                self.cache.destroy(cachename)
            except Exception:
                pass
        return

    @property
    def valid_intervals(self):
        return self.allrings

    @property
    def rimo(self):
        return self.RIMO

    def local_pixels(self, det, name=None):
        if name is None:
            cachename = '{}_{}'.format(self.PIXEL_NAME, det)
        else:
            cachename = '{}_{}'.format(name, det)
        return self.cache.reference(cachename)

    def local_weights(self, det, name=None):
        if name is None:
            cachename = '{}_{}'.format(self.WEIGHT_NAME, det)
        else:
            cachename = '{}_{}'.format(name, det)
        return self.cache.reference(cachename)

    def local_timestamps(self, name=None, **kwargs):
        """ Planck flavor of local_timestamps always caches both the
        times and the common flags.

        """
        if name is None:
            cachename = self.TIMESTAMP_NAME
            if not self.cache.exists(cachename):
                timestamps, commonflags = self.read_times(and_flags=True,
                                                          **kwargs)
                self.cache.put(self.TIMESTAMP_NAME, timestamps)
                self.cache.put(self.COMMON_FLAG_NAME, commonflags)
        else:
            cachename = name
        return self.cache.reference(cachename)

    def local_signal(self, det, name=None, **kwargs):
        """ Planck flavor of local_signal always caches both the signal
        and the flags.

        """
        if name is None:
            todname = '{}_{}'.format(self.SIGNAL_NAME, det)
            if not self.cache.exists(todname):
                signal, flags = self.read(detector=det, and_flags=True,
                                          **kwargs)
                self.cache.put(todname, signal)
                flagname = '{}_{}'.format(self.FLAG_NAME, det)
                if not self.cache.exists(flagname):
                    self.cache.put(flagname, flags)
        else:
            todname = '{}_{}'.format(name, det)
        return self.cache.reference(todname)

    def local_flags(self, det, name=None, **kwargs):
        """ Planck flavor of local_flags always caches both the signal
        and the flags.

        """
        if name is None:
            flagname = '{}_{}'.format(self.FLAG_NAME, det)
            if not self.cache.exists(flagname):
                local_start = 0
                n = self.local_samples[1] - local_start
                signal, flags = self._get(det, local_start, n, and_flags=True,
                                          **kwargs)
                self.cache.put(flagname, flags)
                todname = '{}_{}'.format(self.SIGNAL_NAME, det)
                if not self.cache.exists(todname):
                    self.cache.put(todname, signal)
        else:
            flagname = '{}_{}'.format(name, det)
        return self.cache.reference(flagname)

    def _get(self, detector, local_start, n, margin=0, and_flags=False,
             effdir=None, file_pattern=None):
        # The backend is set up to cache both the TOD and flags so we'll just
        # read both and return the data

        if self.lfi_raw:
            dets = to_diodes(detector)
            lfi_data = []
        else:
            dets = detector,

        if effdir is not None:
            effdir_in = effdir
            file_pattern = file_pattern
        else:
            if detector[-1] in '01' and detector[-2] != '-':
                # This is a single diode
                if self.lfi_raw:
                    raise Exception(
                        'Cannot read individual detectors in LFI RAW mode.')
                if detector[-1] == '0':
                    effdir_in = self.effdir_in_diode0
                else:
                    effdir_in = self.effdir_in_diode1
                # The EFF files refer to the radiometers, not diodes
                dets = to_radiometer(detector),
                file_pattern = None
            else:
                if and_flags and self.effdir_in != self.effdir_flags:
                    raise RuntimeError(
                        'Reading TOD and flags simultaneously is not supported '
                        'when effdir_in({}) != effdir_flags ({})'.format(
                            self.effdir_in, self.effdir_flags))
                effdir_in = self.effdir_in
                file_pattern = self.effdir_in_pattern

        for det in dets:
            data, flags = read_eff(
                local_start - margin, n + 2 * margin, self.globalfirst,
                self.local_samples[0], self.ringdb, self.ringdb_path,
                self.freq, effdir_in, det.lower(), self.flagmask,
                self.eff_cache, self.cache, self.filenames[effdir_in],
                file_pattern=file_pattern)

            if self.lfi_raw:
                lfi_data.append(data.T)

        if self.lfi_raw:
            data = np.hstack(lfi_data).copy()

        if and_flags:
            return data, flags
        else:
            return data

    def _get_flags(self, detector, local_start, n, margin=0):
        # The backend is set up to cache both the TOD and flags so we'll just
        # read both and return the flags

        if self.lfi_raw:
            dets = to_diodes(detector)
        else:
            dets = detector,

        if detector[-1] in '01' and detector[-2] != '-':
            # This is a single diode
            if self.lfi_raw:
                raise Exception(
                    'Cannot read individual detectors in LFI RAW mode.')
            if detector[-1] == '0':
                effdir_in = self.effdir_in_diode0
            else:
                effdir_in = self.effdir_in_diode1
            # The EFF files refer to the radiometers, not diodes
            dets = to_radiometer(detector),
            file_pattern = None
        else:
            effdir_in = self.effdir_flags
            file_pattern = self.effdir_flags_pattern

        flags = np.zeros(n + 2 * margin, dtype=np.byte)

        for det in dets:
            _, flag = read_eff(
                local_start - margin, n + 2 * margin, self.globalfirst,
                self.local_samples[0], self.ringdb, self.ringdb_path,
                self.freq, effdir_in, det.lower(), self.flagmask,
                self.eff_cache, self.cache, self.filenames[effdir_in],
                file_pattern=file_pattern)
            flags |= flag

        return flags

    def _put(self, detector, local_start, data, effdir_out=None,
             file_pattern=None):
        if effdir_out is None:
            effdir_out = self.effdir_out
            file_pattern = self.effdir_out_pattern

        if detector[-1] in '01' and detector[-2] != '-':
            det = to_radiometer(detector)
        else:
            det = detector

        write_eff(
            local_start, data, None, self.globalfirst, self.local_samples[0],
            self.ringdb, self.ringdb_path, self.freq, effdir_out,
            det.lower(), self.filenames[effdir_out], file_pattern=file_pattern)

        return

    def _put_tod_and_flags(self, detector, local_start, data, flags,
                           effdir_out=None, file_pattern=None):

        if effdir_out is None:
            effdir_out = self.effdir_out
            file_pattern = self.effdir_out_pattern

        if detector[-1] in '01' and detector[-2] != '-':
            det = to_radiometer(detector)
        else:
            det = detector

        write_eff(
            local_start, data, flags, self.globalfirst, self.local_samples[0],
            self.ringdb, self.ringdb_path, self.freq, effdir_out, det.lower(),
            self.filenames[effdir_out], file_pattern=file_pattern)

        return

    def _put_flags(self, detector, local_start, flags, effdir_out=None):

        if effdir_out is None:
            effdir_out = self.effdir_out
            file_pattern = self.effdir_out_pattern

        if detector[-1] in '01' and detector[-2] != '-':
            det = to_radiometer(detector)
        else:
            det = detector

        write_eff(
            local_start, None, flags, self.globalfirst, self.local_samples[0],
            self.ringdb, self.ringdb_path, self.freq, effdir_out,
            det.lower(), self.filenames[effdir_out], file_pattern=file_pattern)

        return

    def _get_pntg(self, detector, local_start, n, deaberrate=True, margin=0,
                  velocity=None, full_output=False, satquats=None):
        if detector[-1] in '01' and detector[-2] != '-':
            # Single diode, use common radiometer pointing
            det = to_radiometer(detector)
        else:
            det = detector

        detquat = self.RIMO[det].quat

        if satquats is None:
            # Get the satellite attitude
            satquats, _ = read_eff(
                local_start - margin, n + 2 * margin, self.globalfirst,
                self.local_samples[0], self.ringdb, self.ringdb_path,
                self.freq, self.effdir_pntg, 'attitude', self.satquatmask,
                self.eff_cache, self.cache, self.filenames[self.effdir_pntg])
            satquats = satquats.T.copy()

        # Rotate into detector frame and convert to desired format

        quats = qa.mult(qa.norm(satquats), detquat)

        if deaberrate:
            # Correct for aberration
            if velocity is None:
                velocity = self._get_velocity(local_start, n, margin=margin)

        # Manipulate the quaternions in buffers not to allocate excessive
        # Python memory
        buflen = 10000
        for istart in range(0, len(quats), buflen):
            istop = min(istart + buflen, len(quats))
            ind = slice(istart, istop)
            if deaberrate:
                vec = qa.rotate(quats[ind], ZAXIS)
                abvec = np.cross(vec, velocity[ind])
                lens = np.linalg.norm(abvec, axis=1)
                ang = lens * CINV
                abvec /= np.tile(lens, (3, 1)).T  # Normalize for direction
                abquat = qa.rotation(abvec, -ang)
                quats[ind] = qa.mult(abquat, quats[ind])
            if self.coordquat is not None:
                quats[ind] = qa.mult(self.coordquat, quats[ind])

        if full_output:
            return quats, satquats
        else:
            return quats

    def _put_pntg(self, _, local_start, data, effdir_out=None):

        if self.sim:
            raise RuntimeError('Cannot write pointing when doing simulations')

        if effdir_out is None:
            effdir_out = self.effdir_out

        write_eff(
            local_start, data, None, self.globalfirst, self.local_samples[0],
            self.ringdb, self.ringdb_path, self.freq, effdir_out, 'attitude',
            self.filenames[effdir_out])

        return

    def _get_common_flags(self, local_start, n, margin=0):

        if self.lfi_raw:
            obtextname = 'time'
        else:
            obtextname = 'obt'

        # read_eff caches everything so we can read here both the timestamps
        # and the common flags

        _, commonflags = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            self.freq, self.effdir_in, obtextname, self.obtmask,
            self.eff_cache, self.cache, self.filenames[self.effdir_in],
            file_pattern=self.effdir_in_pattern)

        self._add_ACMS_flags(commonflags, local_start, n, margin)

        return commonflags

    def _add_ACMS_flags(self, commonflag, local_start, n, margin=0):

        # Mask out samples that have unreliable pointing
        # (HCM, OCM or interpolated across interval ends)
        # Automatically flags margin's worth of samples outside
        # of the global requested data span
        #
        # We need to process one extra sample in both ends of the
        # interval because we want to extend the "none" flags

        is_hcm = np.zeros(len(commonflag) + 2, dtype=bool)
        is_scm = np.zeros(len(commonflag) + 2, dtype=bool)
        is_ocm = np.zeros(len(commonflag) + 2, dtype=bool)
        is_none = np.ones(len(commonflag) + 2, dtype=bool)

        first = self.local_samples[0] + local_start - margin - 1
        last = first + n + 2 * margin + 2
        ind = np.arange(first, last)
        for interval, name in zip(self.allmodes, self.modenames):
            mode_first = interval.first
            mode_last = interval.last
            if mode_first <= last and mode_last >= first:
                if 'H' in name:
                    is_hcm[np.logical_and(
                        ind >= mode_first, ind <= mode_last)] = True
                elif 'S' in name:
                    is_scm[np.logical_and(
                        ind >= mode_first, ind <= mode_last)] = True
                elif 'O' in name:
                    is_ocm[np.logical_and(
                        ind >= mode_first, ind <= mode_last)] = True

        is_none[is_hcm] = False
        is_none[is_scm] = False
        is_none[is_ocm] = False

        # Extend the "none" flags by one sample in both directions

        temp = is_none.copy()
        is_none[:-1] += temp[1:]
        is_none[1:] += temp[:-1]

        commonflag[is_ocm[1:-1]] |= np.uint8(255)
        commonflag[is_none[1:-1]] |= np.uint8(255)
        commonflag[is_hcm[1:-1]] |= np.uint8(self.pntflagmask)

        return

    def _put_common_flags(self, local_start, flags, effdir_out=None,
                          file_pattern=None):
        if effdir_out is None:
            effdir_out = self.effdir_out
            file_pattern = self.effdir_out_pattern

        write_eff(
            local_start, None, flags, self.globalfirst, self.local_samples[0],
            self.ringdb, self.ringdb_path, self.freq, effdir_out, 'obt',
            self.filenames[effdir_out], file_pattern=file_pattern)

        return

    def _get_times(self, local_start, n, margin=0, and_flags=False):

        if self.lfi_raw:
            obtextname = 'time'
        else:
            obtextname = 'obt'

        data, commonflags = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            self.freq, self.effdir_in, obtextname, self.obtmask,
            self.eff_cache, self.cache, self.filenames[self.effdir_in],
            file_pattern=self.effdir_in_pattern)

        if self.lfi_raw:
            # The second column contained SCET time stamps
            data = data[0].astype(np.float64)
        else:
            data = data.astype(np.float64)

        if np.amax(data) > 1e16:
            # OBT nanoseconds
            data *= 1e-9
        elif np.amax(data) > 1e10:
            # OBT ticks
            data *= 2 ** -16

        if and_flags:
            self._add_ACMS_flags(commonflags, local_start, n, margin)
            return data, commonflags
        else:
            return data

    def _put_times(self, local_start, stamps, effdir_out=None,
                   file_pattern=None):
        if effdir_out is None:
            effdir_out = self.effdir_out
            file_pattern = self.effdir_out_pattern

        write_eff(
            local_start, stamps, None, self.globalfirst, self.local_samples[0],
            self.ringdb, self.ringdb_path, self.freq, effdir_out, 'obt',
            self.filenames[effdir_out], file_pattern=file_pattern)

        return

    def local_phase(self, name=None, **kwargs):
        if name is None:
            cachename = self.PHASE_NAME
            if not self.cache.exists(cachename):
                phase = self.read_phase(**kwargs)
                self.cache.put(cachename, phase)
        else:
            cachename = name
        return self.cache.reference(cachename)

    def read_phase(self, local_start=0, n=0, margin=0):
        """ Read satellite spin phase

        """
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                'cannot read phases- process has no assigned local samples')
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(
                local_start, local_start + n - 1))

        phase, _ = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            self.freq, self.effdir_pntg, 'phase', 0,
            self.eff_cache, self.cache, self.filenames[self.effdir_pntg])
        phase = np.radians(phase.ravel())
        return phase

    def _get_position(self, local_start, n, margin=0):
        position, _ = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            self.freq, self.effdir_pntg, 'position', 0,
            self.eff_cache, self.cache, self.filenames[self.effdir_pntg])
        position = position.T.copy()
        return position

    def _get_velocity(self, local_start, n, margin=0, no_cache=False):
        velocity, _ = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            self.freq, self.effdir_pntg, 'velocity', 0,
            self.eff_cache, self.cache, self.filenames[self.effdir_pntg])
        velocity = velocity.T.copy()
        return velocity

    def local_dark(self, det, name=None, **kwargs):
        if det not in ['Dark-1', 'Dark-2']:
            raise RuntimeError(
                '{} is not a valid dark bolometer name'.format(det))
        if name is None:
            cachename = '{}_{}'.format(self.SIGNAL_NAME, det)
            if not self.cache.exists(cachename):
                dark1, dark2, flag1, flag2 = self.read_dark(**kwargs)
                self.cache.put('{}_{}'.format(self.SIGNAL_NAME, 'Dark-1'),
                               dark1)
                self.cache.put('{}_{}'.format(self.SIGNAL_NAME, 'Dark-2'),
                               dark2)
                self.cache.put('{}_{}'.format(self.FLAG_NAME, 'Dark-1'), flag1)
                self.cache.put('{}_{}'.format(self.FLAG_NAME, 'Dark-2'), flag2)
        else:
            cachename = '{}_{}'.format(name, det)
        return self.cache.reference(cachename)

    def read_dark(self, local_start=0, n=0, margin=0):
        """ Read dark bolometer TOD

        """
        if n == 0:
            n = self.local_samples[1] - local_start

        data1, flag1 = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            100, self.effdir_dark, 'dark-1', self.darkmask,
            self.eff_cache, self.cache, self.filenames[self.effdir_dark])

        data2, flag2 = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            100, self.effdir_dark, 'dark-2', self.darkmask,
            self.eff_cache, self.cache, self.filenames[self.effdir_dark])
        return data1, data2, flag1, flag2

    def local_fsl(self, det, name=None, **kwargs):
        if name is None:
            cachename = '{}_{}'.format(self.FSL_NAME, det)
            if not self.cache.exists(cachename):
                fsl = self.read_fsl(detector=det, **kwargs)
                self.cache.put(cachename, fsl)
        else:
            cachename = '{}_{}'.format(name, det)
        return self.cache.reference(cachename)

    def read_fsl(self, detector=None, local_start=0, n=0, margin=0):
        """ Read the far side lobe TOD

        """
        if self.effdir_fsl is None:
            raise Exception('read_fsl: effdir_fsl is not set.')
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                'cannot read- process has no assigned local samples')
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(
                local_start, local_start + n - 1))

        if detector[-1] in '01' and detector[-2] != '-':
            det = to_radiometer(detector)
        else:
            det = detector

        data, _ = read_eff(
            local_start - margin, n + 2 * margin, self.globalfirst,
            self.local_samples[0], self.ringdb, self.ringdb_path,
            self.freq, self.effdir_fsl, det.lower(), self.flagmask,
            self.eff_cache, self.cache, self.filenames[self.effdir_fsl])
        return data

    def read_fsl_primary(self, detector=None, local_start=0, n=0, margin=0):
        raise Exception('exchange TOD does not yet implement read_fsl_primary')

    def read_fsl_secondary(self, detector=None, local_start=0, n=0, margin=0):
        raise Exception(
            'exchange TOD does not yet implement read_fsl_secondary')

    def read_fsl_baffle(self, detector=None, local_start=0, n=0, margin=0):
        raise Exception('exchange TOD does not yet implement read_fsl_baffle')

    def write_dark(self, local_start=0, dark1=None, darkflag1=None, dark2=None,
                   darkflag2=None, effdir_out=None):
        if effdir_out is None:
            effdir_out = self.effdir_out

        if dark1 is not None or darkflag1 is not None:
            write_eff(
                local_start, dark1, darkflag1, self.globalfirst,
                self.local_samples[0], self.ringdb, self.ringdb_path, 100,
                effdir_out, 'dark-1', self.filenames[effdir_out])

        if dark2 is not None or darkflag2 is not None:
            write_eff(
                local_start, dark2, darkflag2, self.globalfirst,
                self.local_samples[0], self.ringdb, self.ringdb_path, 100,
                effdir_out, 'dark-2', self.filenames[effdir_out])
        return

    def write_tod_and_flags(
            self, detector=None, local_start=0, data=None, flags=None,
            effdir_out=None):
        """ Write detector data and flags.

        Args:
            detector (str): the name of the detector.
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            data (array): the data array.
            flags (array): the flag array.
        """
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if data is None:
            raise ValueError('data array must be specified')
        if flags is None:
            raise ValueError('flags array must be specified')
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                'cannot write- process has no assigned local samples')
        if (local_start < 0) \
           or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError(
                'local sample range {} - {} is invalid'.format(
                    local_start, local_start + data.shape[0] - 1))
        self._put_tod_and_flags(
            detector, local_start, data, flags, effdir_out=effdir_out)
        return
