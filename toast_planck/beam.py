# Copyright (c) 2016-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

import time
from toast_planck.beam_modules import Beam_Reconstructor
from toast_planck.utilities import bolo_to_pnt

from toast import qarray as qa
import toast
from toast_planck.imo import IMO
from toast_planck.preproc_modules import (
    Pnt2Planeter, MapSampler, Dipoler, RingMasker)

import numpy as np
import toast.timing as timing


# import warnings
# warnings.filterwarnings('error')
class OpBeamReconstructor(toast.Operator):
    """
    Operator for building beams.

    Args:
        input (str):  if None, read input TOD, otherwise the name of
            the cached data.
        output (str):  if None, write TOD, otherwise the name to use
            in the cache.
    """

    def __init__(self, imofile, freq, effdir_out=None, bg_map_path=None,
                 bg_pol=False, bg_has_dipole=False, bg_nside=1024, detmask=1,
                 commonmask=1, pntmask=2, ssomask=2, maskfile=None,
                 nbin_phase=3000, margin=0, bad_rings=None, out='.',
                 verbose=True, targets='JUPITER,SATURN,MARS,URANUS,NEPTUNE',
                 bsiter=3, beamtofits=None, beamtofits_polar=None,
                 savebeamobj=False, datarad=100, dstrrad=80, dstrtol=0.9,
                 radrms=100, xtfthr=4, jupthr=[2, 3, 5, 10, 100, 100],
                 optical=False, pixsize=10,
                 knotstep=[2, 1.5, 1.25, 1.25, 1.25, 0.8], sqmaphpix=240,
                 knotextframe=80, bsorder=4, hrmax=100,
                 nslices=[16, 32, 32, 32, 64, 64], hybthd=[9, 9, 7, 7, 5, 5],
                 ntf=[1, 1, 3, 3, 3, 3], knotextframex=24, knotextframey=120,
                 rectmaphpixx=72, rectmaphpixy=360, trans=2):
        # if bg_pol == True and input_weights is None:
        #    raise Exception('OpBeamReconstuctor: Cannot interpolate polarized '
        #                    'map (bg_pol==True) without IQU weights '
        #                    '(input_weights==None).' )
        self._imo = IMO(imofile)
        self._freq = int(freq)
        self._lfi_mode = freq < 100
        self._bg_map_path = bg_map_path
        self._bg_pol = bg_pol
        self._bg_has_dipole = bg_has_dipole
        self._bg_nside = bg_nside
        self._detmask = detmask  # Default value (1) assumes preprocessing
        self._commonmask = commonmask  # Default value (1) assumes preprocessing
        self._pntmask = pntmask  # Default value (2) assumes preprocessing
        self._ssomask = ssomask  # Default value (2) assumes preprocessing
        self._maskfile = maskfile
        self._nbin_phase = nbin_phase
        self._margin = margin
        self._effdir_out = effdir_out
        self._bad_rings = bad_rings
        self._out = out
        self._verbose = verbose
        if targets is None:
            raise RuntimeError('BeamReconstructor requires a list of planets '
                               'to use as targets.')
        else:
            self._targets = []
            for target in targets.split(','):
                self._targets.append(target.upper())
        self._bsiter = bsiter
        self._beamtofits = beamtofits
        self._beamtofits_polar = beamtofits_polar
        self._savebeamobj = savebeamobj
        self._datarad = datarad
        self._dstrrad = dstrrad
        self._dstrtol = dstrtol
        self._radrms = radrms
        self._xtfthr = xtfthr
        self._jupthr = jupthr
        self._optical = optical
        self._bsorder = bsorder
        self._hrmax = hrmax
        self._nslices = nslices
        self._hybthd = hybthd
        self._ntf = ntf
        self._knotstep = knotstep
        self._knotextframe = knotextframe
        self._knotextframex = knotextframex
        self._knotextframey = knotextframey
        self._pixsize = pixsize
        self._sqmaphpix = sqmaphpix
        self._rectmaphpixx = rectmaphpixx
        self._rectmaphpixy = rectmaphpixy
        self._trans = trans
        super().__init__()

    # @profile
    def exec(self, data):
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world

        rank = cworld.Get_rank()

        margin = self._margin

        pnt2planets = []
        for target in self._targets:
            pnt2planets.append(Pnt2Planeter(target))

        for obs in data.obs:

            tod = obs['tod']
            nsamp = tod.local_samples[1]

            if self._effdir_out is not None:
                tod.cache_metadata(self._effdir_out, comm=cworld)

            timestamps = tod.local_times(margin=margin)
            if len(timestamps) != nsamp + 2 * margin:
                raise Exception('Cached time stamps do not include margins.')

            commonflags = tod.local_common_flags(margin=margin)
            if len(commonflags) != nsamp + 2 * margin:
                raise Exception('Cached common flags do not include margins.')

            phase = tod.local_phase(margin=margin)
            if len(phase) != nsamp + 2 * margin:
                raise Exception('Cached phases do not include margins.')

            velocity = tod.local_velocity(margin=margin)
            if len(velocity) != nsamp + 2 * margin:
                raise Exception('Cached velocities do not include margins.')

            if self._pntmask is not None:
                pntflag = (commonflags & self._pntmask) != 0
            else:
                pntflag = None

            intervals = tod.local_intervals(obs['intervals'])
            starts = [ival.start for ival in intervals]
            stops = [ival.stop + 1 for ival in intervals]
            local_starts = np.array(starts)
            local_stops = np.array(stops)

            ring_offset = tod.globalfirst_ring
            for interval in intervals:
                if interval.last < tod.local_samples[0]:
                    ring_offset += 1

            if self._lfi_mode:
                dipoler = Dipoler(full4pi=True, comm=cworld, RIMO=tod.RIMO)
            else:
                dipoler = Dipoler(freq=int(self._freq))

            if self._bad_rings is not None:
                ringmasker = RingMasker(self._bad_rings)
            else:
                ringmasker = None

            if self._bg_map_path is not None \
               and 'DETECTOR' not in self._bg_map_path:
                # All detectors share the same template map
                mapsampler = MapSampler(
                    self._bg_map_path, pol=self._bg_pol, nside=self._bg_nside,
                    comm=cworld, cache=tod.cache)
            else:
                mapsampler = None

            if self._maskfile:
                masksampler = MapSampler(
                    self._maskfile, comm=cworld, dtype=np.byte,
                    cache=tod.cache)
                maskflag = np.zeros(nsamp + 2 * margin, dtype=np.bool)
            else:
                masksampler = None
                maskflag = None

            # Now the optical channels

            for det in tod.local_dets:

                if rank == 0:
                    print('Setting up processing for {}'.format(det),
                          flush=True)

                if self._lfi_mode:
                    bolo_id = None
                else:
                    bolo_id = bolo_to_pnt(det)

                psi_pol = np.radians(
                    (tod.RIMO[det].psi_uv + tod.RIMO[det].psi_pol))
                beampar = {}
                beampar['savebeamobj'] = self._savebeamobj
                beampar['bsverbose'] = self._verbose
                beampar['savepath'] = self._out
                beampar['savedir'] = 'beams'
                beampar['prefix'] = ''
                beampar['boloID'] = bolo_id
                beampar['bsdebug'] = False
                beampar['datarad'] = self._datarad
                beampar['dstrrad'] = self._dstrrad
                beampar['dstrtol'] = self._dstrtol
                beampar['jupthr'] = self._jupthr
                beampar['radrms'] = self._radrms
                beampar['xtfthr'] = self._xtfthr
                beampar['nslices'] = self._nslices
                beampar['ntf'] = self._ntf
                beampar['hybthd'] = self._hybthd
                beampar['xtfthr'] = self._xtfthr
                beampar['trans'] = self._trans
                beampar['hrmax'] = self._hrmax
                beampar['optical'] = self._optical
                beampar['bsorder'] = self._bsorder
                beampar['pixsize'] = self._pixsize
                beampar['sqmaphpix'] = self._sqmaphpix
                beampar['rectmaphpixx'] = self._rectmaphpixx
                beampar['rectmaphpixy'] = self._rectmaphpixy
                beampar['knotstep'] = self._knotstep
                beampar['knotextframe'] = self._knotextframe
                beampar['knotextframex'] = self._knotextframex
                beampar['knotextframey'] = self._knotextframey

                beamobj = Beam_Reconstructor(bolo_id, beampar, mpicomm=cworld)

                if self._bg_map_path is not None \
                   and 'DETECTOR' in self._bg_map_path:
                    # Detectors have separete template maps
                    mapsampler = MapSampler(
                        self._bg_map_path.replace('DETECTOR', det),
                        pol=self._bg_pol, nside=self._bg_nside,
                        comm=cworld, cache=tod.cache)

                # Read all of the data for this process and process
                # interval by interval

                signal = tod.local_signal(det, margin=margin)
                if len(signal) != nsamp + 2 * margin:
                    raise Exception('Cached signal does not include margins.')

                flags = tod.local_flags(det, margin=margin)
                if len(flags) != nsamp + 2 * margin:
                    raise Exception('Cached flags do not include margins.')

                quat = tod.local_pointing(det, margin=margin)
                if len(quat) != nsamp + 2 * margin:
                    raise Exception('Cached quats do not include margins.')

                iquweights = tod.local_weights(det)
                if len(iquweights) != nsamp + 2 * margin:
                    raise Exception('Cached weights do not include margins.')

                # Cast the flags into boolean vectors

                if self._detmask:
                    detflag = (flags & self._detmask) != 0
                else:
                    detflag = flags != 0

                if self._commonmask is not None:
                    detflag[(commonflags & self._commonmask) != 0] = True

                if self._ssomask is not None:
                    ssoflag = (flags & self._ssomask) != 0
                else:
                    ssoflag = np.zeros(nsamp, dtype=np.bool)

                # Add ring flags

                if ringmasker is not None:
                    ringflag = ringmasker.get_mask(timestamps, det)
                    detflag[ringflag] = True
                else:
                    ringflag = None

                # Process

                if rank == 0:
                    print('Processing {}'.format(det), flush=True)

                signal_out = np.zeros(nsamp + 2 * margin, dtype=np.float64)
                flags_out = np.zeros(nsamp + 2 * margin, dtype=np.uint8)

                ring_number = ring_offset - 1

                for ring_start, ring_stop in zip(local_starts, local_stops):

                    ring_number += 1

                    if self._verbose:
                        print('{:4} : Processing ring {:4}'.format(
                            rank, ring_number), flush=True)

                    # Slice without margins
                    ind = slice(ring_start + margin, ring_stop + margin)

                    tme = timestamps[ind]
                    sig = signal[ind].copy()
                    flg = detflag[ind].copy()

                    # Require at least 10% of the signal to be unflagged to
                    # even attempt processing.

                    if np.sum(flg == 0) < 0.1 * len(flg):
                        raise Exception('Too many samples are flagged.')

                    if pntflag is not None:
                        pntflg = pntflag[ind].copy()
                    else:
                        pntflg = np.zeros_like(flg)
                    if margin > 0:
                        pntflg[:margin] = True
                        pntflg[-margin:] = True

                    if np.sum(pntflg == 0) < 10000:
                        raise Exception('Pointing is unstable')

                    q = quat[ind]
                    if self._bg_pol:
                        iquw = iquweights[ind]
                    v = velocity[ind]

                    # Get the dipole

                    dipo = dipoler.dipole(q, velocity=v, det=det)
                    if self._bg_has_dipole:
                        dipo -= dipoler.dipole(q, det=det)

                    # Convert pointing to angles

                    theta, phi = qa.to_position(q)

                    # Sample the (polarized) background map

                    if mapsampler is not None:
                        if self._bg_pol:
                            bg = mapsampler.atpol(theta, phi, iquw)
                        else:
                            bg = mapsampler.at(theta, phi)
                    else:
                        bg = None

                    # Sample the processing mask

                    if masksampler is not None:
                        maskflg = masksampler.at(theta, phi) < 0.5
                        maskflag[ind] = maskflg
                    else:
                        maskflg = None

                    # Cache planet data here

                    for pnt2planet in pnt2planets:
                        target = pnt2planet.target
                        az, el = pnt2planet.translate(q, tme, psi_pol=psi_pol)

                        beamobj.preproc(tme, sig - dipo - bg, flg + maskflg,
                                        az, el, target)

                # Collect planet data

                beamobj.cache = [beamobj.cache]
                cworld.Barrier()
                beamobj.cache = cworld.gather(beamobj.cache, root=0)

                # Build the beam and/or solve for the transfer function

                if rank == 0:
                    try:
                        beamobj.recache()
                        if len(beamobj.cache) == 0:
                            print('WARNING: No samples in cache for {}. No '
                                  'beam reconstructed.'.format(det))
                            continue
                        beamobj.scache = beamobj.cache
                        for iiter in range(int(self._bsiter)):
                            if iiter > 0:
                                print('Beam reconstruction, iteration # {}'
                                      ''.format(iiter + 1))
                                t_iter_start = time.time()
                                beamobj.cache = beamobj.scache
                            beamobj.mergedata()
                            beamobj.reconstruct(iiter)
                            t_iter_finish = time.time()
                            print('Beam Reconstruction iteration # {} '
                                  'completed in a total of {:.2f} s.'.format(
                                      iiter, t_iter_finish - t_iter_start))
                            if iiter < int(self._bsiter):
                                beamobj.update(iiter)
                        if self._beamtofits:
                            beamobj.hmapsave(self._beamtofits, polar=False)
                        if self._beamtofits_polar:
                            beamobj.hmapsave(self._beamtofits_polar, polar=True)
                    except Exception as e:
                        print('Beam reconsruction for {} failed: "{}"'
                              ''.format(det, e))

                # If necessary pass through the data again, applying the new
                # transfer function

                # If new transfer function was applied, cache and optionally
                # write out the new TOD

                # Write detector data

                signal_out[:] = signal[:]
                flags_out |= np.uint8(1) * detflag
                if ssoflag is not None:
                    flags_out |= np.uint8(2) * ssoflag
                if maskflag is not None:
                    flags_out |= np.uint8(4) * maskflag

                ind = slice(margin, len(signal_out) - margin)

                cachename = "{}_{}".format(tod.SIGNAL_NAME, det)
                tod.cache.put(cachename, signal_out[ind], replace=True)

                cachename = "{}_{}".format(tod.FLAG_NAME, det)
                tod.cache.put(cachename, flags_out[ind], replace=True)

                if self._effdir_out is not None:
                    if rank == 0:
                        print('Saving detector data to {}'.format(
                            self._effdir_out), flush=True)
                    tod.write_tod_and_flags(
                        detector=det, data=signal_out[ind],
                        flags=flags_out[ind],
                        effdir_out=self._effdir_out)

            commonflags_out = np.zeros_like(commonflags)
            commonflags_out |= np.uint8(1) * (
                (commonflags & (255 - self._pntmask)) != 0)
            if pntflag is not None:
                commonflags_out |= np.uint8(2) * pntflag

            if self._output_common_flags is not None:
                tod.cache.put(tod.COMMON_FLAG_NAME, commonflags_out[ind],
                              replace=True)

            if self._effdir_out is not None:
                tod.write_common_flags(flags=commonflags_out[ind],
                                       effdir_out=self._effdir_out)
