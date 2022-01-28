# Copyright (c) 2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from toast import qarray as qa
import toast
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.utils import Logger, Environment

from toast.mpi import MPI

import healpy as hp
from libsharp import packed_real_order, synthesis
import numpy as np
import toast.timing as timing

from .preproc_modules import MapSampler, Dipoler
from .reproc_modules.rings import DistRings

cached_skymodels = {}


CMBCACHE = "./cmb_cache"


class OpSignalSim(toast.Operator):
    """ Operator to scan a CMB sky onto a TOD.

    Args:
        almfile(str): Path to precomputed a_lm expansion FITS file
        fwhm(float): Full Width Half maximum beam to smooth the a_alm
            with [arc minutes]
        freq(float): Observing frequency [GHz] for the foregrounds
        comm(MPIComm): MPI communicator to use (for node memory)
        freqs(dict): Detector central frequencies for foregrounds
        pol(bool): Treat the a_lm as polarized
        nside(int): Resolution of the intermediate map that will be
            interpolated.
        margin(int): Extra margin to simulate outside the local
            sample range
        out(str): Cache object to use for returning the simulated
            signal
        add(bool): Add or replace an existing cache object.
        dipole(bool): Add the solar system and orbital dipole to the
            simulated signal.
        rimo(dict): Planck RIMO dictionary for 4pi dipole calculation
        fsl(bool): Add the CMB+foreground FSL signal from the TOD object
        foreground(bool): Add the foreground signal from SkyModel
        mapdir(str): Directory for saving the synthesized maps.
        skip_reproc(bool): No reprocessing.
        scale_skymodel(float): Scale the skymodel map

    """

    mapsampler = None
    skymodel = None
    skymodel_deriv = None

    def __init__(
        self,
        almfile,
        fwhm,
        freq,
        comm,
        freqs=None,
        pol=False,
        nside=2048,
        margin=0,
        out=None,
        add=False,
        dipole=True,
        rimo=None,
        fsl=True,
        foreground=False,
        mapdir=None,
        groupnodes=8,
        skip_reproc=False,
        scale_skymodel=None,
        quickpolbeam=None,
        skymodelfile=None,
        skymodelderivfile=None,
        mc=None,
        bpm_extra=None,
    ):
        # We have to split the communicator because there is not enough
        # work for every process.
        nodecomm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        ntask_node = nodecomm.size
        groupsize = groupnodes * ntask_node
        del nodecomm
        self._comm = comm.Split(color=(comm.rank // groupsize), key=comm.rank)
        self._global_rank = comm.rank
        self._rank = self._comm.rank
        self._ntask = self._comm.size

        self.skip_reproc = skip_reproc
        self.scale_skymodel = scale_skymodel
        self._mapdir = mapdir
        if self._global_rank == 0 and mapdir is not None:
            os.makedirs(mapdir, exist_ok=True)
        self._almfile = almfile
        self._fwhm_in = fwhm
        self._fwhm = np.radians(self._fwhm_in / 60)
        self._quickpolbeam = quickpolbeam
        self._fn_skymodel = skymodelfile
        self._fn_skymodel_deriv = skymodelderivfile
        # Determine sufficient lmax
        self._nside = nside
        self._npix = 12 * self._nside ** 2
        if self._quickpolbeam is None:
            beam = hp.gauss_beam(self._fwhm, 3 * self._nside)
            self._lmax = np.argmin(np.abs(beam - 1e-4)) + 1
        else:
            self._lmax = 2 * self._nside
        self._freq = freq
        self._freqs = freqs
        self._pol = pol
        self._nnz = 1
        self._mc = mc
        self._bpm_extra = bpm_extra
        if self._pol:
            self._nnz = 3
        self._margin = margin
        if self._almfile is not None:
            # Synthesize or load a CMB map into a node-shared mapsampler
            if self._quickpolbeam is None:
                fn_cmb = os.path.join(
                    CMBCACHE,
                    "{}_nside{:04}_fwhm{:.1f}.fits".format(
                        os.path.basename(self._almfile).replace(".fits", ""),
                        self._nside,
                        self._fwhm_in,
                    ),
                )
            else:
                fn_cmb = os.path.join(
                    CMBCACHE,
                    "{}_nside{:04}_quickpol.fits".format(
                        os.path.basename(self._almfile).replace(".fits", ""),
                        self._nside,
                    ),
                )
            if comm.rank == 0:
                there = os.path.isfile(fn_cmb)
            else:
                there = None
            there = comm.bcast(there)
            if not there:
                # Set up a libsharp processing grid
                self._dist_rings = DistRings(
                    self._comm, nside=self._nside, nnz=self._nnz
                )
                self._grid = self._dist_rings.libsharp_grid
                self._load_alm()
                self._distribute_alm()
            self._synthesize_map(fn_cmb, there)
        else:
            # CMB was already simulated with Conviqt.
            # Store an empty CMB map in node memory to add with the foregrounds
            maptemp = np.zeros([self._nnz, self._npix], dtype=np.float32)
            self.mapsampler = MapSampler(
                None,
                pol=self._pol,
                comm=self._comm,
                preloaded_map=maptemp,
                nest=True,
                plug_holes=False,
                use_shmem=True,
            )
            del maptemp
        self._out = out
        self._add = add
        self._rimo = rimo
        if dipole:
            # Create the dipoler object without reference frequency.
            # The CMB realizations already include the frequency-dependent
            # part of the quadrupole
            if self._rimo is None:
                self._dipoler = Dipoler()
            else:
                # We use the pencil beam version of the dipole and adjust
                # reproc to match it.  The FSL template already has the
                # FSL-convolved part of the solar system dipole and the
                # effect on orbital dipole is negligible.
                self._dipoler = Dipoler(full4pi=False, comm=self._comm, RIMO=self.rimo)
        else:
            self._dipoler = None
        self._fsl = fsl
        if foreground:
            self._load_skymodel()
        else:
            self._skymodel = None
        return

    def __delete__(self):
        self.comm.Free()
        return

    def _load_skymodel(self):
        """ Load a pre-computed sky model (and derivative) map
        and store as a mapsampler object.

        """
        if self._fn_skymodel is None:
            raise RuntimeError("Cannot load sky model, no file specified.")
        if self._fn_skymodel_deriv is None:
            raise RuntimeError("Cannot load sky model deriv, no file specified.")

        if self._fn_skymodel not in cached_skymodels:
            # Sky model at freq
            if not os.path.isfile(self._fn_skymodel):
                raise RuntimeError(
                    'Invalid sky model file: "{}"'.format(self._fn_skymodel)
                )
            if self._global_rank == 0:
                print(
                    "Loading cached sky model from {}".format(self._fn_skymodel),
                    flush=True,
                )
            cached_skymodels[self._fn_skymodel] = MapSampler(
                self._fn_skymodel,
                pol=self._pol,
                comm=self._comm,
                nest=True,
                use_shmem=True,
            )
        self.skymodel = cached_skymodels[self._fn_skymodel]

        if self._fn_skymodel_deriv not in cached_skymodels:
            # Sky model derivative at freq
            if not os.path.isfile(self._fn_skymodel_deriv):
                raise RuntimeError(
                    'Invalid sky model deriv file: "{}"'.format(self._fn_skymodel_deriv)
                )
            if self._global_rank == 0:
                print(
                    "Loading cached sky model deriv from {}".format(
                        self._fn_skymodel_deriv
                    ),
                    flush=True,
                )
            cached_skymodels[self._fn_skymodel_deriv] = MapSampler(
                self._fn_skymodel_deriv,
                pol=self._pol,
                comm=self._comm,
                nest=True,
                use_shmem=True,
            )
        self.skymodel_deriv = cached_skymodels[self._fn_skymodel_deriv]

        # if self._global_rank == 0 and self._mapdir is not None:
        if (
            self._almfile is not None
            and self._mapdir is not None
            and self._global_rank == 0
        ):
            # Save a combined map for comparing to the simulation results
            maptemp = self.mapsampler.Map[:] + self.skymodel.Map[:]
            if self._pol:
                qmap = self.mapsampler.Map_Q[:] + self.skymodel.Map_Q[:]
                umap = self.mapsampler.Map_U[:] + self.skymodel.Map_U[:]
                maptemp = [maptemp, qmap, umap]
            fname = os.path.join(
                self._mapdir, "fg_and_map_from_" + os.path.basename(self._almfile)
            )
            header = [("fwhm", np.degrees(self._fwhm), "gaussian smoothing (deg)")]
            hp.write_map(fname, maptemp, extra_header=header, overwrite=True, nest=True)
            del maptemp
            print("Total map saved in {}".format(fname))
        return

    def _synthesize_map(self, fn_cmb, there):
        """ Synthesize the stored alm expansion into a map
        and place the map in node-shared memory.

        """
        timer = Timer()
        timer.start()
        if not there:
            # Use libsharp to perform the synthesis across the communicator
            if self._quickpolbeam is None:
                beam = hp.gauss_beam(fwhm=self._fwhm, lmax=self._lmax, pol=True)
                beam = beam[:, 0:3].copy()
            else:
                beam = np.array(hp.read_cl(self._quickpolbeam))
                if beam.ndim == 1:
                    beam = np.vstack([beam, beam, beam])
                beam = beam[:, : self._lmax + 1].T.copy()
            almT = self._alm[0].reshape(1, 1, -1)
            self._alminfo.almxfl(almT, np.ascontiguousarray(beam[:, 0:1]))
            my_outmap = synthesis(
                self._grid, self._alminfo, almT, spin=0, comm=self._comm
            )[0]
            my_outmap = [my_outmap]
            if self._pol:
                almP = self._alm[1:3].reshape(1, 2, -1)
                self._alminfo.almxfl(almP, np.ascontiguousarray(beam[:, (1, 2)]))
                my_outmap.append(
                    synthesis(self._grid, self._alminfo, almP, spin=2, comm=self._comm)[
                        0
                    ]
                )
            # Discard the a_lm
            del self._alm
            my_outmap = np.vstack(my_outmap)
            my_pixels = self._dist_rings.local_pixels
            my_maptemp = np.zeros([self._nnz, self._npix], dtype=float)
            maptemp = np.zeros([self._nnz, self._npix], dtype=float)
            my_maptemp[:, my_pixels] = my_outmap
            self._comm.Reduce(my_maptemp, maptemp)
            del my_maptemp
            maptemp = hp.reorder(maptemp, r2n=True)
            timer.stop()
            if self._global_rank == 0:
                timer.report("synthesize CMB map")
                # Save the CMB map
                os.makedirs(CMBCACHE, exist_ok=True)
                header = [("fwhm", np.degrees(self._fwhm), "gaussian smoothing (deg)")]
                hp.write_map(
                    fn_cmb, maptemp, extra_header=header, overwrite=True, nest=True
                )
                print("CMB map saved in {}".format(fn_cmb), flush=True)
        else:
            if self._global_rank == 0:
                print("Loading cached CMB map from {}".format(fn_cmb), flush=True)
            if self._rank == 0:
                maptemp = hp.read_map(
                    fn_cmb, None, nest=True, verbose=False, dtype=np.float32
                )
                if not self._pol:
                    maptemp = maptemp[0]
            else:
                maptemp = None
        self.mapsampler = MapSampler(
            None,
            pol=self._pol,
            comm=self._comm,
            preloaded_map=maptemp,
            nest=True,
            plug_holes=False,
            use_shmem=True,
        )
        del maptemp
        return

    def _load_alm(self):
        """ Load the alm expansion and place it in the node-shared memory

        """
        if self._rank == 0:
            timer = Timer()
            timer.start()
            alm, mmax = hp.read_alm(self._almfile, return_mmax=True)
            nalm = len(alm)
            lmax = hp.Alm.getlmax(nalm, mmax)
            alm = [alm]
            if self._pol:
                for hdu in [2, 3]:
                    alm.append(hp.read_alm(self._almfile, hdu=hdu))
            alm = np.vstack(alm)
            nalm = len(alm)
            # If necessary, truncate the expansion to sufficient lmax
            self._lmax = min(lmax, self._lmax)
            self._mmax = min(mmax, self._lmax)
            if self._lmax < lmax:
                sz = hp.Alm.getsize(self._lmax, self._mmax)
                new_alm = np.zeros([nalm, sz], dtype=np.complex)
                for ell in range(self._lmax + 1):
                    for m in range(min(ell, self._mmax)):
                        i = hp.Alm.getidx(self._lmax, ell, m)
                        j = hp.Alm.getidx(lmax, ell, m)
                        new_alm[:, i] = alm[:, j]
                alm = new_alm
                lmax = self._lmax
                mmax = self._mmax
            # Suppress any primordial monopole or dipole
            for ell in range(min(2, lmax + 1)):
                for m in range(min(ell + 1, mmax + 1)):
                    ind = hp.Alm.getidx(lmax, 1, m)
                    alm[0, ind] = 0
            timer.stop()
            timer.report("load CMB alm")
        else:
            alm, lmax, mmax = None, None, None
        self._alm = self._comm.bcast(alm)
        self._lmax = self._comm.bcast(lmax)
        self._mmax = self._comm.bcast(mmax)
        return

    def _distribute_alm(self):
        """ Distribute the a_lm across the communicator.  This
        includes translating the complex a_lm in to real
        coefficients

        """
        self._local_m_indices = np.arange(
            self._rank, self._mmax + 1, self._ntask, dtype=np.int32
        )
        self._alminfo = packed_real_order(self._lmax, ms=self._local_m_indices)
        my_nalm = 0
        for m in self._local_m_indices:
            # All but the m=0 mode create two entries in the
            # real a_lm array
            if m == 0:
                my_nalm += self._lmax + 1 - m
            else:
                my_nalm += 2 * (self._lmax + 1 - m)
        my_alm = np.zeros([self._nnz, my_nalm])
        sqrt2 = np.sqrt(2)
        for comp in range(self._nnz):
            i = 0
            ii = 0
            for m in range(self._mmax + 1):
                if m % self._ntask != self._rank:
                    # not a local m-mode
                    i += self._lmax + 1 - m
                    continue
                for _ in range(self._lmax + 1 - m):
                    if m == 0:
                        my_alm[comp, ii] = self._alm[comp, i].real
                    else:
                        my_alm[comp, ii] = self._alm[comp, i].real * sqrt2
                        ii += 1
                        my_alm[comp, ii] = self._alm[comp, i].imag * sqrt2
                    ii += 1
                    i += 1
        self._alm = my_alm
        return

    def _check_len(self, vec, nsamp, name):
        """ Ensure the vector includes the margins

        """
        if len(vec) != nsamp + 2 * self._margin:
            raise RuntimeError(
                "Cached {} do not match margin={}" "".format(name, self._margin)
            )
        return

    def _sample_maps(self, tod, det, quat, weights=None):
        """ Perform bilinear interpolation of the stored map.

        """
        thetaname = "theta_{}".format(det)
        phiname = "phi_{}".format(det)
        if tod.cache.exists(thetaname):
            theta = tod.cache.reference(thetaname)
            phi = tod.cache.reference(phiname)
        else:
            theta, phi = qa.to_position(quat)
            theta = tod.cache.put(thetaname, theta.astype(np.float32, copy=False))
            phi = tod.cache.put(phiname, phi.astype(np.float32, copy=False))
        if self.scale_skymodel:
            self.skymodel *= self.scale_skymodel
            self.skymodel_deriv *= self.scale_skymodel
        # Temporarily co-add the CMB and the foregrounds
        self.mapsampler += self.skymodel
        # bandpass mismatch
        try:
            freq = self._freqs[det]
        except Exception:
            freq = self._freq
        delta = freq - self._freq
        if delta != 0:
            self.skymodel_deriv *= delta
            self.mapsampler += self.skymodel_deriv
        if self._bpm_extra:
            fn_bpm = self._bpm_extra.replace("DETECTOR", det)
            if self._global_rank == 0:
                print("  Adding bandpass mismatch from {}".format(fn_bpm), flush=True)
            bpm = MapSampler(
                fn_bpm,
                pol=False,
                comm=self._comm,
                nest=True,
                nside=self._nside,
                plug_holes=False,
                # Work around a bug in the default cray-mpich library that does
                # not allow releasing MPI shared memory.
                use_shmem=False,
            )
            self.mapsampler += bpm
        # Sample the aggregate map
        if self._global_rank == 0:
            print("  Sampling signal map", flush=True)
        signal = self.mapsampler.atpol(theta, phi, weights)
        # restore original CMB
        if self._bpm_extra:
            self.mapsampler -= bpm
            del bpm
        self.mapsampler -= self.skymodel
        if delta != 0:
            self.mapsampler -= self.skymodel_deriv
            self.skymodel_deriv /= delta
        if self.scale_skymodel:
            self.skymodel /= self.scale_skymodel
            self.skymodel_deriv /= self.scale_skymodel
        return signal

    def exec(self, data):
        """ Apply the OpSignalSim operator on data

        """
        for obs in data.obs:
            tod = obs["tod"]
            nsamp = tod.local_samples[1]
            for det in tod.local_dets:
                timer = Timer()
                timer.start()
                if self._global_rank == 0:
                    print("Processing {}".format(det), flush=True)

                quat = tod.local_pointing(det, margin=self._margin)
                self._check_len(quat, nsamp, "detector quaternions")
                iquweights = tod.local_weights(det)
                self._check_len(iquweights, nsamp, "detector weights")

                sampled = self._sample_maps(tod, det, quat, iquweights)

                if self._dipoler is not None:
                    if self._global_rank == 0:
                        print("  Adding dipole", flush=True)
                    if self.skip_reproc:
                        velocity = None
                    else:
                        velocity = tod.local_velocity(margin=self._margin)
                        self._check_len(velocity, nsamp, "velocity")
                    sampled += self._dipoler.dipole(quat, velocity=velocity, det=det)

                if self._fsl and not self.skip_reproc:
                    if self._global_rank == 0:
                        print("  Adding FSL", flush=True)
                    local_fsl = tod.local_fsl(det, margin=self._margin)
                    self._check_len(local_fsl, nsamp, "FSL")
                    sampled += local_fsl

                if self._out is not None:
                    cachename = "{}_{}".format(self._out, det)
                    if not tod.cache.exists(cachename):
                        tod.cache.create((nsamp + 2 * self._margin,), dtype=np.float64)
                local_signal = tod.local_signal(
                    det, name=self._out, margin=self._margin
                )
                self._check_len(iquweights, nsamp, "signal")
                if self._add:
                    local_signal += sampled
                else:
                    local_signal[:] = sampled
                del local_signal
                timer.stop()
                if self._global_rank == 0:
                    timer.report("Process {}".format(det))
        return
