# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from toast_planck import utilities
from toast_planck.reproc_modules.destripe_tools import fast_scanning32
import warnings

from scipy.constants import arcmin
from toast.mpi import MPIShared
from toast.cache import Cache

import healpy as hp
import numpy as np
import toast.timing as timing

DTYPE = np.float32


class MapSampler:

    Map = None
    Map_Q = None
    Map_U = None

    def __init__(
        self,
        map_path,
        pol=False,
        pol_fwhm=None,
        no_temperature=False,
        dtype=None,
        plug_holes=True,
        verbose=False,
        nside=None,
        comm=None,
        cache=None,
        preloaded_map=None,
        buflen=1000000,
        nest=False,
        pscorrect=False,
        psradius=30,
        use_shmem=True,
    ):
        """
        Instantiate the map sampler object, load a healpix
        map in a file located at map_path

        if pol==True, reads I,Q,U maps from extensions 0, 1, 2
        """

        if not pol and no_temperature:
            raise RuntimeError("You cannot have pol=False, " "no_temperature=True")

        self.path = map_path
        self.pol = pol
        self.pol_fwhm = pol_fwhm
        self.nest = nest
        if nest:
            self.order = "NESTED"
        else:
            self.order = "RING"
        self.pscorrect = pscorrect
        self.psradius = psradius
        self.buflen = buflen
        # Output data type, internal is always DTYPE
        if dtype is not None:
            warnings.warn("MapSampler no longer supports dtype", DeprecationWarning)

        # Use healpy to load the map into memory.

        if comm is None:
            self.comm = None
            self.rank = 0
            self.ntask = 1
        else:
            self.comm = comm
            self.rank = comm.Get_rank()
            self.ntask = comm.Get_size()

        self.shmem = self.ntask > 1 and use_shmem
        self.pol = pol

        if self.rank == 0:
            if map_path is None and preloaded_map is None:
                raise RuntimeError("Either map_path or preloaded_map must be provided")
            if map_path is not None and preloaded_map is not None:
                if os.path.isfile(map_path):
                    raise RuntimeError(
                        "Both map_path and preloaded_map cannot be provided"
                    )
            if self.pol:
                if preloaded_map is not None:
                    if pscorrect or plug_holes or self.pol_fwhm is not None:
                        copy = True
                    else:
                        copy = False
                    if no_temperature:
                        self.Map_Q = preloaded_map[0].astype(DTYPE, copy=copy)
                        self.Map_U = preloaded_map[1].astype(DTYPE, copy=copy)
                    else:
                        self.Map = preloaded_map[0].astype(DTYPE, copy=copy)
                        self.Map_Q = preloaded_map[1].astype(DTYPE, copy=copy)
                        self.Map_U = preloaded_map[2].astype(DTYPE, copy=copy)
                else:
                    if no_temperature:
                        self.Map_Q, self.Map_U = hp.read_map(
                            self.path,
                            field=[1, 2],
                            dtype=DTYPE,
                            verbose=verbose,
                            memmmap=True,
                            nest=self.nest,
                        )
                    else:
                        try:
                            self.Map, self.Map_Q, self.Map_U = hp.read_map(
                                self.path,
                                field=[0, 1, 2],
                                dtype=DTYPE,
                                verbose=verbose,
                                memmap=True,
                                nest=self.nest,
                            )
                        except IndexError:
                            print(
                                "WARNING: {} is not polarized".format(self.path),
                                flush=True,
                            )
                            self.pol = False
                            self.Map = hp.read_map(
                                self.path,
                                dtype=DTYPE,
                                verbose=verbose,
                                memmap=True,
                                nest=self.nest,
                            )

                if nside is not None:
                    if not no_temperature:
                        self.Map = hp.ud_grade(
                            self.Map,
                            nside,
                            dtype=DTYPE,
                            order_in=self.order,
                            order_out=self.order,
                        )
                    if self.pol:
                        self.Map_Q = hp.ud_grade(
                            self.Map_Q,
                            nside,
                            dtype=DTYPE,
                            order_in=self.order,
                            order_out=self.order,
                        )
                        self.Map_U = hp.ud_grade(
                            self.Map_U,
                            nside,
                            dtype=DTYPE,
                            order_in=self.order,
                            order_out=self.order,
                        )

                if self.pscorrect:
                    if not no_temperature:
                        utilities.remove_bright_sources(
                            self.Map, nest=self.nest, fwhm=self.psradius
                        )
                    if self.pol:
                        utilities.remove_bright_sources(
                            self.Map_Q, nest=self.nest, fwhm=self.psradius
                        )
                        utilities.remove_bright_sources(
                            self.Map_U, nest=self.nest, fwhm=self.psradius
                        )
                elif plug_holes or self.pol_fwhm is not None:
                    if not no_temperature:
                        utilities.plug_holes(self.Map, verbose=verbose, nest=self.nest)
                    if self.pol:
                        utilities.plug_holes(
                            self.Map_Q, verbose=verbose, nest=self.nest
                        )
                        utilities.plug_holes(
                            self.Map_U, verbose=verbose, nest=self.nest
                        )
            else:
                if preloaded_map is not None:
                    self.Map = np.array(preloaded_map, dtype=DTYPE)
                else:
                    self.Map = hp.read_map(
                        map_path,
                        field=[0],
                        dtype=DTYPE,
                        verbose=verbose,
                        memmap=True,
                        nest=self.nest,
                    )
                if nside is not None:
                    self.Map = hp.ud_grade(
                        self.Map,
                        nside,
                        dtype=DTYPE,
                        order_in=self.order,
                        order_out=self.order,
                    )
                if self.pscorrect:
                    utilities.remove_bright_sources(
                        self.Map, nest=self.nest, fwhm=self.psradius
                    )
                elif plug_holes:
                    utilities.plug_holes(self.Map, verbose=verbose, nest=self.nest)

        if self.ntask > 1:
            self.pol = comm.bcast(self.pol, root=0)
            npix = 0
            if self.rank == 0:
                if self.pol:
                    npix = len(self.Map_Q)
                else:
                    npix = len(self.Map)
            npix = comm.bcast(npix, root=0)
            if self.shmem:
                shared = MPIShared((npix,), np.dtype(DTYPE), comm)
                if not no_temperature:
                    if self.rank == 0 and self.Map is None:
                        raise RuntimeError("Cannot set shared map from None")
                    shared.set(self.Map, (0,), fromrank=0)
                    self.Map = shared
                if self.pol:
                    if self.rank == 0 and self.Map_Q is None:
                        raise RuntimeError("Cannot set shared map from None")
                    shared_Q = MPIShared((npix,), np.dtype(DTYPE), comm)
                    shared_Q.set(self.Map_Q, (0,), fromrank=0)
                    self.Map_Q = shared_Q
                    shared_U = MPIShared((npix,), np.dtype(DTYPE), comm)
                    shared_U.set(self.Map_U, (0,), fromrank=0)
                    self.Map_U = shared_U
            else:
                if self.rank != 0:
                    if not no_temperature:
                        self.Map = np.zeros(npix, dtype=DTYPE)
                    if self.pol:
                        self.Map_Q = np.zeros(npix, dtype=DTYPE)
                        self.Map_U = np.zeros(npix, dtype=DTYPE)

                if not no_temperature:
                    comm.Bcast(self.Map, root=0)
                if self.pol:
                    comm.Bcast(self.Map_Q, root=0)
                    comm.Bcast(self.Map_U, root=0)

        if self.pol:
            self.npix = len(self.Map_Q[:])
        else:
            self.npix = len(self.Map[:])
        self.nside = hp.npix2nside(self.npix)

        if cache is None:
            self.cache = Cache()
        else:
            self.cache = cache
        self.instance = 0
        if not self.shmem:
            # Increase the instance counter until we find an unused
            # instance.  If the user did not want to store duplicates,
            # they would not have created two identical mapsampler
            # objects.
            while self.cache.exists(self._cachename("I")):
                self.instance += 1
            if not no_temperature:
                self.Map = self.cache.put(self._cachename("I"), self.Map)
            if self.pol:
                self.Map_Q = self.cache.put(self._cachename("Q"), self.Map_Q)
                self.Map_U = self.cache.put(self._cachename("U"), self.Map_U)

        if self.pol_fwhm is not None:
            self.smooth(self.pol_fwhm, pol_only=True)
        return

    def smooth(self, fwhm, lmax=None, pol_only=False):
        """ Smooth the map with a Gaussian kernel.
        """
        if self.rank == 0:
            if pol_only:
                print(
                    "Smoothing the polarization to {} arcmin".format(fwhm), flush=True
                )
            else:
                print("Smoothing the map to {} arcmin".format(fwhm), flush=True)

        if lmax is None:
            lmax = min(np.int(fwhm / 60 * 512), 2 * self.nside)

        # If the map is in node-shared memory, only the root process on each
        # node does the smoothing.
        if not self.shmem or self.Map.nodecomm.rank == 0:
            if self.pol:
                m = np.vstack([self.Map[:], self.Map_Q[:], self.Map_U[:]])
            else:
                m = self.Map[:]
            if self.nest:
                m = hp.reorder(m, n2r=True)
            smap = hp.smoothing(m, fwhm=fwhm * arcmin, lmax=lmax, verbose=False)
            del m
            if self.nest:
                smap = hp.reorder(smap, r2n=True)
        else:
            # Convenience dummy variable
            smap = np.zeros([3, 12])

        if not pol_only:
            if self.shmem:
                self.Map.set(smap[0].astype(DTYPE, copy=False), (0,), fromrank=0)
            else:
                self.Map[:] = smap[0]

        if self.pol:
            if self.shmem:
                self.Map_Q.set(smap[1].astype(DTYPE, copy=False), (0,), fromrank=0)
                self.Map_U.set(smap[2].astype(DTYPE, copy=False), (0,), fromrank=0)
            else:
                self.Map_Q[:] = smap[1]
                self.Map_U[:] = smap[2]

        self.pol_fwhm = fwhm
        return

    def _cachename(self, stokes):
        """
        Construct a cache name string for the selected Stokes map
        """
        return "{}_ns{:04}_{}_{:04}".format(
            self.path, self.nside, stokes, self.instance
        )

    def __del__(self):
        """
        Explicitly free memory taken up in the cache.
        """
        if not self.shmem:
            # Ensure the cache objects are destroyed after their references
            self.Map = None
            self.Map_Q = None
            self.Map_U = None
            self.cache.destroy(self._cachename("I"))
            if self.pol:
                self.cache.destroy(self._cachename("Q"))
                self.cache.destroy(self._cachename("U"))

    def __iadd__(self, other):
        """ Accumulate provided Mapsampler object with this one.
        """
        if self.shmem:
            # One process does the manipulation on each node
            self.Map._nodecomm.Barrier()
            if self.Map._noderank == 0:
                self.Map.data[:] += other.Map[:]
            if self.pol and other.pol:
                if self.Map_Q._noderank == (1 % self.Map_Q._nodeprocs):
                    self.Map_Q.data[:] += other.Map_Q[:]
                if self.Map_U._noderank == (2 % self.Map_U._nodeprocs):
                    self.Map_U.data[:] += other.Map_U[:]
            self.Map._nodecomm.Barrier()
        else:
            self.Map += other.Map
            if self.pol and other.pol:
                self.Map_Q += other.Map_Q
                self.Map_U += other.Map_U
        return self

    def __isub__(self, other):
        """ Subtract provided Mapsampler object from this one.
        """
        if self.shmem:
            # One process does the manipulation on each node
            self.Map._nodecomm.Barrier()
            if self.Map._noderank == 0:
                self.Map.data[:] -= other.Map[:]
            if self.pol and other.pol:
                if self.Map_Q._noderank == (1 % self.Map_Q._nodeprocs):
                    self.Map_Q.data[:] -= other.Map_Q[:]
                if self.Map_U._noderank == (2 % self.Map_U._nodeprocs):
                    self.Map_U.data[:] -= other.Map_U[:]
            self.Map._nodecomm.Barrier()
        else:
            self.Map -= other.Map
            if self.pol and other.pol:
                self.Map_Q -= other.Map_Q
                self.Map_U -= other.Map_U
        return self

    def __imul__(self, other):
        """ Scale the maps in this MapSampler object
        """
        if self.shmem:
            # One process does the manipulation on each node
            self.Map._nodecomm.Barrier()
            if self.Map._noderank == 0:
                self.Map.data[:] *= other
            if self.pol:
                if self.Map_Q._noderank == (1 % self.Map_Q._nodeprocs):
                    self.Map_Q.data[:] *= other
                if self.Map_U._noderank == (2 % self.Map_U._nodeprocs):
                    self.Map_U.data[:] *= other
            self.Map._nodecomm.Barrier()
        else:
            self.Map *= other
            if self.pol:
                self.Map_Q *= other
                self.Map_U *= other
        return self

    def __itruediv__(self, other):
        """ Divide the maps in this MapSampler object
        """
        if self.shmem:
            self.Map._nodecomm.Barrier()
            if self.Map._noderank == 0:
                self.Map.data[:] /= other
            if self.pol:
                if self.Map_Q._noderank == (1 % self.Map_Q._nodeprocs):
                    self.Map_Q.data[:] /= other
                if self.Map_U._noderank == (2 % self.Map_U._nodeprocs):
                    self.Map_U.data[:] /= other
            self.Map._nodecomm.Barrier()
        else:
            self.Map /= other
            if self.pol:
                self.Map_Q /= other
                self.Map_U /= other
        return self

    def at(self, theta, phi, interp_pix=None, interp_weights=None):
        """
        Use healpy bilinear interpolation to interpolate the
        map.  User must make sure that coordinate system used
        for theta and phi matches the map coordinate system.
        """

        if self.Map is None:
            raise RuntimeError("No temperature map to sample")

        n = len(theta)
        stepsize = self.buflen
        signal = np.zeros(n, dtype=np.float32)

        for istart in range(0, n, stepsize):
            istop = min(istart + stepsize, n)
            ind = slice(istart, istop)
            if interp_pix is None or interp_weights is None:
                p, w = hp.get_interp_weights(
                    self.nside, theta[ind], phi[ind], nest=self.nest
                )
            else:
                p = np.ascontiguousarray(interp_pix[:, ind])
                w = np.ascontiguousarray(interp_weights[:, ind])
            buffer = np.zeros(istop - istart, dtype=np.float64)
            fast_scanning32(buffer, p, w, self.Map[:])
            signal[ind] = buffer
        return signal

    def atpol(
        self,
        theta,
        phi,
        IQUweight,
        onlypol=False,
        interp_pix=None,
        interp_weights=None,
        pol=True,
        pol_deriv=False,
    ):
        """
        Use healpy bilinear interpolation to interpolate the
        map.  User must make sure that coordinate system used
        for theta and phi matches the map coordinate system.
        IQUweight is an array of shape (nsamp,3) returned by the
        pointing library that gives the weights of the I,Q, and U maps.

        Args:
            pol_deriv(bool):  Return the polarization angle derivative
                of the signal instead of the actual signal.

        """

        if onlypol and not self.pol:
            return None

        if not self.pol or not pol:
            return self.at(
                theta, phi, interp_pix=interp_pix, interp_weights=interp_weights
            )

        if np.shape(IQUweight)[1] != 3:
            raise RuntimeError(
                "Cannot sample polarized map with only " "intensity weights"
            )

        n = len(theta)
        stepsize = self.buflen
        signal = np.zeros(n, dtype=np.float32)

        for istart in range(0, n, stepsize):
            istop = min(istart + stepsize, n)
            ind = slice(istart, istop)

            if interp_pix is None or interp_weights is None:
                p, w = hp.get_interp_weights(
                    self.nside, theta[ind], phi[ind], nest=self.nest
                )
            else:
                p = np.ascontiguousarray(interp_pix[:, ind])
                w = np.ascontiguousarray(interp_weights[:, ind])

            weights = np.ascontiguousarray(IQUweight[ind].T)

            buffer = np.zeros(istop - istart, dtype=np.float64)
            fast_scanning32(buffer, p, w, self.Map_Q[:])
            if pol_deriv:
                signal[ind] = -2 * weights[2] * buffer
            else:
                signal[ind] = weights[1] * buffer

            buffer[:] = 0
            fast_scanning32(buffer, p, w, self.Map_U[:])
            if pol_deriv:
                signal[ind] += 2 * weights[1] * buffer
            else:
                signal[ind] += weights[2] * buffer

            if not onlypol:
                if self.Map is None:
                    raise RuntimeError("No temperature map to sample")
                buffer[:] = 0
                fast_scanning32(buffer, p, w, self.Map[:])
                signal[ind] += weights[0] * buffer

        return signal
