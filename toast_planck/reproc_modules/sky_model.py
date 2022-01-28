# Copyright (c) 2016-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# Parallel sky model class to yield estimates of the sky at desired frequencies

import os
from scipy.constants import arcmin
from scipy.constants import h, k

from toast.cache import Cache
from .rings import DistRings
from toast.mpi import MPI

import astropy.io.fits as pf
import healpy as hp
from libsharp import packed_real_order, analysis, synthesis
import numpy as np
import toast.timing as timing

TCMB = 2.72548
DTYPE = np.float32

__path__ = os.path.dirname(__file__)
DATADIR = os.path.join(__path__, 'sky_model_data')


class SkyModel():

    def __init__(
            self, nside, file_sync, file_sync_pol, file_freefree, file_ame,
            file_dust, file_dust_pol, comm, fwhm=0, verbose=False,
            groupsize=256, quickpolbeam=None):
        # We have to split the communicator because there is not enough
        # work for every process.
        self.comm = comm.Split(color=(comm.rank // groupsize), key=comm.rank)
        self.global_rank = comm.rank
        self.rank = self.comm.rank
        self.ntask = self.comm.size

        self.cache = Cache()
        self.verbose = verbose

        # Divide effort for loading, smoothing and downgrading the input maps
        id_step = 16
        self.id_sync = 0
        self.id_sync_pol = (self.id_sync + id_step) % self.ntask
        self.id_ff = (self.id_sync_pol + id_step) % self.ntask
        self.id_ame = (self.id_ff + id_step) % self.ntask
        self.id_dust = (self.id_ame + id_step) % self.ntask
        self.id_dust_pol = (self.id_dust + id_step) % self.ntask

        # Set up a libsharp processing grid
        self.nside = nside
        self.npix = 12 * self.nside ** 2
        self.dist_rings = {}
        self.my_pix = self.get_my_pix(self.nside)
        self.my_npix = len(self.my_pix)

        # Minimum smoothing in every returned component
        self.fwhm = fwhm
        self.quickpolbeam = quickpolbeam

        if self.verbose and self.global_rank == 0:
            print('Initializing SkyModel. nside = {}, fwhm = {}, '
                  'quickpolbeam = {}'.format(
                      self.nside, self.fwhm, self.quickpolbeam), flush=True)

        self.file_sync = file_sync
        self.file_sync_pol = file_sync_pol
        self.file_freefree = file_freefree
        self.file_ame = file_ame
        self.file_dust = file_dust
        self.file_dust_pol = file_dust_pol

        # Load the model components

        self.load_synchrotron_temperature()
        self.load_synchrotron_polarization()
        self.load_freefree()
        self.load_ame()
        self.load_dust_temperature()
        self.load_dust_polarization()

        # Distribute the inputs

        self.comm.Barrier()
        self.broadcast_synchrotron()
        self.broadcast_freefree()
        self.broadcast_ame()
        self.broadcast_dust_temperature()
        self.broadcast_dust_polarization()

        return

    def __del__(self):

        del self.my_pix
        del self.dist_rings
        self.comm.Free()
        del self.sync_As
        del self.sync_beta
        del self.sync_As_Q
        del self.sync_As_U
        del self.sync_pol_beta
        del self.ff_em
        del self.ff_amp
        del self.ff_T_e
        del self.ame_1
        del self.ame_2
        del self.ame_nu_p_1
        del self.dust_Ad
        del self.dust_Ad_Q
        del self.dust_Ad_U
        del self.dust_temp
        del self.dust_beta
        del self.dust_temp_pol
        del self.dust_beta_pol

        self.cache.clear()

    def optimal_lmax(self, fwhm_in, nside_in):
        lmax = 2 * min(nside_in, self.nside)
        if fwhm_in < self.fwhm and self.quickpolbeam is None:
            beam = hp.gauss_beam(self.fwhm * arcmin, lmax=lmax, pol=False)
            better_lmax = np.argmin(np.abs(beam - 1e-4)) + 1
            if better_lmax < lmax:
                lmax = better_lmax
        return lmax

    def total_beam(self, fwhm_in, lmax, pol=False):
        total_beam = None
        if fwhm_in < .99 * self.fwhm:
            if self.quickpolbeam is None:
                total_beam = hp.gauss_beam(self.fwhm * arcmin, lmax=lmax, pol=True)
                total_beam = total_beam[:, 0:3].copy()
            else:
                total_beam = np.array(hp.read_cl(self.quickpolbeam))
                if total_beam.ndim == 1:
                    total_beam = np.vstack([total_beam, total_beam, total_beam])
                total_beam = total_beam[:, :lmax + 1].T.copy()
            beam_in = hp.gauss_beam(fwhm_in * arcmin, lmax=lmax, pol=True)
            beam_in = beam_in[:, 0:3].copy()
            good = beam_in != 0
            total_beam[good] /= beam_in[good]
            if pol:
                total_beam = np.ascontiguousarray(total_beam[:, (1, 2)])
            else:
                total_beam = np.ascontiguousarray(total_beam[:, 0:1])
        return total_beam

    def smooth(self, fwhm_in, nside_in, maps_in):
        """ Smooth a distributed map and change the resolution.
        """
        if fwhm_in > .9 * self.fwhm and nside_in == self.nside:
            return maps_in
        if fwhm_in > .9 * self.fwhm and self.nside < nside_in:
            # Simple ud_grade
            if self.global_rank == 0 and self.verbose:
                print('Downgrading Nside {} -> {}'
                      ''.format(nside_in, self.nside), flush=True)
            maps_out = []
            npix_in = hp.nside2npix(nside_in)
            my_pix_in = self.get_my_pix(nside_in)
            for m in maps_in:
                my_outmap = np.zeros(npix_in, dtype=float)
                outmap = np.zeros(npix_in, dtype=float)
                my_outmap[my_pix_in] = m
                self.comm.Allreduce(my_outmap, outmap)
                del my_outmap
                maps_out.append(hp.ud_grade(outmap, self.nside)[self.my_pix])
        else:
            # Full smoothing
            lmax = self.optimal_lmax(fwhm_in, nside_in)
            total_beam = self.total_beam(fwhm_in, lmax, pol=False)
            if self.global_rank == 0 and self.verbose:
                print('Smoothing {} -> {}. lmax = {}. Nside {} -> {}'
                      ''.format(fwhm_in, self.fwhm, lmax, nside_in,
                                self.nside), flush=True)
            local_m = np.arange(self.rank, lmax + 1, self.ntask, dtype=np.int32)
            alminfo = packed_real_order(lmax, ms=local_m)
            grid_in = self.get_grid(nside_in)
            grid_out = self.get_grid(self.nside)
            maps_out = []
            for local_map in maps_in:
                map_I = np.ascontiguousarray(local_map.reshape([1, 1, -1]),
                                             dtype=np.float64)
                alm_I = analysis(grid_in, alminfo, map_I, spin=0,
                                 comm=self.comm)
                if total_beam is not None:
                    alminfo.almxfl(alm_I, total_beam)
                map_I = synthesis(grid_out, alminfo, alm_I, spin=0,
                                  comm=self.comm)[0][0]
                maps_out.append(map_I)
        return maps_out

    def smooth_pol(self, fwhm_in, nside_in, maps_in):
        if fwhm_in > .9 * self.fwhm and nside_in == self.nside:
            return maps_in
        if fwhm_in > .9 * self.fwhm and self.nside < nside_in:
            # Simple ud_grade
            if self.global_rank == 0 and self.verbose:
                print('Downgrading Nside {} -> {}'
                      ''.format(nside_in, self.nside), flush=True)
            maps_out = []
            npix_in = hp.nside2npix(nside_in)
            my_pix_in = self.get_my_pix(nside_in)
            for (qmap, umap) in maps_in:
                my_mapout = np.zeros(npix_in, dtype=float)
                qmapout = np.zeros(npix_in, dtype=float)
                umapout = np.zeros(npix_in, dtype=float)
                my_mapout[my_pix_in] = qmap
                self.comm.Allreduce(my_mapout, qmapout)
                my_mapout[my_pix_in] = umap
                self.comm.Allreduce(my_mapout, umapout)
                del my_mapout
                maps_out.append((
                    hp.ud_grade(qmapout, self.nside)[self.my_pix],
                    hp.ud_grade(umapout, self.nside)[self.my_pix]))
        else:
            # Full smoothing
            lmax = self.optimal_lmax(fwhm_in, nside_in)
            total_beam = self.total_beam(fwhm_in, lmax, pol=True)
            if self.global_rank == 0 and self.verbose:
                print('Smoothing {} -> {}. lmax = {}. Nside {} -> {}'
                      ''.format(fwhm_in, self.fwhm, lmax, nside_in,
                                self.nside), flush=True)
            local_m = np.arange(self.rank, lmax + 1, self.ntask, dtype=np.int32)
            alminfo = packed_real_order(lmax, ms=local_m)
            grid_in = self.get_grid(nside_in)
            grid_out = self.get_grid(self.nside)
            maps_out = []
            for (local_map_Q, local_map_U) in maps_in:
                map_P = np.ascontiguousarray(
                        np.vstack([local_map_Q, local_map_U]
                                  ).reshape((1, 2, -1)),
                        dtype=np.float64)
                alm_P = analysis(grid_in, alminfo, map_P, spin=2,
                                 comm=self.comm)
                if total_beam is not None:
                    alminfo.almxfl(alm_P, total_beam)
                map_P = synthesis(grid_out, alminfo, alm_P, spin=2,
                                  comm=self.comm)[0]
                maps_out.append(map_P)
        return maps_out

    def load_synchrotron_temperature(self):
        # # Synchrotron temperature
        if self.rank == self.id_sync:
            try:
                # Try old format first
                with pf.open(self.file_sync, 'readonly') as h:
                    self.sync_psd_freq = h[2].data.field(0)
                    self.sync_psd = h[2].data.field(1)
                    self.sync_nu_ref = float(
                        h[1].header['nu_ref'].split()[0]) * 1e-3  # To GHz
                    self.sync_fwhm = h[1].header['fwhm']
                self.sync_As = hp.read_map(
                    self.file_sync, verbose=False, dtype=DTYPE, memmap=True)
                self.sync_nside = hp.get_nside(self.sync_As)
                self.sync_beta = None
                if self.verbose:
                    print('Loaded synchrotron T: nside = {}, fwhm = {}'.format(
                        self.sync_nside, self.sync_fwhm), flush=True)
            except Exception as e:
                if self.verbose:
                    print('Old synchrotron T format failed ("{}"). Trying new '
                          'format'.format(e), flush=True)
                self.sync_fwhm = 60.
                self.sync_nu_ref = 0.408
                self.sync_As = hp.read_map(
                    self.file_sync, verbose=False, dtype=DTYPE, memmap=True)
                self.sync_beta = hp.read_map(
                    self.file_sync.replace('synch_', 'synch_beta_'),
                    verbose=False, dtype=DTYPE, memmap=True)
                self.sync_nside = hp.get_nside(self.sync_As)
                self.sync_psd_freq = None
                self.sync_psd = None
                if self.verbose:
                    print('Loaded synchrotron T: nside = {}, fwhm = {}'.format(
                        self.sync_nside, self.sync_fwhm), flush=True)
        else:
            self.sync_As = None
            self.sync_beta = None
            self.sync_psd_freq = None
            self.sync_psd = None
            self.sync_nu_ref = None
            self.sync_fwhm = None
            self.sync_nside = None
        return

    def load_synchrotron_polarization(self):
        # # Synchrotron polarization
        if self.rank == self.id_sync_pol:
            try:
                # Try old format first
                with pf.open(self.file_sync_pol) as h:
                    self.sync_pol_nu_ref = float(
                        h[1].header['nu_ref'].split()[0])  # In GHz
                    self.sync_pol_fwhm = h[1].header['fwhm']
                self.sync_As_Q, self.sync_As_U = hp.read_map(
                    self.file_sync_pol, [0, 1], verbose=False, dtype=DTYPE,
                    memmap=True)
                self.sync_pol_nside = hp.get_nside(self.sync_As_Q)
                self.sync_pol_psd_freq, self.sync_pol_psd = np.genfromtxt(
                    os.path.join(DATADIR, 'synchrotron_psd_2015.dat')).T
                self.sync_pol_beta = None
                if self.verbose:
                    print('Loaded synchrotron P: nside = {}, fwhm = {}'.format(
                        self.sync_pol_nside, self.sync_pol_fwhm),
                        flush=True)
            except Exception as e:
                if self.verbose:
                    print('Old synchrotron T format failed ("{}"). Trying new '
                          'format'.format(e), flush=True)
                self.sync_pol_fwhm = 40.
                self.sync_pol_nu_ref = 30.
                self.sync_As_Q, self.sync_As_U = hp.read_map(
                    self.file_sync_pol, [1, 2], verbose=False, dtype=DTYPE,
                    memmap=True)
                self.sync_pol_beta = hp.read_map(
                    self.file_sync.replace('synch_', 'synch_beta_'),
                    verbose=False, dtype=DTYPE, memmap=True)
                self.sync_pol_nside = hp.get_nside(self.sync_As_Q)
                self.sync_pol_psd_freq = None
                self.sync_pol_psd = None
                if self.verbose:
                    print('Loaded synchrotron P: nside = {}, fwhm = {}'.format(
                        self.sync_pol_nside, self.sync_pol_fwhm), flush=True)
        else:
            self.sync_As_Q = None
            self.sync_As_U = None
            self.sync_pol_beta = None
            self.sync_pol_psd_freq = None
            self.sync_pol_psd = None
            self.sync_pol_nu_ref = None
            self.sync_pol_fwhm = None
            self.sync_pol_nside = None
        return

    def load_freefree(self):
        # # free-free
        if self.rank == self.id_ff:
            try:
                # Try old format first
                with pf.open(self.file_freefree) as h:
                    self.ff_fwhm = h[1].header['fwhm']
                self.ff_em, self.ff_T_e = hp.read_map(
                    self.file_freefree, [0, 3], verbose=False, dtype=DTYPE,
                    memmap=True)
                self.ff_nside = hp.get_nside(self.ff_em)
                self.ff_nu_ref = None
                self.ff_amp = None
                if self.verbose:
                    print('Loaded freefree: nside = {}, fwhm = {}'.format(
                        self.ff_nside, self.ff_fwhm), flush=True)
            except Exception as e:
                if self.verbose:
                    print('Old freefree format failed ("{}"). Trying new '
                          'format'.format(e), flush=True)
                self.ff_fwhm = 20.
                self.ff_nu_ref = 1.4
                self.ff_amp = hp.read_map(
                    self.file_freefree, verbose=False, dtype=DTYPE, memmap=True)
                # self.ff_em = hp.read_map(
                #    self.file_freefree.replace('ff_', 'ff_EM'), verbose=False,
                #    dtype=DTYPE, memmap=True)
                self.ff_T_e = hp.read_map(
                    self.file_freefree.replace('ff_', 'ff_Te_'), verbose=False,
                    dtype=DTYPE, memmap=True)
                self.ff_nside = hp.get_nside(self.ff_amp)
                self.ff_em = None
                if self.verbose:
                    print('Loaded freefree: nside = {}, fwhm = {}'.format(
                        self.ff_nside, self.ff_fwhm), flush=True)
        else:
            self.ff_amp = None
            self.ff_em = None
            self.ff_T_e = None
            self.ff_fwhm = None
            self.ff_nside = None
            self.ff_nu_ref = None
        return

    def load_ame(self):
        # # spinning dust
        if self.rank == self.id_ame:
            try:
                # Try old format first
                self.ame_nu_p0 = 30.
                with pf.open(self.file_ame) as h:
                    self.ame_fwhm = h[1].header['fwhm']
                    # All frequencies are in GHz
                    self.ame_nu_ref1 = float(
                        h[1].header['nu_ref'].split()[0])
                    self.ame_nu_ref2 = float(
                        h[2].header['nu_ref'].split()[0])
                    self.ame_nu_p_2 = float(h[2].header['nu_p'].split()[0])
                    self.ame_psd_freq = h[3].data.field(0)
                    self.ame_psd = h[3].data.field(1)
                self.ame_1, self.ame_nu_p_1 = hp.read_map(
                    self.file_ame, [0, 3], verbose=False, dtype=DTYPE,
                    memmap=True)
                self.ame_2 = hp.read_map(
                    self.file_ame, hdu=2, verbose=False, dtype=DTYPE,
                    memmap=True)
                self.ame_nside = hp.get_nside(self.ame_1)
                if self.verbose:
                    print('Loaded AME: nside = {}, fwhm = {}'.format(
                        self.ame_nside, self.ame_fwhm), flush=True)
            except Exception as e:
                if self.verbose:
                    print('Old AME format failed ("{}"). Trying new '
                          'format'.format(e), flush=True)
                self.ame_nu_p0 = 22.2  # GHz
                self.ame_fwhm = 30.  # GHz
                self.ame_nu_ref1 = 30.  # arc min
                self.ame_1 = hp.read_map(
                    self.file_ame, verbose=False, dtype=DTYPE, memmap=True)
                self.ame_nu_p_1 = hp.read_map(
                    self.file_ame.replace('ame_', 'ame_nu_p_'), verbose=False,
                    dtype=DTYPE, memmap=True)
                self.ame_nu_ref2 = None
                self.ame_2 = None
                self.ame_nu_p_2 = None
                self.ame_psd_freq, self.ame_psd = np.genfromtxt(
                    os.path.join(DATADIR, 'spdust2_cnm.dat')).T
                self.ame_nside = hp.get_nside(self.ame_1)
                if self.verbose:
                    print('Loaded AME: nside = {}, fwhm = {}'.format(
                        self.ame_nside, self.ame_fwhm), flush=True)
        else:
            self.ame_1 = None
            self.ame_2 = None
            self.ame_nu_p0 = None
            self.ame_nu_p_1 = None
            self.ame_nu_p_2 = None
            self.ame_nu_ref1 = None
            self.ame_nu_ref2 = None
            self.ame_fwhm = None
            self.ame_nside = None
            self.ame_psd_freq = None
            self.ame_psd = None
        return

    def load_dust_temperature(self):
        # # Thermal dust temperature
        if self.rank == self.id_dust:
            try:
                # Try old format first
                with pf.open(self.file_dust) as h:
                    self.dust_fwhm = h[1].header['fwhm']
                    self.dust_nu_ref = float(
                        h[1].header['nu_ref'].split()[0])  # in GHz
                self.dust_Ad, self.dust_temp, self.dust_beta = hp.read_map(
                    self.file_dust, [0, 3, 6], verbose=False, dtype=DTYPE,
                    memmap=True)
            except Exception as e:
                if self.verbose:
                    print('Old dust T format failed ("{}"). Trying new '
                          'format'.format(e), flush=True)
                self.dust_fwhm = 5.
                self.dust_nu_ref = 857
                self.dust_Ad = hp.read_map(
                    self.file_dust, verbose=False, dtype=DTYPE, memmap=True)
                self.dust_temp = hp.read_map(
                    self.file_dust.replace('dust', 'dust_T'),
                    verbose=False, dtype=DTYPE, memmap=True)
                self.dust_beta = hp.read_map(
                    self.file_dust.replace('dust', 'dust_beta'),
                    verbose=False, dtype=DTYPE, memmap=True)
            self.dust_nside = hp.get_nside(self.dust_Ad)
            if self.verbose:
                print('Loaded dust T: nside = {}, fwhm = {}'.format(
                    self.dust_nside, self.dust_fwhm), flush=True)
        else:
            self.dust_Ad = None
            self.dust_temp = None
            self.dust_beta = None
            self.dust_fwhm = None
            self.dust_nside = None
            self.dust_nu_ref = None
        return

    def load_dust_polarization(self):
        # # Thermal dust polarization
        if self.rank == self.id_dust_pol:
            try:
                # Try old format first
                with pf.open(self.file_dust_pol) as h:
                    self.dust_pol_fwhm = h[1].header['fwhm']
                    self.dust_pol_nu_ref = float(
                        h[1].header['nu_ref'].split()[0])  # in GHz
                self.dust_temp_pol = None
                self.dust_beta_pol = None
                self.dust_Ad_Q, self.dust_Ad_U = hp.read_map(
                    self.file_dust_pol, range(2), verbose=False, dtype=DTYPE,
                    memmap=True)
            except Exception as e:
                if self.verbose:
                    print('Old dust P format failed ("{}"). Trying new '
                          'format'.format(e), flush=True)
                self.dust_pol_fwhm = 5
                self.dust_pol_nu_ref = 353
                self.dust_Ad_Q, self.dust_Ad_U = hp.read_map(
                    self.file_dust_pol, [1, 2], verbose=False, dtype=DTYPE,
                    memmap=True)
                fname = self.file_dust_pol.replace('dust', 'dust_T')
                self.dust_temp_pol = hp.read_map(
                    fname, 1, verbose=False, dtype=DTYPE, memmap=True)
                tlim = 12
                bad = self.dust_temp_pol < tlim
                nbad = np.sum(bad)
                if nbad > 0:
                    print('WARNING: regularizing {} cold pixels in {}'
                          ''.format(nbad, fname), flush=True)
                    self.dust_temp_pol[bad] = tlim
                self.dust_beta_pol = hp.read_map(
                    self.file_dust_pol.replace('dust', 'dust_beta'), 1,
                    verbose=False, dtype=DTYPE, memmap=True)
            self.dust_pol_nside = hp.get_nside(self.dust_Ad_Q)
            if self.verbose:
                print('Loaded dust P: nside = {}, fwhm = {}'.format(
                    self.dust_pol_nside, self.dust_pol_fwhm),
                    flush=True)
        else:
            self.dust_Ad_Q = None
            self.dust_Ad_U = None
            self.dust_temp_pol = None
            self.dust_beta_pol = None
            self.dust_pol_fwhm = None
            self.dust_pol_nside = None
            self.dust_pol_nu_ref = None
        return

    def get_my_pix(self, nside):
        if nside not in self.dist_rings:
            self.dist_rings[nside] = DistRings(self.comm, nside=nside, nnz=3)
        return self.dist_rings[nside].local_pixels

    def get_grid(self, nside):
        if nside not in self.dist_rings:
            self.dist_rings[nside] = DistRings(self.comm, nside=nside, nnz=3)
        return self.dist_rings[nside].libsharp_grid

    def broadcast_synchrotron(self):
        # Broadcast synchrotron temperature
        root = self.id_sync
        self.sync_psd_freq = self.comm.bcast(self.sync_psd_freq, root=root)
        self.sync_psd = self.comm.bcast(self.sync_psd, root=root)
        self.sync_nu_ref = self.comm.bcast(self.sync_nu_ref, root=root)
        self.sync_fwhm = self.comm.bcast(self.sync_fwhm, root=root)
        self.sync_nside = self.comm.bcast(self.sync_nside, root=root)
        my_pix = self.get_my_pix(self.sync_nside)
        self.sync_As = self.cache.put(
            'sync_As', self.comm.bcast(self.sync_As, root=root)[my_pix])
        if self.sync_psd is None:
            # New format
            self.sync_beta = self.cache.put(
                'sync_beta', self.comm.bcast(self.sync_beta, root=root)[my_pix])
        # Broadcast synchrotron polarization
        root = self.id_sync_pol
        self.sync_pol_psd_freq = self.comm.bcast(self.sync_pol_psd_freq,
                                                 root=root)
        self.sync_pol_psd = self.comm.bcast(self.sync_pol_psd, root=root)
        self.sync_pol_nu_ref = self.comm.bcast(self.sync_pol_nu_ref, root=root)
        self.sync_pol_fwhm = self.comm.bcast(self.sync_pol_fwhm, root=root)
        self.sync_pol_nside = self.comm.bcast(self.sync_pol_nside, root=root)
        my_pix = self.get_my_pix(self.sync_pol_nside)
        self.sync_pol_beta = self.comm.bcast(self.sync_pol_beta, root=root)
        if self.sync_pol_beta is not None:
            self.sync_pol_beta = self.cache.put(
                'sync_pol_beta', self.sync_pol_beta[my_pix])
        self.sync_As_Q = self.cache.put(
            'sync_As_Q', self.comm.bcast(self.sync_As_Q, root=root)[my_pix])
        self.sync_As_U = self.cache.put(
            'sync_As_U', self.comm.bcast(self.sync_As_U, root=root)[my_pix])
        return

    def broadcast_freefree(self):
        # Broadcast free-free
        root = self.id_ff
        self.ff_nu_ref = self.comm.bcast(self.ff_nu_ref, root=root)
        self.ff_fwhm = self.comm.bcast(self.ff_fwhm, root=root)
        self.ff_nside = self.comm.bcast(self.ff_nside, root=root)
        my_pix = self.get_my_pix(self.ff_nside)
        self.ff_amp = self.comm.bcast(self.ff_amp, root=root)
        if self.ff_amp is not None:
            self.ff_amp = self.cache.put('ff_amp', self.ff_amp[my_pix])
        self.ff_em = self.comm.bcast(self.ff_em, root=root)
        if self.ff_em is not None:
            self.ff_em = self.cache.put('ff_em', self.ff_em[my_pix])
        self.ff_T_e = self.cache.put(
            'ff_T_e', self.comm.bcast(self.ff_T_e, root=root)[my_pix])
        return

    def broadcast_ame(self):
        # Broadcast AME
        root = self.id_ame
        self.ame_nu_p0 = self.comm.bcast(self.ame_nu_p0, root=root)
        self.ame_nu_p_2 = self.comm.bcast(self.ame_nu_p_2, root=root)
        self.ame_nu_ref1 = self.comm.bcast(self.ame_nu_ref1, root=root)
        self.ame_nu_ref2 = self.comm.bcast(self.ame_nu_ref2, root=root)
        self.ame_fwhm = self.comm.bcast(self.ame_fwhm, root=root)
        self.ame_nside = self.comm.bcast(self.ame_nside, root=root)
        self.ame_psd_freq = self.comm.bcast(self.ame_psd_freq, root=root)
        self.ame_psd = self.comm.bcast(self.ame_psd, root=root)
        my_pix = self.get_my_pix(self.ame_nside)
        self.ame_1 = self.cache.put(
            'ame_1', self.comm.bcast(self.ame_1, root=root)[my_pix])
        self.ame_2 = self.comm.bcast(self.ame_2, root=root)
        if self.ame_2 is not None:
            self.ame_2 = self.cache.put('ame_2', self.ame_2[my_pix])
        self.ame_nu_p_1 = self.cache.put(
            'ame_freq_1', self.comm.bcast(self.ame_nu_p_1, root=root)[my_pix])
        return

    def broadcast_dust_temperature(self):
        # Broadcast dust
        root = self.id_dust
        self.dust_fwhm = self.comm.bcast(self.dust_fwhm, root=root)
        self.dust_nside = self.comm.bcast(self.dust_nside, root=root)
        self.dust_nu_ref = self.comm.bcast(self.dust_nu_ref, root=root)
        my_pix = self.get_my_pix(self.dust_nside)
        self.dust_Ad = self.cache.put(
            'dust_Ad', self.comm.bcast(self.dust_Ad, root=root)[my_pix])
        # We need to store dust temperature and beta fully in case
        # the dust inputs are in the old format and the polarization
        # has different resolution than the temperature
        self.dust_temp = self.comm.bcast(self.dust_temp, root=root)
        self.dust_beta = self.comm.bcast(self.dust_beta, root=root)
        return

    def broadcast_dust_polarization(self):
        root = self.id_dust_pol
        self.dust_pol_fwhm = self.comm.bcast(self.dust_pol_fwhm, root=root)
        self.dust_pol_nside = self.comm.bcast(self.dust_pol_nside, root=root)
        self.dust_pol_nu_ref = self.comm.bcast(self.dust_pol_nu_ref, root=root)
        self.dust_temp_pol = self.comm.bcast(self.dust_temp_pol, root=root)
        my_pix = self.get_my_pix(self.dust_pol_nside)
        # amplitude
        self.dust_Ad_Q = self.cache.put(
            'dust_Ad_Q', self.comm.bcast(self.dust_Ad_Q, root=root)[my_pix])
        self.dust_Ad_U = self.cache.put(
            'dust_Ad_U', self.comm.bcast(self.dust_Ad_U, root=root)[my_pix])
        # temperature
        if self.dust_temp_pol is None:
            # Old dust inputs
            self.dust_temp_pol = hp.ud_grade(self.dust_temp,
                                             self.dust_pol_nside)
        self.dust_temp_pol = self.cache.put(
                'dust_temp_pol', self.dust_temp_pol[my_pix])
        # beta
        self.dust_beta_pol = self.comm.bcast(self.dust_beta_pol,
                                             root=root)
        if self.dust_beta_pol is None:
            # Old dust inputs
            self.dust_beta_pol = hp.ud_grade(self.dust_beta,
                                             self.dust_pol_nside)
        self.dust_beta_pol = self.cache.put(
                'dust_beta_pol', self.dust_beta_pol[my_pix])
        # Only now, can we extract the local portion of dust
        # temperature and beta.
        my_pix = self.get_my_pix(self.dust_nside)
        self.dust_temp = self.cache.put('dust_temp', self.dust_temp[my_pix])
        self.dust_beta = self.cache.put('dust_beta', self.dust_beta[my_pix])
        return

    def add_synchrotron(self, map_I, map_Q, map_U, freq, krj2kcmb):
        # synchrotron temperature
        if self.sync_beta is None:
            # Old format
            psd_sync_ref = np.exp(
                np.interp(np.log(self.sync_nu_ref),
                          np.log(self.sync_psd_freq), np.log(self.sync_psd)))
            psd_sync = np.exp(
                np.interp(np.log(freq),
                          np.log(self.sync_psd_freq), np.log(self.sync_psd)))
            scale = (self.sync_nu_ref / freq) ** 2 * psd_sync / psd_sync_ref
        else:
            # New format
            scale = (freq / self.sync_nu_ref) \
                ** self.sync_beta.astype(np.float64)
        sync = self.sync_As * scale
        sync = (sync * krj2kcmb).astype(DTYPE)
        if self.verbose and self.global_rank == 0:
            print('Smoothing synchrotron T', flush=True)
        sync = self.smooth(self.sync_fwhm, self.sync_nside, [sync])[0]
        map_I += sync
        # synchrotron polarization
        if self.sync_pol_beta is None:
            # Old format
            psd_sync_ref = np.exp(
                np.interp(np.log(self.sync_pol_nu_ref),
                          np.log(self.sync_pol_psd_freq),
                          np.log(self.sync_pol_psd)))
            psd_sync = np.exp(
                np.interp(np.log(freq),
                          np.log(self.sync_pol_psd_freq),
                          np.log(self.sync_pol_psd)))
            scale = (self.sync_pol_nu_ref / freq) ** 2 * psd_sync / psd_sync_ref
        else:
            # New format
            scale = (freq / self.sync_pol_nu_ref) \
                ** self.sync_pol_beta.astype(np.float64)
        sync_Q = self.sync_As_Q * scale
        sync_U = self.sync_As_U * scale
        sync_Q = (sync_Q * krj2kcmb).astype(DTYPE)
        sync_U = (sync_U * krj2kcmb).astype(DTYPE)
        if self.verbose and self.global_rank == 0:
            print('Smoothing synchrotron P', flush=True)
        sync_Q, sync_U = self.smooth_pol(
            self.sync_pol_fwhm, self.sync_pol_nside, [(sync_Q, sync_U)])[0]
        map_Q += sync_Q
        map_U += sync_U
        return

    def add_freefree(self, map_I, freq, krj2kcmb):
        # freefree temperature
        if self.ff_amp is None:
            # 2015 model
            gff = np.log(
                np.exp(
                    5.960 - np.sqrt(3) / np.pi
                    * np.log(
                        freq * (self.ff_T_e.astype(np.float64) * 1e-4) ** (-1.5)
                    )
                ) +
                np.exp(1)
            )
            tau = (0.05468 * self.ff_T_e.astype(np.float64) ** (-1.5)
                   * freq ** (-2) * self.ff_em * gff)
            ff = 1e6 * self.ff_T_e.astype(np.float64) * (1 - np.exp(-tau))
        else:
            # 2018 model
            S = np.log(np.exp(5.960 - np.sqrt(3) / np.pi * np.log(
                freq * (self.ff_T_e.astype(np.float64) * 1e-4
                        ) ** (-1.5))) + np.exp(1))
            S_ref = np.log(np.exp(5.960 - np.sqrt(3) / np.pi * np.log(
                self.ff_nu_ref * (self.ff_T_e.astype(np.float64) * 1e-4
                                  ) ** (-1.5))) + np.exp(1))
            ff = self.ff_amp * S / S_ref * np.exp(
                -h * (freq - self.ff_nu_ref) / k /
                self.ff_T_e.astype(np.float64)
                ) * (freq / self.ff_nu_ref) ** (-2)
        # K_RJ -> K_CMB
        ff = (ff * krj2kcmb).astype(DTYPE)
        if self.verbose and self.global_rank == 0:
            print('Smoothing freefree', flush=True)
        ff = self.smooth(self.ff_fwhm, self.ff_nside, [ff])[0]
        map_I += ff
        return

    def add_ame(self, map_I, freq, krj2kcmb):
        # Prepare for logarithmic interpolation
        x, y = np.log(self.ame_psd_freq), np.log(self.ame_psd)
        # spinning dust, first component
        scale = self.ame_nu_p0 / self.ame_nu_p_1.astype(np.float64)
        arg1 = freq * scale
        arg2 = self.ame_nu_ref1 * scale
        psd_ame = np.exp(np.interp(np.log(arg1), x, y))
        psd_ame_ref = np.exp(np.interp(np.log(arg2), x, y))
        ame1 = self.ame_1 * (self.ame_nu_ref1 / freq) ** 2 * \
            psd_ame / psd_ame_ref
        ame1 = (ame1 * krj2kcmb).astype(DTYPE)
        if self.ame_2 is not None:
            # spinning dust, second component
            scale = self.ame_nu_p0 / self.ame_nu_p_2
            arg1 = freq * scale
            arg2 = self.ame_nu_ref2 * scale
            psd_ame = np.exp(np.interp(np.log(arg1), x, y))
            psd_ame_ref = np.exp(np.interp(np.log(arg2), x, y))
            ame2 = self.ame_2 * (self.ame_nu_ref2 / freq) ** 2 * \
                psd_ame / psd_ame_ref
            ame2 = (ame2 * krj2kcmb).astype(DTYPE)
        if self.verbose and self.global_rank == 0:
            print('Smoothing AME', flush=True)
        if self.ame_2 is None:
            ame1 = self.smooth(self.ame_fwhm, self.ame_nside, [ame1])[0]
            ame2 = 0
        else:
            ame1, ame2 = self.smooth(self.ame_fwhm, self.ame_nside,
                                     [ame1, ame2])
        map_I += ame1 + ame2
        return

    def add_dust(self, map_I, map_Q, map_U, freq, krj2kcmb):
        # thermal dust temperature
        gamma = h / k / self.dust_temp.astype(np.float64)
        scale = (
            (freq / self.dust_nu_ref) ** (self.dust_beta.astype(np.float64) + 1)
            * (np.exp(gamma * self.dust_nu_ref * 1e9) - 1)
            / (np.exp(gamma * freq * 1e9) - 1))
        dust = self.dust_Ad * scale
        dust = (dust * krj2kcmb).astype(DTYPE)
        if self.verbose and self.global_rank == 0:
            print('Smoothing dust T', flush=True)
        dust = self.smooth(self.dust_fwhm, self.dust_nside, [dust])[0]
        map_I += dust
        # thermal dust polarization
        gamma = h / k / self.dust_temp_pol.astype(np.float64)
        scale = (
            (freq / self.dust_pol_nu_ref) ** (
                self.dust_beta_pol.astype(np.float64) + 1)
            * (np.exp(gamma * self.dust_pol_nu_ref * 1e9) - 1)
            / (np.exp(gamma * freq * 1e9) - 1))
        dust_Q = self.dust_Ad_Q * scale
        dust_U = self.dust_Ad_U * scale
        dust_Q = (dust_Q * krj2kcmb).astype(DTYPE)
        dust_U = (dust_U * krj2kcmb).astype(DTYPE)
        if self.verbose and self.global_rank == 0:
            print('Smoothing dust P', flush=True)
        dust_Q, dust_U = self.smooth_pol(
            self.dust_pol_fwhm, self.dust_pol_nside, [(dust_Q, dust_U)])[0]
        map_Q += dust_Q
        map_U += dust_U
        return

    def eval(self, freq, synchrotron=True, freefree=True, ame=True, dust=True):
        """
        Evaluate the total sky model.
        Args:
        freq [GHz]
        """

        my_mtot = np.zeros(self.my_npix)
        my_mtot_Q = np.zeros(self.my_npix)
        my_mtot_U = np.zeros(self.my_npix)

        # uK_RJ -> K_CMB
        x = h * freq * 1e9 / k / TCMB
        # delta T_CMB / delta T_RJ
        g = (np.exp(x) - 1) ** 2 / (x ** 2 * np.exp(x))
        # uK_CMB -> K_CMB
        krj2kcmb = g * 1e-6

        if synchrotron:
            self.add_synchrotron(my_mtot, my_mtot_Q, my_mtot_U, freq, krj2kcmb)
        if freefree:
            self.add_freefree(my_mtot, freq, krj2kcmb)
        if ame:
            self.add_ame(my_mtot, freq, krj2kcmb)
        if dust:
            self.add_dust(my_mtot, my_mtot_Q, my_mtot_U, freq, krj2kcmb)

        # Gather the pieces, each process gets a copy of the full map

        my_pix = self.get_my_pix(self.nside)
        my_outmap = np.zeros([3, self.npix], dtype=float)
        outmap = np.zeros([3, self.npix], dtype=float)
        my_outmap[0, my_pix] = my_mtot
        my_outmap[1, my_pix] = my_mtot_Q
        my_outmap[2, my_pix] = my_mtot_U
        self.comm.Allreduce(my_outmap, outmap)
        del my_outmap

        return outmap


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    nside = 256
    file_sync = '/Users/reijo/data/PR2/foregroundmaps/' \
                'COM_CompMap_Synchrotron-commander_0256_R2.00.fits'
    file_sync_pol = '/Users/reijo/data/PR2/foregroundmaps/' \
                    'COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits'
    file_freefree = '/Users/reijo/data/PR2/foregroundmaps/' \
                    'COM_CompMap_freefree-commander_0256_R2.00.fits'
    file_ame = '/Users/reijo/data/PR2/foregroundmaps/' \
               'COM_CompMap_AME-commander_0256_R2.00.fits'
    file_dust = '/Users/reijo/data/PR2/foregroundmaps/' \
                'COM_CompMap_dust-commander_0256_R2.00.fits'
    file_dust_pol = '/Users/reijo/data/PR2/foregroundmaps/' \
                    'COM_CompMap_DustPol-commander_1024_R2.00.fits'

    t1 = MPI.Wtime()
    skymodel = SkyModel(nside, file_sync, file_sync_pol, file_freefree,
                        file_ame, file_dust, file_dust_pol, MPI.COMM_WORLD)
    t2 = MPI.Wtime()
    print('{:4} : Initialized sky model in {:.2f} s'.format(
        skymodel.rank, t2 - t1), flush=True)

    skymodel.cache.report()

    norm = 1e6
    amp = 3e2
    pol_amp = 5e1
    deriv_amp = 10
    pol_deriv_amp = 1
    freqs = [30, 44, 70, 100, 143, 217, 353]
    nfreq = len(freqs)
    nrow = 8
    ncol = nfreq
    for ifreq, freq in enumerate(freqs):

        if skymodel.rank == -1:
            try:
                if freq < 100:
                    m = hp.ud_grade(hp.read_map(
                        '/Users/reijo/data/PR2/frequencymaps/'
                        'LFI_SkyMap_{:03}_1024_R2.01_full.fits'.format(freq),
                        range(3), verbose=False), nside)
                else:
                    m = hp.ud_grade(hp.read_map(
                        '/Users/reijo/data/PR2/frequencymaps/'
                        'HFI_SkyMap_{:03}_2048_R2.02_full.fits'.format(freq),
                        range(3), verbose=False), nside)
                m = hp.smoothing(m, fwhm=4 * np.pi / 180, lmax=512,
                                 verbose=False)
                m = np.array(m)
                m[0] = hp.remove_monopole(m[0], gal_cut=80)
                hp.mollview(m[0] * norm, min=-amp, max=amp,
                            sub=[nrow, ncol, ifreq + 1],
                            title='PR2 {}GHz'.format(freq))
                hp.mollview(np.sqrt(m[1] ** 2 + m[2] ** 2) * norm, min=0,
                            max=pol_amp,
                            sub=[nrow, ncol, ifreq + ncol + 1],
                            title='PR2 {}GHz P'.format(freq))
            except Exception:
                m = None
                pass

        t1 = MPI.Wtime()
        mtot1 = skymodel.eval(freq)
        t2 = MPI.Wtime()
        print('{:4} : Evaluated sky model (1/2) in {:.2f} s'.format(
            skymodel.rank, t2 - t1), flush=True)
        mtot2 = skymodel.eval(freq + 1)
        t1 = MPI.Wtime()
        print('{:4} : Evaluated sky model (2/2) in {:.2f} s'.format(
            skymodel.rank, t1 - t2), flush=True)

        if skymodel.rank == -1:
            mtot1 = np.array(hp.smoothing(
                mtot1, fwhm=4 * np.pi / 180, lmax=512, verbose=False))
            mtot2 = np.array(hp.smoothing(
                mtot2, fwhm=4 * np.pi / 180, lmax=512, verbose=False))
            hp.mollview(mtot1[0] * norm, sub=[nrow, ncol, ifreq + 2 * ncol + 1],
                        min=-amp, max=amp, title='FG {}GHz'.format(freq))
            hp.mollview(np.sqrt(mtot1[1] ** 2 + mtot1[2] ** 2) * norm,
                        sub=[nrow, ncol, ifreq + 3 * ncol + 1], min=0,
                        max=pol_amp, title='FG {}GHz P'.format(freq))
            deriv = mtot2 - mtot1
            hp.mollview(deriv[0] * norm, sub=[nrow, ncol, ifreq + 4 * ncol + 1],
                        min=-deriv_amp, max=deriv_amp,
                        title='FG deriv {}GHz'.format(freq))
            hp.mollview(np.sqrt(deriv[1] ** 2 + deriv[2] ** 2) * norm,
                        sub=[nrow, ncol, ifreq + 5 * ncol + 1], min=0,
                        max=pol_deriv_amp,
                        title='FG deriv {}GHz P'.format(freq))
            try:
                resid = m - mtot1
                resid[0] = hp.remove_monopole(resid[0], gal_cut=80)
                hp.mollview(resid[0] * norm,
                            sub=[nrow, ncol, ifreq + 6 * ncol + 1],
                            min=-amp, max=amp, title='Resid {}GHz'.format(freq))
                hp.mollview(np.sqrt(resid[1] ** 2 + resid[2] ** 2) * norm,
                            sub=[nrow, ncol, ifreq + 7 * ncol + 1],
                            min=0, max=pol_amp,
                            title='Resid {}GHz P'.format(freq))
            except Exception:
                pass

    plt.show()
