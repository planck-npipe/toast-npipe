# Upgrades to the 2018 runs:
#   - switch from FFP9 to FFP10
#   - add tensor modes matching r=0.01
#   - include T->EB leakage based on the QuickPol window functions

import healpy as hp
import numpy as np
from time import time
import os

if True:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
    ntask = comm.size
else:
    comm = None
    rank = 0
    ntask = 1

bldir = "/project/projectdirs/planck/data/npipe/npipe6v20/quickpol"

ijob = -1
#for mc in range(1000):
#for mc in range(900, 910):
for mc in range(200, 210):
    tstart_mc = time()
    for freq in [30, 44, 70, 100, 143, 217, 353, 545, 857]:
    #for freq in [545, 857]:
        ijob += 1
        if ijob % ntask != rank:
            continue
        print("\n{:4} : Processing MC = {}, freq = {}GHz".format(rank, mc, freq))
        tstart = time()
        if freq < 100:
            nside = 1024
        else:
            nside = 2048
        # Scale all maps to K_CMB
        #if freq == 545:
        #    # Planck 2013 Results. IX. HFI Spectral response. Table 6.
        #    scale = 1 / 58.04
        #elif freq == 857:
        #    # Ditto ^
        #    scale = 1 / 2.27
        #else:
        scale = 1
        pol = True
        fname_wl = os.path.join(
            bldir, "Wl_npipe6v20_{:03}GHzx{:03}GHz.fits".format(freq, freq)
        )
        fname_wlA = os.path.join(
            bldir, "Wl_npipe6v20_{:03}Ax{:03}A.fits".format(freq, freq)
        )
        fname_wlB = os.path.join(
            bldir, "Wl_npipe6v20_{:03}Bx{:03}B.fits".format(freq, freq)
        )
        if freq < 545:
            teb = '_TEB'
        else:
            teb = ''
        fname_bl = os.path.join(
            bldir, "Bl{}_npipe6v20_{:03}GHzx{:03}GHz.fits".format(teb, freq, freq)
        )
        fname_blA = os.path.join(
            bldir, "Bl{}_npipe6v20_{:03}Ax{:03}A.fits".format(teb, freq, freq)
        )
        fname_blB = os.path.join(
            bldir, "Bl{}_npipe6v20_{:03}Bx{:03}B.fits".format(teb, freq, freq)
        )
        # almfile = 'mc_cmb/scl/{:03}/ffp9_cmb_scl_{:03}_alm_mc_{:04}.fits' \
        #          ''.format(freq, freq, mc)
        almfile1 = (
            "/project/projectdirs/planck/data/ffp10/sky/CMB/mc/scl/"
            "lensed/{:03}/ffp10_lensed_scl_cmb_{:03}_alm_mc_{:04}.fits"
            "".format(freq, freq, mc)
        )
        almfile2 = (
            "/project/projectdirs/planck/data/ffp10/sky/CMB/mc/ten/"
            "ffp10_ten_cmb_000_alm_mc_{:04}.fits".format(mc)
        )
        cachefile_map = (
            "cmb_cache/"
            "ffp10_cmb_{:03}_alm_mc_{:04}_nside{:04}_quickpol.fits"
            "".format(freq, mc, nside)
        )
        cachefile_mapA = (
            "cmb_cache/"
            "ffp10_cmb_{:03}A_alm_mc_{:04}_nside{:04}_quickpol.fits"
            "".format(freq, mc, nside)
        )
        cachefile_mapB = (
            "cmb_cache/"
            "ffp10_cmb_{:03}B_alm_mc_{:04}_nside{:04}_quickpol.fits"
            "".format(freq, mc, nside)
        )
        cachefile_alm = (
            "cmb_cache/ffp10_cmb_{:03}_alm_mc_{:04}.fits".format(freq, mc, nside)
        )
        #if os.path.isfile(cachefile_map):
        #    print('{:4} : {} exists! Skipping.'.format(rank, cachefile_map))
        #    continue
        lmax = 2 * nside  # avoid aliasing
        if not os.path.isfile(cachefile_alm):
            t1 = time()
            if ntask == 1:
                print("    Loading", almfile1)
            alm1, mmax_file1 = hp.read_alm(almfile1, return_mmax=True)
            if ntask == 1:
                print("    Loading", almfile2)
            alm2, mmax_file2 = hp.read_alm(almfile2, return_mmax=True)
            lmax_file1 = hp.Alm.getlmax(len(alm1), mmax_file1)
            lmax_file2 = hp.Alm.getlmax(len(alm2), mmax_file2)
            if ntask == 1:
                print("    lmax_file1 = {}, mmax_file1 = {}".format(lmax_file1, mmax_file1))
                print("    lmax_file2 = {}, mmax_file2 = {}".format(lmax_file2, mmax_file2))
            alm1 = [alm1]
            alm2 = [alm2]
            if pol:
                for hdu in [2, 3]:
                    alm1.append(hp.read_alm(almfile1, hdu=hdu))
                    alm2.append(hp.read_alm(almfile2, hdu=hdu))
            t2 = time()
            print("{:4} :    Loaded in {:.2f} s".format(rank, t2 - t1))
            alm = np.vstack(alm1)
            alm2 = np.vstack(alm2)
            # Combine the alm
            for ialm in range(len(alm)):
                for ell in range(lmax_file2 + 1):
                    for m in range(ell + 1):
                        ind = hp.Alm.getidx(lmax_file1, ell, m)
                        ind2 = hp.Alm.getidx(lmax_file2, ell, m)
                        alm[ialm][ind] += alm2[ialm][ind2]
            # Remove intrinsic monopole and dipole
            if ntask == 1:
                print("    Suppressing monopole and dipole")
            t1 = time()
            for ell in range(min(2, lmax_file1 + 1)):
                for m in range(min(ell + 1, mmax_file1 + 1)):
                    ind = hp.Alm.getidx(lmax_file1, ell, m)
                    alm[0, ind] = 0
            t2 = time()
            if ntask == 1:
                print("    Suppressed in {:.2f} s".format(t2 - t1))
            """
            # 900-series only: scale up the polarization for transfer function
            if mc >= 900:
                if ntask == 1:
                    print("    Scaling up the CMB polarization for transfer function")
                t1 = time()
                alm[1] *= 100
                alm[2] *= 100
                t2 = time()
                if ntask == 1:
                    print("    Scaled in {:.2f} s".format(t2 - t1))
            """
            # Write the combined alm expansion
            if os.path.isfile(cachefile_alm):
                os.remove(cachefile_alm)
            hp.write_alm(cachefile_alm, alm, mmax_in=mmax_file1)
            print("{:4} : Wrote {}.".format(rank, cachefile_alm))
        else:
            alm = [hp.read_alm(cachefile_alm, 1)]
            alm.append(hp.read_alm(cachefile_alm, 2))
            alm.append(hp.read_alm(cachefile_alm, 3))
            alm = np.vstack(alm)
        # Make a quickpol-convolved CMB map to use as reference
        # if os.path.isfile(cachefile_map):
        #    os.remove(cachefile_map)
        # if os.path.isfile(cachefile_map):
        #    print('{:4} : {} exists! Skipping.'.format(rank, cachefile_map))
        #    continue
        nalm = len(alm)
        lmax_file = hp.Alm.getlmax(alm[0].size)
        lmax = min(lmax_file, lmax)
        mmax = min(lmax_file, lmax)
        if lmax < lmax_file:
            if ntask == 1:
                print(
                    "    Truncating the loaded alm from lmax = {} -> {}"
                    "".format(lmax_file, lmax)
                )
            sz = hp.Alm.getsize(lmax, mmax)
            new_alm = np.zeros([nalm, sz], dtype=np.complex)
            for ell in range(lmax + 1):
                for m in range(min(ell, mmax)):
                    i = hp.Alm.getidx(lmax, ell, m)
                    j = hp.Alm.getidx(lmax_file, ell, m)
                    new_alm[:, i] = alm[:, j]
            alm = new_alm
            t1 = time()
            if ntask == 1:
                print("    Truncated in {:.2f} s".format(t1 - t2))
        for mapfile, wlfile, blfile in [
                (cachefile_map, fname_wl, fname_bl),
                (cachefile_mapA, fname_wlA, fname_blA),
                (cachefile_mapB, fname_wlB, fname_blB)]:
            if os.path.isfile(mapfile):
                if ntask == 1:
                    print("    {} exists".format(mapfile))
                continue
            if ntask == 1:
                print("    Applying beam from {}".format(blfile))
            t2 = time()
            bl = hp.read_cl(blfile)
            # Apply TEB beam
            almout = alm.copy()
            for i in range(3):
                hp.almxfl(almout[i], bl[i], mmax=mmax, inplace=True)
            header = [("bl", blfile, "Beam window function")]
            if os.path.isfile(wlfile):
                wl = hp.read_cl(wlfile)
                # Add T->EB leakage
                alm_leak = np.zeros_like(alm)
                for i in range(1, 3):
                    alm_leak[i] = alm[0].copy()
                    w = wl[i].ravel().copy()
                    w[w < 0] = 0
                    w = np.sqrt(w)
                    hp.almxfl(alm_leak[i], w, mmax=mmax, inplace=True)
                almout += alm_leak
                header.append(("wl", wlfile, "Beam window function"))
            if ntask == 1:
                print("    Synthesizing alm to nside = {}".format(nside))
            m = hp.alm2map(list(almout), nside, pixwin=True, verbose=False, pol=pol)
            t1 = time()
            print("{:4} :    Synthesized in {:.2f} s".format(rank, t1 - t2))
            m = hp.reorder(m, r2n=True)
            t2 = time()
            print("{:4} :    Ordered to NEST in {:.2f} s".format(rank, t2 - t1))
            hp.write_map(
                mapfile, m * scale, extra_header=header, nest=True, overwrite=True
            )
            t1 = time()
            if ntask == 1:
                print("    Saved to {} in {:.2f} s".format(mapfile, t1 - t2))
            tstop = time()
            print(
                "{:4} : {} + {} -> {} in {:.2f} s".format(
                    rank, almfile1, almfile2, mapfile, tstop - tstart
                ),
                flush=True,
            )
    tstop_mc = time()
    if ntask == 1:
        print("\nMC = {} done in {:.2f} s".format(mc, tstop_mc - tstart_mc))

if comm is not None:
    comm.Barrier()
