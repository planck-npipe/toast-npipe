import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys

from toast_planck.reproc_modules import SkyModel
from toast.mpi import MPI

from toast_planck.utilities import freq_to_fwhm


comm = MPI.COMM_WORLD
rank = comm.rank

datadir = "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20"
auxdir = "/global/cfs/cdirs/cmb/data/planck2020/npipe/aux"
datadir3 = "/global/cfs/cdirs/cmb/data/planck2020/npipe/aux/npipe5v21_compsep"

delta = 0.01

# for freq in 30, 44, 70, 100, 143, 217, 353, 545, 857:
for freq in sys.argv[1:]:
    freq = int(freq)
    band = f"{freq:03}"
    flight = hp.read_map(f"{datadir}/npipe6v20_{band}_map.fits")
    fwhm = freq_to_fwhm(freq)
    quickpol = (f"{datadir}/quickpol/Bl_TEB_npipe6v20_{freq:03}GHzx{freq:03}GHz.fits")

    nside = hp.get_nside(flight)
    npix = 12 * nside ** 2
    fwhm_rad = np.radians(fwhm / 60)


    if freq == 100:
        co = hp.read_map(f"{datadir3}/co10_joint_2048_merged.smoothed.fits")
    elif freq == 217:
        co = hp.read_map(f"{datadir3}/co21_joint_2048_merged.smoothed.fits")
    elif freq == 353:
        co = hp.read_map(f"{datadir3}/co32_joint_2048_merged.smoothed.fits")
    else:
        co = None

    dipo = hp.ud_grade(
        hp.read_map(
            f"{auxdir}/dipole_nside{nside:04}.fits",
            nest=True,
            dtype=np.float32,
        ),
        nside,
        order_in="nest",
        order_out="ring",
    )

    zodi = hp.read_map(f"{auxdir}/commander_zodi_maps/commander_zodi_{band}.fits")

    cmb = (
        hp.read_map(
            f"{datadir3}/npipe5v21_compsep/cmb_npipe5v21_joint_fullres.fits",
            dtype=np.float32,
        )
        * 1e-6
    )
    if fwhm > 7:
        cmb = hp.smoothing(cmb, fwhm=fwhm_rad, lmax=2048, iter=0)
    cmb = hp.ud_grade(cmb, nside)

    if rank == 0:
        print("Creating sky model", flush=True)

    skymodel = SkyModel(
        nside,
        f"{datadir3}/synch_npipe5v21_joint_fullres.fits",
        f"{datadir3}/v6.2_pol/synch_c0001_k000001.smoothed.fits",
        f"{datadir3}/ff_npipe5v21_joint_fullres.smoothed.fits",
        f"{datadir3}/ame_npipe5v21_joint_fullres.smoothed.fits",
        f"{datadir3}/dust_npipe5v21_joint_fullres.fits",
        f"{datadir3}/v6.2_pol/dust_c0001_k000001.smoothed.fits",
        comm,
        fwhm=fwhm,
        quickpolbeam=quickpol,
        verbose=False,
    )

    mask = (
        hp.ud_grade(
            hp.read_map(f"{auxdir}/bandpass_mask_{band}.fits", nest=True),
            nside,
            order_in="nest",
            order_out="ring",
        )
        > 0.75
    )

    # evaluating

    if rank == 0:
        print("Evaluating", flush=True)

    cfreq = freq
    # Account for the changed main beam efficiencies.  The skymodel was derived from npipe5-data
    gain = {
        30: 1 / 0.992,
        44: 1 / 0.999,
        70: 1 / 0.994,
        100: 1 / 0.998,
        143: 1 / 0.998,
        217: 1 / 0.998,
        353: 1 / 0.9998,
        545: 1 / 0.9985,
        857: 1 / 0.9955,
    }[freq]

    for i in range(10):
        if rank == 0:
            print(f"{i} : cfreq = {cfreq}, gain = {gain}", flush=True)
        model = skymodel.eval(cfreq)[0] * gain
        deriv = (skymodel.eval(cfreq + delta)[0] * gain - model) / delta
        if rank == 0:
            target = flight - cmb - model
            templates = [np.ones(npix)[mask], dipo[mask], deriv[mask]]
            if zodi is not None:
                templates.append(zodi[mask])
            if co is not None:
                templates.append(co[mask])
            templates = np.vstack(templates)
            invcov = np.dot(templates, templates.T)
            cov = np.linalg.inv(invcov)
            proj = np.dot(templates, target[mask])
            coeff = np.dot(cov, proj)
            if co is None:
                offset, dipoamp, dfreq, zodiamp = coeff
                coamp = 0
            else:
                offset, dipoamp, dfreq, zodiamp, coamp = coeff
            print(
                f"offset = {offset}, dipo = {dipoamp}, dfreq = {dfreq}, "
                f"zodi = {zodiamp}, co : {coamp}",
                flush=True,
            )
        else:
            offset, dipoamp, dfreq, zodiamp, coamp = [None] * 5
        offset = comm.bcast(offset)
        dipoamp = comm.bcast(dipoamp)
        dfreq = comm.bcast(dfreq)
        zodiamp = comm.bcast(zodiamp)
        coamp = comm.bcast(coamp)
        cfreq += dfreq
        if i > 2 and np.abs(dfreq) < 0.01:
            break

    model = skymodel.eval(cfreq) * gain
    deriv = skymodel.eval(cfreq + delta) * gain

    if rank == 0:
        deriv = (deriv - model) / delta
        if zodi is not None:
            model[0] += zodiamp * zodi
        if co is not None:
            model[0] += coamp * co
        model = hp.reorder(model, r2n=True)
        deriv = hp.reorder(deriv, r2n=True)
        hdr = [
            ("cfreq", cfreq, "Central frequency [GHz]"),
            ("gain", gain, "Gain"),
            ("zodi", zodiamp, "Zodi template amplitude"),
            ("co", coamp, "CO template amplitude"),
        ]
        fname_out = (
            f"sky_model_{band}_nside{nside:04}_cfreq_smoothed.fits"
        )
        fname_out_deriv = (
            f"sky_model_deriv_{band}_nside{nside:04}_cfreq_smoothed.fits"
        )
        hp.write_map(
            fname_out,
            model,
            nest=True,
            extra_header=hdr,
            coord="G",
            overwrite=True,
            dtype=np.float32,
        )
        print("Wrote", fname_out, flush=True)
        hp.write_map(
            fname_out_deriv,
            deriv,
            nest=True,
            extra_header=hdr,
            coord="G",
            overwrite=True,
            dtype=np.float32,
        )
        print("Wrote", fname_out_deriv, flush=True)

comm.Barrier()
