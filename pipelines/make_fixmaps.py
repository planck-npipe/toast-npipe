import os
import sys

import numpy as np
import healpy as hp

if len(sys.argv) == 3:
    mcstart = int(sys.argv[1])
    mcstop = int(sys.argv[2])
else:
    mcstart = 800
    mcstop = 900

mapdir_in = "/global/cscratch1/sd/keskital/npipe_maps/npipe6v20"
mapdir_out = "fixmaps"

fn_mask = "/global/cscratch1/sd/keskital/hfi_pipe/clmask_30pc_ns{:4}.fits"

for freq in 100, 143, 217, 353:
    if freq < 100:
        nside = 1024
    else:
        nside = 2048
    npix = 12 * nside ** 2
    lmax = 3 * nside

    mask = hp.read_map(fn_mask.format(nside)) > 0.5
    fsky = np.sum(mask) / mask.size

    fn_clfix = "cl_fix_{:03}.fits".format(freq)
    clfix = np.atleast_2d(hp.read_cl(fn_clfix)) / 2

    lmax = clfix[0].size - 1
    while np.all(clfix[:2, lmax] < 1e-30):
        lmax -= 1
    print("lmax = {}".format(lmax), flush=True)

    for isubset, subset in enumerate("AB"):
        fn_wcov = os.path.join(
            mapdir_in + subset,
            "npipe6v20{}_{:03}_wcov_mcscaled.fits".format(subset, freq),
        )
        print("Reading", fn_wcov, flush=True)
        wcov = hp.read_map(fn_wcov, [0, 3, 5], verbose=False)

        for mc in range(mcstart, mcstop):
            fn_out = os.path.join(
                mapdir_out, "noisefix_{:03}{}_{:04}.fits".format(freq, subset, mc)
            )
            if os.path.isfile(fn_out):
                continue

            seed = 1000 * (freq + isubset) + mc
            np.random.seed(seed)
            noisemap_in = np.random.randn(3, npix)
            noisemap_in *= np.sqrt(wcov)
            noisealm = hp.map2alm(noisemap_in, lmax=lmax, iter=0)
            # noisecl = hp.alm2cl(noisealm)
            noisecl = (
                hp.anafast(noisemap_in * mask, lmax=2 * nside, iter=0, pol=True) / fsky
            )
            for i in range(3):
                noisealm[i] /= np.mean(noisecl[i][100:]) ** 0.5
            # noisecl2 = hp.alm2cl(noisealm)

            noisealm[0] = hp.almxfl(noisealm[0], clfix[0] ** 0.5)
            noisealm[1] = hp.almxfl(noisealm[1], clfix[1] ** 0.5)
            noisealm[2] = hp.almxfl(noisealm[2], clfix[2] ** 0.5)

            noisemap_out = hp.alm2map(noisealm, nside)
            print("Writing", fn_out, flush=True)
            hp.write_map(
                fn_out, hp.reorder(noisemap_out, r2n=True), coord="G", nest=True
            )

    for mc in range(mcstart, mcstop):
        fn_out = os.path.join(mapdir_out, "noisefix_{:03}_{:04}.fits".format(freq, mc))
        if os.path.isfile(fn_out):
            continue
        fna = os.path.join(
            mapdir_out, "noisefix_{:03}{}_{:04}.fits".format(freq, "A", mc)
        )
        fnb = os.path.join(
            mapdir_out, "noisefix_{:03}{}_{:04}.fits".format(freq, "B", mc)
        )
        if not os.path.isfile(fna):
            raise RuntimeError("File not found: " + fna)
        if not os.path.isfile(fnb):
            raise RuntimeError("File not found: " + fnb)
        print("Reading", fna, flush=True)
        noisemap_out = hp.read_map(fna, None, nest=True, verbose=False)
        print("Reading", fnb, flush=True)
        noisemap_out += hp.read_map(fnb, None, nest=True, verbose=False)
        noisemap_out *= 0.5
        print("Writing", fn_out, flush=True)
        hp.write_map(fn_out, noisemap_out, coord="G", nest=True)
