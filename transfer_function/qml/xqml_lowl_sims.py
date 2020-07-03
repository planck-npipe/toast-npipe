#!/usr/bin/env python
# $ -pe multicores 16
# $ -q mc_long
# $ -N xqml_sims
# $ -l h_rss=3500M
# $ -l sps=1
# $ -R y
# $ -j y
# $ -V
# $ -o /sps/planck/Users/tristram/Soft/Planck/NPIPE/output
# $ -e /sps/planck/Users/tristram/Soft/Planck/NPIPE/output
from numpy import *
import sys
import os

sys.path.insert(0, "/sps/planck/Users/tristram/Soft/LAL/libs/python")
sys.path.insert(0, "/sps/planck/Users/tristram/Soft/lib/python2.7/site-packages")
os.environ["OMP_NUM_THREADS"] = (
    os.environ["NSLOTS"] if os.environ.has_key("NSLOTS") else "16"
)

from healpy import *
from astropy.io import fits

# from planck import *
import xqml
import npipe

nside = 16
lmax = 45

bsimu = 0
esimu = 100

# datatype = "FFP10"; esimu=300
datatype = "NPIPE"
esimu = 100

galcut = 10

freq = [70, 100, 143, 217]

# beam
bl = npipe.cosbeam(nside)

clth = read_cl(
    "/sps/planck/Users/tristram/Models/planck_base_planck_2018_TTTEEElowllowE.fits"
)

if datatype == "FFP10":
    DBdir = (
        "/sps/planck/Users/tristram/Planck/NPIPE/Simu_%s_mask52_skymodel_ns512"
        % datatype
    )
    MAPdir = "%s/cleanmaps/ring" % (DBdir)
else:
    DBdir = (
        "/sps/planck/Users/tristram/Planck/NPIPE/Simu_%s_mask52_sync_ns512" % datatype
    )
    COVdir = "/sps/planck/Users/tristram/Planck/NPIPE/npipe6v19/lowres"
    MAPdir = "%s/cleanmaps" % (DBdir)

# CLdir  = "%s/xqml_%dpc" % (DBdir,galcut)
# os.system( 'mkdir '+CLdir)

nmaps = len(freq)
ntags = 6

# Masks
MASKFILE = (
    "/sps/planck/Users/tristram/Planck/Mask/lowl/mask_PR3_lowl_%dpc_ns512.fits"
    % (galcut)
)
mask = ud_grade(read_map(MASKFILE, verbose=False), nside_out=nside) > 0.5

print(MASKFILE)


# construct QML
print("Construct QML")
ellbins = arange(2, lmax + 2)
cross = xqml.xQML(
    mask, ellbins, clth, bell=bl, lmax=lmax, temp=False, polar=True, corr=False
)
deltaN = 1e-16


allcl = zeros((esimu - bsimu, len(freq), len(freq), 6, lmax + 1))
for f1 in range(len(freq)):
    N1 = (
        npipe.readcov(
            "%s/npipe6v19_ncm_ns0016_smoothed_%03d_bin.dat" % (COVdir, freq[f1]), mask
        )
        + identity(cross.npix * cross.nstokes) * deltaN
    )
    for f2 in range(f1, len(freq)):
        print("Compute xQML (%sx%s)" % (freq[f1], freq[f2]))
        N2 = (
            npipe.readcov(
                "%s/npipe6v19_ncm_ns0016_smoothed_%03d_bin.dat" % (COVdir, freq[f2]),
                mask,
            )
            + identity(cross.npix * cross.nstokes) * deltaN
        )
        cross.construct_esti(N1, N2)

        for n in range(bsimu, esimu):
            print("Simu %d" % n)

            if datatype == "NPIPE":
                simuname = "npipe6v19_map_mc_%04d" % n
                mapfile1 = "npipe6v19_%03d_map_%04d" % (freq[f1], n)
                mapfile2 = "npipe6v19_%03d_map_%04d" % (freq[f2], n)
            if datatype == "FFP10":
                simuname = "full_map_mc_%05d" % n
                mapfile1 = "%03d_%s" % (freq[f1], simuname)
                mapfile2 = "%03d_%s" % (freq[f2], simuname)

            m1 = npipe.smooth_and_degrade(
                "%s/%s_clean1.fits" % (MAPdir, mapfile1), bl, nside
            )
            m2 = npipe.smooth_and_degrade(
                "%s/%s_clean2.fits" % (MAPdir, mapfile2), bl, nside
            )
            clA = cross.get_spectra(m1, m2)

            m1 = npipe.smooth_and_degrade(
                "%s/%s_clean2.fits" % (MAPdir, mapfile1), bl, nside
            )
            m2 = npipe.smooth_and_degrade(
                "%s/%s_clean1.fits" % (MAPdir, mapfile2), bl, nside
            )
            clB = cross.get_spectra(m1, m2)

            allcl[n, f1, f2, 1:3, 2:] = (clA + clB) / 2.0
            allcl[n, f2, f1, 1:3, 2:] = (clA + clB) / 2.0

fits.writeto(
    "%s/allcl_npipe6v19_xqml_%dpc.fits" % (DBdir, galcut), allcl, overwrite=True
)
