#!/usr/bin/env python
# $ -pe multicores 32
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
DBdir = "/sps/planck/Users/tristram/Planck/NPIPE"


# --------------------------------------------------------------
dataname = "npipe6v20"
galcut = 10
if len(sys.argv) >= 2:
    dataname = str(sys.argv[1])
if len(sys.argv) == 3:
    galcut = int(sys.argv[2])

if dataname == "PR3":
    MAPdir = "%s/Clean_%s_mask52_sync_ns512" % (DBdir, dataname)
    splits = ["O", "E"]
elif dataname == "npipe6v20":
    MAPdir = "%s/Clean_NPIPE_mask52_sync_ns512" % (DBdir)
    splits = ["A", "B"]
else:
    print("Error in dataname: %s" % dataname)
    exit()
# --------------------------------------------------------------

print("dataname: %s " % dataname)
print("galcut: %d" % galcut)

freq = [70, 100, 143, 217]
nmaps = len(freq)
ntags = 6

# beam
bl = npipe.cosbeam(nside)
clth = read_cl(
    "/sps/planck/Users/tristram/Models/planck_base_planck_2018_TTTEEElowllowE.fits"
)

# Masks
MASKFILE = (
    "/sps/planck/Users/tristram/Planck/Mask/lowl/mask_PR3_lowl_%dpc_ns512.fits"
    % (galcut)
)
mask = ud_grade(read_map(MASKFILE, verbose=False), nside_out=nside) > 0.5

print(MASKFILE)

# construct QML
ellbins = arange(2, lmax + 2)
cross = xqml.xQML(
    mask, ellbins, clth, bell=bl, lmax=lmax, temp=False, polar=True, corr=False
)
deltaN = 1e-16


cl = zeros((len(freq), len(freq), 6, lmax + 1))
for f1 in range(len(freq)):
    N1 = npipe.readcov(
        "%s/%s/lowres/npipe6v20_ncm_ns0016_smoothed_%03d_bin.dat"
        % (DBdir, dataname, freq[f1]),
        mask,
    )
    for f2 in range(f1, len(freq)):
        print("Compute xQML (%sx%s)" % (freq[f1], freq[f2]))
        N2 = npipe.readcov(
            "%s/%s/lowres/npipe6v20_ncm_ns0016_smoothed_%03d_bin.dat"
            % (DBdir, dataname, freq[f2]),
            mask,
        )

        dN = identity(cross.npix * cross.nstokes) * deltaN
        cross.construct_esti(N1 + dN, N2 + dN)

        m1 = npipe.smooth_and_degrade(
            "%s/%s_%03d_clean%s.fits" % (MAPdir, dataname, freq[f1], splits[0]),
            bl,
            nside,
        )
        m2 = npipe.smooth_and_degrade(
            "%s/%s_%03d_clean%s.fits" % (MAPdir, dataname, freq[f2], splits[1]),
            bl,
            nside,
        )
        clA = cross.get_spectra(m1, m2)

        m1 = npipe.smooth_and_degrade(
            "%s/%s_%03d_clean%s.fits" % (MAPdir, dataname, freq[f1], splits[1]),
            bl,
            nside,
        )
        m2 = npipe.smooth_and_degrade(
            "%s/%s_%03d_clean%s.fits" % (MAPdir, dataname, freq[f2], splits[0]),
            bl,
            nside,
        )
        clB = cross.get_spectra(m1, m2)

        cl[f1, f2, 1:3, 2:] = (clA + clB) / 2.0
        cl[f2, f1, 1:3, 2:] = (clA + clB) / 2.0

fits.writeto("%s/cl_%s_xqml_%dpc.fits" % (MAPdir, dataname, galcut), cl, overwrite=True)
