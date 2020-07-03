from __future__ import print_function
from __future__ import division
import healpy as hp
import healpy.projaxes as projaxes
import healpy.visufunc as visufunc
import healpy.rotator as rotator
import numpy as np
import pylab
import os

try:
    import pyfits
except:
    import astropy.io.fits as pyfits
from scipy.constants import degree, arcmin, arcsec, c

def plug_holes(m, verbose=False, in_place=True, nest=False):
    """
    Use simple downgrading to derive estimates of the missing pixel values
    """
    nbad_start = np.sum(np.isclose(m, hp.UNSEEN))

    if nbad_start == m.size:
        if verbose:
            print('plug_holes: All map pixels are empty. Cannot plug holes',
                  flush=True)
        return

    if nbad_start == 0:
        return

    nside = hp.get_nside(m)
    npix = m.size
    if nest:
        mnest = m.copy()
    else:
        mnest = hp.reorder(m, r2n=True)

    lowres = mnest
    nside_lowres = nside
    bad = np.isclose(mnest, hp.UNSEEN)
    while np.any(bad) and nside_lowres > 1:
        nside_lowres //= 2
        lowres = hp.ud_grade(lowres, nside_lowres, order_in='NESTED')
        hires = hp.ud_grade(lowres, nside, order_in='NESTED')
        bad = np.isclose(mnest, hp.UNSEEN)
        mnest[bad] = hires[bad]

    nbad_end = np.sum(bad)

    if nbad_end != 0:
        mn = np.mean(mnest[np.logical_not(bad)])
        mnest[bad] = mn

    if not in_place:
        m = m.copy()
    if nest:
        m[:] = mnest
    else:
        m[:] = hp.reorder(mnest, n2r=True)

    if verbose and nbad_start != 0:
        print(
            "Filled {} missing pixels ({:.2f}%), lowest resolution was "
            "Nside={}.".format(nbad_start, (100.0 * nbad_start) // npix, nside_lowres)
        )

    return


def get_dipole_map(nside):

    npix = 12 * nside ** 2

    solsys_velocity = 370.082  # km / s
    solsys_glon = 264.00  # galactic, degrees
    solsys_glat = 48.24
    TCMB = 2.72548  # Kelvin
    cinv = 1e3 / c  # Inverse light speed in km / s ( the assumed unit for velocity )

    baryvel = np.zeros(3)
    dipole_proj = solsys_velocity * np.sin((90 - solsys_glat) * degree)
    baryvel[0] = dipole_proj * np.cos(solsys_glon * degree)
    baryvel[1] = dipole_proj * np.sin(solsys_glon * degree)
    baryvel[2] = solsys_velocity * np.cos((90 - solsys_glat) * degree)

    pix = np.arange(npix, dtype=np.int32)
    pixdir = np.array(hp.pix2vec(nside, pix)).T

    proper = np.tile(baryvel, (npix, 1))  # We will ignore satellite velocity for now
    speed = np.sqrt(np.sum(proper ** 2, axis=1))
    invspeed = 1.0 / speed
    proper_dir = np.tile(invspeed, (3, 1)).T * proper
    beta = speed * cinv
    num = 1 - beta * np.sum(proper_dir * pixdir, axis=1)
    invgamma = np.sqrt(1 - beta ** 2)

    dipole = TCMB / num * invgamma - TCMB

    return dipole


def read_map(fname):

    if not os.path.isfile(fname):
        raise Exception("File does not exist: " + fname)

    h = pyfits.open(fname, "readonly")

    nside = h[1].header["NSIDE"]
    npix = 12 * nside ** 2

    if "grain" in h[1].header:
        grain = h[1].header["GRAIN"]
    else:
        grain = 0

    if grain > 1:
        raise Exception("Grain > 1 not implemented!")

    nmap = h[1].header["TFIELDS"]
    if grain != 0:
        ind = h[1].data.field(0)
        nmap -= 1

    # Rename the columns to make sure there are no duplicate names
    for i in range(1, nmap + 1):
        s = str(i)
        h[1].header["TTYPE" + s] = "column" + s

    m = np.zeros([nmap, npix])

    for imap in range(nmap):
        if grain == 0:
            m[imap] = h[1].data.field(imap).ravel()
        elif grain == 1:
            m[imap] += hp.UNSEEN
            m[imap][ind] = h[1].data.field(imap + 1)

    h.close()

    return m


detsets = {
    "030": 30,
    "044": 44,
    "070": 70,
    "070_18_23": ["LFI18M", "LFI18S", "LFI23M", "LFI23S"],
    "070_19_22": ["LFI19M", "LFI19S", "LFI22M", "LFI22S"],
    "070_20_21": ["LFI20M", "LFI20S", "LFI21M", "LFI21S"],
    "100": 100,
    "100_1_4": ["100-1a", "100-1b", "100-4a", "100-4b"],
    "100_2_3": ["100-2a", "100-2b", "100-3a", "100-3b"],
    "143": 143,
    "143_1_3": ["143-1a", "143-1b", "143-3a", "143-3b"],
    "143_2_4": ["143-2a", "143-2b", "143-4a", "143-4b"],
    "143_5": ["143-5"],
    "143_6": ["143-6"],
    "143_7": ["143-7"],
    "217": 217,
    "217_1": ["217-1"],
    "217_2": ["217-2"],
    "217_3": ["217-3"],
    "217_4": ["217-4"],
    "217_5_7": ["217-5a", "217-5b", "217-7a", "217-7b"],
    "217_6_8": ["217-6a", "217-6b", "217-8a", "217-8b"],
    "353": 353,
    "353_1": ["353-1"],
    "353_2": ["353-2"],
    "353_5_7": ["353-3a", "353-3b", "353-5a", "353-5b"],
    "353_6_8": ["353-4a", "353-4b", "353-6a", "353-6b"],
    "353_7": ["353-7"],
    "353_8": ["353-8"],
    "545": 545,
    "545_1": ["545-1"],
    "545_2": ["545-2"],
    "545_4": ["545-4"],
    "857": 857,
    "857_1": ["857-1"],
    "857_2": ["857-2"],
    "857_3": ["857-3"],
    "857_4": ["857-4"],
}


def list_planck(detset, good=True, subset=None, extend_857=True, extend_545=False):
    detectors = []
    if subset == None:
        subset = 0

    if detset in (30, "30", "030", "30GHz"):
        horns = range(27, 29)
        instrument = "LFI"
    elif detset in (44, "44", "044", "44GHz"):
        horns = range(24, 27)
        instrument = "LFI"
    elif detset in (70, "70", "070", "70GHz"):
        horns = range(18, 24)
        if subset == 1:
            horns = [18, 23]
        elif subset == 2:
            horns = [19, 22]
        elif subset == 3:
            horns = [20, 21]
        instrument = "LFI"
    elif isinstance(detset, str) and detset.upper() == "LFI":
        detectors.extend(list_planck(30, good=good))
        detectors.extend(list_planck(44, good=good))
        detectors.extend(list_planck(70, good=good))
        return detectors
    elif detset in (100, "100", "100GHz"):
        psb_horns = range(1, 5)
        swb_horns = []
        if subset == 1:
            psb_horns = [1, 4]
        elif subset == 2:
            psb_horns = [2, 3]
        instrument = "HFI"
        freq = "100-"
    elif detset in (143, "143", "143GHz"):
        psb_horns = np.arange(1, 5)
        if good:
            swb_horns = range(5, 8)
        else:
            swb_horns = range(5, 9)
        if subset == 1:
            psb_horns, swb_horns = [1, 3], []
        elif subset == 2:
            psb_horns, swb_horns = [2, 4], []
        elif subset == 3:
            psb_horns, swb_horns = [], [5, 6, 7]
        instrument = "HFI"
        freq = "143-"
    elif detset in (217, "217", "217GHz"):
        psb_horns = np.arange(5, 9)
        swb_horns = np.arange(1, 5)
        if subset == 1:
            psb_horns, swb_horns = [5, 7], []
        elif subset == 2:
            psb_horns, swb_horns = [6, 8], []
        elif subset == 3:
            psb_horns, swb_horns = [], [1, 2, 3, 4]
        instrument = "HFI"
        freq = "217-"
    elif detset in (353, "353", "353GHz"):
        psb_horns = np.arange(3, 7)
        swb_horns = [1, 2, 7, 8]
        if subset == 1:
            psb_horns, swb_horns = [3, 5], []
        elif subset == 2:
            psb_horns, swb_horns = [4, 6], []
        elif subset == 3:
            psb_horns, swb_horns = [], [1, 2, 7, 8]
        instrument = "HFI"
        freq = "353-"
    elif detset in (545, "545", "545GHz"):
        psb_horns = []
        if good and not extend_545:
            swb_horns = [1, 2, 4]
        else:
            swb_horns = np.arange(1, 5)
        instrument = "HFI"
        freq = "545-"
    elif detset in (857, "857", "857GHz"):
        psb_horns = []
        if good and not extend_857:
            swb_horns = [1, 2, 3]
        else:
            swb_horns = np.arange(1, 5)
        instrument = "HFI"
        freq = "857-"
    elif isinstance(detset, str) and detset.upper() == "HFI":
        detectors.extend(list_planck(100, good=good, extend_857=extend_857))
        detectors.extend(list_planck(143, good=good, extend_857=extend_857))
        detectors.extend(list_planck(217, good=good, extend_857=extend_857))
        detectors.extend(list_planck(353, good=good, extend_857=extend_857))
        detectors.extend(list_planck(545, good=good, extend_857=extend_857))
        detectors.extend(list_planck(857, good=good, extend_857=extend_857))
        return detectors
    elif isintance(detset, str) and detset.upper() == "PLANCK":
        detectors.extend(list_planck("LFI", good=good, extend_857=extend_857))
        detectors.extend(list_planck("HFI", good=good, extend_857=extend_857))
        return detectors
    elif isinstance(detset, str) and detset.upper() == "ROWS":
        if True:
            return [
                ["LFI27M", "LFI27S", "LFI28M", "LFI28S"],
                ["LFI24M", "LFI24S"],
                ["LFI25M", "LFI25S", "LFI26M", "LFI26S"],
                ["LFI18M", "LFI23S"],
                ["LFI19M", "LFI22S"],
                ["LFI20M", "LFI21S"],
                ["100-1a", "100-1b", "100-4a", "100-4b"],
                ["100-2a", "100-2b", "100-3a", "100-3b"],
                ["143-1a", "143-1b", "143-3a", "143-3b"],
                ["143-2a", "143-2b", "143-4a", "143-4b"],
                ["143-5", "143-7"],
                ["143-6"],
                ["217-1", "217-3"],
                ["217-2", "217-4"],
                ["217-5a", "217-5b", "217-7a", "217-7b"],
                ["217-6a", "217-6b", "217-8a", "217-8b"],
                ["353-1", "353-7"],
                ["353-3a", "353-3b", "353-5a", "353-5b"],
                ["353-4a", "353-4b", "353-6a", "353-6b"],
                ["353-2", "353-8"],
                ["545-1"],
                ["545-2", "545-4"],
                ["857-1", "857-3"],
                ["857-2"],
            ]
        else:
            return [
                list_planck(30),
                ["LFI24M", "LFI24S"],
                ["LFI25M", "LFI25S", "LFI26M", "LFI26S"],
                list_planck(70),
                list_planck(100),
                [
                    "143-1a",
                    "143-1b",
                    "143-2a",
                    "143-2b",
                    "143-3a",
                    "143-3b",
                    "143-4a",
                    "143-4b",
                ],
                ["143-5", "143-7", "143-6"],
                ["217-1", "217-2", "217-3", "217-4"],
                [
                    "217-5a",
                    "217-5b",
                    "217-6a",
                    "217-6b",
                    "217-7a",
                    "217-7b",
                    "217-8a",
                    "217-8b",
                ],
                list_planck(353),
                list_planck(545),
                list_planck(857, extend_857=extend_857),
            ]
    else:
        print("ERROR: unknown detector set: ", detset)
        return -1

    if instrument == "LFI":
        for horn in horns:
            for arm in ["S", "M"]:
                detectors.append("LFI" + str(horn) + arm)
    elif instrument == "HFI":
        for horn in psb_horns:
            for arm in ["a", "b"]:
                detectors.append(freq + str(horn) + arm)
        for horn in swb_horns:
            detectors.append(freq + str(horn))

    return detectors


def det2detset(det):
    if "LFI27" in det:
        return "030_27"
    if "LFI28" in det:
        return "030_28"

    if "LFI25" in det or "LFI26" in det:
        return "044_25_26"
    if "LFI24" in det:
        return "044_24"

    if "LFI18" in det or "LFI23" in det:
        return "070_18_23"
    if "LFI19" in det or "LFI22" in det:
        return "070_19_22"
    if "LFI20" in det or "LFI21" in det:
        return "070_20_21"

    if "100-1" in det or "100-4" in det:
        return "100_1_4"
    if "100-2" in det or "100-3" in det:
        return "100_2_3"

    if "143-1" in det or "143-3" in det:
        return "143_1_3"
    if "143-2" in det or "143-4" in det:
        return "143_2_4"
    if "143" in det:
        return "143_swb"

    if "217-5" in det or "217-7" in det:
        return "217_5_7"
    if "217-6" in det or "217-8" in det:
        return "217_6_8"
    if "217" in det:
        return "217_swb"

    if "353-3" in det or "353-5" in det:
        return "353_3_5"
    if "353-4" in det or "353-6" in det:
        return "353_4_6"
    if "353" in det:
        return "353_swb"

    if "545" in det:
        return det.replace("-", "_")

    if "857" in det:
        return det.replace("-", "_")

    return None


def det2row(det):
    if "LFI25" in det or "LFI26" in det:
        return "044_25_26"
    if "LFI24" in det:
        return "044_24"

    if "LFI18" in det or "LFI23" in det:
        return "070_18_23"
    if "LFI19" in det or "LFI22" in det:
        return "070_19_22"
    if "LFI20" in det or "LFI21" in det:
        return "070_20_21"

    if "100-1" in det or "100-4" in det:
        return "100_1_4"
    if "100-2" in det or "100-3" in det:
        return "100_2_3"

    if "143-1" in det or "143-2" in det or "143-3" in det or "143-4" in det:
        return "143_psb"
    if "143" in det:
        return "143_swb"

    if "217-5" in det or "217-6" in det or "217-7" in det or "217-8" in det:
        return "217_psb"
    if "217" in det:
        return "217_swb"

    # if '353' in det: return '353'
    # if '545' in det: return '545'
    # if '857' in det: return '857'

    return None


def to_bolo_id(bolo):
    bolo_ids = {
        "100-1a": "00_100_1a",
        "100-1b": "01_100_1b",
        "100-2a": "20_100_2a",
        "100-2b": "21_100_2b",
        "100-3a": "40_100_3a",
        "100-3b": "41_100_3b",
        "100-4a": "80_100_4a",
        "100-4b": "81_100_4b",
        "143-1a": "02_143_1a",
        "143-1b": "03_143_1b",
        "143-2a": "30_143_2a",
        "143-2b": "31_143_2b",
        "143-3a": "50_143_3a",
        "143-3b": "51_143_3b",
        "143-4a": "82_143_4a",
        "143-4b": "83_143_4b",
        "143-5": "10_143_5",
        "143-6": "42_143_6",
        "143-7": "60_143_7",
        "143-8": "70_143_8",
        "217-5a": "11_217_5a",
        "217-5b": "12_217_5b",
        "217-6a": "43_217_6a",
        "217-6b": "44_217_6b",
        "217-7a": "61_217_7a",
        "217-7b": "62_217_7b",
        "217-8a": "71_217_8a",
        "217-8b": "72_217_8b",
        "217-1": "04_217_1",
        "217-2": "22_217_2",
        "217-3": "52_217_3",
        "217-4": "84_217_4",
        "353-3a": "23_353_3a",
        "353-3b": "24_353_3b",
        "353-4a": "32_353_4a",
        "353-4b": "33_353_4b",
        "353-5a": "53_353_5a",
        "353-5b": "54_353_5b",
        "353-6a": "63_353_6a",
        "353-6b": "64_353_6b",
        "353-1": "05_353_1",
        "353-2": "13_353_2",
        "353-7": "45_353_7",
        "353-8": "85_353_8",
        "545-1": "14_545_1",
        "545-2": "34_545_2",
        "545-3": "55_545_3",
        "545-4": "73_545_4",
        "857-1": "25_857_1",
        "857-2": "35_857_2",
        "857-3": "65_857_3",
        "857-4": "74_857_4",
    }
    if bolo not in bolo_ids:
        print("ERROR: " + bolo + " is not a valid bolometer")
        return None

    return bolo_ids[bolo]


def to_bolo(bolo_id):
    parts = bolo_id.split("_")
    return parts[1] + "-" + parts[2]


def get_pair(det):
    if det.endswith("M"):
        return det.replace("M", "S")
    elif det.endswith("S"):
        return det.replace("S", "M")
    elif det.endswith("a"):
        return det.replace("a", "b")
    elif det.endswith("b"):
        return det.replace("b", "a")
    else:
        return None


def horn2freq(horn):
    ihorn = int(horn)
    if ihorn not in range(18, 29):
        raise Exception("Not a valid LFI horn")
    if ihorn < 24:
        return 70
    elif ihorn < 27:
        return 44
    else:
        return 30


def det2freq(det):
    if "LFI" in str(det):
        horn = det[3:5]
        return horn2freq(horn)
    else:
        return int(det[0:3])


def det2fsample(det):
    freq = det2freq(det)
    if freq == 30:
        return 32.5079
    elif freq == 44:
        return 46.5455
    elif freq == 70:
        return 78.7692
    else:
        return 180.374


def input_map(file_map):
    import pycfitsio

    f = pycfitsio.open(file_map)
    map = f[f.HDUs.keys()[0]].read_column(0)
    f.close()
    return map


def crosshair(phi0, theta0, radii, fmt="k-"):
    """Add a cross-hair to a healpix plot

    Arguments:
    phi0 -- longitude of the target [degrees]
    theta0 -- colatitude of the target [degrees]
    radii -- sizes of the cocentric circles [arc min]
    """
    deg2rad = np.pi / 180

    # construct a rotation from the North pole to the target
    rot1 = rotator.Rotator(rot=-phi0)
    rot2 = rotator.Rotator(rot=[0, 90 - theta0])
    rot = rot1 * rot2

    # meridian
    theta = np.arange(-90, 90, 0.1)
    visufunc.projplot(theta * 0 + phi0, theta, "k-", lonlat=True)

    # iso-latitude curve
    phi = np.arange(0, 360, 0.1)
    visufunc.projplot(phi, phi * 0 + theta0, "k-", lonlat=True)

    # circles
    for radius in radii:
        # first define the circle around the North pole
        phi = np.arange(0, 360, 0.1) * deg2rad
        theta = (0 * phi + radius / 60) * deg2rad
        # then rotate it to circle the target
        # notice that theta and phi flip without lonlat=true
        theta, phi = rot(theta, phi)
        # and plot
        visufunc.projplot(theta, phi, fmt)


def get_plot_ranges(map, longitude, latitude, inner_radius, outer_radius):
    """ get good plotting ranges.

    Arguments:
    map -- the single column map to use
    longitude -- target longitude [degrees]
    latitude -- target co-latitude [degrees]
    inner_radius -- radius of disc to use for peak-to-peak [arc min]
    outer_radius -- radius of disc to use for background offset [arc min]
    """

    npix = len(map)
    nside = hp.npix2nside(npix)

    phi = longitude * np.pi / 180
    theta = (90 - latitude) * np.pi / 180
    vec0 = hp.ang2vec(theta, phi)

    inner = hp.query_disc(nside, vec0, inner_radius / 60, nest=True, deg=True)
    aperture = map[inner]
    aperture = aperture[np.where(aperture != hp.UNSEEN)]
    if len(aperture) == 0:
        raise ValueError("Empty aperture")
    aperture_min = aperture.min()
    aperture_max = aperture.max()

    outer = hp.query_disc(nside, vec0, outer_radius / 60, nest=True, deg=True)
    annulus = map[np.setdiff1d(outer, inner)]
    annulus = annulus[np.where(annulus != hp.UNSEEN)]
    if len(annulus) == 0:
        raise ValueError("Empty annulus")

    aperture_median = np.median(annulus)
    amplitude = (
        aperture_max - aperture_median
    ) * 0.9  # saturate the color scale a little bit

    plot_min = aperture_median - amplitude
    plot_max = aperture_median + amplitude

    return plot_min, plot_max


def log_bin(data, nbin=100):
    # To get the bin positions, you must call log_bin twice: first with x and then y vectors
    n = len(data)

    ind = np.arange(n) + 1

    bins = np.logspace(
        np.log(ind[0]), np.log(ind[-1]), num=nbin + 1, endpoint=True, base=np.e
    )

    locs = np.digitize(ind, bins)

    hits = np.zeros(nbin + 2, dtype=np.integer)
    binned = np.zeros(nbin + 2)

    # Always merge the last two bins to avoid undersampling
    locs[locs == nbin + 1] = nbin

    for i, ibin in enumerate(locs):
        hits[ibin] += 1
        binned[ibin] += data[i]

    ind = hits > 0
    binned[ind] /= hits[ind]

    return binned[ind], hits[ind]


def log_bin_complex(data, nbin=100):
    # To get the bin positions, you must call log_bin twice: first with x and then y vectors
    n = len(data)

    ind = np.arange(n) + 1

    bins = np.logspace(
        np.log(ind[0]), np.log(ind[-1]), num=nbin + 1, endpoint=True, base=np.e
    )

    locs = np.digitize(ind, bins)

    hits = np.zeros(nbin + 2, dtype=np.integer)
    binned = np.zeros(nbin + 2)

    for ibin in range(nbin + 2):
        ind = locs == ibin
        nhit = np.sum(ind)
        if nhit > 0:
            binned[ibin] = np.std(np.append(data[ind].real, data[ind].imag)) * np.sqrt(
                2.0
            )
            hits[ibin] = nhit

    ind = hits > 0

    return binned[ind], hits[ind]
