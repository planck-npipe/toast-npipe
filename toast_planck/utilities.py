# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import namedtuple
import gc
import io
import os
import pickle
import re
from scipy.constants import degree
import sqlite3
from time import time

from toast.tod.interval import Interval
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers

import astropy.io.fits as pf
import healpy as hp
import numpy as np
import toast.qarray as qa


XAXIS, YAXIS, ZAXIS = np.eye(3)
SPINANGLE = 85.0 * degree
SPINROT = qa.rotation(YAXIS, np.pi / 2 - SPINANGLE)

ringdb_cache = {}

# define a named tuple to contain detector data and RIMO in a standardized
# format

DetectorData = namedtuple(
    "DetectorData",
    "detector phi_uv theta_uv psi_uv psi_pol epsilon fsample "
    "fknee alpha net fwhm quat",
)

past_queries = {}

# Relative differences [GHz] to effective cental frequency from npipe6v20
cfreq_deltas = {
    "LFI27M": -0.067,
    "LFI27S": 0.325,
    "LFI28M": 0.256,
    "LFI28S": -0.514,
    "LFI24M": -0.312,
    "LFI24S": 0.065,
    "LFI25M": 0.198,
    "LFI25S": 0.284,
    "LFI26M": -0.161,
    "LFI26S": 0.023,
    "LFI18M": -0.343,
    "LFI18S": -0.499,
    "LFI19M": -2.018,
    "LFI19S": -0.036,
    "LFI20M": -1.041,
    "LFI20S": -0.357,
    "LFI21M": 0.730,
    "LFI21S": -1.665,
    "LFI22M": 2.149,
    "LFI22S": 2.294,
    "LFI23M": 0.227,
    "LFI23S": 0.559,
    "100-1a": -0.990,
    "100-1b": -0.863,
    "100-2a": 0.408,
    "100-2b": 0.347,
    "100-3a": 0.048,
    "100-3b": -0.283,
    "100-4a": 0.512,
    "100-4b": 0.822,
    "143-1a": -0.658,
    "143-1b": 0.060,
    "143-2a": -0.919,
    "143-2b": -0.535,
    "143-3a": -0.888,
    "143-3b": -0.054,
    "143-4a": 0.041,
    "143-4b": -0.344,
    "143-5": 1.153,
    "143-6": 0.726,
    "143-7": 1.418,
    "217-1": 1.265,
    "217-2": 1.416,
    "217-3": 1.853,
    "217-4": 1.364,
    "217-5a": -0.711,
    "217-5b": -0.488,
    "217-6a": -0.490,
    "217-6b": -0.191,
    "217-7a": -1.094,
    "217-7b": -1.057,
    "217-8a": -1.298,
    "217-8b": -0.571,
    "353-1": -0.094,
    "353-2": 0.006,
    "353-3a": -0.904,
    "353-3b": -1.235,
    "353-4a": 1.749,
    "353-4b": 2.160,
    "353-5a": -2.118,
    "353-5b": -1.917,
    "353-6a": -1.082,
    "353-6b": -4.644,
    "353-7": 2.904,
    "353-8": 5.175,
    "545-1": 1.421,
    "545-2": -1.454,
    "545-4": 0.033,
    "857-1": -0.498,
    "857-2": 0.503,
    "857-3": 0.489,
    "857-4": -0.494,
}


def list_files(path, cache=True):
    """
    Generate a list of files with full paths
    """

    fn_cache = "cached_listing_" + path.replace("/", "_").replace("\\", "_") + ".pck"
    if os.path.isfile(fn_cache) and cache:
        print("Loading list of cached files from {}".format(fn_cache), flush=True)
        files = pickle.load(open(fn_cache, "rb"))
    else:
        files = []
        # Use os.scandir instead of os.walk because it is more efficient
        with os.scandir(path) as it:
            ndir = 0
            nfile = 0
            for entry in it:
                if entry.is_dir():
                    if "AVR" in entry.name:
                        continue  # No compression diagnostics
                    files += list_files(entry.path, cache=False)
                    ndir += 1
                elif entry.is_file():
                    files.append(entry.path)
                    nfile += 1
            print(f"Found {nfile} files {ndir} and directories in {path}", flush=True)
        if cache:
            print("Writing list of cached files to {}".format(fn_cache), flush=True)
            pickle.dump(files, open(fn_cache, "wb"))

    return files


def list_planck(detset, good=True, subset=None, extend_857=True, extend_545=False):
    detectors = []
    if subset is None:
        subset = 0

    if detset in (30, "30", "030", "30GHz", "030GHz", "30A", "030A", "30B", "030B"):
        horns = range(27, 29)
        instrument = "LFI"
    elif detset in (44, "44", "044", "44GHz", "044GHz", "44A", "044A", "44B", "044B"):
        horns = range(24, 27)
        instrument = "LFI"
    elif detset in (70, "70", "070", "70GHz", "070GHz"):
        horns = range(18, 24)
        if subset == 1:
            horns = [18, 23]
        elif subset == 2:
            horns = [19, 22]
        elif subset == 3:
            horns = [20, 21]
        instrument = "LFI"
    elif detset in ["70A", "070A"]:
        horns = [18, 20, 23]
        instrument = "LFI"
    elif detset in ["70B", "070B"]:
        horns = [19, 21, 22]
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
    elif detset == "100A":
        psb_horns = [1, 4]
        swb_horns = []
        instrument = "HFI"
        freq = "100-"
    elif detset == "100B":
        psb_horns = [2, 3]
        swb_horns = []
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
    elif detset == "143A":
        psb_horns = [1, 3]
        swb_horns = [5, 7]
        instrument = "HFI"
        freq = "143-"
    elif detset == "143B":
        psb_horns = [2, 4]
        swb_horns = [6]
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
    elif detset == "217A":
        psb_horns = [5, 7]
        swb_horns = [1, 3]
        instrument = "HFI"
        freq = "217-"
    elif detset == "217B":
        psb_horns = [6, 8]
        swb_horns = [2, 4]
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
    elif detset == "353A":
        psb_horns = [3, 5]
        swb_horns = [1, 7]
        instrument = "HFI"
        freq = "353-"
    elif detset == "353B":
        psb_horns = [4, 6]
        swb_horns = [2, 8]
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
    elif detset == "545A":
        psb_horns = []
        swb_horns = [1]
        instrument = "HFI"
        freq = "545-"
    elif detset == "545B":
        psb_horns = []
        swb_horns = [2, 4]
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
    elif detset == "857A":
        psb_horns = []
        swb_horns = [1, 3]
        instrument = "HFI"
        freq = "857-"
    elif detset == "857B":
        psb_horns = []
        swb_horns = [2, 4]
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
    elif isinstance(detset, str) and detset.upper() == "PLANCK":
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
        # single detectors and horns
        lfidets = list_planck("LFI")
        hfidets = list_planck("HFI")
        if detset in lfidets or detset in hfidets:
            return [detset]
        if detset + "M" in lfidets:
            return [detset + "M", detset + "S"]
        if detset + "a" in hfidets:
            return [detset + "a", detset + "b"]
        # All other cases
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


def remove_bright_sources(m, nest=False, fwhm=60, in_place=True):
    """
    Remove known bright sources from a map.  Useful for regularizing
    bandpass mismatch templates.
    """
    if not in_place:
        m = m.copy()
    sources = {
        # Coordinates are from http://simbad.u-strasbg.fr
        # name : (lon [deg], lat [deg], radius [arc min]
        "Crab": (184.5575, -05.7844, 7),
        "3C 405": (76.1899, 5.7554, 1.5),
        "3C 454.3": (86.1111, -38.1838, 1),
        "3C 273": (289.9508, 64.3600, 1),
        "3C 279": (305.1043, 57.0624, 1),
        "Cen A": (309.5159, 19.4173, 10),
        "QSO J0635-7516": (286.3683, -27.1584, 1),
        "3C 84": (150.5758, -13.2612, 3),
        "QSO B1921-293": (9.3441, -19.6068, 1),
        "30 Dor": (-80.532, -31.6720, 10),
        "QSO B0537-441": (250.0828, -31.0896, 1),
        "QSO B0420-0127": (195.2901, -33.1399, 1),
        "M87": (283.7777, 74.4912, 5),
        "7C 164117.60+395412.00": (63.4550, 40.9489, 1),
        "7C 163330.69+381410.00": (61.0856, 42.3364, 1),
        "ICRF J184916.0+670541": (97.4913, 25.0437, 1),
        "QSO J0854+2006": (206.8121, 35.8209, 1),
        "4C 01.28": (251.5106, 52.7740, 1),
        "ICRF J092703.0+390220": (183.7085, 46.1637, 1),
        "ICRF J192748.4+735801": (105.6259, 23.5409, 1),
        "QSO J0403-3605": (237.7429, -48.4833, 1),
        "QSO B0521-365": (240.6077, -32.7160, 1),
        "QSO B1334-127": (320.0240, 48.3749, 1),
        "3C 446": (58.9597, -48.8428, 1),
    }
    nside = hp.get_nside(m)
    for name in sources:
        lon, lat, radius = sources[name]
        total_radius = np.radians((fwhm + radius) / 60)
        vec = hp.ang2vec(lon, lat, lonlat=True)
        pix = hp.query_disc(nside, vec, total_radius, inclusive=False, nest=nest)
        for mm in np.atleast_2d(m):
            mm[pix] = hp.UNSEEN
    plug_holes(m, nest=nest)
    return m


def plug_holes(m, verbose=False, in_place=True, nest=False):
    """
    Use simple downgrading to derive estimates of the missing pixel values
    """
    nbad_start = np.sum(np.isclose(m, hp.UNSEEN))

    if nbad_start == m.size:
        if verbose:
            print("plug_holes: All map pixels are empty. Cannot plug holes", flush=True)
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
        lowres = hp.ud_grade(lowres, nside_lowres, order_in="NESTED")
        hires = hp.ud_grade(lowres, nside, order_in="NESTED")
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
            "plug_holes: Filled {} missing pixels ({:.2f}%), lowest "
            "resolution was Nside={}.".format(
                nbad_start, (100.0 * nbad_start) // npix, nside_lowres
            )
        )
    return m


ADU2Volt = {
    "LFI2700": (5.491826e-05, 0.958340),
    "LFI2701": (5.454350e-05, 1.076279),
    "LFI2710": (5.473547e-05, 1.018206),
    "LFI2711": (5.430311e-05, 0.769237),
    "LFI2800": (5.460166e-05, 0.696949),
    "LFI2801": (5.435220e-05, 1.098157),
    "LFI2810": (5.477642e-05, 0.715768),
    "LFI2811": (5.486962e-05, 0.568779),
    "LFI2400": (1.362262e-05, -0.097502),
    "LFI2401": (1.373675e-05, -0.099307),
    "LFI2410": (2.756824e-05, -0.212564),
    "LFI2411": (2.714697e-05, -0.209627),
    "LFI2500": (4.130199e-05, -0.325112),
    "LFI2501": (4.080861e-05, -0.321322),
    "LFI2510": (4.085333e-05, -0.320733),
    "LFI2511": (4.101210e-05, -0.321683),
    "LFI2600": (2.725836e-05, -0.209177),
    "LFI2601": (4.132134e-05, -0.324404),
    "LFI2610": (2.748296e-05, -0.211944),
    "LFI2611": (4.054677e-05, -0.319058),
    "LFI1800": (8.209513e-05, 1.203361),
    "LFI1801": (1.645581e-04, 0.639331),
    "LFI1810": (5.471130e-05, 1.418874),
    "LFI1811": (5.465637e-05, 0.735420),
    "LFI1900": (5.489444e-05, 0.628215),
    "LFI1901": (5.479791e-05, 1.135533),
    "LFI1910": (5.464215e-05, 0.618559),
    "LFI1911": (5.475693e-05, 0.627994),
    "LFI2000": (5.438797e-05, 1.279399),
    "LFI2001": (5.439007e-05, 1.257471),
    "LFI2010": (5.512579e-05, 0.762344),
    "LFI2011": (5.441487e-05, 1.148864),
    "LFI2100": (5.456505e-05, 0.598286),
    "LFI2101": (5.458101e-05, 0.063775),
    "LFI2110": (5.463337e-05, 0.539929),
    "LFI2111": (5.478074e-05, 0.528317),
    "LFI2200": (5.481768e-05, -0.182580),
    "LFI2201": (5.500792e-05, -0.133994),
    "LFI2210": (5.480497e-05, -0.122493),
    "LFI2211": (5.435908e-05, 0.048182),
    "LFI2300": (5.493823e-05, 0.650412),
    "LFI2301": (5.477657e-05, 0.862792),
    "LFI2310": (5.477666e-05, 0.789134),
    "LFI2311": (5.497197e-05, -0.134015),
}

ADU2Volt_post953 = {  # After 1703263248.3820317
    "LFI2700": (5.491826e-05, 0.958340),
    "LFI2701": (5.454350e-05, 1.076279),
    "LFI2710": (5.473547e-05, 1.018206),
    "LFI2711": (5.430311e-05, 0.769237),
    "LFI2800": (5.460166e-05, 0.696949),
    "LFI2801": (5.435220e-05, 1.098157),
    "LFI2810": (5.477642e-05, 0.715768),
    "LFI2811": (5.486962e-05, 0.568779),
    "LFI2400": (1.362262e-05, -0.097502),
    "LFI2401": (1.373675e-05, -0.099307),
    "LFI2410": (2.756824e-05, -0.212564),
    "LFI2411": (2.714697e-05, -0.209627),
    "LFI2500": (4.130199e-05, -0.325112),
    "LFI2501": (4.080861e-05, -0.321322),
    "LFI2510": (4.085333e-05, -0.320733),
    "LFI2511": (4.101210e-05, -0.321683),
    "LFI2600": (2.725836e-05, -0.209177),
    "LFI2601": (4.132134e-05, -0.324404),
    "LFI2610": (2.748296e-05, -0.211944),
    "LFI2611": (4.054677e-05, -0.319058),
    # OFFSET MISMATCH: LFI1800 1.203361 1.440341832640983
    "LFI1800": (8.209513e-05, 1.440342),
    # OFFSET MISMATCH: LFI1801 0.639331 1.1668567309015119
    "LFI1801": (1.645581e-04, 1.166857),
    "LFI1810": (5.471130e-05, 1.418874),
    # OFFSET MISMATCH: LFI1811 0.73542 1.0082754440692991
    "LFI1811": (5.465637e-05, 1.008275),
    # OFFSET MISMATCH: LFI1900 0.628215 0.881012773335376
    "LFI1900": (5.489444e-05, 0.881013),
    "LFI1901": (5.479791e-05, 1.135533),
    "LFI1910": (5.464215e-05, 0.618559),
    "LFI1911": (5.475693e-05, 0.627994),
    "LFI2000": (5.438797e-05, 1.279399),
    "LFI2001": (5.439007e-05, 1.257471),
    # OFFSET MISMATCH: LFI2010 0.762344 0.9987390372012804
    "LFI2010": (5.512579e-05, 0.998739),
    "LFI2011": (5.441487e-05, 1.148864),
    "LFI2100": (5.456505e-05, 0.598286),
    # OFFSET MISMATCH: LFI2101 0.063775 0.5897512588882274
    "LFI2101": (5.458101e-05, 0.589751),
    "LFI2110": (5.463337e-05, 0.539929),
    "LFI2111": (5.478074e-05, 0.528317),
    # OFFSET MISMATCH: LFI2200 -0.18258 0.042919264724727596
    "LFI2200": (5.481768e-05, 0.042919),
    # OFFSET MISMATCH: LFI2201 -0.133994 0.10020897148397322
    "LFI2201": (5.500792e-05, 0.100209),
    # OFFSET MISMATCH: LFI2210 -0.122493 0.11308988059095518
    "LFI2210": (5.480497e-05, 0.113090),
    # OFFSET MISMATCH: LFI2211 0.048182 0.5854576226463757
    "LFI2211": (5.435908e-05, 0.585458),
    "LFI2300": (5.493823e-05, 0.650412),
    "LFI2301": (5.477657e-05, 0.862792),
    "LFI2310": (5.477666e-05, 0.789134),
    # OFFSET MISMATCH: LFI2311 -0.134015 0.09076310563082374
    "LFI2311": (5.497197e-05, 0.090763),
}

diode_gains = {
    "LFI1800": 1.132,
    "LFI1801": 0.867,
    "LFI1810": 0.878,
    "LFI1811": 1.099,
    "LFI1900": 1.055,
    "LFI1901": 0.953,
    "LFI1910": 1.097,
    "LFI1911": 0.906,
    "LFI2000": 0.982,
    "LFI2001": 1.025,
    "LFI2010": 0.968,
    "LFI2011": 1.037,
    "LFI2100": 0.942,
    "LFI2101": 1.070,
    "LFI2110": 0.994,
    "LFI2111": 1.011,
    "LFI2200": 1.061,
    "LFI2201": 0.940,
    "LFI2210": 1.096,
    "LFI2211": 0.904,
    "LFI2300": 1.099,
    "LFI2301": 0.914,
    "LFI2310": 0.709,
    "LFI2311": 1.302,
    "LFI2400": 1.046,
    "LFI2401": 0.944,
    "LFI2410": 1.023,
    "LFI2411": 0.987,
    "LFI2500": 0.995,
    "LFI2501": 1.001,
    "LFI2510": 0.905,
    "LFI2511": 1.068,
    "LFI2600": 1.038,
    "LFI2601": 0.946,
    "LFI2610": 1.012,
    "LFI2611": 0.992,
    "LFI2700": 1.046,
    "LFI2701": 0.967,
    "LFI2710": 0.936,
    "LFI2711": 1.091,
    "LFI2800": 1.134,
    "LFI2801": 0.887,
    "LFI2810": 0.957,
    "LFI2811": 1.038,
}

diode_weights = {
    "LFI1800": 0.499,
    "LFI1801": 0.501,
    "LFI1810": 0.455,
    "LFI1811": 0.545,
    "LFI1900": 0.450,
    "LFI1901": 0.550,
    "LFI1910": 0.519,
    "LFI1911": 0.481,
    "LFI2000": 0.520,
    "LFI2001": 0.480,
    "LFI2010": 0.482,
    "LFI2011": 0.518,
    "LFI2100": 0.526,
    "LFI2101": 0.474,
    "LFI2110": 0.575,
    "LFI2111": 0.425,
    "LFI2200": 0.502,
    "LFI2201": 0.498,
    "LFI2210": 0.502,
    "LFI2211": 0.498,
    "LFI2300": 0.457,
    "LFI2301": 0.543,
    "LFI2310": 0.506,
    "LFI2311": 0.494,
    "LFI2400": 0.569,
    "LFI2401": 0.431,
    "LFI2410": 0.433,
    "LFI2411": 0.567,
    "LFI2500": 0.490,
    "LFI2501": 0.510,
    "LFI2510": 0.419,
    "LFI2511": 0.581,
    "LFI2600": 0.545,
    "LFI2601": 0.455,
    "LFI2610": 0.399,
    "LFI2611": 0.601,
    "LFI2700": 0.499,
    "LFI2701": 0.501,
    "LFI2710": 0.524,
    "LFI2711": 0.476,
    "LFI2800": 0.493,
    "LFI2801": 0.507,
    "LFI2810": 0.489,
    "LFI2811": 0.511,
}

PLANCK_DETINDX = {
    "LFI18M": 0,
    "LFI18S": 1,
    "LFI19M": 2,
    "LFI19S": 3,
    "LFI20M": 4,
    "LFI20S": 5,
    "LFI21M": 6,
    "LFI21S": 7,
    "LFI22M": 8,
    "LFI22S": 9,
    "LFI23M": 10,
    "LFI23S": 11,
    "LFI24M": 12,
    "LFI24S": 13,
    "LFI25M": 14,
    "LFI25S": 15,
    "LFI26M": 16,
    "LFI26S": 17,
    "LFI27M": 18,
    "LFI27S": 19,
    "LFI28M": 20,
    "LFI28S": 21,
    "100-1a": 22,
    "100-1b": 23,
    "100-2a": 24,
    "100-2b": 25,
    "100-3a": 26,
    "100-3b": 27,
    "100-4a": 28,
    "100-4b": 29,
    "143-1a": 30,
    "143-1b": 31,
    "143-2a": 32,
    "143-2b": 33,
    "143-3a": 34,
    "143-3b": 35,
    "143-4a": 36,
    "143-4b": 37,
    "143-5": 38,
    "143-6": 39,
    "143-7": 40,
    "143-8": 41,
    "217-1": 42,
    "217-2": 43,
    "217-3": 44,
    "217-4": 45,
    "217-5a": 46,
    "217-5b": 47,
    "217-6a": 48,
    "217-6b": 49,
    "217-7a": 50,
    "217-7b": 51,
    "217-8a": 52,
    "217-8b": 53,
    "353-1": 54,
    "353-2": 55,
    "353-3a": 56,
    "353-3b": 57,
    "353-4a": 58,
    "353-4b": 59,
    "353-5a": 60,
    "353-5b": 61,
    "353-6a": 62,
    "353-6b": 63,
    "353-7": 64,
    "353-8": 65,
    "545-1": 66,
    "545-2": 67,
    "545-3": 68,
    "545-4": 69,
    "857-1": 70,
    "857-2": 71,
    "857-3": 72,
    "857-4": 73,
    "Dark-1": 74,
    "Dark-2": 75,
}


def to_diodes(det):
    radiometer = det.replace("M", "0").replace("S", "1")
    return radiometer + "0", radiometer + "1"


def to_radiometer(det):
    radiometer = det[:-1]  # Strip the diode identifier
    radiometer = radiometer.replace("0", "M").replace("1", "S")
    return radiometer


def read_gains(calfile, det, tstart=None, tstop=None):
    if not os.path.isfile(calfile):
        raise RuntimeError("Calibration file not found: {}".format(calfile))

    if (tstart is None and tstop is not None) or (tstart is not None and tstop is None):
        raise RuntimeError(
            "read_gains requires setting both tstart and tstop at the same time."
        )

    h = pf.open(calfile)

    # Translate the timestamps into TAI seconds

    times = h[1].data.field(0).ravel()
    if times[-1] > 1e17:
        times = times * 1e-9
    elif times[-1] > 1e14:
        times = times * 2 ** -16

    # Load the right gains

    cal = None
    for hdu in h[2:]:
        if det in hdu.header["extname"] or det.upper() in hdu.header["extname"]:
            cal = hdu.data.field(0).ravel()

    h.close()

    if cal is None:
        raise RuntimeError("{} not found in {}".format(det, calfile))

    if tstart is not None and tstop is not None:
        # Prune the unnecessary gains
        istart, istop = np.searchsorted(times, [tstart, tstop], side="right") - 1
        if istart > 0:
            istart -= 1
        if istop < len(cal):
            istop += 1
        ind = slice(istart, istop)
        gains = np.vstack([times[ind], cal[ind]])
    else:
        gains = np.vstack([times, cal])
    return gains


def read_eff(
    local_start,
    n,
    globalfirst,
    local_offset,
    ringdb,
    ringdb_path,
    freq,
    effdir,
    extname,
    flagmask,
    eff_cache,
    tod_cache,
    filenames,
    debug=0,
    file_pattern=None,
):
    t0 = time()

    if n < 1:
        raise Exception("ERROR: cannot read negative number of samples: {}".format(n))

    ringtable = ringdb_table_name(freq)

    # Convert to global indices

    start = globalfirst + local_offset + local_start
    # start = local_offset + local_start
    stop = start + n

    # Start, stop interval follows Python conventions.
    # (stop-start) samples are read starting at "start".
    # Sample at "stop" is omitted

    # Determine first and last ring to read

    t1 = time()

    if ringdb not in past_queries:
        past_queries[ringdb] = {}

    cmd = (
        "select start_time, start_index, start_row from {} where "
        "start_index <= {} order by start_index".format(ringtable, start)
    )
    if cmd in past_queries[ringdb]:
        start_time, start_index, start_row = past_queries[ringdb][cmd]
    else:
        try:
            x = ringdb.execute(cmd).fetchall()[-1]
        except IndexError:
            raise RuntimeError(
                '{} returned nothing for query "{}"'.format(ringdb_path, cmd)
            )
        start_time, start_index, start_row = x
        past_queries[ringdb][cmd] = x

    cmd = (
        "select stop_time from {} where stop_index >= {} order by "
        "stop_time".format(ringtable, stop)
    )
    if cmd in past_queries[ringdb]:
        stop_time, = past_queries[ringdb][cmd]
    else:
        try:
            x = ringdb.execute(cmd).fetchall()[0]
        except IndexError:
            raise RuntimeError(
                '{} returned nothing for query "{}"'.format(ringdb_path, cmd)
            )
        stop_time, = x
        past_queries[ringdb][cmd] = x

    # Determine the list of ODs that contain the rings

    ods = []
    if int(freq) < 100:
        query = ringdb.execute(
            "select eff_od, nrow from eff_files where stop_time >= {} and "
            "start_time <= {} and freq == {}".format(start_time, stop_time, freq)
        )
    else:
        query = ringdb.execute(
            "select eff_od, nrow from eff_files where stop_time >= {} and "
            "start_time <= {} and freq == 100".format(start_time, stop_time)
        )
    for q in query:
        ods.append([int(q[0]), int(q[1])])

    if debug > 3:
        print("SQL queries completed in {:.2f} s".format(time() - t1))

    data = []
    flag = []
    first_row = start_row + (start - start_index)
    nleft = stop - start
    nread = 0

    ods_read = []  # just for diagnostics
    files_read = []  # just for diagnostics

    while len(ods) > 0:

        od, nrow = ods[0]
        ods_read.append(od)

        if nrow <= first_row:
            ods = ods[1:]
            first_row -= nrow
            continue

        if nrow - first_row > nleft:
            last_row = first_row + nleft
        else:
            last_row = nrow

        nbuff = last_row - first_row

        if nbuff < 1:
            raise Exception(
                "Empty read on OD {}: indices {} - {}, rows {} - {}, "
                "timestamps {} - {}, "
                "ods_read = {}, files_read = {}, nleft = {}, ods = {}, "
                "start = {}, start_row = {}, start_index = {}".format(
                    od, start, stop, first_row, last_row, start_time, stop_time,
                    ods_read, files_read, nleft, ods,
                    start, start_row, start_index,
                )
            )

        if (
            eff_cache is None
            or effdir not in eff_cache
            or od not in eff_cache[effdir]
            or extname not in eff_cache[effdir][od]
        ):

            if debug > 3:
                print(
                    "Caching new data for {}:{}:{}".format(effdir, od, extname),
                    flush=True,
                )

            if eff_cache is not None and effdir not in eff_cache:
                eff_cache[effdir] = {}
            if eff_cache is not None and od not in eff_cache[effdir]:
                eff_cache[effdir][od] = {}

            if extname.lower() in ["attitude", "velocity", "phase", "position"]:
                if int(freq) < 100:
                    pattern = effdir + "/{:04}/pointing*-{:03}-*fits".format(od, freq)
                else:
                    pattern = effdir + "/{:04}/pointing*fits".format(od, freq)
            elif "dark" in extname.lower():
                pattern = effdir + "/{:04}/HDRK*fits".format(od)
            else:
                pattern = effdir + "/{:04}/?{:03}*fits".format(od, freq)
            try:
                regexp_pattern = pattern.replace("*", ".*").replace("?", ".")
                regexp = re.compile(regexp_pattern)
                all_fn = sorted(
                    [filename for filename in filenames if regexp.match(filename)]
                )
                if file_pattern is not None:
                    all_fn = [fn for fn in all_fn if file_pattern in fn]
                fn = all_fn[-1]
            except Exception:
                raise Exception(
                    "Error: failed to find a file to read matching: {} ({})"
                    "".format(pattern, regexp_pattern)
                )

            files_read.append(fn)

            if debug > 3:
                print("Opening {}".format(fn), flush=True)

            hdulist = pf.open(fn, "readonly", memmap=True, lazy_load_hdus=True)
            found = False
            for hdu in hdulist:
                if "extname" not in hdu.header:
                    continue
                if extname in hdu.header["extname"].strip().lower():
                    found = True
                    break
            if not found:
                raise Exception(
                    "No HDU matching extname = {} in {}".format(extname, fn)
                )
            ncol = len(hdu.columns)

            if eff_cache is not None:
                # Cache the entire column
                if extname not in eff_cache[effdir][od]:
                    if ncol == 2:
                        temp = np.array(hdu.data.field(0), dtype=np.float64)
                    else:
                        temp = np.array(
                            [hdu.data.field(col).ravel() for col in range(ncol - 1)],
                            dtype=np.float64,
                        )
                    if debug > 3:
                        print(
                            "Storing {:.2f} MB to eff_cache:{}:{}:{}".format(
                                temp.nbytes / 2.0 ** 20, effdir, od, extname
                            ),
                            flush=True,
                        )
                    cachename = ":".join([effdir, str(od), extname])
                    eff_cache[effdir][od][extname] = tod_cache.put(
                        cachename, temp, replace=True
                    )
                if ncol == 2:
                    dat = eff_cache[effdir][od][extname][first_row:last_row].copy()
                else:
                    dat = eff_cache[effdir][od][extname][:, first_row:last_row].copy()

                if debug > 3:
                    print(
                        "Retrieved {:.2f} MB from eff_cache:{}:{}:{}".format(
                            dat.nbytes / 2.0 ** 20, effdir, od, extname
                        ),
                        flush=True,
                    )
            else:
                if ncol == 2:
                    dat = np.array(
                        hdu.data.field(0)[first_row:last_row], dtype=np.float64
                    )
                else:
                    dat = np.array(
                        [
                            hdu.data.field(col)[first_row:last_row].ravel()
                            for col in range(ncol - 1)
                        ],
                        dtype=np.float64,
                    )

            if np.shape(dat)[-1] != last_row - first_row:
                raise Exception(
                    "Got wrong number of samples: shape(dat)={}, "
                    "first_row = {}, last_row={}, last_row-first_row={}. "
                    "fn = {}, nrow = {}, shape(h[hdu].data.field(0)) = {}"
                    "".format(
                        np.shape(dat),
                        first_row,
                        last_row,
                        last_row - first_row,
                        fn,
                        nrow,
                        np.shape(hdu.data.field(0)),
                    )
                )

            flg = np.zeros(last_row - first_row, dtype=np.uint8)

            if flagmask != 0:
                if eff_cache is not None:
                    # Cache the detector flags
                    if extname + "flag" not in eff_cache[effdir][od]:
                        temp = np.array(hdu.data.field(ncol - 1), dtype=np.uint8)
                        # PyFITS seems to return wrong endian or otherwise
                        # deformed arrays
                        temp2 = np.empty(temp.shape, dtype=np.uint8)
                        temp2[:] = temp[:]
                        temp = temp2
                        if debug > 3:
                            print(
                                "Storing {:.2f} MB to eff_cache:{}:{}:{}"
                                "".format(
                                    temp.nbytes / 2.0 ** 20,
                                    effdir,
                                    od,
                                    extname + "flag",
                                ),
                                flush=True,
                            )  # DEBUG
                        cachename = ":".join([effdir, str(od), extname + "flag"])
                        eff_cache[effdir][od][extname + "flag"] = tod_cache.put(
                            cachename, temp, replace=True
                        )
                    detflg = eff_cache[effdir][od][extname + "flag"][
                        first_row:last_row
                    ].copy()
                    if debug > 3:
                        print(
                            "Retrieved {:.2f} MB from eff_cache:{}:{}:{}"
                            "".format(
                                detflg.nbytes / 2.0 ** 20, effdir, od, extname + "flag"
                            ),
                            flush=True,
                        )
                else:
                    detflg = np.array(
                        hdu.data.field(ncol - 1)[first_row:last_row], dtype=np.uint8
                    )
                flg[:] = detflg & flagmask

            hdulist.close()
            del hdu.data
            del hdu
        else:
            # get the requested TOI from the cache
            if debug > 3:
                print(
                    "Cache already contains {}:{}:{}".format(effdir, od, extname),
                    flush=True,
                )
            if len(np.shape(eff_cache[effdir][od][extname])) == 1:
                ncol = 2
            else:
                ncol = np.shape(eff_cache[effdir][od][extname])[0] + 1

            if ncol == 2:
                dat = eff_cache[effdir][od][extname][first_row:last_row].copy()
            else:
                dat = eff_cache[effdir][od][extname][:, first_row:last_row].copy()
            if debug > 3:
                print(
                    "Retrieved {:.2f} MB from eff_cache:{}:{}:{}".format(
                        dat.nbytes / 2.0 ** 20, effdir, od, extname
                    ),
                    flush=True,
                )

            flg = np.zeros(last_row - first_row, dtype=np.uint8)

            if flagmask != 0:
                detflg = eff_cache[effdir][od][extname + "flag"][
                    first_row:last_row
                ].copy()
                if debug > 3:
                    print(
                        "Retrieved {:.2f} MB from eff_cache:{}:{}:{}".format(
                            detflg.nbytes / 2.0 ** 20, effdir, od, extname + "flag"
                        ),
                        flush=True,
                    )  # DEBUG
                flg += detflg & flagmask

        data.append(dat)
        flag.append(flg)

        ods = ods[1:]
        first_row = 0

        nread += nbuff
        nleft -= nbuff

        if nleft == 0:
            break

    if len(data) > 0:
        data = np.hstack(data)
    if len(flag) > 0:
        flag = np.hstack(flag)

    if np.shape(flag)[-1] != stop - start:
        raise Exception(
            "ERROR: inconsistent dimensions: shape(data) = {}, shape(flag) "
            "=  {}, stop-start = {}, ods = {}, files = {}".format(
                np.shape(data), np.shape(flag), stop - start, ods_read, files_read
            )
        )

    if debug > 3:
        print(
            "Read {} samples from extension {}. Number of flagged samples "
            "= {} ({:.2f}%)".format(
                n, extname, np.sum(flag != 0), np.sum(flag != 0) * 100.0 / n
            ),
            flush=True,
        )  # DEBUG

    if debug > 3:
        print("read_eff completed in {:.2f} s".format(time() - t0))

    return (data, flag)


def write_eff(
    local_start,
    data,
    flags,
    globalfirst,
    local_offset,
    ringdb,
    ringdb_path,
    freq,
    effdir,
    extname,
    filenames,
    debug=0,
    file_pattern=None,
):
    ringtable = ringdb_table_name(freq)

    ntot = 0
    if data is not None:
        ntot = len(data)
    elif flags is not None:
        ntot = len(flags)
    if ntot == 0:
        raise Exception("write_eff: no samples to write")

    # Convert to global indices

    start = globalfirst + local_offset + local_start
    stop = start + ntot

    if data is not None:
        # Promote all input data to 2 dimensions to support multicolumn writing
        rms0 = np.std(data)
        data = data.copy()
        rms = np.std(data)
        if np.isfinite(rms0) and (rms0 != rms):
            raise Exception(
                "write_eff: Something is wrong! rms(data) = {}, "
                "rms(data.copy()) = {}".format(rms0, rms)
            )
        data2d = np.atleast_2d(data)
        rms2d = np.std(data2d)
        if np.isfinite(rms) and (rms != rms2d):
            raise Exception(
                "write_eff: Something is wrong! rms(data) = {}, "
                "rms(data2d) = {}".format(rms, rms2d)
            )
    else:
        data2d = None

    cmd = (
        "select start_time, start_index, start_row from {} where "
        "start_index <= {} order by start_index".format(ringtable, start)
    )
    if cmd in past_queries[ringdb]:
        start_time, start_index, start_row = past_queries[ringdb][cmd]
    else:
        x = ringdb.execute(cmd).fetchall()[-1]
        start_time, start_index, start_row = x
        past_queries[ringdb][cmd] = x

    cmd = (
        "select stop_time from {} where stop_index >= {} order by "
        "stop_time".format(ringtable, stop)
    )
    if cmd in past_queries[ringdb]:
        stop_time, = past_queries[ringdb][cmd]
    else:
        x = ringdb.execute(cmd).fetchall()[0]
        stop_time, = x
        past_queries[ringdb][cmd] = x

    ods = []
    if freq < 100:
        cmd2 = (
            "select eff_od, nrow from eff_files where stop_time >= {} and "
            "start_time <= {} and freq == {}".format(start_time, stop_time, freq)
        )
        query = ringdb.execute(cmd2)
    else:
        cmd2 = (
            "select eff_od, nrow from eff_files where stop_time >= {} and "
            "start_time <= {} and freq == 100".format(start_time, stop_time)
        )
        query = ringdb.execute(cmd2)
    for q in query:
        ods.append([int(q[0]), int(q[1])])

    if len(ods) == 0:
        raise Exception("write_eff: No ODs match query:", cmd2)

    nleft = stop - start
    nwrote = 0

    first_row = start_row + (start - start_index)
    offset = 0

    while len(ods) > 0:
        od, nrow = ods[0]

        if nrow < first_row:
            ods = ods[1:]
            first_row -= nrow
            continue

        if extname.lower() in ["attitude", "velocity", "phase", "position"]:
            pattern = effdir + "/{:04}/pointing*fits".format(od)
        elif "dark" in extname.lower():
            pattern = effdir + "/{:04}/HDRK*fits".format(od)
        else:
            pattern = effdir + "/{:04}/?{:03}*fits".format(od, freq)
        try:
            regexp_pattern = pattern.replace("*", ".*").replace("?", ".")
            regexp = re.compile(regexp_pattern)
            all_fn = sorted(
                [filename for filename in filenames if regexp.match(filename)]
            )
            if file_pattern is not None:
                all_fn = [fn for fn in all_fn if file_pattern in fn]
            fn = all_fn[-1]
        except Exception:
            raise Exception(
                "Error: failed to find a file to write matching: {} ({})"
                "".format(pattern, regexp_pattern)
            )

        h = pf.open(fn, "update")
        hdu = 1
        while extname not in h[hdu].header["extname"].strip().lower():
            hdu += 1
            if hdu == len(h):
                raise Exception(
                    "No HDU matching extname = {} in {}".format(extname, fn)
                )

        if nrow - first_row > nleft:
            last_row = first_row + nleft
        else:
            last_row = nrow

        nwrite = last_row - first_row
        ncol = len(h[hdu].columns)

        if data2d is not None:
            ncol_data = len(data2d)
            if ncol - 1 != ncol_data:
                raise Exception(
                    "Expected {} columns to write data but got {}."
                    "".format(ncol - 1, ncol_data)
                )

        if nwrite > 0:
            try:
                if data2d is not None:
                    for col in range(len(data2d)):
                        h[hdu].data.field(col)[first_row:last_row] = data2d[col][
                            offset : offset + nwrite
                        ]
                if flags is not None:
                    h[hdu].data.field(ncol - 1)[first_row:last_row] = flags[
                        offset : offset + nwrite
                    ]
            except Exception as e:
                raise Exception(
                    "Failed to write rows {} - {} to {}[{}] : {}".format(
                        first_row, last_row, fn, hdu, e
                    )
                )

            offset += nwrite

        ods = ods[1:]
        first_row = 0
        result = h.flush()
        h.close()
        if debug > 3:
            print(
                "Wrote {} samples of {} TOD to {}. Rows {} - {}".format(
                    nwrite, extname, fn, first_row, last_row
                )
            )

        nwrote += nwrite
        nleft -= nwrite

        if nleft == 0:
            break

    return result


def ringdb_table_name(freq):
    if int(freq) < 100:
        return "ring_times_lfi{}".format(freq)
    else:
        return "ring_times_hfi"


def load_ringdb(path, comm, freq):
    """
    Load and broadcast the ring database.
    """
    if path in ringdb_cache:
        return ringdb_cache[path]

    timer = Timer()
    timer.start()

    # Read relevant parts of the database to tempfile and broadcast

    tempfile = ""
    tables_include = ["ods", ringdb_table_name(freq), "eff_files", "rings"]

    expr = r""
    for table in tables_include:
        if len(expr) != 0:
            expr += "|"
        expr += ".*" + table + ".*"
    linetest = re.compile(expr)

    if comm is None or comm.rank == 0:
        conn = sqlite3.connect(path)
        tempfile = io.StringIO()
        for line in conn.iterdump():
            if linetest.match(line):
                tempfile.write("{}\n".format(line))
        conn.close()
        tempfile.seek(0)

    if comm is not None:
        tempfile = comm.bcast(tempfile, root=0)

    # Create a database in memory and import from tempfile

    ringdb = sqlite3.connect(":memory:")
    ringdb.cursor().executescript(tempfile.read())
    ringdb.commit()

    if comm is None or comm.rank == 0:
        timer.report_clear("Load and broadcast ring database from {}".format(path))

    ringdb_cache[path] = ringdb
    return ringdb


def load_RIMO(path, comm=None):
    """
    Load and broadcast the reduced instrument model,
    a.k.a. focal plane database.
    """

    # Read database, parse and broadcast

    if comm is not None:
        comm.Barrier()
    timer = Timer()
    timer.start()

    RIMO = {}
    if comm is None or comm.rank == 0:
        print("Loading RIMO from {}".format(path), flush=True)
        hdulist = pf.open(path, "readonly")
        detectors = hdulist[1].data.field("detector").ravel()
        phi_uvs = hdulist[1].data.field("phi_uv").ravel()
        theta_uvs = hdulist[1].data.field("theta_uv").ravel()
        psi_uvs = hdulist[1].data.field("psi_uv").ravel()
        psi_pols = hdulist[1].data.field("psi_pol").ravel()
        epsilons = hdulist[1].data.field("epsilon").ravel()
        fsamples = hdulist[1].data.field("f_samp").ravel()
        fknees = hdulist[1].data.field("f_knee").ravel()
        alphas = hdulist[1].data.field("alpha").ravel()
        nets = hdulist[1].data.field("net").ravel()
        fwhms = hdulist[1].data.field("fwhm").ravel()

        for i in range(len(detectors)):
            phi = (phi_uvs[i]) * degree
            theta = theta_uvs[i] * degree
            # Make sure we don't double count psi rotation already
            # included in phi
            psi = (psi_uvs[i] + psi_pols[i]) * degree - phi
            quat = np.zeros(4)
            # ZYZ conversion from
            # http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
            # Note: The above document has the scalar part of the quaternion at
            # first position but quaternionarray module has it at the end, we
            # use the quaternionarray convention
            # scalar part:
            quat[3] = np.cos(0.5 * theta) * np.cos(0.5 * (phi + psi))
            # vector part
            quat[0] = -np.sin(0.5 * theta) * np.sin(0.5 * (phi - psi))
            quat[1] = np.sin(0.5 * theta) * np.cos(0.5 * (phi - psi))
            quat[2] = np.cos(0.5 * theta) * np.sin(0.5 * (phi + psi))
            # apply the bore sight rotation to the detector quaternion
            quat = qa.mult(SPINROT, quat)
            RIMO[detectors[i]] = DetectorData(
                detectors[i],
                phi_uvs[i],
                theta_uvs[i],
                psi_uvs[i],
                psi_pols[i],
                epsilons[i],
                fsamples[i],
                fknees[i],
                alphas[i],
                nets[i],
                fwhms[i],
                quat,
            )

        hdulist.close()

    if comm is not None:
        RIMO = comm.bcast(RIMO, root=0)

    if comm is None or comm.rank == 0:
        timer.report_clear("Load and broadcast RIMO")

    return RIMO


def count_samples(ringdb, freq, obt_range, ring_range, od_range):
    """
    Query the ring database to determine which global samples the requested
    time ranges point to.  The ranges are checked in order:  OD, ring, then
    OBT.  If none are specified, then all data are selected.
    """
    ringtable = ringdb_table_name(freq)

    rings = []
    modes = []
    modenames = []

    select_range = None

    # Begin by querying the rings from the beginning of the database
    # to the end of the desired span.

    if od_range is not None:
        cmd = (
            "select start_time, stop_time from ods where od >= {} and "
            "od <= {} order by od".format(od_range[0], od_range[1])
        )
        ods = ringdb.execute(cmd).fetchall()
        start, _ = ods[0]
        _, stop = ods[-1]
        select_range = (start, stop)

    if ring_range is not None:
        # Ring numbers are not recorded into the database. Simply get a list
        # of the repointing maneuvers at the start of each pointing period
        cmd = (
            "select start_time, stop_time from {} where pointID_unique "
            'like "%-H%" order by start_time'.format(ringtable)
        )
        ringlist = ringdb.execute(cmd).fetchall()
        try:
            start1, _ = ringlist[ring_range[0]]
            if ring_range[1] + 1 >= len(ringlist):
                start2, _ = ringlist[-1]
            else:
                start2, _ = ringlist[ring_range[1] + 1]
        except Exception:
            raise Exception(
                "Failed to determine the ring span {} from the "
                "datababase".format(ring_range)
            )
        select_range = (start1, start2)

    if obt_range is not None:
        select_range = obt_range

    cmd = ""

    if select_range is not None:
        start, stop = select_range
        cmd = (
            "select start_time, stop_time, start_index, stop_index, "
            "start_row, stop_row, pointID_unique from {} where "
            "start_time < {} order by start_index".format(ringtable, stop)
        )
    else:
        cmd = (
            "select start_time, stop_time, start_index, stop_index, "
            "start_row, stop_row, pointID_unique from {} order by "
            "start_index".format(ringtable)
        )

    intervals = ringdb.execute(cmd).fetchall()

    if len(intervals) < 1:
        raise Exception(
            "Warning: failed to find any intervals with the query: {}".format(cmd)
        )

    for i in range(len(intervals) - 1):
        last = intervals[i][3]
        first = intervals[i + 1][2]
        if first <= last:
            print(
                "WARNING: Overlap of {} samples in RingDB intervals:"
                "".format(first - last + 1)
            )
            print(i, intervals[i])
            print(i + 1, intervals[i + 1], flush=True)
            new_entry = list(intervals[i + 1])
            new_entry[2] = last + 1
            intervals[i + 1] = tuple(new_entry)

    global_start = None
    global_first = 0

    hcm_interval = None
    hcm_start_time = None
    hcm_start_index = None
    hcm_name = None

    # Get the global start time and index

    for interval in intervals:
        start_time, stop_time, start_index, stop_index, _, _, name = interval

        if "H" in name:
            hcm_interval = interval
            (hcm_start_time, _, hcm_start_index, _, _, _, hcm_name) = hcm_interval

        if stop_time > start:
            # First of the queried rings that is part of the desired range
            if hcm_interval is None:
                # Start from an HCM or an orphan SCM
                global_start = start_time
                global_first = start_index
            else:
                # User provided a time range that starts during an SCM
                # we will include the preceeding HCM
                global_start = hcm_start_time
                global_first = hcm_start_index
            break

    # Create a list of rings where each HCM (repointing maneuver) is joined
    # with the subsequent SCM or OCM.

    hcm_interval = None
    last_stop_time = None
    last_stop_index = None
    hcm_start_time = None
    hcm_start_index = None
    hcm_name = None

    for interval in intervals:
        start_time, stop_time, start_index, stop_index, _, _, name = interval

        if stop_time > start:
            modes.append(
                Interval(
                    start=start_time,
                    stop=stop_time,
                    first=start_index - global_first,
                    last=stop_index - global_first,
                )
            )
            modenames.append(name)

        if "H" in name and interval is not intervals[-1]:
            if hcm_interval is not None:
                print(
                    "WARNING: Two consecutive HCMs: {} ({}) and {} ({})"
                    "".format(hcm_name, hcm_interval, name, interval)
                )
            else:
                hcm_interval = interval
                (hcm_start_time, _, hcm_start_index, _, _, _, hcm_name) = hcm_interval
        else:
            if hcm_interval is not None:
                if (
                    last_stop_time is not None
                    and hcm_start_time - last_stop_time > 75 * 60
                ):
                    # The unnamed gap between two rings is so long that it
                    # needs to be split
                    nstep = int(
                        np.ceil((hcm_start_time - last_stop_time) / (75 * 60))
                    )
                    steptime = (hcm_start_time - last_stop_time) / nstep
                    steplen = int(
                        np.ceil((hcm_start_index - last_stop_index) / nstep)
                    )
                    if steplen > 1:
                        for istep in range(nstep):
                            rings.append(
                                Interval(
                                    start=last_stop_time + istep * steptime,
                                    stop=last_stop_time + (istep + 1) * steptime,
                                    first=(
                                        last_stop_index
                                        + 1
                                        + istep * steplen
                                        - global_first
                                    ),
                                    last=min(
                                        last_stop_index + (istep + 1) * steplen,
                                        hcm_start_index - 1,
                                    )
                                    - global_first,
                                )
                            )
                rings.append(
                    Interval(
                        start=hcm_start_time,
                        stop=stop_time,
                        first=hcm_start_index - global_first,
                        last=stop_index - global_first,
                    )
                )
                last_stop_time = stop_time
                last_stop_index = stop_index
                hcm_interval = None
            else:
                if last_stop_time is not None and start_time - last_stop_time > 75 * 60:
                    # The unnamed gap between two rings is so long that it
                    # needs to be split
                    nstep = int(np.ceil((start_time - last_stop_time) / (75 * 60)))
                    steptime = (start_time - last_stop_time) / nstep
                    steplen = int(np.ceil((start_index - last_stop_index) / nstep))
                    if steplen > 1:
                        for istep in range(nstep):
                            rings.append(
                                Interval(
                                    start=last_stop_time + istep * steptime,
                                    stop=last_stop_time + (istep + 1) * steptime,
                                    first=(
                                        last_stop_index
                                        + 1
                                        + istep * steplen
                                        - global_first
                                    ),
                                    last=min(
                                        last_stop_index + (istep + 1) * steplen,
                                        start_index - 1,
                                    )
                                    - global_first,
                                )
                            )
                rings.append(
                    Interval(
                        start=start_time,
                        stop=stop_time,
                        first=start_index - global_first,
                        last=stop_index - global_first,
                    )
                )
                last_stop_time = stop_time
                last_stop_index = stop_index

    ring_offset = 0
    while rings[ring_offset].stop < start:
        ring_offset += 1
    rings = rings[ring_offset:]
    samples = rings[-1].last - rings[0].first + 1

    return (global_start, global_first, samples, rings, modes, modenames, ring_offset)


def bolos_by_type(bolotype):
    vals = {
        "P100": [
            "100-1a",
            "100-1b",
            "100-2a",
            "100-2b",
            "100-3a",
            "100-3b",
            "100-4a",
            "100-4b",
        ],
        "P143": [
            "143-1a",
            "143-1b",
            "143-2a",
            "143-2b",
            "143-3a",
            "143-3b",
            "143-4a",
            "143-4b",
        ],
        "S143": ["143-5", "143-6", "143-7", "143-8"],
        "P217": [
            "217-5a",
            "217-5b",
            "217-6a",
            "217-6b",
            "217-7a",
            "217-7b",
            "217-8a",
            "217-8b",
        ],
        "S217": ["217-1", "217-2", "217-3", "217-4"],
        "P353": [
            "353-3a",
            "353-3b",
            "353-4a",
            "353-4b",
            "353-5a",
            "353-5b",
            "353-6a",
            "353-6b",
        ],
        "S353": ["353-1", "353-2", "353-7", "353-8"],
        "S545": ["545-1", "545-2", "545-3", "545-4"],
        "S857": ["857-1", "857-2", "857-3", "857-4"],
        "SDRK": ["Dark-1", "Dark-2"],
    }
    return vals[bolotype]


def bolos_by_p(bolotype):
    vals = {
        "P": [
            "100-1a",
            "100-1b",
            "100-2a",
            "100-2b",
            "100-3a",
            "100-3b",
            "100-4a",
            "100-4b",
            "143-1a",
            "143-1b",
            "143-2a",
            "143-2b",
            "143-3a",
            "143-3b",
            "143-4a",
            "143-4b",
            "217-5a",
            "217-5b",
            "217-6a",
            "217-6b",
            "217-7a",
            "217-7b",
            "217-8a",
            "217-8b",
            "353-3a",
            "353-3b",
            "353-4a",
            "353-4b",
            "353-5a",
            "353-5b",
            "353-6a",
            "353-6b",
        ],
        "S": [
            "143-5",
            "143-6",
            "143-7",
            "143-8",
            "217-1",
            "217-2",
            "217-3",
            "217-4",
            "353-1",
            "353-2",
            "353-7",
            "353-8",
            "545-1",
            "545-2",
            "545-3",
            "545-4",
            "857-1",
            "857-2",
            "857-3",
            "857-4",
            "Dark-1",
            "Dark-2",
        ],
    }
    return vals[bolotype]


def bolo_types():
    return [
        "P100",
        "P143",
        "S143",
        "P217",
        "S217",
        "P353",
        "S353",
        "S545",
        "S857",
        "SDRK",
    ]


def bolos_by_freq(freq):
    ret = []
    for t in ["P", "S"]:
        key = "{}{}".format(t, freq)
        if key in bolo_types():
            ret.extend(bolos_by_type(key))
    return ret


def bolos():
    return [
        "100-1a",
        "100-1b",
        "143-1a",
        "143-1b",
        "217-1",
        "353-1",
        "143-5",
        "217-5a",
        "217-5b",
        "353-2",
        "545-1",
        "Dark-1",
        "100-2a",
        "100-2b",
        "217-2",
        "353-3a",
        "353-3b",
        "857-1",
        "143-2a",
        "143-2b",
        "353-4a",
        "353-4b",
        "545-2",
        "857-2",
        "100-3a",
        "100-3b",
        "143-6",
        "217-6a",
        "217-6b",
        "353-7",
        "143-3a",
        "143-3b",
        "217-3",
        "353-5a",
        "353-5b",
        "545-3",
        "143-7",
        "217-7a",
        "217-7b",
        "353-6a",
        "353-6b",
        "857-3",
        "143-8",
        "217-8a",
        "217-8b",
        "545-4",
        "857-4",
        "Dark-2",
        "100-4a",
        "100-4b",
        "143-4a",
        "143-4b",
        "217-4",
        "353-8",
    ]


def bolo_to_BC(bolo):
    vals = {
        "100-1a": "00",
        "100-1b": "01",
        "143-1a": "02",
        "143-1b": "03",
        "217-1": "04",
        "353-1": "05",
        "143-5": "10",
        "217-5a": "11",
        "217-5b": "12",
        "353-2": "13",
        "545-1": "14",
        "Dark-1": "15",
        "100-2a": "20",
        "100-2b": "21",
        "217-2": "22",
        "353-3a": "23",
        "353-3b": "24",
        "857-1": "25",
        "143-2a": "30",
        "143-2b": "31",
        "353-4a": "32",
        "353-4b": "33",
        "545-2": "34",
        "857-2": "35",
        "100-3a": "40",
        "100-3b": "41",
        "143-6": "42",
        "217-6a": "43",
        "217-6b": "44",
        "353-7": "45",
        "143-3a": "50",
        "143-3b": "51",
        "217-3": "52",
        "353-5a": "53",
        "353-5b": "54",
        "545-3": "55",
        "143-7": "60",
        "217-7a": "61",
        "217-7b": "62",
        "353-6a": "63",
        "353-6b": "64",
        "857-3": "65",
        "143-8": "70",
        "217-8a": "71",
        "217-8b": "72",
        "545-4": "73",
        "857-4": "74",
        "Dark-2": "75",
        "100-4a": "80",
        "100-4b": "81",
        "143-4a": "82",
        "143-4b": "83",
        "217-4": "84",
        "353-8": "85",
    }
    return vals[bolo]


def bolo_to_pnt(bolo):
    vals = {
        "100-1a": "00_100_1a",
        "100-1b": "01_100_1b",
        "143-1a": "02_143_1a",
        "143-1b": "03_143_1b",
        "217-1": "04_217_1",
        "353-1": "05_353_1",
        "143-5": "10_143_5",
        "217-5a": "11_217_5a",
        "217-5b": "12_217_5b",
        "353-2": "13_353_2",
        "545-1": "14_545_1",
        "Dark-1": "15_Dark1",
        "100-2a": "20_100_2a",
        "100-2b": "21_100_2b",
        "217-2": "22_217_2",
        "353-3a": "23_353_3a",
        "353-3b": "24_353_3b",
        "857-1": "25_857_1",
        "143-2a": "30_143_2a",
        "143-2b": "31_143_2b",
        "353-4a": "32_353_4a",
        "353-4b": "33_353_4b",
        "545-2": "34_545_2",
        "857-2": "35_857_2",
        "100-3a": "40_100_3a",
        "100-3b": "41_100_3b",
        "143-6": "42_143_6",
        "217-6a": "43_217_6a",
        "217-6b": "44_217_6b",
        "353-7": "45_353_7",
        "143-3a": "50_143_3a",
        "143-3b": "51_143_3b",
        "217-3": "52_217_3",
        "353-5a": "53_353_5a",
        "353-5b": "54_353_5b",
        "545-3": "55_545_3",
        "143-7": "60_143_7",
        "217-7a": "61_217_7a",
        "217-7b": "62_217_7b",
        "353-6a": "63_353_6a",
        "353-6b": "64_353_6b",
        "857-3": "65_857_3",
        "143-8": "70_143_8",
        "217-8a": "71_217_8a",
        "217-8b": "72_217_8b",
        "545-4": "73_545_4",
        "857-4": "74_857_4",
        "Dark-2": "75_Dark2",
        "100-4a": "80_100_4a",
        "100-4b": "81_100_4b",
        "143-4a": "82_143_4a",
        "143-4b": "83_143_4b",
        "217-4": "84_217_4",
        "353-8": "85_353_8",
    }
    return vals[bolo]


def bolo_to_ADU(bolo):
    return "HFI_" + bolo_to_BC(bolo) + "_C"


def freq_to_fwhm(freq):
    ifreq = int(freq)
    fwhms = {
        30: 33,
        44: 24,
        70: 14,
        100: 10,
        143: 7.1,
        217: 5.5,
        353: 5,
        545: 5,
        857: 5,
    }
    if ifreq not in fwhms:
        raise RuntimeError("Unknown frequency: {}".format(freq))
    return fwhms[ifreq]


DEFAULT_PARAMETERS = {
    "solsys_speed": 370.082,
    "solsys_glon": 264.00,
    "solsys_glat": 48.24,
}

# Detector noise weights measured by libmadam and used in
# npipe6v19 production

detector_weights = {
    # 30 GHz
    "LFI27": 0.40164e06,
    "LFI28": 0.36900e06,
    # 44 GHz
    "LFI24": 0.12372e06,
    "LFI25": 0.14049e06,
    "LFI26": 0.11233e06,
    # 70 GHz
    "LFI18": 53650,
    "LFI19": 42141,
    "LFI20": 36579,
    "LFI21": 50355,
    "LFI22": 49363,
    "LFI23": 47966,
    # 100 GHz
    "100-1": 0.76343e06,
    "100-2": 0.12661e07,
    "100-3": 0.10631e07,
    "100-4": 0.10532e07,
    # 143 GHz
    "143-1": 0.16407e07,
    "143-2": 0.18577e07,
    "143-3": 0.16439e07,
    "143-4": 0.14458e07,
    "143-5": 0.27630e07,
    "143-6": 0.26942e07,
    "143-7": 0.28599e07,
    # 217 GHz
    "217-1": 0.11058e07,
    "217-2": 0.10261e07,
    "217-3": 0.10958e07,
    "217-4": 0.10593e07,
    "217-5": 0.67318e06,
    "217-6": 0.71092e06,
    "217-7": 0.76576e06,
    "217-8": 0.71226e06,
    # 353 GHz
    "353-1": 0.12829e06,
    "353-2": 0.13475e06,
    "353-3": 48067,
    "353-4": 42187,
    "353-5": 56914,
    "353-6": 25293,
    "353-7": 87730,
    "353-8": 74453,
    # 545 GHz
    "545-1": 4475.5,
    "545-2": 5540.3,
    "545-4": 4321.0,
    # 857 GHz
    "857-1": 6.8895,
    "857-2": 6.3108,
    "857-3": 6.5964,
    "857-4": 3.6785,
}


def qp_file(
    outdir,
    dets,
    lmax=2000,
    smax=6,
    angle_shift=0.0,
    full=True,
    pconv="cmbfast",
    force_det=None,
    release=None,
    rhobeam=None,
    rhohit=None,
):
    """ qb_file()
    returns quickpol filename
    """
    if not os.path.isfile(outdir):
        os.makedirs(outdir, exist_ok=True)

    lmax_def = 3000

    if lmax is None:
        lmax = lmax_def
    if smax is None:
        smax = 6
    if angle_shift is None:
        angle_shift = 0.0
    if full is None:
        full = True
    if force_det is not None:
        sfd = "_FD%s" % (force_det)
    else:
        sfd = ""

    if rhobeam == "Ideal":
        srb = ""
    elif rhobeam == "IMO":
        srb = "_rbIMO"
    else:
        raise RuntimeError("Unknown rhobeam: {}".format(rhobeam))
        print('Unknown rhobeam:"{}"'.format(rhobeam))

    if rhohit == "Ideal":
        srh = ""
    elif rhohit == "IMO":
        srh = "_rhIMO"
    else:
        raise RuntimeError("Unknown rhohit: {}".format(rhohit))

    angst = "%+03d" % (angle_shift)
    angst = angst.replace("+180", "180")
    angst = angst.replace("-180", "180")
    angst = angst.replace("+00", "000")

    fz = os.path.join(
        outdir,
        "beam_matrix_{}x{}_l{}_s{}_A{}_{}_{}{}{}{}.npz".format(
            dets[0],
            dets[1],
            str(lmax),
            str(smax),
            angst,
            pconv,
            str(full * 1),
            sfd,
            srb,
            srh,
        ),
    )

    return fz
