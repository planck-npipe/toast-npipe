#!/usr/bin/env python

# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
if 'TOAST_STARTUP_DELAY' in os.environ:
    import numpy as np
    import time
    delay = float(os.environ['TOAST_STARTUP_DELAY'])
    wait = np.random.rand() * delay
    print('Sleeping for {} seconds before importing TOAST'.format(wait),
          flush=True)
    time.sleep(wait)
"""
# from memory_profiler import profile
# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)

# TOAST must be imported before anaconda Python to ensure
# the right MKL library is used
import toast

import argparse
import datetime
import os
import re
import socket
import sys
import traceback

import numpy as np

from toast import Comm, Data, distribute_discrete
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm
from toast.todmap import OpMadam, OpSimConviqt
from toast.tod import OpSimNoise

from toast_planck.utilities import (
    load_RIMO,
    to_radiometer,
    list_planck,
    cfreq_deltas,
)
import toast_planck as tp
from toast_planck.signal_sim import CMBCACHE
from toast_planck.preproc_modules import MapSampler


def test_performance(mpiworld):
    if "TOAST_PLANCK_TEST_PERFORMANCE" not in os.environ:
        return
    if int(os.environ["TOAST_STARTUP_DELAY"]) == 0:
        return
    n = 2 ** 10
    x = np.arange(n ** 2).reshape([n, n])
    y = np.arange(n ** 2).reshape([n, n])
    timer = Timer()
    timer.start()
    x = np.dot(x, y)
    timer.stop()
    t = timer.seconds()
    if mpiworld is None:
        all_t = t
    else:
        all_t = np.array(mpiworld.allgather(t))
    tmin = np.amin(all_t)
    tmax = np.amax(all_t)
    if mpiworld is None or mpiworld.rank == 0:
        print(
            "Matrix product best time: {:.3f} s. Worst time = {:.3f} s. "
            "Mean = {:.3f} s. Median = {:.3f} s. Sdev = {:.3f} s"
            "".format(tmin, tmax, np.mean(all_t), np.median(all_t), np.std(all_t)),
            flush=True,
        )
    host = socket.gethostname()
    if t > tmin * 1.2:
        print(
            "Rank {} on {} ran {:.3f} times slower than the fastest test."
            "".format(comm.rank, host, t / tmin),
            flush=True,
        )
    if mpiworld is not None:
        mpiworld.Barrier()
    if tmax / tmin > 2:
        raise RuntimeError(
            "Performance varies by a factor of {:.3f} across "
            "the communicator.".format(tmax / tmin)
        )
    return


def add_sim_params(parser):
    # CMB parameters
    parser.add_argument(
        "--cmb_alm",
        required=False,
        default=None,
        help="Input a_lm for simulated signal. "
        "MC ID is Python-formatted into the string",
    )
    parser.add_argument(
        "--sim_nside",
        required=False,
        default=2048,
        type=int,
        help="Resolution for interpolation",
    )
    # Conviqt parameters
    parser.add_argument(
        "--conviqt_lmax", default=0, type=int, help="Simulation lmax"
    )
    parser.add_argument(
        "--conviqt_fwhm",
        default=0,
        type=float,
        help="Sky FWHM [arcmin] to deconvolve",
    )
    parser.add_argument("--conviqt_mmax", default=0, type=int, help="Beam mmax")
    parser.add_argument(
        "--conviqt_order", default=11, type=int, help="Iteration order"
    )
    parser.add_argument(
        "--conviqt_pxx",
        default=False,
        action="store_true",
        help="Beams are in Pxx frame, not Dxx",
    )
    parser.add_argument(
        "--conviqt_sky",
        required=False,
        help="Path to sky alm files. Tag DETECTOR will be "
        "replaced with detector name.",
    )
    parser.add_argument(
        "--conviqt_fsl",
        required=False,
        help="Path to beam alm files. Tag DETECTOR will be "
        "replaced with detector name.  Use ',' to list several FSL flavors",
    )
    parser.add_argument(
        "--conviqt_beamfile",
        required=False,
        help="Path to beam alm files. Tag DETECTOR will be "
        "replaced with detector name.",
    )
    # FSL beam mask
    parser.add_argument(
        "--fslbeam_mask",
        required=False,
        help="Path to pixelized FSL beam mask",
    )
    # Bandpass for foreground
    parser.add_argument(
        "--freq_sigma",
        required=False,
        default=1.00,
        type=float,
        help="Width of the center frequency "
        "distribution. 0 = use hard-coded values.",
    )
    # Optional transfer function input for simulation
    parser.add_argument(
        "--sim_tf",
        required=False,
        default=None,
        help="Input TF for simulated signal. DETECTOR is replaced with actual name",
    )
    # Additional bandpass mismatch maps for simulation
    parser.add_argument(
        "--bpm_extra",
        required=False,
        default=None,
        help="Additional BPM templates for sim. "
        "DETECTOR is replaced with actual name",
    )
    # Monte Carlo parameters
    parser.add_argument(
        "--MC_start",
        required=False,
        default=0,
        type=int,
        help="First Monte Carlo noise realization",
    )
    parser.add_argument(
        "--MC_count",
        required=False,
        default=1,
        type=int,
        help="Number of Monte Carlo noise realizations",
    )
    # Noise parameters
    parser.add_argument(
        "--noisefile",
        required=False,
        default="RIMO",
        help="Path to noise PSD files for noise filter. "
        "Tag DETECTOR will be replaced with detector name.",
    )
    parser.add_argument(
        "--noisefile_simu",
        required=False,
        default="RIMO",
        help="Path to noise PSD files for noise simulation. "
        "Tag DETECTOR will be replaced with detector name.",
    )
    parser.add_argument(
        "--simulate_noise",
        dest="simulate_noise",
        default=False,
        action="store_true",
        help="Simulate and add noise to signal.",
    )
    # Decalibration
    parser.add_argument(
        "--decalibrate",
        required=False,
        help="Path to calibration file to decalibrate with. You can use python string "
        "formatting, assuming .format(mc)",
    )
    # quickpol beam window function for CMB and sky model
    parser.add_argument(
        "--quickpolbeam",
        required=False,
        help="Path to symmetric b_ell window function file. "
        "Tag DETECTOR will be replaced with detector name.",
    )
    # Cached sky model files for simulated foregrounds
    parser.add_argument(
        "--skymodel_sim",
        required=False,
        help="Path to cached sky model file for simulation.",
    )
    parser.add_argument(
        "--skymodelderiv_sim",
        required=False,
        help="Path to cached sky model file for simulation.",
    )
    # Cached sky model files for simulated foregrounds
    parser.add_argument(
        "--skymodel",
        required=False,
        help="Path to cached sky model file for simulation.",
    )
    parser.add_argument(
        "--skymodelderiv",
        required=False,
        help="Path to cached sky model file for simulation.",
    )
    return


def add_input_params(parser):
    parser.add_argument(
        "--effdir_in", required=True, help="Input Exchange Format File directory"
    )
    parser.add_argument(
        "--effdir_in_diode0",
        required=False,
        default=None,
        help="Input Exchange Format File directory, LFI diode 0",
    )
    parser.add_argument(
        "--effdir_in_diode1",
        required=False,
        default=None,
        help="Input Exchange Format File directory, LFI diode 1",
    )
    parser.add_argument(
        "--effdir_fsl",
        required=False,
        help="Input Exchange Format File directory for far side  lobe TOD",
    )
    parser.add_argument(
        "--effdir_pntg",
        required=False,
        help="Input Exchange Format File directory for pointing",
    )
    parser.add_argument("--ringdb", required=True, help="Ring DB file")
    parser.add_argument(
        "--odfirst", required=False, default=None, type=int, help="First OD to use"
    )
    parser.add_argument(
        "--odlast", required=False, default=None, type=int, help="Last OD to use"
    )
    parser.add_argument(
        "--ringfirst",
        required=False,
        default=None,
        type=int,
        help="First ring to use",
    )
    parser.add_argument(
        "--ringlast", required=False, default=None, type=int, help="Last ring to use"
    )
    parser.add_argument(
        "--obtfirst",
        required=False,
        default=None,
        type=float,
        help="First OBT to use",
    )
    parser.add_argument(
        "--obtlast", required=False, default=None, type=float, help="Last OBT to use"
    )
    parser.add_argument(
        "--effdir_dark",
        required=False,
        help="Input Exchange Format File directory for dark bolometer data",
    )
    parser.add_argument(
        "--darkmask", required=False, default=1, type=int, help="Dark flag mask"
    )
    parser.add_argument(
        "--lfi_raw",
        dest="lfi_raw",
        default=False,
        action="store_true",
        help="Input LFI data is undifferenced",
    )
    parser.add_argument(
        "--margin",
        required=False,
        default=1000,
        type=int,
        help="Preprocessing margin.",
    )
    parser.add_argument(
        "--flagmask", required=False, default=1, type=int, help="Quality flag mask"
    )
    parser.add_argument(
        "--obtmask", required=False, default=1, type=int, help="OBT flag mask"
    )
    parser.add_argument(
        "--pntmask", required=False, default=2, type=int, help="Pointing flag mask"
    )
    parser.add_argument("--ssomask", required=False, type=int, help="SSO flag mask")
    return


def add_beam_params(parser):
    parser.add_argument(
        "--beam_targets",
        required=False,
        default="JUPITER,SATURN,MARS,URANUS,NEPTUNE",
        help="Beam target list (comma separated)",
    )
    parser.add_argument(
        "--beam_iter",
        required=False,
        default=3,
        type=int,
        help="Number of beam iterations",
    )
    parser.add_argument("--beam_fits", required=False, help="Beam output file.")
    parser.add_argument(
        "--beam_fits_polar", required=False, help="Polar beam output file."
    )
    return


def add_shdet_params(parser):
    parser.add_argument(
        "--effdir_optical",
        required=False,
        help="Input Exchange Format File directory for optical signal (for SHDET)",
    )
    # The following options are for SHDet; all bolometers get the same value
    parser.add_argument(
        "--shdet_adc_on",
        default=False,
        action="store_true",
        required=False,
        help="use ADC in SHDet, returning digits instead of volts.",
    )
    parser.add_argument(
        "--shdet_seed",
        required=False,
        default=0,
        type=float,
        help="SHDet random number generator base seed.",
    )
    parser.add_argument(
        "--shdet_offset_file",
        required=False,
        help="Pre and post modulation offset pickle file. "
        "Tag DETECTOR will be replaced with the appropriate "
        "detector name.",
    )
    parser.add_argument(
        "--shdet_switch_optical_offset",
        required=False,
        default=1,
        help="Switch on/off the pre modulation offset. on=1, off=0.",
    )
    parser.add_argument(
        "--shdet_switch_raw_offset",
        required=False,
        default=1,
        help="Switch on/off the post modulation offset. on=1, off=0.",
    )

    # The following options are for SHDet, taking comma-seperated lists as a
    # rguments, expecting either one item or one item per bolometer.
    parser.add_argument(
        "--shdet_noise_dsn",
        default=None,
        required=False,
        help="White noise generated by SHDet in DSN",
    )
    parser.add_argument(
        "--shdet_adc_table",
        default=None,
        required=False,
        help="Path to ADC tables for SHDet or DPC-style ADC NL correction.",
    )
    parser.add_argument(
        "--shdet_optical_load_watts",
        default=None,
        required=False,
        help="Optical constant load in W.",
    )
    parser.add_argument(
        "--shdet_gain_w_per_kcmb",
        default=None,
        required=False,
        help="Gain of the detector in W per K CMB.",
    )
    parser.add_argument(
        "--shdet_bdac",
        default=None,
        required=False,
        help="BDAC value for the compensation signal",
    )
    parser.add_argument(
        "--shdet_cstray",
        default=None,
        required=False,
        help="Stray capacitance value (F)",
    )
    parser.add_argument(
        "--shdet_raw_model",
        default=None,
        required=False,
        help="Substitute the bolo+electronics by the linear raw model",
    )

    # The following options are for preproc, taking comma-seperated
    # lists as arguments, expecting either one item or one item per
    # bolometer.
    parser.add_argument(
        "--shdet_tffile",
        default=None,
        required=False,
        help="Path to transfer function text file for use with SHDet.",
    )
    parser.add_argument(
        "--shdet_global_phase_shift",
        required=False,
        type=float,
        default=0,
        help="An global phase shift (in samples) to use in reproc Fourier filter.",
    )
    parser.add_argument(
        "--shdet_g0",
        required=False,
        default=None,
        help="The Watts/Volt conversion factor.  By default read this from the IMO.",
    )
    parser.add_argument(
        "--shdet_v0",
        required=False,
        default=None,
        help="Nonlinear gain correction.  By default this is read from the IMO.",
    )
    return


def add_preproc_params(parser):
    parser.add_argument(
        "--effdir_out_preproc", required=False, help="Output directory for preproc"
    )
    parser.add_argument(
        "--preproc_mask", required=False, help="Preproc processing mask file"
    )
    parser.add_argument(
        "--preproc_mask_adc", required=False, help="Preproc ADC NL processing mask file"
    )
    parser.add_argument(
        "--preproc_dark",
        dest="preproc_dark",
        default=False,
        action="store_true",
        help="Preprocess dark data",
    )
    parser.add_argument(
        "--preproc_common",
        dest="preproc_common",
        default=False,
        action="store_true",
        help="Preprocess common data",
    )
    parser.add_argument(
        "--nbin",
        required=False,
        default=10000,
        type=int,
        help="Number of phase bins",
    )
    parser.add_argument(
        "--jump_filter_len",
        required=False,
        default=40000,
        type=int,
        help="Jump filter length",
    )
    parser.add_argument(
        "--jump_threshold",
        required=False,
        default=5.0,
        type=float,
        help="Jump detection threshold",
    )
    parser.add_argument(
        "--preproc_timeout",
        required=False,
        default=120,
        type=int,
        help="Maximum time allowed for preprocessing a ring",
    )
    parser.add_argument(
        "--preproc_timeout_intermediate",
        required=False,
        default=60,
        type=int,
        help="Maximum time allowed for preprocessing a ring before last iteration",
    )
    parser.add_argument(
        "--adc_correction", required=False, help="Full (new) NL correction file."
    )
    parser.add_argument(
        "--measure_ADC",
        dest="measure_ADC",
        default=False,
        action="store_true",
        help="Measure ADC NL",
    )
    parser.add_argument(
        "--niter_ADC", default=1, type=int, help="Number of ADC NL iterations"
    )
    parser.add_argument(
        "--delta_ADC", default=1.0, type=float, help="Width of ADC bin in ADU"
    )
    parser.add_argument(
        "--nphase4k",
        required=False,
        default=2,
        type=int,
        help="Number of 4K cooler phases to measure ADC NL for.",
    )
    parser.add_argument(
        "--skip_preproc",
        dest="skip_preproc",
        default=False,
        action="store_true",
        help="Do not pre-process the TOD",
    )
    parser.add_argument(
        "--flag_planets",
        dest="flag_planets",
        default=False,
        action="store_true",
        help="Derive planet flags",
    )
    parser.add_argument(
        "--planet_flag_radius",
        required=False,
        default=2.0,
        type=float,
        help="New planet flag radius (in FWHM) when --flag_planets",
    )
    parser.add_argument(
        "--detmask", required=False, type=int, help="Detector flag mask"
    )
    parser.add_argument(
        "--intense_threshold",
        required=False,
        default=1e10,
        type=float,
        help="Intense signal threshold [K_CMB]",
    )
    parser.add_argument(
        "--preproc_async_time",
        required=False,
        default=1000,
        type=int,
        help="Initial asynchronous processing time before load balancing",
    )
    parser.add_argument(
        "--preproc_async_time_intermediate",
        required=False,
        default=800,
        type=int,
        help="Initial asynchronous processing time before "
        "load balancing before last iteration",
    )
    return


def add_reproc_params(parser):
    parser.add_argument(
        "--tfmode",
        dest="tfmode",
        default=False,
        action="store_true",
        help="Run the simulation in transfer function measurement mode",
    )
    parser.add_argument(
        "--polparammode",
        dest="polparammode",
        default=False,
        action="store_true",
        help="Run the simulation in pol. params measurement mode",
    )
    parser.add_argument(
        "--save_destriper_data",
        dest="save_destriper_data",
        default=False,
        action="store_true",
        help="Save all of the phase-ordered destriper data for plotting",
    )
    parser.add_argument(
        "--reproc_effdir_out", required=False, help="Output directory for reproc"
    )
    parser.add_argument(
        "--reproc_polmap",
        required=False,
        default="",
        help="Reprocessing polarization template",
    )
    parser.add_argument(
        "--reproc_polmap2",
        required=False,
        default="",
        help="Reprocessing polarization template2",
    )
    parser.add_argument(
        "--reproc_polmap3",
        required=False,
        default="",
        help="Reprocessing polarization template3",
    )
    parser.add_argument(
        "--reproc_force_polmaps",
        dest="reproc_force_polmaps",
        default=False,
        action="store_true",
        help="Use the polmaps without fitting to construct the polarization prior",
    )
    parser.add_argument(
        "--reproc_pol_fwhm",
        required=False,
        default=60,
        type=float,
        help="Reproc polarization resolution",
    )
    parser.add_argument(
        "--reproc_pol_lmax",
        required=False,
        default=512,
        type=int,
        help="Reproc polarization resolution",
    )
    parser.add_argument(
        "--reproc_pol_nside",
        required=False,
        default=256,
        type=int,
        help="Reproc polarization resolution",
    )
    parser.add_argument(
        "--reproc_pixlim",
        required=False,
        default=1e-2,
        type=float,
        help="Reproc destriper pixel rejection threshold",
    )
    parser.add_argument(
        "--reproc_detmask",
        required=False,
        default=1 + 8,
        type=int,
        help="Reproc detector flag mask",
    )
    parser.add_argument(
        "--reproc_nharm",
        required=False,
        default=20,
        type=int,
        help="Number of passbands in differentiation.",
    )
    parser.add_argument(
        "--reproc_do_bands",
        dest="reproc_do_bands",
        default=False,
        action="store_true",
        help="Include passband calibration with nharm != 0.",
    )
    parser.add_argument(
        "--skip_reproc",
        dest="skip_reproc",
        default=False,
        action="store_true",
        help="Do not re-process the TOD",
    )
    parser.add_argument(
        "--reproc_zodi",
        dest="reproc_zodi",
        default=False,
        action="store_true",
        help="Subtract Zodiacal light in reproc",
    )
    parser.add_argument(
        "--reproc_zodi_total",
        dest="reproc_zodi_total",
        default=False,
        action="store_true",
        help="Subtract total Zodiacal model, not just seasonally varying part",
    )
    parser.add_argument(
        "--reproc_zodi_detector",
        dest="reproc_zodi_detector",
        default=False,
        action="store_true",
        help="Fit Zodi for each detector independently",
    )
    parser.add_argument(
        "--reproc_zodi_cache", default="./zodi_cache", help="Zodi cache directory"
    )
    parser.add_argument(
        "--reproc_quss_correct",
        dest="reproc_quss_correct",
        default=False,
        action="store_true",
        help="Use QUSS to improve bandpass correction",
    )
    parser.add_argument(
        "--reproc_temperature_only",
        dest="reproc_temperature_only",
        default=False,
        action="store_true",
        help="Run reproc unpolarized",
    )
    parser.add_argument(
        "--reproc_temperature_only_destripe",
        dest="reproc_temperature_only_destripe",
        default=False,
        action="store_true",
        help="Run reproc fitting unpolarized",
    )
    parser.add_argument(
        "--reproc_temperature_only_intermediate",
        dest="reproc_temperature_only_intermediate",
        default=False,
        action="store_true",
        help="Run reproc intermediate mapmaking unpolarized",
    )
    parser.add_argument(
        "--reproc_CMB", required=False, help="CMB map file for calibration."
    )
    parser.add_argument(
        "--reproc_CO", required=False, help="CO map file for bandpass correction."
    )
    parser.add_argument(
        "--reproc_CO2",
        required=False,
        help="Second CO map file for bandpass correction.",
    )
    parser.add_argument(
        "--reproc_CO3",
        required=False,
        help="Third CO map file for bandpass correction.",
    )
    parser.add_argument(
        "--reproc_dust", required=False, help="Dust map file for bandpass correction."
    )
    parser.add_argument(
        "--reproc_dust_pol",
        required=False,
        help="Dust polarization map file for bandpass correction.",
    )
    parser.add_argument(
        "--reproc_sync",
        required=False,
        help="Synchrotron map file for bandpass correction.",
    )
    parser.add_argument(
        "--reproc_sync_pol",
        required=False,
        help="Synchrotron polarization_map file for bandpass correction.",
    )
    parser.add_argument(
        "--reproc_AME", required=False, help="AME map file for bandpass correction."
    )
    parser.add_argument(
        "--reproc_freefree",
        required=False,
        help="Free-free map file for bandpass correction.",
    )
    parser.add_argument(
        "--reproc_nside",
        required=False,
        default=512,
        type=int,
        help="Reprocessing resolution",
    )
    parser.add_argument(
        "--reproc_nside_bandpass",
        required=False,
        default=256,
        type=int,
        help="Reproc bandpass resolution",
    )
    parser.add_argument(
        "--reproc_fwhm_bandpass",
        required=False,
        default=60,
        type=float,
        help="Reproc bandpass resolution",
    )
    parser.add_argument(
        "--reproc_symmetrize",
        dest="reproc_symmetrize",
        default=False,
        action="store_true",
        help="Symmetrize pointing and flags in each horn",
    )
    parser.add_argument(
        "--reproc_restore_dipole",
        dest="reproc_restore_dipole",
        required=False,
        default=False,
        action="store_true",
        help="Restore dipole in the preprocessed TOD",
    )
    parser.add_argument("--reproc_mask", required=False, help="Reproc mask file")
    parser.add_argument(
        "--reproc_bandpass_mask", required=False, help="Reproc bandpass mask file"
    )
    parser.add_argument(
        "--reproc_project_mask", required=False, help="Reproc projection mask file"
    )
    parser.add_argument(
        "--reproc_niter",
        required=False,
        default=1,
        type=int,
        help="Number of reproc iterations",
    )
    parser.add_argument(
        "--reproc_calibrate",
        dest="reproc_calibrate",
        default=False,
        action="store_true",
        help="Variable calibration",
    )
    parser.add_argument(
        "--reproc_fit_distortion",
        dest="reproc_fit_distortion",
        default=False,
        action="store_true",
        help="Variable signal distortion",
    )
    parser.add_argument(
        "--reproc_nlcalibrate",
        dest="reproc_nlcalibrate",
        default=False,
        action="store_true",
        help="Nonlinear calibration",
    )
    parser.add_argument(
        "--reproc_effective_amp_limit",
        required=False,
        type=float,
        default=0.02,
        help="Effective dipole amplitude limit.",
    )
    parser.add_argument(
        "--reproc_gain_step_mode",
        required=False,
        help="Either none, mission, years or surveys",
    )
    parser.add_argument(
        "--reproc_min_step_length",
        required=False,
        type=int,
        default=10,
        help="Minimum gain step length.",
    )
    parser.add_argument(
        "--reproc_max_step_length",
        required=False,
        type=int,
        default=100,
        help="Maximum gain step length.",
    )
    parser.add_argument(
        "--reproc_outlier_threshold",
        required=False,
        type=float,
        default=10,
        help="Outlier ring flagging threshold in units of RMS.",
    )
    parser.add_argument(
        "--reproc_recalibrate",
        dest="reproc_recalibrate",
        default=False,
        action="store_true",
        help="Adjust overall calibration using orbital dipole fit.",
    )
    parser.add_argument(
        "--reproc_fgdipole",
        dest="reproc_fgdipole",
        default=False,
        action="store_true",
        help="Fit and correct for orbital Doppler effect on the Galactic foreground",
    )
    parser.add_argument(
        "--reproc_save_maps",
        dest="reproc_save_maps",
        default=False,
        action="store_true",
        help="Save intermediate maps from reproc.",
    )
    parser.add_argument(
        "--reproc_save_survey_maps",
        dest="reproc_save_survey_maps",
        default=False,
        action="store_true",
        help="Save single detector, single survey maps from reproc.",
    )
    parser.add_argument(
        "--reproc_save_template_maps",
        dest="reproc_save_template_maps",
        default=False,
        action="store_true",
        help="Save single detector template maps from reproc.",
    )
    parser.add_argument(
        "--reproc_save_single_maps",
        dest="reproc_save_single_maps",
        default=False,
        action="store_true",
        help="Save single detector maps from reproc.",
    )
    parser.add_argument(
        "--reproc_nside_single",
        required=False,
        type=int,
        help="Single detector map resolution",
    )
    parser.add_argument(
        "--reproc_forcepol",
        dest="reproc_forcepol",
        default=False,
        action="store_true",
        help="Force polarization template to unity amplitude in reproc.",
    )
    parser.add_argument(
        "--reproc_forcefsl",
        dest="reproc_forcefsl",
        default=False,
        action="store_true",
        help="Force FSL template(s) to unity amplitude in reproc.",
    )
    parser.add_argument(
        "--reproc_asymmetric_fsl",
        dest="reproc_asymmetric_fsl",
        default=False,
        action="store_true",
        help="Fit FSL independently, not by horn",
    )
    parser.add_argument(
        "--reproc_bpcorrect",
        dest="reproc_bpcorrect",
        default=False,
        action="store_true",
        help="Apply templated bandpass correction in reproc",
    )
    parser.add_argument(
        "--reproc_pscorrect",
        dest="reproc_pscorrect",
        default=False,
        action="store_true",
        help="Remove brightest sources from the sky model",
    )
    parser.add_argument(
        "--reproc_psradius",
        required=False,
        type=float,
        default=30,
        help="Radius to excise in arc min when --reproc_pscorrect.",
    )
    parser.add_argument(
        "--reproc_force_conserve_memory",
        dest="reproc_force_conserve_memory",
        default=False,
        action="store_true",
        help="Force libmadam conserve_memory.",
    )
    return


def parse_shdet_params(comm, args, detectors):
    if not args.simulate:
        return None, None, None

    ndets = len(detectors)

    # some arguments need to be split from a string into a list of floats
    arglist = {}
    for argname, arg_to_check in [
        ("optical_load_watts", args.shdet_optical_load_watts),
        ("gain_w_per_kcmb", args.shdet_gain_w_per_kcmb),
        ("cstray", args.shdet_cstray),
        ("bdac", args.shdet_bdac),
        ("noise_dsn", args.shdet_noise_dsn),
        ("g0", args.shdet_g0),
        ("v0", args.shdet_v0),
        ("global_phase_shift", str(args.shdet_global_phase_shift)),
        ("raw_model", args.shdet_raw_model),
    ]:
        if arg_to_check is not None:
            tmplist = re.split(",", arg_to_check)
            npars = len(tmplist)
            if (npars != 1) and (npars != ndets):
                raise Exception(
                    "{:4} : ERROR: list parameters {} needs to "
                    "have either 1 argument or ndet arguments"
                    "".format(comm.comm_world.rank, argname)
                )
            # convert to floats
            arglist[argname] = [float(i) for i in tmplist]
        else:
            arglist[argname] = None

    for argname, arg_to_check in [
        ("adc_table", args.shdet_adc_table),
        ("shdet_tffile", args.shdet_tffile),
    ]:
        if arg_to_check is not None:
            tmplist = re.split(",", arg_to_check)
            npars = len(tmplist)
            if (npars != 1) & (npars != ndets):
                raise Exception(
                    "{:4} : ERROR: list parameters {} needs to "
                    "have either 1 argument or ndet arguments"
                    "".format(comm.comm_world.rank, argname)
                )
            arglist[argname] = tmplist
        else:
            arglist[argname] = None

    shdet_params = {}

    adc_table_path = None
    if "adc_table" in arglist.keys():
        if arglist["adc_table"] is not None:
            adc_table_path = {}

    # TF files
    tf_path = None
    if "shdet_tffile" in arglist.keys():
        if arglist["shdet_tffile"] is not None:
            tf_path = {}

    global_offset_dict = {}
    for idet, det in enumerate(detectors):
        if args.shdet_adc_table is not None:
            npars = len(arglist["adc_table"])
            if npars == 1:
                adc_table_path[det] = arglist["adc_table"][0]
                if npars == ndets:
                    adc_table_path[det] = arglist["adc_table"][idet]

        if args.shdet_tffile is not None:
            npars = len(arglist["shdet_tffile"])
            if npars == 1:
                tf_path[det] = arglist["shdet_tffile"][0]
                if npars == ndets:
                    tf_path[det] = arglist["shdet_tffile"][idet]

        shdet_params[det] = {}

        # base seed, always just set to the value specified on the
        # command line
        shdet_params[det]["seed"] = args.shdet_seed
        shdet_params[det]["switch_optical_offset"] = args.shdet_switch_optical_offset
        shdet_params[det]["switch_raw_offset"] = args.shdet_switch_raw_offset

        if args.adc_on:
            shdet_params[det]["adc_on"] = 1

        if arglist["noise_dsn"] is not None:
            npars = len(arglist["noise_dsn"])
            if npars == 1:
                shdet_params[det]["noise_dsn"] = (
                    arglist["noise_dsn"][0] / np.sqrt(40) * 10.2 / 2 ** 16
                )
            if npars == ndets:
                shdet_params[det]["noise_dsn"] = (
                    arglist["noise_dsn"][idet] / np.sqrt(40.0) * 10.2 / 2 ** 16
                )

        if arglist["cstray"] is not None:
            npars = len(arglist["cstray"])
            if npars == 1:
                shdet_params[det]["cstray"] = arglist["cstray"][0]
            if npars == ndets:
                shdet_params[det]["cstray"] = arglist["cstray"][idet]

        if arglist["bdac"] is not None:
            npars = len(arglist["bdac"])
            if npars == 1:
                shdet_params[det]["bdac"] = arglist["bdac"][0]
            if npars == ndets:
                shdet_params[det]["bdac"] = arglist["bdac"][idet]

        if arglist["optical_load_watts"] is not None:
            npars = len(arglist["optical_load_watts"])
            if npars == 1:
                shdet_params[det]["optical_load_watts"] = arglist["optical_load_watts"][
                    0
                ]
            if npars == ndets:
                shdet_params[det]["optical_load_watts"] = arglist["optical_load_watts"][
                    idet
                ]

        if arglist["gain_w_per_kcmb"] is not None:
            npars = len(arglist["gain_w_per_kcmb"])
            if npars == 1:
                shdet_params[det]["gain_w_per_kcmb"] = arglist["gain_w_per_kcmb"][0]
            if npars == ndets:
                shdet_params[det]["gain_w_per_kcmb"] = arglist["gain_w_per_kcmb"][idet]

        if arglist["raw_model"] is not None:
            npars = len(arglist["raw_model"])
            if npars == 1:
                shdet_params[det]["raw_model"] = arglist["raw_model"][0]
            if npars == ndets:
                shdet_params[det]["raw_model"] = arglist["raw_model"][idet]

        if arglist["global_phase_shift"] is not None:
            npars = len(arglist["global_phase_shift"])
            if npars == 1:
                global_offset_dict[det] = arglist["global_phase_shift"][0]
            if npars == ndets:
                global_offset_dict[det] = arglist["global_phase_shift"][idet]

    return adc_table_path, shdet_params, tf_path


def add_madam_params(parser):
    parser.add_argument(
        "--madampar", required=False, default=None, help="Madam parameter file"
    )
    parser.add_argument("--madam_prefix", required=False, help="map prefix")
    parser.add_argument(
        "--skip_madam",
        dest="skip_madam",
        default=False,
        action="store_true",
        help="Disable Madam mapmaking and only process and save the TOD.",
    )
    parser.add_argument(
        "--map_dir", required=False, default=".", help="Map output directory"
    )
    return


def parse_madam_params(comm, args):
    map_dir = os.path.join(args.out, args.map_dir)
    if not map_dir.endswith(os.sep):
        map_dir += os.sep

    if comm.comm_world.rank == 0:
        os.makedirs(map_dir, exist_ok=True)

    # Read in madam parameter file
    # Allow more than one entry, gather into a list
    repeated_keys = ["detset", "detset_nopol", "survey"]
    pars = {}

    if comm.comm_world.rank == 0:
        pars["kfirst"] = False
        pars["temperature_only"] = True
        pars["base_first"] = 60.0
        pars["nside_submap"] = 16
        pars["write_map"] = False
        pars["write_binmap"] = True
        pars["write_matrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = True
        pars["kfilter"] = False
        pars["info"] = 3
        if args.madampar:
            pat = re.compile(r"\s*(\S+)\s*=\s*(\S+(\s+\S+)*)\s*")
            comment = re.compile(r"^#.*")
            with open(args.madampar, "r") as f:
                for line in f:
                    if not comment.match(line):
                        result = pat.match(line)
                        if result:
                            key, value = result.group(1), result.group(2)
                            if key in repeated_keys:
                                if key not in pars:
                                    pars[key] = []
                                pars[key].append(value)
                            else:
                                pars[key] = value
        # Command line parameters override the ones in the madam parameter file
        if "file_root" not in pars:
            pars["file_root"] = "madam"
        if args.madam_prefix is not None:
            pars["file_root"] = args.madam_prefix
        sfreq = "{:03}".format(args.freq)
        if sfreq not in pars["file_root"]:
            pars["file_root"] += "_" + sfreq
        try:
            fsample = {30: 32.51, 44: 46.55, 70: 78.77}[args.freq]
        except Exception:
            fsample = 180.3737
        pars["fsample"] = fsample
        pars["nside_map"] = args.nside
        if "nside_cross" not in pars:
            pars["nside_cross"] = pars["nside_map"] // 2
        else:
            pars["nside_cross"] = int(pars["nside_cross"])
        if pars["nside_cross"] > pars["nside_map"]:
            print(
                "WARNING: {} has nside_cross = {} > {}, setting it to {}"
                "".format(
                    args.madampar,
                    pars["nside_cross"],
                    pars["nside_map"],
                    pars["nside_map"],
                ),
                flush=True,
            )
            pars["nside_cross"] = pars["nside_map"]
        pars["path_output"] = map_dir

    pars = comm.comm_world.bcast(pars, root=0)

    return map_dir, pars


def parse_arguments():
    comm = Comm()
    if comm.comm_world.rank == 0:
        print(
            "Running with {} processes at {}".format(
                comm.comm_world.size, str(datetime.datetime.now())
            )
        )

    parser = argparse.ArgumentParser(
        description="Planck Data Reduction", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--simulate",
        dest="simulate",
        required=False,
        default=False,
        action="store_true",
        help="Perform an SHDet simulation instead",
    )
    parser.add_argument("--rimo", required=True, help="RIMO file")
    parser.add_argument("--imo", required=True, help="IMO file")
    parser.add_argument("--freq", required=True, type=int, help="Frequency")
    parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="Write data distribution info to file",
    )
    parser.add_argument(
        "--dets", required=False, default=None, help="Detector list (comma separated)"
    )
    parser.add_argument("--out", required=False, default=".", help="Output directory")
    parser.add_argument(
        "--effdir_out_madam", required=False, help="Output directory for Madam"
    )
    parser.add_argument(
        "--nside", required=False, default=1024, type=int, help="Pipeline resolution"
    )
    parser.add_argument("--bg_map", required=False, help="Map template")
    parser.add_argument(
        "--bg_pol",
        dest="bg_pol",
        required=False,
        default=False,
        action="store_true",
        help="Map template polarization",
    )
    parser.add_argument(
        "--bg_has_dipole",
        dest="bg_has_dipole",
        required=False,
        default=False,
        action="store_true",
        help="Background map includes the dipole",
    )
    parser.add_argument(
        "--bg_nside",
        required=False,
        default=1024,
        type=int,
        help="Map template resolution",
    )
    parser.add_argument("--calfile", required=False, help="Calibration file")
    parser.add_argument(
        "--make_rings",
        dest="make_rings",
        default=False,
        action="store_true",
        help="Compile ringsets.",
    )
    parser.add_argument(
        "--nside_ring",
        required=False,
        default=128,
        type=int,
        help="Ringset resolution",
    )
    parser.add_argument(
        "--ring_root",
        required=False,
        default="ringset",
        help="Root filename for ringsets (setting to empty "
        "disables ringset output).",
    )
    parser.add_argument("--bad_rings", required=False, help="Bad ring file.")
    parser.add_argument("--filterfile", required=False, help="Extra filter file.")

    add_input_params(parser)
    add_shdet_params(parser)
    add_preproc_params(parser)
    add_sim_params(parser)
    add_reproc_params(parser)
    add_madam_params(parser)
    add_beam_params(parser)

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if comm.comm_world.rank == 0:
        print(
            "Processing {} at {}GHz. skip_preproc = {}, cmb_alm = {}, "
            "simulate_noise = {}, skip_reproc = {}, skip_madam = {}".format(
                args.dets,
                args.freq,
                args.skip_preproc,
                args.cmb_alm,
                args.simulate_noise,
                args.skip_reproc,
                args.skip_madam,
            )
        )
        print("All parameters:")
        print(args, flush=True)

    # get options

    if comm.comm_world.rank == 0:
        os.makedirs(args.out, exist_ok=True)

    odrange = None
    if args.odfirst is not None and args.odlast is not None:
        odrange = (args.odfirst, args.odlast)
    ringrange = None
    if args.ringfirst is not None and args.ringlast is not None:
        ringrange = (args.ringfirst, args.ringlast)
    obtrange = None
    if args.obtfirst is not None and args.obtlast is not None:
        obtrange = (args.obtfirst, args.obtlast)

    detectors = None
    if args.dets is not None:
        detectors = re.split(",", args.dets)

    if args.skip_preproc:
        # Margins are only used for preprocessing.
        args.margin = 0
    else:
        if args.cmb_alm or args.simulate_noise:
            raise RuntimeError(
                "Preprocessing is not compatible with CMB and noise simulation."
            )

    adc_table_path, shdet_params, tf_path = parse_shdet_params(comm, args, detectors)
    map_dir, madampars = parse_madam_params(comm, args)

    return (
        args,
        map_dir,
        madampars,
        detectors,
        odrange,
        ringrange,
        obtrange,
        adc_table_path,
        shdet_params,
        tf_path,
    )


def create_observations(args, detectors, obtrange, ringrange, odrange):
    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = Comm()

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        print("Computing data distribution ... ", end="", flush=True)
    timer = Timer()

    # The distributed timestream data

    data = Data(comm)

    # Create the single TOD for this observation.  We if are
    # doing a simulation, then just get the pointing and flags
    # from the Exchange data.

    effdir_in = args.effdir_in
    effdir_flags = None
    if args.simulate and args.effdir_optical is not None:
        effdir_in = args.effdir_optical
        effdir_flags = args.effdir_in

    tod = tp.Exchange(
        comm=comm.comm_group,
        detectors=detectors,
        ringdb=args.ringdb,
        effdir_in=effdir_in,
        effdir_flags=effdir_flags,
        effdir_pntg=args.effdir_pntg,
        effdir_fsl=args.effdir_fsl,
        effdir_out=args.effdir_out_preproc,
        effdir_dark=args.effdir_dark,
        obt_range=obtrange,
        ring_range=ringrange,
        od_range=odrange,
        freq=args.freq,
        RIMO=args.rimo,
        obtmask=args.obtmask,
        flagmask=args.flagmask,
        sim=args.simulate,
        darkmask=args.darkmask,
        lfi_raw=args.lfi_raw,
        do_eff_cache=False,
        noisefile=args.noisefile,
        noisefile_simu=args.noisefile_simu,
    )

    if args.tfmode:
        # Suppress noise variance by a factor of 1e6
        for key in tod.noise.keys:
            psd = tod.noise.psd(key)
            psd[:] *= 1e-6
        if args.noisefile != args.noisefile_simu:
            for key in tod.noise_simu.keys:
                psd = tod.noise_simu.psd(key)
                psd[:] *= 1e-6

    ob = {}
    ob["name"] = "mission"
    ob["id"] = 0
    ob["tod"] = tod
    ob["intervals"] = tod.valid_intervals
    ob["baselines"] = None
    ob["noise"] = tod.noise
    ob["noise_simu"] = tod.noise_simu

    # Add the bare minimum focal plane information for the conviqt operator
    focalplane = {}
    for det in tod.detectors:
        focalplane[det] = {
            "pol_leakage" : tod.rimo[det].epsilon,
            "psi_pol_deg" : tod.rimo[det].psi_pol,
            "psi_uv_deg" : tod.rimo[det].psi_uv,
        }
    ob["focalplane"] = focalplane

    fsample = tod.fsample

    data.obs.append(ob)

    detweights = {}
    for d in tod.detectors:
        if d[-1] in "01" and d[-2] != "-":
            det = to_radiometer(d)
        else:
            det = d
        net = tod.rimo[det].net
        detweights[d] = 1.0 / (fsample * net * net)

    timer.stop()
    if comm.comm_world.rank == 0:
        timer.report("Compute data distribution")

    return data, detweights, fsample


def report_stats(data):
    """ Report some data distribution statistics

    """
    if len(data.obs) != 1:
        raise RuntimeError("preproc and reproc expect a single observation")
    obs = data.obs[0]
    tod = obs["tod"]
    my_nsamp = tod.local_samples[1]
    if "intervals" not in obs:
        raise RuntimeError("observation must specify intervals")
    # Get local intervals for statistics.  This will cache the timestamps.
    intervals = tod.local_intervals(obs["intervals"])
    local_starts = [ival.first for ival in intervals]
    local_stops = [ival.last for ival in intervals]
    my_nring = len(local_starts)
    my_longest_ring = np.amax(np.array(local_stops) - np.array(local_starts))
    nsamps = data.comm.comm_world.gather(my_nsamp, root=0)
    nrings = data.comm.comm_world.gather(my_nring, root=0)
    longest_rings = data.comm.comm_world.gather(my_longest_ring, root=0)
    if data.comm.comm_world.rank == 0:
        print("Data distribution stats:")
        for name, vec in [
            ("Nsamp", nsamps),
            ("Nring", nrings),
            ("Longest ring", longest_rings),
        ]:
            print(
                "{:13} min = {:10} max = {:10} mean = {:13.2f} +- {:13.2f}"
                "".format(name, np.amin(vec), np.amax(vec), np.mean(vec), np.std(vec)),
                end="",
            )
            if name != "Longest ring":
                print(" total = {:13}".format(np.sum(vec)))
        print("", flush=True)
    return


def get_pointing(args, madampars, data, rimo, mpiworld):
    """ Expand boresight pointing into detector pixels and IQU weights

    """
    pointingmode = "IQU"
    """
    if args.madampar and pars['temperature_only'] == 'T' and args.skip_reproc \
       and args.skip_preproc:
        pointingmode = 'I'
    """

    pointing = tp.OpPointingPlanck(
        nside=args.nside,
        mode=pointingmode,
        RIMO=rimo,
        margin=args.margin,
        apply_flags=False,
        single_precision=True,
        keep_vel=True,
        keep_pos=True,
        keep_phase=True,
        keep_quats=True,
    )

    if data.comm.comm_world.rank == 0:
        print("Constructing pointing ... ", end="", flush=True)
    timer = Timer()
    timer.start()
    pointing.exec(data)
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        data.obs[0]["tod"].cache.report()
        timer.report("pointing")

    memreport("after pointing", mpiworld)

    return pointing


def purge_caches(data, mcmode, mpiworld):
    """ purge unnecessary cached data
    """
    if not mcmode:
        for obs in data.obs:
            tod = obs["tod"]
            tod.cache.clear(tod.VELOCITY_NAME)
            tod.cache.clear(tod.POSITION_NAME)
            tod.cache.clear(tod.PHASE_NAME)
            tod.cache.clear(tod.POINTING_NAME + "_.*")
    if data.comm.comm_world.rank == 0:
        data.obs[0]["tod"].cache.report()
    memreport("after purge_caches", mpiworld)
    return


def run_madam(args, pars, detweights, data, outdir, mcmode, mpiworld):
    if args.skip_madam:
        return

    map_dir = os.path.join(outdir, args.map_dir)
    if not map_dir.endswith(os.sep):
        map_dir += os.sep
    if data.comm.comm_world.rank == 0:
        os.makedirs(map_dir, exist_ok=True)
    pars["path_output"] = map_dir
    if mcmode:
        pars["write_binmap"] = False
        pars["write_matrix"] = False
        pars["write_leakmatrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = False

    name_out = None
    if not mcmode and args.effdir_out_madam:
        name_out = "signal"

    madam = OpMadam(
        name_out=name_out,
        flag_mask=255,
        common_flag_mask=1,
        params=pars,
        detweights=detweights,
        purge=(not mcmode),
        translate_timestamps=False,
    )

    timer = Timer()
    timer.start()
    madam.exec(data)
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Madam")

    if args.effdir_out_madam is None:
        return

    timer.clear()
    timer.start()
    writer = tp.OpOutputPlanck(
        signal_name="signal", flags_name=None, effdir_out=args.effdir_out_madam
    )
    writer.exec(data)
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Writing")

    memreport("after madam", mpiworld)

    return


def run_shdet(args, adc_table_path, shdet_params, tf_path, detectors, data):
    shdet = None
    if args.simulate:
        if args.effdir_optical is None:
            raise RuntimeError("You must specify the optical input signal for SHDet")
        shdet = tp.OpSimSHDET(
            dets=detectors,
            imofile=args.imo,
            adc_table=adc_table_path,
            margin=args.margin,
            calfile=args.calfile,
            params=shdet_params,
            tffile=tf_path,
            read_no_signal=args.read_no_signal,
            offset_file=args.offset_file,
        )

        timer = Timer()
        timer.start()
        shdet.exec(data)
        data.comm.comm_world.barrier()
        timer.stop()
        if data.comm.comm_world.rank == 0:
            timer.report("SHDet")

    return shdet


def run_beam(args, pntmask, ssomask, data):
    """ Evaluate the detector beams using planet transits

    """
    beam = None
    skip_beam = args.beam_fits is None and args.beam_fits_polar is None
    if not skip_beam:
        beam = tp.OpBeamReconstructor(
            args.imo,
            args.freq,
            bg_map_path=args.bg_map,
            bg_pol=args.bg_pol,
            bg_has_dipole=args.bg_has_dipole,
            bg_nside=args.bg_nside,
            pntmask=pntmask,
            ssomask=ssomask,
            maskfile=args.preproc_mask,
            nbin_phase=args.nbin,
            margin=0,
            effdir_out=args.effdir_out_preproc,
            bad_rings=args.bad_rings,
            out=args.out,
            targets=args.beam_targets,
            bsiter=args.beam_iter,
            beamtofits=args.beam_fits,
            beamtofits_polar=args.beam_fits_polar,
        )
        timer = Timer()
        timer.start()
        beam.exec(data)
        data.comm.comm_world.barrier()
        timer.stop()
        if data.comm.comm_world.rank == 0:
            timer.report("Beam")

        pntmask = 2
        ssomask = ssomask

    return pntmask, ssomask


def run_ringmaker(args, data, outdir):
    """ Make an HDF5 ringset of the signal

    """
    ringmaker = None
    if args.make_rings:
        ringmaker = tp.OpRingMaker(
            args.nside_ring, args.nside, fileroot=args.ring_root, out=outdir
        )
        timer = Timer()
        timer.start()
        ringmaker.exec(data)
        data.comm.comm_world.barrier()
        timer.stop()
        if data.comm.comm_world.rank == 0:
            timer.report("Ringmaking")
    return


def run_preproc(args, shdet, data):
    """ Preproces the raw signal

    """
    if args.skip_preproc:
        return args.pntmask, args.ssomask

    if shdet is not None:
        # grab the transfer function, a dictionary
        shdet_freq, shdet_TF = shdet.get_TF()
        # convert into real and imaginary tables
        shdet_TF_real = {}
        shdet_TF_imag = {}
        for det in shdet_TF.keys():
            tf = shdet_TF[det]
            shdet_TF_real[det] = tf.real
            shdet_TF_imag[det] = tf.imag
        shdet_tf = (shdet_freq, shdet_TF_real, shdet_TF_imag)
    else:
        shdet_tf = None

    preproc = tp.OpPreproc(
        args.imo,
        args.freq,
        bg_map_path=args.bg_map,
        bg_pol=args.bg_pol,
        bg_has_dipole=args.bg_has_dipole,
        bg_nside=args.bg_nside,
        detmask=args.detmask,
        pntmask=args.pntmask,
        ssomask=args.ssomask,
        maskfile=args.preproc_mask,
        maskfile_adc=args.preproc_mask_adc,
        nbin_phase=args.nbin,
        jump_filter_len=args.jump_filter_len,
        jump_threshold=args.jump_threshold,
        timeout=args.preproc_timeout,
        timeout_intermediate=args.preproc_timeout_intermediate,
        preproc_dark=args.preproc_dark,
        preproc_common=args.preproc_common,
        calfile=args.calfile,
        adc_correction=args.adc_correction,
        measure_ADC=args.measure_ADC,
        nadc_iter=args.niter_ADC,
        deltaADU=args.delta_ADC,
        nphase4k=args.nphase4k,
        margin=args.margin,
        effdir_out=args.effdir_out_preproc,
        flag_planets=args.flag_planets,
        planet_flag_radius=args.planet_flag_radius,
        bad_rings=args.bad_rings,
        out=args.out,
        shdet_mode=(shdet is not None),
        g0=args.shdet_g0,
        v0=args.shdet_v0,
        # TODO: make global_phase_shift a dictionary indexed by detector
        global_phase_shift=args.shdet_global_phase_shift,
        tabulated_transfer_function=shdet_tf,
        intense_threshold=args.intense_threshold,
        async_time=args.preproc_async_time,
        async_time_intermediate=args.preproc_async_time_intermediate,
    )

    timer = Timer()
    timer.start()
    preproc.exec(data)
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Preproc")

    return (2, 2)


def apply_filter(args, data):
    """ Apply extra filter to signal
    """
    if args.filterfile is None:
        return
    timer = Timer()
    timer.start()
    convolver = tp.OpConvolvePlanck(args.filterfile)
    convolver.exec(data)
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Convolve with {}".format(args.filterfile))
    return


def run_reproc(
        args,
        pars,
        pntmask,
        ssomask,
        map_dir,
        data,
        outdir,
        mcmode,
        mc,
        fg,
        fg_deriv,
        cmb,
        fslnames,
        fslbeam_mask_path,
):
    """ Reprocess preprocessed or simulated signal.

    """
    if args.skip_reproc:
        return
    if data.comm.comm_world.rank == 0:
        data.obs[0]["tod"].cache.report()
    reproc = tp.OpReprocRing(
        args.freq,
        nharm=args.reproc_nharm,
        do_bands=args.reproc_do_bands,
        do_zodi=args.reproc_zodi,
        differential_zodi=not args.reproc_zodi_total,
        independent_zodi=args.reproc_zodi_detector,
        zodi_cache=args.reproc_zodi_cache,
        pntmask=pntmask,
        # ssomask=ssomask, detmask=1+8,
        ssomask=ssomask,
        detmask=args.reproc_detmask,
        polmap=args.reproc_polmap.format(mc),
        polmap2=args.reproc_polmap2.format(mc),
        polmap3=args.reproc_polmap3.format(mc),
        force_polmaps=args.reproc_force_polmaps,
        pol_fwhm=args.reproc_pol_fwhm,
        pol_lmax=args.reproc_pol_lmax,
        pol_nside=args.reproc_pol_nside,
        cmb=args.reproc_CMB,
        co=args.reproc_CO,
        co2=args.reproc_CO2,
        co3=args.reproc_CO3,
        dust=args.reproc_dust,
        dust_pol=args.reproc_dust_pol,
        sync=args.reproc_sync,
        sync_pol=args.reproc_sync_pol,
        ame=args.reproc_AME,
        freefree=args.reproc_freefree,
        pix_nside=args.nside,
        nside=args.reproc_nside,
        destriper_pixlim=args.reproc_pixlim,
        maskfile=args.reproc_mask,
        maskfile_bp=args.reproc_bandpass_mask,
        bandpass_nside=args.reproc_nside_bandpass,
        bandpass_fwhm=args.reproc_fwhm_bandpass,
        maskfile_project=args.reproc_project_mask,
        symmetrize=args.reproc_symmetrize,
        restore_dipole=args.reproc_restore_dipole,
        effdir_out=args.reproc_effdir_out,
        map_dir=map_dir,
        niter=args.reproc_niter,
        calibrate=args.reproc_calibrate,
        fit_distortion=args.reproc_fit_distortion,
        nlcalibrate=args.reproc_nlcalibrate,
        recalibrate=args.reproc_recalibrate,
        fgdipole=args.reproc_fgdipole,
        save_maps=args.reproc_save_maps,
        save_survey_maps=args.reproc_save_survey_maps,
        save_template_maps=args.reproc_save_template_maps,
        save_single_maps=args.reproc_save_single_maps,
        out=outdir,
        single_nside=args.reproc_nside_single,
        forcepol=args.reproc_forcepol,
        forcefsl=args.reproc_forcefsl,
        fslnames=fslnames,
        fslbeam_mask_path=fslbeam_mask_path,
        asymmetric_fsl=args.reproc_asymmetric_fsl,
        bpcorrect=args.reproc_bpcorrect,
        pscorrect=args.reproc_pscorrect,
        psradius=args.reproc_psradius,
        do_fsl=(args.effdir_fsl is not None),
        madampars=pars,
        bad_rings=args.bad_rings,
        quss_correct=args.reproc_quss_correct,
        temperature_only=args.reproc_temperature_only,
        temperature_only_destripe=args.reproc_temperature_only_destripe,
        temperature_only_intermediate=args.reproc_temperature_only_intermediate,
        calfile=args.calfile,
        effective_amp_limit=args.reproc_effective_amp_limit,
        gain_step_mode=args.reproc_gain_step_mode,
        min_step_length=args.reproc_min_step_length,
        max_step_length=args.reproc_max_step_length,
        outlier_threshold=args.reproc_outlier_threshold,
        mcmode=mcmode,
        fg=fg,
        fg_deriv=fg_deriv,
        cmb_mc=cmb,
        force_conserve_memory=args.reproc_force_conserve_memory,
        polparammode=args.polparammode,
        save_destriper_data=args.save_destriper_data,
    )

    timer = Timer()
    timer.start()
    reproc.exec(data)
    del reproc
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Reproc")
    return


def apply_gains(args, data, mc):
    if not args.cmb_alm and not args.simulate_noise:
        return
    if args.decalibrate is None or args.skip_reproc:
        return
    fn = args.decalibrate.format(mc)
    if data.comm.comm_world.rank == 0:
        print("Decalibrating with {}".format(fn), flush=True)
    timer = Timer()
    timer.start()
    decalibrator = tp.OpCalibPlanck(signal_out="signal", file_gain=fn, decalibrate=True)
    decalibrator.exec(data)
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Decalibrate with {}".format(fn))
    return


def run_noisesim(args, data, fsample, mc, mpiworld):
    if not args.simulate_noise:
        return
    timer = Timer()
    timer.start()
    if not args.cmb_alm:
        # zero the local signal for noise
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                tod.local_signal(det)[:] = 0
    nse = OpSimNoise(
        out="signal", realization=mc, component=0, noise="noise_simu", rate=fsample
    )
    memreport("after initializing OpSimNoise", mpiworld)
    if data.comm.comm_world.rank == 0:
        print("Simulating noise from {}".format(args.noisefile_simu), flush=True)
    nse.exec(data)
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Noise simulation")
    memreport("after running OpSimNoise", mpiworld)
    return


def run_conviqt(
        args,
        data,
        file_slm,
        file_blm,
        rimo,
        mpiworld,
        name="signal",
        pol=True,
        remove_monopole=True,
        remove_dipole=True,
        normalize_beam=True,
):
    if data.comm.comm_world.rank == 0:
        print("Applying Conviqt ...", flush=True)
    if args.conviqt_lmax == 0:
        raise RuntimeError("Must set non-zero conviqt_lmax")
    if args.conviqt_mmax == 0:
        raise RuntimeError("Must set non-zero conviqt_mmax")
    memreport("before Conviqt", mpiworld)
    timer = Timer()
    timer.start()
    # Clear the old signal, if it exists. OpSimConviqt always adds to
    # the existing signal
    for obs in data.obs:
        obs["tod"].cache.clear(f"{name}_.*")
    skyfiles = {}
    beamfiles = {}
    for det in data.obs[0]["tod"].detectors:
        freq = "{:03}".format(tp.utilities.det2freq(det))
        if "LFI" in det:
            if det.endswith("M"):
                arm = "y"
            else:
                arm = "x"
            graspdet = "{}_{}_{}".format(freq[1:], det[3:5], arm)
        else:
            graspdet = det
        skyfile = file_slm.replace("FREQ", freq).replace("DETECTOR", det)
        skyfiles[det] = skyfile
        beamfile = file_blm.replace("GRASPDETECTOR", graspdet).replace(
            "DETECTOR", det
        )
        beamfiles[det] = beamfile
        if data.comm.comm_world.rank == 0:
            print("Convolving {} with {}".format(skyfile, beamfile), flush=True)
    conviqt = OpSimConviqt(
        mpiworld,
        skyfiles,
        beamfiles,
        lmax=args.conviqt_lmax,
        beammmax=args.conviqt_mmax,
        pol=pol,
        fwhm=args.conviqt_fwhm,
        order=args.conviqt_order,
        calibrate=True,
        dxx=not args.conviqt_pxx,
        apply_flags=False,
        out=name,
        remove_monopole=remove_monopole,
        remove_dipole=remove_dipole,
        normalize_beam=normalize_beam,
        verbosity=1,
    )
    conviqt.exec(data)
    if mpiworld is not None:
        mpiworld.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report(f"Conviqt {name}")
    del conviqt
    memreport("after Conviqt", mpiworld)

    return


def check_files(data, args, mc):
    there = True
    # Check CMB input to signal simulation
    if there and args.cmb_alm and data.comm.comm_world.rank == 0:
        fn_cmb = os.path.join(
            CMBCACHE,
            "{}_nside{:04}_quickpol.fits".format(
                os.path.basename(args.cmb_alm.format(mc)).replace(".fits", ""),
                args.sim_nside,
            ),
        )
        there = os.path.isfile(fn_cmb)
    # Check polarization map inputs to reproc
    if there and not args.skip_reproc and data.comm.comm_world.rank == 0:
        polmap1 = args.reproc_polmap.format(mc)
        polmap2 = args.reproc_polmap2.format(mc)
        polmap3 = args.reproc_polmap3.format(mc)
        for polmap in polmap1, polmap2, polmap3:
            if polmap != "":
                there0 = os.path.isfile(polmap)
                there = there and there0
                if not there0:
                    print("ERROR: polmap does not exist: {}".format(polmap), flush=True)
    there = data.comm.comm_world.bcast(there, root=0)
    return there


def run_signalsim(args, data, mc, outdir, rimo, mpiworld):
    if not args.cmb_alm:
        return None
    if data.comm.comm_world.rank == 0:
        print("Simulating signal ...", flush=True)
    memreport("Before signalsim", mpiworld)
    fwhm = tp.utilities.freq_to_fwhm(args.freq)

    almfile = args.cmb_alm.format(mc)
    if data.comm.comm_world.rank == 0:
        print("Simulating CMB from {}".format(almfile), flush=True)
    if args.freq_sigma is None:
        if data.comm.comm_world.rank == 0:
            print("WARNING: not simulating bandpass mismatch", flush=True)
        freqs = None
    else:
        np.random.seed(1001 * args.freq + mc)
        dets = sorted(list_planck(args.freq))
        ndet = len(dets)
        if args.freq_sigma > 0:
            # Draw random displacements of the center frequency
            if data.comm.comm_world.rank == 0:
                print(
                    "Drawing center frequencies with sigma = {}"
                    "".format(args.freq_sigma),
                    flush=True,
                )
            centers = args.freq + np.random.randn(ndet) * args.freq_sigma
        else:
            # Use the hard-coded values
            if data.comm.comm_world.rank == 0:
                print("Using hard-coded center frequencies", flush=True)
            centers = np.zeros(ndet) + args.freq
            for idet, det in enumerate(dets):
                centers[idet] += cfreq_deltas[det]
        freqs = {}
        for det, freq in zip(dets, centers):
            freqs[det] = freq
            if data.comm.comm_world.rank == 0:
                print("MC {:4} Det {:8} center freq = {:10.3f}".format(mc, det, freq))
    if args.conviqt_beamfile and not args.tfmode:
        run_conviqt(args, data, almfile, args.conviqt_beamfile, rimo, mpiworld)
        almfile = None
        add = True
    else:
        add = False

    timer = Timer()
    timer.start()
    signalsim = tp.OpSignalSim(
        almfile,
        fwhm,
        args.freq,
        data.comm.comm_world,
        freqs=freqs,
        pol=True,
        add=add,
        nside=args.sim_nside,
        dipole=True,
        rimo=None,
        fsl=(args.effdir_fsl is not None),
        foreground=True,
        mapdir=outdir,
        skip_reproc=args.skip_reproc,
        quickpolbeam=args.quickpolbeam,
        skymodelfile=args.skymodel_sim,
        skymodelderivfile=args.skymodelderiv_sim,
        mc=mc,
        bpm_extra=args.bpm_extra,
    )
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Initialize signalsim")
    memreport("after initializing tp.OpSignalSim", mpiworld)

    timer.clear()
    timer.start()
    signalsim.exec(data)
    data.comm.comm_world.barrier()
    timer.stop()
    if data.comm.comm_world.rank == 0:
        timer.report("Run signalsim")
    memreport("after running tp.OpSignalSim", mpiworld)
    del signalsim
    memreport("after deleting tp.OpSignalSim", mpiworld)

    fn_cmb = os.path.join(
        CMBCACHE,
        "{}_nside{:04}_quickpol.fits".format(
            os.path.basename(args.cmb_alm.format(mc)).replace(".fits", ""),
            args.sim_nside,
        ),
    )
    if data.comm.comm_world.rank == 0:
        if not os.path.isfile(fn_cmb):
            print(
                "ERROR: Even when using Conviqt, there must also be a cached version "
                'of the CMB map. File not found: "{}"'.format(fn_cmb),
                flush=True,
            )
            data.comm.comm_world.Abort("No cached CMB map")
        print("Loading pre-computed CMB map from {}".format(fn_cmb), flush=True)
    cmb = MapSampler(
        fn_cmb,
        pol=True,
        comm=data.comm.comm_world,
        nest=True,
        nside=args.reproc_nside_bandpass,
        plug_holes=False,
    )

    if args.sim_tf is not None:
        if data.comm.comm_world.rank == 0:
            print("Convolving signal ...", flush=True)
        timer = Timer()
        timer.start()
        convolver = tp.OpConvolvePlanck(
            filterfile=args.sim_tf, normalize=False, extend_flags=False
        )
        convolver.exec(data)
        data.comm.comm_world.barrier()
        timer.stop()
        if data.comm.comm_world.rank == 0:
            timer.report("Convolve with {}".format(args.sim_tf))

    return cmb


# @profile
def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_planck_reduce (total)")

    mpiworld, procs, rank, comm = get_comm()

    memreport("at beginning of main", mpiworld)

    pipeline_timer = Timer()
    pipeline_timer.start()
    (
        args,
        map_dir,
        madampars,
        detectors,
        odrange,
        ringrange,
        obtrange,
        adc_table_path,
        shdet_params,
        tf_path,
    ) = parse_arguments()

    data, detweights, fsample = create_observations(
        args, detectors, obtrange, ringrange, odrange
    )

    report_stats(data)

    rimo = load_RIMO(args.rimo, mpiworld)
    get_pointing(args, madampars, data, rimo, mpiworld)

    shdet = run_shdet(args, adc_table_path, shdet_params, tf_path, detectors, data)
    pntmask, ssomask = run_preproc(args, shdet, data)
    pntmask, ssomask = run_beam(args, pntmask, ssomask, data)

    if comm.world_rank == 0:
        pipeline_timer.report_elapsed("Init")

    if args.skymodel:
        if comm.world_rank == 0:
            print("Loading foreground map from {}".format(args.skymodel), flush=True)
        fg = MapSampler(
            args.skymodel,
            pol=True,
            comm=mpiworld,
            nest=True,
            nside=args.reproc_nside_bandpass,
            plug_holes=False,
        )
    else:
        fg = None

    if args.skymodelderiv:
        if comm.world_rank == 0:
            print(
                "Loading foreground derivative map from {}".format(args.skymodelderiv),
                flush=True,
            )
        fg_deriv = MapSampler(
            args.skymodelderiv,
            pol=True,
            comm=mpiworld,
            nest=True,
            nside=args.reproc_nside_bandpass,
            plug_holes=False,
        )
    else:
        fg_deriv = None

    if args.conviqt_fsl:
        if args.conviqt_sky is None:
            raise RuntimeError("Must set conviqt sky to convolve with sidelobes")
        fslnames = []
        for ifsl, fsl in enumerate(args.conviqt_fsl.split(",")):
            fslname = f"fsl{ifsl}"
            run_conviqt(
                args,
                data,
                args.conviqt_sky,
                fsl,
                rimo,
                mpiworld,
                name=fslname,
                pol=False,
                remove_monopole=False,
                remove_dipole=False,
                normalize_beam=False,
            )
            fslnames.append(fslname)
    else:
        fslnames = None
    
    if args.fslbeam_mask:
        fslbeam_mask_path = {}
        for det in data.obs[0]["tod"].detectors:
            fslbeam_mask_path[det] = args.fslbeam_mask.replace("DETECTOR", det)
    else:
        fslbeam_mask_path = None

    for mc in range(args.MC_start, args.MC_start + args.MC_count):
        mpiworld.Barrier()
        there = check_files(data, args, mc)
        if not there:
            continue
        mctimer = Timer()
        mctimer.start()
        if args.cmb_alm or args.simulate_noise:
            outdir = os.path.join(args.out, "{:04}".format(mc))
            if comm.world_rank == 0:
                os.makedirs(outdir, exist_ok=True)
            mcmode = True
        else:
            outdir = args.out
            mcmode = False
        cmb = run_signalsim(args, data, mc, outdir, rimo, mpiworld)
        run_noisesim(args, data, fsample, mc, mpiworld)
        apply_gains(args, data, mc)
        run_ringmaker(args, data, outdir)
        apply_filter(args, data)
        run_reproc(
            args,
            madampars,
            pntmask,
            ssomask,
            map_dir,
            data,
            outdir,
            mcmode,
            mc,
            fg,
            fg_deriv,
            cmb,
            fslnames,
            fslbeam_mask_path,
        )
        del cmb
        sys.exit(0)
        purge_caches(data, mcmode, mpiworld)
        run_madam(args, madampars, detweights, data, outdir, mcmode, mpiworld)
        if mpiworld is not None:
            mpiworld.barrier()
        mctimer.stop()
        if comm.world_rank == 0:
            mctimer.report("MC = {} iteration".format(mc))

    data.obs[0]["tod"].cache.clear()

    pipeline_timer.stop()
    if data.comm.comm_world.rank == 0:
        pipeline_timer.report("Pipeline")

    memreport("at end of main", mpiworld)

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()
    timer = Timer()
    timer.start()
    alltimers = gather_timers(comm=mpiworld)
    if comm.world_rank == 0:
        out = os.path.join(args.out, "timing")
        dump_timing(alltimers, out)
        timer.stop()
        timer.report("Gather and dump timing info")
    return


if __name__ == "__main__":
    try:
        mpiworld, procs, rank, comm = get_comm()
        test_performance(mpiworld)
        main()
    except Exception as e:
        print('Exception occurred: "{}"'.format(e), flush=True)
        if procs == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("*** print_exception:")
        traceback.print_exception(
            exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout
        )
        print("*** print_exc:")
        traceback.print_exc()
        print("*** format_exc, first and last line:")
        formatted_lines = traceback.format_exc().splitlines()
        print(formatted_lines[0])
        print(formatted_lines[-1])
        print("*** format_exception:")
        print(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        print("*** extract_tb:")
        print(repr(traceback.extract_tb(exc_traceback)))
        print("*** format_tb:")
        print(repr(traceback.format_tb(exc_traceback)))
        print("*** tb_lineno:", exc_traceback.tb_lineno, flush=True)
        mpiworld.Abort()
