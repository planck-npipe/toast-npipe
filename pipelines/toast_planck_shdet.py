#!/usr/bin/env python

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

#from memory_profiler import profile

import sys
import os
from toast.mpi import MPI
import traceback

import re
import argparse
import resource
import datetime

import numpy as np

import toast
import toast.tod as tt
import toast.map as tm

import toast_planck as tp

from toast_planck.utilities import load_RIMO

#@profile
def main():

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        print("Running with {} processes at {}".format(
            comm.comm_world.size, str(datetime.datetime.now())))

    global_start = MPI.Wtime()

    parser = argparse.ArgumentParser(description='Planck Data Reduction',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--simulate', dest='simulate', required=False,
                         default=False, action='store_true',
                        help='Perform a simulation instead')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--imo', required=True, help='IMO file')
    parser.add_argument('--freq', required=True, type=np.int, help='Frequency')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true',
                        help='Write data distribution info to file')
    parser.add_argument('--dets', required=False, default = None,
                        help='Detector list (comma separated)')
    parser.add_argument('--effdir_in', required=False,
                        help='Input Exchange Format File directory')
    parser.add_argument('--effdir_optical', required=True,
                        help='Input Exchange Format File directory for optical '
                        'signal (for SHDET)')
    parser.add_argument('--effdir_fsl', required=False,
                        help='Input Exchange Format File directory for far '
                        'side lobe TOD')
    parser.add_argument('--effdir_pntg', required=False,
                        help='Input Exchange Format File directory for pointing')
    parser.add_argument('--ringdb', required=True, help='Ring DB file')
    parser.add_argument('--odfirst', required=False, default=None, type=np.int,
                        help='First OD to use')
    parser.add_argument('--odlast', required=False, default=None, type=np.int,
                        help='Last OD to use')
    parser.add_argument('--ringfirst', required=False, default=None,
                        type=np.int, help='First ring to use')
    parser.add_argument('--ringlast', required=False, default=None, type=np.int,
                        help='Last ring to use')
    parser.add_argument('--obtfirst', required=False, default=None,
                        type=np.float, help='First OBT to use')
    parser.add_argument('--obtlast', required=False, default=None,
                        type=np.float, help='Last OBT to use')
    parser.add_argument('--madampar', required=False, default=None,
                        help='Madam parameter file')
    parser.add_argument('--out', required=False, default='.',
                        help='Output directory')
    parser.add_argument('--madam_prefix', required=False, help='map prefix')
    parser.add_argument('--effdir_out_shdet', required=False,
                        help='Output directory for SHDet')
    parser.add_argument('--effdir_in_shdet', required=False,
                        help='Output directory for SHDet input signal')
    parser.add_argument('--effdir_out_preproc', required=False,
                        help='Output directory for preproc')
    parser.add_argument('--effdir_out_reproc', required=False,
                        help='Output directory for reproc')
    parser.add_argument('--effdir_out_madam', required=False,
                        help='Output directory for Madam')
    parser.add_argument('--nside', required=False, default=1024, type=np.int,
                        help='Pipeline resolution')
    parser.add_argument('--bg_map', required=False, help='Map template')
    parser.add_argument('--bg_nside', required=False, default=1024, type=np.int,
                        help='Map template resolution')
    parser.add_argument('--bg_pol', dest='bg_pol', required=False,
                        default=False, action='store_true',
                        help='Map template polarization')
    parser.add_argument('--bg_has_dipole', dest='bg_has_dipole', required=False,
                        default=False, action='store_true',
                        help='Background map includes the dipole')
    parser.add_argument('--calfile', required=False, help='Calibration file')
    parser.add_argument('--obtmask', required=False, default=1, type=np.int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=1, type=np.int,
                        help='Quality flag mask')
    parser.add_argument('--detmask', required=False, type=np.int,
                        help='Detector flag mask')
    parser.add_argument('--pntmask', required=False, default=2, type=np.int,
                        help='Pointing flag mask')
    parser.add_argument('--ssomask', required=False, type=np.int,
                        help='SSO flag mask')
    parser.add_argument('--preproc_mask', required=False,
                        help='Preproc processing mask file')
    parser.add_argument('--effdir_dark', required=False,
                        help='Input Exchange Format File directory for dark '
                        'bolometer data')
    parser.add_argument('--darkmask', required=False, default=1, type=np.int,
                        help='Dark flag mask')
    parser.add_argument('--preproc_dark', dest='preproc_dark', default=False,
                        action='store_true', help='Preprocess dark data')
    parser.add_argument('--nbin', required=False, default=10000, type=np.int,
                        help='Number of phase bins')
    parser.add_argument('--jump_filter_len', required=False, default=40000,
                        type=np.int, help='Jump filter length')
    parser.add_argument('--despike', dest='despike', default=False,
                        action='store_true', help='Use despike instead of '
                        'glitch_remover')
    parser.add_argument('--despiker_timeout', required=False, default=60.,
                        type=np.float, help='Maximum time allowed for despiker')
    parser.add_argument('--even_correction', required=False,
                        help='Even NL correction file.')
    parser.add_argument('--odd_correction', required=False,
                        help='Odd NL correction file.')
    parser.add_argument('--measure_ADC', dest='measure_ADC', default=False,
                        action='store_true', help='Measure ADC NL')
    parser.add_argument('--niter_ADC', default=1, type=np.int,
                        help='Number of ADC NL iterations')
    parser.add_argument('--delta_ADC', default=0.1, type=np.float,
                        help='Width of ADC bin in ADU')
    parser.add_argument('--nphase4k', required=False, default=2, type=np.int,
                        help='Number of 4K cooler phases to measure ADC NL for.')
    parser.add_argument('--nharm', required=False, default=20, type=np.int,
                        help='Number of passbands in differentiation.')
    parser.add_argument('--band_calibrate', dest='band_calibrate',
                        default=False, action='store_true',
                        help='Measure spin harmonic calibration')
    parser.add_argument('--skip_preproc', dest='skip_preproc', default=False,
                        action='store_true', help='Do not pre-process the TOD')
    parser.add_argument('--skip_reproc', dest='skip_reproc', default=False,
                        action='store_true', help='Do not re-process the TOD')
    parser.add_argument('--skip_madam', dest='skip_madam', default=False,
                        action='store_true',
                        help='Disable Madam mapmaking and only process and '
                        'save the TOD.')
    parser.add_argument('--make_rings', dest='make_rings', default=False,
                        action='store_true', help='Compile ringsets.')
    parser.add_argument('--nside_ring', required=False, default=128,
                        type=np.int, help='Ringset resolution')
    parser.add_argument('--ring_root', required=False, default='ringset',
                        help='Root filename for ringsets (setting to empty '
                        'disables ringset output).')
    parser.add_argument('--lfi_raw', dest='lfi_raw', default=False,
                        action='store_true',
                        help='Input LFI data is undifferenced')
    parser.add_argument('--zodi', dest='zodi', default=False,
                        action='store_true',
                        help='Subtract Zodiacal light in reproc')
    parser.add_argument('--swap', dest='swap', default=False,
                        action='store_true',
                        help='Use swap files to reduce memory footprint.')
    parser.add_argument('--margin', required=False, default=1000, type=np.int,
                        help='Preprocessing margin.')
    parser.add_argument('--CO', required=False,
                        help='CO map file for bandpass correction.')
    parser.add_argument('--CO2', required=False,
                        help='Second CO map file for bandpass correction.')
    parser.add_argument('--dust', required=False,
                        help='Dust map file for bandpass correction.')
    parser.add_argument('--dust_pol', required=False,
                        help='Dust polarization map file for bandpass correction.')
    parser.add_argument('--sync', required=False,
                        help='Synchrotron map file for bandpass correction.')
    parser.add_argument('--sync_pol', required=False, help='Synchrotron '
                        'polarization_map file for bandpass correction.')
    parser.add_argument('--AME', required=False,
                        help='AME map file for bandpass correction.')
    parser.add_argument('--freefree', required=False,
                        help='Free-free map file for bandpass correction.')
    parser.add_argument('--reproc_nside', required=False, default=512,
                        type=np.int, help='Reprocessing resolution')
    parser.add_argument('--bandpass_nside', required=False, default=128,
                        type=np.int, help='Bandpass template resolution')
    parser.add_argument('--bandpass_fwhm', required=False, default=240.,
                        type=np.float,
                        help='Bandpass template smoothing [arc min]')
    parser.add_argument('--bandpass_lmax', required=False, default=256,
                        type=np.int, help='Bandpass template smoothing lmax')
    parser.add_argument('--reproc_destripe_mask', required=False,
                        help='Reproc destriping mask file')
    parser.add_argument('--reproc_first_calib_mask', required=False,
                        help='Reproc first calibration mask file')
    parser.add_argument('--reproc_calib_mask', required=False,
                        help='Reproc calibration mask file')
    parser.add_argument('--reproc_bandpass_mask', required=False,
                        help='Reproc bandpass correction mask file')
    parser.add_argument('--map_dir', required=False, default='.',
                        help='Map output directory')
    parser.add_argument('--niter_reproc', required=False, default=1,
                        type=np.int, help='Number of calibration iterations')
    parser.add_argument('--niter_reproc_multi', required=False, default=1,
                         type=np.int,
                         help='Number of multi-detector calibration iterations')
    parser.add_argument('--single_cal', dest='single_cal', default=False,
                        action='store_true', help='Single detector calibration')
    parser.add_argument('--multi_cal', dest='multi_cal', default=False,
                        action='store_true', help='Multi-detector calibration')
    parser.add_argument('--flag_planets', dest='flag_planets', default=False,
                        action='store_true', help='Derive planet flags')
    parser.add_argument('--planet_flag_radius', required=False, default=2.0,
                        type=np.float,
                        help='New planet flag radius (in FWHM) when '
                        '--flag_planets')
    parser.add_argument('--recalibrate', dest='recalibrate', default=False,
                        action='store_true',
                        help='Adjust overall calibration using orbital '
                        'dipole fit.')
    parser.add_argument('--reproc_no_destripe', dest='reproc_no_destripe',
                        default=False, action='store_true',
                        help='Disable reproc destriping (TOD is already '
                        'destriped).')
    parser.add_argument('--bad_rings', required=False, help='Bad ring file.')
    parser.add_argument('--save_maps', dest='save_maps', default=False,
                        action='store_true', help='Save single detector maps '
                        'from reproc.')
    parser.add_argument('--save_survey_maps', dest='save_survey_maps',
                        default=False, action='store_true',
                        help='Save single detector, single survey maps from '
                        'reproc.')
    parser.add_argument('--save_template_maps', dest='save_template_maps',
                        default=False, action='store_true',
                        help='Save single detector template maps from reproc.')

    parser.add_argument('--jump_threshold', required=False, default=4.0,
                        type=np.float, help='Jump detection threshold')
    parser.add_argument('--forcepol', dest='forcepol', default=False,
                        action='store_true', help='Force polarization template'
                        ' to unity amplitude in reproc.')
    parser.add_argument('--forcedipo', dest='forcedipo', default=1, type=np.int,
                        help='Force orbital dipole template to unity amplitude '
                        'in reproc this multi iteration onwards.')
    parser.add_argument('--forcefsl', dest='forcefsl', default=False,
                        action='store_true', help='Force FSL template(s) to '
                        'unity amplitude in reproc.')
    parser.add_argument('--pol_deriv', dest='pol_deriv', type=np.int,
                        help='Fit polarization derivative template in reproc '
                        'during and after this multi iteration.')
    parser.add_argument('--bpcorrect', dest='bpcorrect', default=False,
                        action='store_true', help='Apply templated bandpass '
                        'correction in reproc (implies harmonize).')
    parser.add_argument('--harmonize', dest='harmonize', default=False,
                        action='store_true',
                        help='Measure and correct relative gain in reproc.')
    parser.add_argument('--use_iiqu', dest='use_iiqu', default=False,
                        action='store_true', help='Use IIQU approach for '
                        'bandpass mismatch correction maps.')
    parser.add_argument('--fgdipo', dest='fgdipo', default=False,
                        action='store_true',
                        help='Separate foreground dipole template in reproc.')
    parser.add_argument('--pol_map_bpcorr_path', required=False, help='Path to '
                        'per-detector polarization bandpass mismatch correction'
                        '. DETECTOR will be replaced with the detector name.')
    parser.add_argument('--sigmalim_cal', required=False, type=np.float,
                        help='Gain smoothing noise limit')
    parser.add_argument('--reproc_lowpassfreq', required=False, default=1.0,
                        type=np.float, help='Polarized destriper lowpass freq.')
    parser.add_argument('--read_no_signal',  default=False, action='store_true',
                        required=False,
                        help='read no input optical signal for SHDet tests.')
    parser.add_argument('--generate_dipole', default=False, action='store_true',
                        help='Generate dipole (solsys+orbital)')

    # The following options are for SHDet; all bolometers get the same value
    parser.add_argument('--adc_on', default=False, action='store_true',
                        required=False,
                        help='use ADC in SHDet, returning digits instead '
                        'of volts.')
    parser.add_argument('--seed', required=False, default=0, type=np.float,
                        help='SHDet random number generator base seed.')
    parser.add_argument('--offset_file', required=False,
                        help='Pre and post modulation offset pickle file. '
                        'Tag DETECTOR will be replaced with the appropriate '
                        'detector name.')
    parser.add_argument('--switch_optical_offset', required=False, default=1,
                        help='Switch on/off the pre modulation offset. on=1, '
                        'off=0.')
    parser.add_argument('--switch_raw_offset', required=False, default=1,
                        help='Switch on/off the post modulation offset. on=1, '
                        'off=0.')

    # The following options are for SHDet, taking comma-seperated lists as a
    # rguments, expecting either one item or one item per bolometer.
    parser.add_argument('--noise_dsn', default=None,required=False,
                        help='White noise generated by SHDet in DSN')
    parser.add_argument('--adc_table', default=None, required=False,
                        help='Path to ADC tables for SHDet or DPC-style ADC '
                        'NL correction.')
    parser.add_argument('--optical_load_watts',  default=None, required=False,
                        help='Optical constant load in W.')
    parser.add_argument('--gain_w_per_kcmb', default=None, required=False,
                        help='Gain of the detector in W per K CMB.')
    parser.add_argument('--bdac', default=None, required=False,
                        help='BDAC value for the compensation signal')
    parser.add_argument('--cstray', default=None, required=False,
                        help='Stray capacitance value (F)')
    parser.add_argument('--raw_model', default=None, required=False,
                        help='Substitute the bolo+electronics by the linear '
                        'raw model')

    # The following options are for preproc, taking comma-seperated
    # lists as arguments, expecting either one item or one item per
    # bolometer.
    parser.add_argument('--shdet_tffile', default=None, required=False,
                        help='Path to transfer function text file for '
                        'use with SHDet.')
    parser.add_argument('--global_phase_shift', required=False, default='0.0',
                        help='An global phase shift (in samples) to use in '
                        'reproc Fourier filter.')
    parser.add_argument('--g0', required=False, default=None,
                        help='The Watts/Volt conversion factor.  By default '
                        'read this from the IMO.')
    parser.add_argument('--v0', required=False, default=None,
                        help='Nonlinear gain correction.  By default this is '
                        'read from the IMO.')

    args = parser.parse_args()

    if comm.comm_world.rank == 0:
        print('Processing {} at {}GHz. skip_preproc = {}, skip_reproc = {}, '
              'skip_madam = {}'.format(
                args.dets, args.freq, args.skip_preproc, args.skip_reproc,
                  args.skip_madam))
        print('All parameters:')
        print(args, flush=True)

    # get options

    map_dir = os.path.join(args.out, args.map_dir)
    if not map_dir.endswith(os.sep):
        map_dir += os.sep

    if not os.path.isdir(args.out) and comm.comm_world.rank == 0:
        os.makedirs(args.out)

    if not os.path.isdir(map_dir) and comm.comm_world.rank == 0:
        os.makedirs(map_dir)

    odrange = None
    if args.odfirst is not None and args.odlast is not None:
        odrange = (args.odfirst, args.odlast)

    ringrange = None
    if args.ringfirst is not None and args.ringlast is not None:
        ringrange = (args.ringfirst, args.ringlast)

    obtrange = None
    if args.obtfirst is not None and args.obtlast is not None:
        obtrange = (args.obtfirst, args.obtlast)

    nside = args.nside

    detectors = None
    if args.dets is not None:
        detectors = re.split(',', args.dets)
        ndets = len(detectors)
    else:
        ndets = 0

    # some arguments need to be split from a string into a list of floats
    arglist = {}
    for argname, arg_to_check in [
            ('optical_load_watts', args.optical_load_watts),
            ('gain_w_per_kcmb', args.gain_w_per_kcmb), ('cstray',args.cstray),
            ('bdac',args.bdac), ('noise_dsn',args.noise_dsn), ('g0', args.g0),
            ('v0', args.v0), ('global_phase_shift',args.global_phase_shift),
            ('raw_model',args.raw_model)]:
         if arg_to_check is not None:
             tmplist = re.split(',', arg_to_check)
             npars = len(tmplist)
             if (npars != 1) and (npars != ndets):
                raise Exception('{:4} : ERROR: list parameters {} needs to '
                                'have either 1 argument or ndet arguments'
                                ''.format(comm.comm_world.rank, argname))

             # convert to floats
             arglist[argname] = [float(i) for i in tmplist]
         else:
             arglist[argname] = None

    for argname, arg_to_check in [('adc_table', args.adc_table),
                                  ('shdet_tffile', args.shdet_tffile)]:
        if arg_to_check is not None:
             tmplist = re.split(',', arg_to_check)
             npars = len(tmplist)
             if ((npars != 1) & (npars != ndets)):
                raise Exception('{:4} : ERROR: list parameters {} needs to '
                                'have either 1 argument or ndet arguments'
                                ''.format(comm.comm_world.rank,argname))
             arglist[argname] = tmplist
        else:
             arglist[argname] = None

    if args.skip_preproc:
        # Margins are only used for preprocessing.
        args.margin = 0

    # Read in madam parameter file
    # Allow more than one entry, gather into a list
    repeated_keys = ['detset', 'detset_nopol', 'survey']
    pars = {}

    if comm.comm_world.rank == 0:
        pars['kfirst'] = False
        pars['temperature_only'] = True
        pars['base_first'] = 60.0
        pars['nside_submap'] = 16
        pars['write_map'] = False
        pars['write_binmap'] = True
        pars['write_matrix'] = False
        pars['write_wcov'] = False
        pars['write_hits'] = True
        pars['kfilter'] = False
        pars['info'] = 3
        if args.madampar:
            pat = re.compile(r'\s*(\S+)\s*=\s*(\S+(\s+\S+)*)\s*')
            comment = re.compile(r'^#.*')
            with open(args.madampar, 'r') as f:
                for line in f:
                    if not comment.match(line):
                        result = pat.match(line)
                        if result:
                            key, value = result.group(1), result.group(2)
                            if key in repeated_keys:
                                if key not in pars: pars[key] = []
                                pars[key].append(value)
                            else:
                                pars[key] = value
        # Command line parameters override the ones in the madam parameter file
        if 'file_root' not in pars:
            pars['file_root'] = 'madam'
        if args.madam_prefix is not None:
            pars['file_root'] = args.madam_prefix
        sfreq = '{:03}'.format(args.freq)
        if sfreq not in pars['file_root']:
            pars['file_root'] += '_' + sfreq
        try:
            fsample = {30:32.51, 44:46.55, 70:78.77}[args.freq]
        except:
            fsample = 180.3737
        pars['fsample'] = fsample
        pars['nside_map'] = nside
        pars['nside_cross'] = nside//2
        pars['path_output'] = map_dir

    pars = comm.comm_world.bcast(pars, root=0)

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        print('Computing data distribution ...', end='', flush=True)
    start = MPI.Wtime()

    # Since madam only supports a single observation, we use
    # that here.  Normally we would have multiple observations
    # with some subset assigned to each process group.

    # The distributed timestream data

    data = toast.Data(comm)

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
   )

    # Create the (single) observation.  Normally we would get the
    # intervals from somewhere else, but since the Exchange TOD
    # already had to get that information, we can get it from there.

    ob = {}
    ob['name'] = 'mission'
    ob['id'] = 0
    ob['tod'] = tod
    ob['intervals'] = tod.valid_intervals
    ob['baselines'] = None
    ob['noise'] = tod.noise

    data.obs.append(ob)

    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Compute data distribution took {:.3f} seconds".format(stop-start), flush=True)
    start = stop

    # Report some data distribution statistics

    my_nsamp = tod.local_samples[1]
    intervals = None
    if 'intervals' in data.obs[0].keys():
        intervals = data.obs[0]['intervals']
    if intervals is None:
        intervals = [Interval(start=0.0, stop=0.0, first=0, last=(tod.total_samples-1))]
    local_starts = [(t.first - tod.local_samples[0]) if (t.first > tod.local_samples[0]) else 0 for t in intervals if (t.last >= tod.local_samples[0]) and (t.first < (tod.local_samples[0] + tod.local_samples[1]))]
    local_stops = local_starts[1:] + [my_nsamp]
    my_nring = len(local_starts)
    my_longest_ring = np.amax(np.array(local_stops) - np.array(local_starts))
    nsamps = None
    nrings = None
    longest_rings = None
    nsamps = comm.comm_world.gather(my_nsamp, root=0)
    nrings = comm.comm_world.gather(my_nring, root=0)
    longest_rings = comm.comm_world.gather(my_longest_ring, root=0)
    if comm.comm_world.rank == 0:
        print('Data distribution stats:')
        for name, vec in [('Nsamp',nsamps),('Nring',nrings),('Longest ring',longest_rings)]:
            print('{:13} min = {:10} max = {:10} mean = {:13.2f} +- {:.2f}'.format(
                    name, np.amin(vec), np.amax(vec), np.mean(vec), np.std(vec)))
        print('',flush=True)

    pointingmode = 'IQU'
    if args.madampar and pars['temperature_only'] == 'T':
        pointingmode = 'I'

    # Construct pixel numbers and pointing weights but hold off on applying detector flags to pixel numbers

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        print('Constructing pointing ...', end='', flush=True)
    start = MPI.Wtime()

    rimo = load_RIMO(args.rimo, comm.comm_world)
    pointing = tp.OpPointingPlanck(nside=nside, mode=pointingmode, RIMO=rimo, margin=args.margin,
                                   apply_flags=False, single_precision=True)

    pointing.exec(data)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Pointing matrix generation took {:.3f} s".format(elapsed), flush=True)
        tod.cache.report()
        print('Memory high water mark: {:.2f} MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))
    start = stop

    if args.simulate:
        # We are running a simulation.  Fill in the cached TOD.
        if args.read_no_signal:
            shdet_input = None
        else:
            if args.effdir_optical is None:
                # first run conviqt to get the sky signal
                # FIXME:  will have to get options for constructor from command line
                # or a parameter file...
                sky = tt.conviqt.OpSimConviqt()
                sky.exec(data)

                comm.comm_world.barrier()
                stop = MPI.Wtime()
                elapsed = stop - start
                if comm.comm_world.rank == 0:
                    print("Sky simulation took {:.3f} s".format(elapsed), flush=True)
                start = stop

                shdet_input = 'conviqt_tod'
            else:
                # read the data here
                shdet_input = 'sky_tod'
                loader = tp.OpInputPlanck(signal_name=shdet_input, margin=args.margin)
                loader.exec(data)

                comm.comm_world.barrier()
                stop = MPI.Wtime()
                elapsed = stop - start
                if comm.comm_world.rank == 0:
                    print("Optical power input data read and cache took {:.3f} s".format(elapsed), flush=True)
                #tod.cache.report()
                start = stop

        if args.generate_dipole:
            if shdet_input is None:
                shdet_input = 'dipole_tod'
            dipole = tp.OpDipolePlanck(args.freq, margin=args.margin,
                                       output='dipole_tod', mode='TOTAL',
                                       npipe_mode=True)
            dipole.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("Dipole calculation took {:.3f} s".format(elapsed),
                      flush=True)
            start = stop

            if shdet_input != 'dipole_tod':
                adder = tp.OpCacheMath(in1=shdet_input, in2='dipole_tod',
                                       add=True, out=shdet_input)
                adder.exec(data)
                stop = MPI.Wtime()
                elapsed = stop - start
                if comm.comm_world.rank == 0:
                    print("Dipole adding took {:.3f} s".format(elapsed),
                          flush=True)
                start = stop

        if args.effdir_in_shdet is not None:
            tod.set_effdir_out(args.effdir_in_shdet)

            writer = tp.OpOutputPlanck(signal_name=shdet_input, margin=args.margin)
            writer.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("SHDet input timeline output to disk took {:.3f} s".format(elapsed), flush=True)
            start = stop

        # Now run detector simulation


        # get SHDet parameters to pass to the SHDet constructor
        shdet_params = {}

        # ADC table files
        adc_table_path = None
        if 'adc_table' in arglist.keys():
            if arglist['adc_table'] is not None:
                adc_table_path = {}

        # TF files
        tf_path = None
        if 'shdet_tffile' in arglist.keys():
            if arglist['shdet_tffile'] is not None:
                tf_path = {}



        global_offset_dict = {}
        for idet, det in enumerate(detectors):

            if args.adc_table is not None:
                npars = len(arglist['adc_table'])
                if npars==1:
                    adc_table_path[det] = arglist['adc_table'][0]
                    if npars==ndets:
                        adc_table_path[det] = arglist['adc_table'][idet]

            if args.shdet_tffile is not None:
                npars = len(arglist['shdet_tffile'])
                if npars==1:
                    tf_path[det] = arglist['shdet_tffile'][0]
                    if npars==ndets:
                        tf_path[det] = arglist['shdet_tffile'][idet]

            shdet_params[det] = {}

            # base seed, always just set to the value specified on the
            # command line
            shdet_params[det]['seed'] = args.seed
            shdet_params[det]['switch_optical_offset'] = args.switch_optical_offset
            shdet_params[det]['switch_raw_offset'] = args.switch_raw_offset

            if args.adc_on:
                shdet_params[det]['adc_on'] = 1

            if arglist['noise_dsn'] is not None:
                npars = len(arglist['noise_dsn'])
                if npars == 1:
                    shdet_params[det]['noise_dsn'] = arglist['noise_dsn'][0] \
                                                     / np.sqrt(40) * 10.2 / 2**16
                if npars == ndets:
                    shdet_params[det]['noise_dsn'] = arglist['noise_dsn'][idet] \
                                                     / np.sqrt(40.) * 10.2 / 2**16

            if arglist['cstray'] is not None:
                npars = len(arglist['cstray'])
                if npars == 1:
                    shdet_params[det]['cstray'] = arglist['cstray'][0]
                if npars==ndets:
                    shdet_params[det]['cstray'] = arglist['cstray'][idet]

            if arglist['bdac'] is not None:
                npars = len(arglist['bdac'])
                if npars == 1:
                    shdet_params[det]['bdac'] = arglist['bdac'][0]
                if npars == ndets:
                    shdet_params[det]['bdac'] = arglist['bdac'][idet]

            if arglist['optical_load_watts'] is not None:
                npars = len(arglist['optical_load_watts'])
                if npars==1:
                    shdet_params[det]['optical_load_watts'] \
                        = arglist['optical_load_watts'][0]
                if npars==ndets:
                    shdet_params[det]['optical_load_watts'] \
                        = arglist['optical_load_watts'][idet]

            if arglist['gain_w_per_kcmb'] is not None:
                npars = len(arglist['gain_w_per_kcmb'])
                if npars == 1:
                    shdet_params[det]['gain_w_per_kcmb'] \
                        = arglist['gain_w_per_kcmb'][0]
                if npars == ndets:
                    shdet_params[det]['gain_w_per_kcmb'] \
                        = arglist['gain_w_per_kcmb'][idet]

            if arglist['raw_model'] is not None:
                npars = len(arglist['raw_model'])
                if npars == 1:
                    shdet_params[det]['raw_model'] = arglist['raw_model'][0]
                if npars == ndets:
                    shdet_params[det]['raw_model'] = arglist['raw_model'][idet]

            if arglist['global_phase_shift'] is not None:
                npars = len(arglist['global_phase_shift'])
                if npars == 1:
                    global_offset_dict[det] = arglist['global_phase_shift'][0]
                if npars == ndets:
                    global_offset_dict[det] = arglist['global_phase_shift'][idet]

        # do the simulation
        simdet = tp.OpSimSHDET(
            dets=detectors, imofile=args.imo, adc_table=adc_table_path,
            input=shdet_input, margin=args.margin, calfile=args.calfile,
            params=shdet_params, tffile=tf_path,
            read_no_signal=args.read_no_signal, offset_file=args.offset_file)
        simdet.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print("Detector simulation took {:.3f} s".format(elapsed), flush=True)
        start = stop

        # now grab the transfer function, a dictionary
        shdet_freq, shdet_TF = simdet.get_TF()

        # convert into real and imaginary tables
        shdet_TF_real = {}
        shdet_TF_imag = {}
        for det in shdet_TF.keys():
            tf = shdet_TF[det]
            shdet_TF_real[det] = tf.real
            shdet_TF_imag[det] = tf.imag

        if comm.comm_world.rank == 0:
            print("Grabbed SHDet time response function", flush=True)

        input_tod = 'shdet_tod'
        input_dark = None # Read dark TOI off the disk, at least for now
        input_timestamps = 'times'
        input_common_flags = 'common_flags'

        if args.effdir_out_shdet is not None:
            tod.set_effdir_out(args.effdir_out_shdet)

            writer = tp.OpOutputPlanck(signal_name='shdet_tod',
                                       margin=args.margin)
            writer.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("SHDet timeline output took {:.3f} s".format(elapsed),
                      flush=True)
            start = stop

    else:
        input_tod = None
        input_dark = None
        input_timestamps = None
        input_common_flags = None

    # Use detector weights from the RIMO

    detweights = {}
    for d in tod.detectors:
        net = tod.rimo[d].net
        fsample = tod.rimo[d].fsample
        detweights[d] = 1.0 / (fsample * net * net)

    # Iterate processing steps

    # FIXME: if these need different constructor parameters
    # for each iteration, then we need to construct them inside
    # the iterative loop.

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        print('Constructing processing objects ...', end='', flush=True)
    start = MPI.Wtime()

    tod_name = None
    flag_name = None
    common_flag_name = None


    if not args.skip_preproc:

        pproc = tp.OpPreproc(
            args.imo, args.freq,
            input_tod=input_tod, input_dark=input_dark,
            input_timestamps=input_timestamps,
            input_common_flags=input_common_flags,
            bg_map_path=args.bg_map, bg_nside=args.bg_nside,
            bg_pol=args.bg_pol, bg_has_dipole=args.bg_has_dipole,
            detmask=args.detmask, pntmask=args.pntmask, ssomask=args.ssomask,
            maskfile=args.preproc_mask,
            nbin_phase=args.nbin,
            jump_filter_len=args.jump_filter_len,
            despike=args.despike,
            despiker_timeout=args.despiker_timeout,
            preproc_dark=args.preproc_dark,
            calfile=args.calfile,
            even_correction=args.even_correction,
            odd_correction=args.odd_correction,
            measure_ADC=args.measure_ADC, nadc_iter=args.niter_ADC,
            deltaADU=args.delta_ADC, nphase4k=args.nphase4k,
            margin=args.margin,
            effdir_out=args.effdir_out_preproc,
            flag_planets=args.flag_planets,
            planet_flag_radius=args.planet_flag_radius,
            bad_rings=args.bad_rings, out=args.out,
            jump_threshold=args.jump_threshold,
            g0 = args.g0,
            v0 = args.v0,
            global_phase_shift=args.global_phase_shift, # TODO: make this a dictionary indexed by detector
            tabulated_transfer_function=(shdet_freq, shdet_TF_real, shdet_TF_imag),
       )
        tod_name = 'preproc_tod'
        flag_name = 'preproc_detflags'
        common_flag_name = 'common_flags'
        pntmask = 2
        ssomask = 2
    else:
        pntmask = args.pntmask
        ssomask = args.ssomask

    if args.make_rings:

        if args.skip_preproc:
            input_tod = None
            input_flags = None
            input_timestamps = None
            input_commonflags = None
        else:
            input_tod = 'preproc_tod'
            input_flags = 'preproc_detflags'
            input_timestamps = 'times'
            input_commonflags = 'common_flags'

        ringmaker = tp.OpRingMaker(args.nside_ring, nside, signal=input_tod, flags=input_flags,
                                    timestamps=input_timestamps, commonflags=input_commonflags,
                                    fileroot=args.ring_root, out=args.out)


    if not args.skip_reproc:

        if args.skip_preproc:
            input_tod = None
            input_flags = None
            input_velocity = None
            input_timestamps = None
            input_commonflags = None
        else:
            input_tod = 'preproc_tod'
            input_flags = 'preproc_detflags'
            input_velocity = 'velocity'
            input_timestamps = 'times'
            input_commonflags = 'common_flags'

        rproc = tp.OpReproc(
            args.freq,
            input=tod_name, input_flags=flag_name,
            input_timestamps=input_timestamps, input_common_flags=input_commonflags,
            nharm=args.nharm, band_calibrate=args.band_calibrate,
            do_zodi=args.zodi, use_swap=args.swap, pntmask=pntmask,
            ssomask=ssomask,
            co=args.CO, co2=args.CO2, dust=args.dust, dust_pol=args.dust_pol,
            sync=args.sync, sync_pol=args.sync_pol, ame=args.AME,
            freefree=args.freefree,
            bg_map_path=args.bg_map, bg_pol=args.bg_pol,
            bg_has_dipole=args.bg_has_dipole,
            pix_nside=nside,
            maskfile_destripe=args.reproc_destripe_mask,
            maskfile_calib=args.reproc_calib_mask,
            maskfile_bandpass=args.reproc_bandpass_mask,
            effdir_out=args.effdir_out_reproc,
            effdir_out_diode0=None,
            effdir_out_diode1=None,
            map_dir=map_dir,
            niter=args.niter_reproc, niter_multi=args.niter_reproc_multi,
            single_cal=args.single_cal, multi_cal=args.multi_cal,
            recalibrate=args.recalibrate,
            save_maps=args.save_maps, save_survey_maps=args.save_survey_maps,
            save_template_maps=args.save_template_maps, out=args.out,
            forcepol=args.forcepol, forcedipo=args.forcedipo,
            forcefsl=args.forcefsl,
            pol_deriv=args.pol_deriv,
            bpcorrect=args.bpcorrect, fg_dipo=args.fgdipo,
            harmonize=args.harmonize, use_iiqu=args.use_iiqu,
            do_fsl=(args.effdir_fsl is not None),
            no_destripe=args.reproc_no_destripe,
            madampars=pars,
            bad_rings=args.bad_rings,
            temperature_only=(pointingmode == 'I'),
            calfile=args.calfile, lowpassfreq=args.reproc_lowpassfreq,
            sigmalim = args.sigmalim_cal,
            pol_map_bpcorr_path=args.pol_map_bpcorr_path,)
        tod_name = 'reproc_tod'
        flag_name = 'reproc_detflags'
        common_flag_name = 'common_flags'

    if not args.skip_madam:
        madam = tm.OpMadam(name=tod_name, name_out='madam_tod',
                           flag_name=flag_name, flag_mask=255,
                           common_flag_name=common_flag_name, common_flag_mask=1,
                           params=pars, detweights=detweights, purge=True,
                           timestamps_name='times')

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Set up processing and mapmaking took {:.3f} s".format(elapsed), flush=True)
    start = stop

    converged = False
    iter = 0

    while not converged:

        if not args.skip_preproc:
            pproc.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("Iter {}: Preproc took {:.3f} s".format(iter, elapsed), flush=True)
                tod.cache.report()
                print('Memory high water mark: {:.2f} MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))
        start = stop


        if args.make_rings:

            ringmaker.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("Iter {}: ringmaking took {:.3f} s".format(iter, elapsed), flush=True)
                tod.cache.report()
                print('Memory high water mark: {:.2f} MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))

        start = stop


        if not args.skip_reproc:

            rproc.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("Iter {}: Reproc took {:.3f} s".format(iter, elapsed), flush=True)

        # purge unnecessary cached data

        for cachename in ['phase', 'velocity']:
            if tod.cache.exists(cachename):
                tod.cache.destroy(cachename)

        for det in tod.local_dets:
            cachename = 'quats_{}'.format(det)
            if tod.cache.exists(cachename):
                tod.cache.destroy(cachename)

        if comm.comm_world.rank == 0:
            tod.cache.report()
            print('Memory high water mark: {:.2f} MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))

        start = stop

        if not args.skip_madam:

            if args.generate_dipole:
                subtractor = tp.OpCacheMath(in1=tod_name, in2='dipole_tod',
                                   subtract=True, out=tod_name)
                subtractor.exec(data)
                stop = MPI.Wtime()
                elapsed = stop - start
                if comm.comm_world.rank == 0:
                    print("Dipole subtraction took {:.3f} s".format(elapsed),
                          flush=True)

            start = stop

            # Make a map
            madam.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("Iter {}: Madam took {:.3f} s".format(iter, elapsed), flush=True)
                tod.cache.report()
                print('Memory high water mark: {:.2f} MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))
            start = stop

            if args.effdir_out_madam is not None:

                # FIXME: Madam needs to preserve the flags and we need to write them out here
                writer = tp.OpOutputPlanck(signal_name='madam_tod', flags_name=None,
                                         commonflags_name=None, effdir_out=args.effdir_out_madam)

                writer.exec(data)

                comm.comm_world.barrier()
                stop = MPI.Wtime()
                elapsed = stop - start
                if comm.comm_world.rank == 0:
                    print("Iter {}: Madam output took {:.3f} s".format(iter, elapsed), flush=True)
                start = stop


        iter += 1

        # FIXME:  use some criterion here
        converged = True


    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - global_start
    if comm.comm_world.rank == 0:
        print("Total Time:  {:.2f} seconds".format(elapsed))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print('Exception occurred: "{}"'.format(e), flush=True)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('*** print_tb:')
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print('*** print_exception:')
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print('*** print_exc:')
        traceback.print_exc()
        print('*** format_exc, first and last line:')
        formatted_lines = traceback.format_exc().splitlines()
        print(formatted_lines[0])
        print(formatted_lines[-1])
        print('*** format_exception:')
        print(repr(traceback.format_exception(exc_type, exc_value,
                                              exc_traceback)))
        print('*** extract_tb:')
        print(repr(traceback.extract_tb(exc_traceback)))
        print('*** format_tb:')
        print(repr(traceback.format_tb(exc_traceback)))
        print('*** tb_lineno:', exc_traceback.tb_lineno, flush=True)
        MPI.COMM_WORLD.Abort()
