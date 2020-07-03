#!/usr/bin/env python

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import datetime
import os
import re
import sys
import traceback

from toast import Comm, Data, distribute_discrete
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm
import toast
from toast.mpi import MPI
from toast_planck.utilities import DEFAULT_PARAMETERS
from toast.tod import OpSimNoise

import numpy as np
import toast.timing as timing
import toast_planck as tp


# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)
def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_planck_reduce (total)")

    mpiworld, procs, rank, comm = get_comm()

    if comm.world_rank == 0:
        print("Running with {} processes at {}".format(
            procs, str(datetime.datetime.now())))

    parser = argparse.ArgumentParser(description='Simple MADAM Mapmaking',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--skip_madam', dest='skip_madam', default=False,
                        action='store_true', help='D not make maps with Madam.')
    parser.add_argument('--skip_noise', dest='skip_noise', default=False,
                        action='store_true',
                        help='Do not add simulated noise to the TOD.')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--freq', required=True, type=np.int, help='Frequency')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true',
                        help='Write data distribution info to file')
    parser.add_argument('--dets', required=False, default=None,
                        help='Detector list (comma separated)')
    parser.add_argument('--effdir', required=True,
                        help='Input Exchange Format File directory')
    parser.add_argument('--effdir2', required=False,
                        help='Additional input Exchange Format File directory')
    parser.add_argument('--effdir_pntg', required=False,
                        help='Input Exchange Format File directory for '
                        'pointing')
    parser.add_argument('--effdir_fsl', required=False,
                        help='Input Exchange Format File directory for '
                        'straylight')
    parser.add_argument('--obtmask', required=False, default=1, type=np.int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=1, type=np.int,
                        help='Quality flag mask')
    parser.add_argument('--pntflagmask', required=False, default=0, type=np.int,
                        help='Which OBT flag bits to raise for HCM maneuvers')
    parser.add_argument('--bad_intervals', required=False,
                        help='Path to bad interval file.')
    parser.add_argument('--ringdb', required=True, help='Ring DB file')
    parser.add_argument('--odfirst', required=False, default=None,
                        help='First OD to use')
    parser.add_argument('--odlast', required=False, default=None,
                        help='Last OD to use')
    parser.add_argument('--ringfirst', required=False, default=None,
                        help='First ring to use')
    parser.add_argument('--ringlast', required=False, default=None,
                        help='Last ring to use')
    parser.add_argument('--obtfirst', required=False, default=None,
                        help='First OBT to use')
    parser.add_argument('--obtlast', required=False, default=None,
                        help='Last OBT to use')
    parser.add_argument('--read_eff', dest='read_eff', default=False,
                        action='store_true',
                        help='Read and co-add the signal from effdir')
    parser.add_argument('--decalibrate', required=False,
                        help='Path to calibration file to decalibrate with. '
                        'You can use python string formatting, assuming '
                        '.format(mc)')
    parser.add_argument('--calibrate', required=False,
                        help='Path to calibration file to calibrate with. '
                        'You can use python string formatting, assuming '
                        '.format(mc)')
    parser.add_argument('--madampar', required=False, default=None,
                        help='Madam parameter file')
    parser.add_argument('--nside', required=False, default=None, type=np.int,
                        help='Madam resolution')
    parser.add_argument('--out', required=False, default='.',
                        help='Output directory')
    parser.add_argument('--madam_prefix', required=False, help='map prefix')
    parser.add_argument('--make_rings', dest='make_rings', default=False,
                        action='store_true', help='Compile ringsets.')
    parser.add_argument('--nside_ring', required=False, default=128,
                        type=np.int, help='Ringset resolution')
    parser.add_argument('--ring_root', required=False, default='ringset',
                        help='Root filename for ringsets (setting to empty '
                        'disables ringset output).')
    parser.add_argument('--MC_start', required=False, default=0, type=np.int,
                        help='First Monte Carlo noise realization')
    parser.add_argument('--MC_count', required=False, default=1, type=np.int,
                        help='Number of Monte Carlo noise realizations')
    # noise parameters
    parser.add_argument('--noisefile', required=False, default='RIMO',
                        help='Path to noise PSD files for noise filter. '
                        'Tag DETECTOR will be replaced with detector name.')
    parser.add_argument('--noisefile_simu', required=False, default='RIMO',
                        help='Path to noise PSD files for noise simulation. '
                        'Tag DETECTOR will be replaced with detector name.')
    # Dipole parameters
    dipogroup = parser.add_mutually_exclusive_group()
    dipogroup.add_argument(
        '--dipole', dest='dipole', required=False,
        default=False, action='store_true',
        help='Simulate dipole')
    dipogroup.add_argument(
        '--solsys_dipole', dest='solsys_dipole',
        required=False, default=False, action='store_true',
        help='Simulate solar system dipole')
    dipogroup.add_argument(
        '--orbital_dipole', dest='orbital_dipole',
        required=False, default=False, action='store_true',
        help='Simulate orbital dipole')
    dipo_parameters_group = parser.add_argument_group('dipole_parameters')
    dipo_parameters_group.add_argument(
        '--solsys_speed', required=False, type=np.float,
        default=DEFAULT_PARAMETERS["solsys_speed"],
        help='Solar system speed wrt. CMB rest frame in km/s. Default is '
        'Planck 2015 best fit value')
    dipo_parameters_group.add_argument(
        '--solsys_glon', required=False, type=np.float,
        default=DEFAULT_PARAMETERS["solsys_glon"],
        help='Solar system velocity direction longitude in degrees')
    dipo_parameters_group.add_argument(
        '--solsys_glat', required=False, type=np.float,
        default=DEFAULT_PARAMETERS["solsys_glat"],
        help='Solar system velocity direction latitude in degrees')

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if comm.world_rank == 0:
        print('All parameters:')
        print(args, flush=True)

    if args.MC_count < 1:
        raise RuntimeError('MC_count = {} < 1. Nothing done.'
                           ''.format(args.MC_count))

    timer = Timer()
    timer.start()

    nrange = 1

    odranges = None
    if args.odfirst is not None and args.odlast is not None:
        odranges = []
        firsts = [int(i) for i in str(args.odfirst).split(',')]
        lasts = [int(i) for i in str(args.odlast).split(',')]
        for odfirst, odlast in zip(firsts, lasts):
            odranges.append((odfirst, odlast))
        nrange = len(odranges)

    ringranges = None
    if args.ringfirst is not None and args.ringlast is not None:
        ringranges = []
        firsts = [int(i) for i in str(args.ringfirst).split(',')]
        lasts = [int(i) for i in str(args.ringlast).split(',')]
        for ringfirst, ringlast in zip(firsts, lasts):
            ringranges.append((ringfirst, ringlast))
        nrange = len(ringranges)

    obtranges = None
    if args.obtfirst is not None and args.obtlast is not None:
        obtranges = []
        firsts = [float(i) for i in str(args.obtfirst).split(',')]
        lasts = [float(i) for i in str(args.obtlast).split(',')]
        for obtfirst, obtlast in zip(firsts, lasts):
            obtranges.append((obtfirst, obtlast))
        nrange = len(obtranges)

    if odranges is None:
        odranges = [None] * nrange

    if ringranges is None:
        ringranges = [None] * nrange

    if obtranges is None:
        obtranges = [None] * nrange

    detectors = None
    if args.dets is not None:
        detectors = re.split(',', args.dets)

    # create the TOD for this observation

    if args.noisefile != 'RIMO' or args.noisefile_simu != 'RIMO':
        do_eff_cache = True
    else:
        do_eff_cache = False

    tods = []

    for obtrange, ringrange, odrange in zip(obtranges, ringranges, odranges):
        # create the TOD for this observation
        tods.append(tp.Exchange(
            comm=comm.comm_group,
            detectors=detectors,
            ringdb=args.ringdb,
            effdir_in=args.effdir,
            extra_effdirs=[args.effdir2, args.effdir_fsl],
            effdir_pntg=args.effdir_pntg,
            obt_range=obtrange,
            ring_range=ringrange,
            od_range=odrange,
            freq=args.freq,
            RIMO=args.rimo,
            obtmask=args.obtmask,
            flagmask=args.flagmask,
            pntflagmask=args.pntflagmask,
            do_eff_cache=do_eff_cache))

    # Make output directory

    if not os.path.isdir(args.out) and comm.world_rank == 0:
        os.makedirs(args.out)

    # Read in madam parameter file
    # Allow more than one entry, gather into a list
    repeated_keys = ['detset', 'detset_nopol', 'survey']
    pars = {}

    if comm.world_rank == 0:
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
                                if key not in pars:
                                    pars[key] = []
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
            fsample = {30: 32.51, 44: 46.55, 70: 78.77}[args.freq]
        except Exception:
            fsample = 180.3737
        pars['fsample'] = fsample
        pars['path_output'] = args.out

    pars = comm.comm_world.bcast(pars, root=0)

    madam_mcmode = True
    if 'nsubchunk' in pars and int(pars['nsubchunk']) > 1:
        madam_mcmode = False

    if args.noisefile != 'RIMO' or args.noisefile_simu != 'RIMO':
        # We split MPI_COMM_WORLD into single process groups, each of
        # which is assigned one or more observations (rings)
        comm = toast.Comm(groupsize=1)

    # This is the distributed data, consisting of one or
    # more observations, each distributed over a communicator.
    data = toast.Data(comm)

    for iobs, tod in enumerate(tods):
        if args.noisefile != 'RIMO' or args.noisefile_simu != 'RIMO':
            # Use a toast helper method to optimally distribute rings between
            # processes.
            dist = distribute_discrete(tod.ringsizes, comm.world_size)
            my_first_ring, my_n_ring = dist[comm.world_rank]

            for my_ring in range(my_first_ring, my_first_ring + my_n_ring):
                ringtod = tp.Exchange.from_tod(
                    tod, my_ring, comm.comm_group, noisefile=args.noisefile,
                    noisefile_simu=args.noisefile_simu)
                ob = {}
                ob['name'] = 'ring{:05}'.format(ringtod.globalfirst_ring)
                ob['id'] = ringtod.globalfirst_ring
                ob['tod'] = ringtod
                ob['intervals'] = ringtod.valid_intervals
                ob['baselines'] = None
                ob['noise'] = ringtod.noise
                ob['noise_simu'] = ringtod.noise_simu
                data.obs.append(ob)
        else:
            ob = {}
            ob['name'] = 'observation{:04}'.format(iobs)
            ob['id'] = 0
            ob['tod'] = tod
            ob['intervals'] = tod.valid_intervals
            ob['baselines'] = None
            ob['noise'] = tod.noise
            ob['noise_simu'] = tod.noise

            data.obs.append(ob)

    rimo = tods[0].rimo

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Metadata queries")

    # Always read the signal and flags, even if the signal is later
    # overwritten.  There is no overhead for the signal because it is
    # interlaced with the flags.

    tod_name = 'signal'
    timestamps_name = 'timestamps'
    flags_name = 'flags'
    common_flags_name = 'common_flags'
    reader = tp.OpInputPlanck(signal_name=tod_name, flags_name=flags_name,
                              timestamps_name=timestamps_name,
                              commonflags_name=common_flags_name)
    if comm.world_rank == 0:
        print('Reading input signal from {}'.format(args.effdir),
              flush=True)
    reader.exec(data)
    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Read")

    # Clear the signal if we don't need it

    if not args.read_eff:
        eraser = tp.OpCacheMath(in1=tod_name, in2=0, multiply=True,
                                out=tod_name)
        if comm.world_rank == 0:
            print('Erasing TOD', flush=True)
        eraser.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Erase")

    # Optionally flag bad intervals

    if args.bad_intervals is not None:
        flagger = tp.OpBadIntervals(path=args.bad_intervals)
        flagger.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Apply {}".format(args.bad_intervals))

    # Now read an optional second TOD to add with the first

    if args.effdir2 is not None:
        # Read the extra TOD and add it to the first one
        reader = tp.OpInputPlanck(signal_name='tod2', flags_name=None,
                                  timestamps_name=None, commonflags_name=None,
                                  effdir=args.effdir2)
        if comm.world_rank == 0:
            print('Reading extra TOD from {}'.format(args.effdir2),
                  flush=True)
        reader.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            print("Reading took {:.3f} s".format(elapsed), flush=True)

        adder = tp.OpCacheMath(in1=tod_name, in2='signal2', add=True,
                               out=tod_name)
        if comm.world_rank == 0:
            print('Adding TODs', flush=True)
        adder.exec(data)

        # Erase the extra cache object
        for ob in data.obs:
            tod = ob['tod']
            tod.cache.clear('signal2_.*')

    if args.effdir_fsl is not None:
        # Read the straylight signal into the tod cache under
        # "fsl_<detector>"
        reader = tp.OpInputPlanck(signal_name='fsl', flags_name=None,
                                  timestamps_name=None, commonflags_name=None,
                                  effdir=args.effdir_fsl)
        if comm.world_rank == 0:
            print('Reading straylight signal from {}'.format(args.effdir_fsl),
                  flush=True)
        reader.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Read FSL")
        do_fsl = True
    else:
        do_fsl = False

    # make a planck Healpix pointing matrix
    mode = 'IQU'
    if pars['temperature_only'] == 'T':
        mode = 'I'

    if args.nside is None:
        if 'nside_map' in pars:
            nside = int(pars['nside_map'])
        else:
            raise RuntimeError(
                'Nside must be set either in the Madam parameter file or on '
                'the command line')
    else:
        nside = args.nside
        pars['nside_map'] = nside
    if 'nside_cross' not in pars or pars['nside_cross'] > pars['nside_map']:
        pars['nside_cross'] = pars['nside_map']

    do_dipole = args.dipole or args.solsys_dipole or args.orbital_dipole

    pointing = tp.OpPointingPlanck(
        nside=nside, mode=mode, RIMO=rimo, margin=0, apply_flags=True,
        keep_vel=do_dipole, keep_pos=False, keep_phase=False,
        keep_quats=do_dipole)
    pointing.exec(data)
    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Pointing Matrix")

    flags_name = 'flags'
    common_flags_name = 'common_flags'

    # for now, we pass in the noise weights from the RIMO.
    detweights = {}
    for d in tod.detectors:
        net = tod.rimo[d].net
        fsample = tod.rimo[d].fsample
        detweights[d] = 1.0 / (fsample * net * net)

    if args.debug:
        with open("debug_planck_exchange_madam.txt", "w") as f:
            data.info(f)

    if do_dipole:
        # Simulate the dipole
        if args.dipole:
            dipomode = 'total'
        elif args.solsys_dipole:
            dipomode = 'solsys'
        else:
            dipomode = 'orbital'
        dipo = tp.OpDipolePlanck(
            args.freq, solsys_speed=args.solsys_speed,
            solsys_glon=args.solsys_glon, solsys_glat=args.solsys_glat,
            mode=dipomode, output='dipole', keep_quats=False)
        dipo.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Dipole")

    # Loop over Monte Carlos

    madam = None

    for mc in range(args.MC_start, args.MC_start + args.MC_count):

        out = "{}/{:05d}".format(args.out, mc)
        if comm.world_rank == 0:
            if not os.path.isdir(out):
                os.makedirs(out)

        # clear all noise data from the cache, so that we can generate
        # new noise timestreams.

        for ob in data.obs:
            ob['tod'].cache.clear("noise_.*")
        tod_name = 'signal'

        if do_dipole:
            adder = tp.OpCacheMath(in1=tod_name, in2='dipole',
                                   add=True, out='noise')
            adder.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Add dipole".format(mc))
            tod_name = 'noise'

        # Simulate noise

        if not args.skip_noise:
            tod_name = 'noise'
            nse = toast.tod.OpSimNoise(
                out=tod_name, realization=mc, component=0,
                noise='noise_simu', rate=fsample)
            if comm.world_rank == 0:
                print('Simulating noise from {}'.format(args.noisefile_simu),
                      flush=True)
            nse.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Noise simulation".format(mc))

            # If we didn't add the dipole, we need to add the input
            # signal with the noise we just simulated

            if args.read_eff and not do_dipole:
                adder = tp.OpCacheMath(in1=tod_name, in2='signal',
                                       add=True, out=tod_name)
                adder.exec(data)
                if mpiworld is not None:
                    mpiworld.barrier()
                if comm.world_rank == 0:
                    timer.report_clear("MC {}:  Add input signal".format(mc))

        # Make rings

        if args.make_rings:
            ringmaker = tp.OpRingMaker(
                args.nside_ring, nside, signal=tod_name,
                fileroot=args.ring_root, out=out,
                commonmask=args.obtmask, detmask=args.flagmask)
            ringmaker.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Ringmaking".format(mc))

        # Apply calibration errors

        if args.decalibrate is not None:
            fn = args.decalibrate
            try:
                fn = fn.format(mc)
            except Exception:
                pass
            if comm.world_rank == 0:
                print('Decalibrating with {}'.format(fn), flush=True)
            decalibrator = tp.OpCalibPlanck(
                signal_in=tod_name, signal_out='noise',
                file_gain=fn, decalibrate=True)
            decalibrator.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Decalibrate".format(mc))
            tod_name = 'noise'

        if args.calibrate is not None:
            fn = args.calibrate
            try:
                fn = fn.format(mc)
            except Exception:
                pass
            if comm.world_rank == 0:
                print('Calibrating with {}'.format(fn), flush=True)
            calibrator = tp.OpCalibPlanck(
                signal_in=tod_name, signal_out='noise', file_gain=fn)
            calibrator.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Calibrate".format(mc))
            tod_name = 'noise'

        # Subtract the dipole and straylight

        if do_dipole:
            subtractor = tp.OpCacheMath(in1=tod_name, in2='dipole',
                                        subtract=True, out='noise')
            subtractor.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Subtract dipole".format(mc))
            tod_name = 'noise'

        if do_fsl:
            subtractor = tp.OpCacheMath(in1=tod_name, in2='fsl',
                                        subtract=True, out='noise')
            subtractor.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Subtract straylight".format(mc))
            tod_name = 'noise'

        # Make the map

        if not args.skip_madam:
            # Make maps
            if madam is None:
                try:
                    madam = toast.todmap.OpMadam(
                        params=pars, detweights=detweights, purge_tod=True,
                        name=tod_name, apply_flags=False,
                        name_out=None, noise='noise', mcmode=madam_mcmode,
                        translate_timestamps=False)
                except Exception as e:
                    raise Exception('{:4} : ERROR: failed to initialize Madam: '
                                    '{}'.format(comm.world_rank, e))
            madam.params['path_output'] = out
            madam.exec(data)
            if mpiworld is not None:
                mpiworld.barrier()
            if comm.world_rank == 0:
                timer.report_clear("MC {}:  Mapmaking".format(mc))

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
        main()
    except Exception as e:
        print('Exception occurred: "{}"'.format(e), flush=True)
        if procs == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2,
                                  file=sys.stdout)
        print("*** print_exc:")
        traceback.print_exc()
        print("*** format_exc, first and last line:")
        formatted_lines = traceback.format_exc().splitlines()
        print(formatted_lines[0])
        print(formatted_lines[-1])
        print("*** format_exception:")
        print(repr(traceback.format_exception(exc_type, exc_value,
                                              exc_traceback)))
        print("*** extract_tb:")
        print(repr(traceback.extract_tb(exc_traceback)))
        print("*** format_tb:")
        print(repr(traceback.format_tb(exc_traceback)))
        print("*** tb_lineno:", exc_traceback.tb_lineno, flush=True)
        mpiworld.Abort()
