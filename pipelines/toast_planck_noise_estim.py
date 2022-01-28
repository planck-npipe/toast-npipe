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

import numpy as np

import toast
from toast.mpi import MPI
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm
import toast_planck as tp


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


def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_planck_reduce (total)")

    mpiworld, procs, rank, comm = get_comm()

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        print('Running with {} processes at {}'
              ''.format(procs, str(datetime.datetime.now())))

    parser = argparse.ArgumentParser(description='Planck Ringset making',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--freq', required=True, type=int, help='Frequency')
    parser.add_argument('--dets', required=False, default=None,
                        help='Detector list (comma separated)')
    parser.add_argument('--nosingle', dest='nosingle', required=False,
                        default=False, action='store_true',
                        help='Do not compute single detector PSDs')
    parser.add_argument('--effdir', required=True,
                        help='Input Exchange Format File directory')
    parser.add_argument('--effdir_pntg', required=False,
                        help='Input Exchange Format File directory '
                        'for pointing')
    parser.add_argument('--obtmask', required=False, default=1, type=int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=1, type=int,
                        help='Quality flag mask')
    parser.add_argument('--skymask', required=False, help='Pixel mask file')
    parser.add_argument('--skymap', required=False, help='Sky estimate file')
    parser.add_argument('--skypol', dest='skypol', required=False,
                        default=False, action='store_true',
                        help='Sky estimate is polarized')
    parser.add_argument('--no_spin_harmonics', dest='no_spin_harmonics',
                        required=False, default=False, action='store_true',
                        help='Do not include PSD bins with spin harmonics')
    parser.add_argument('--calibrate', required=False,
                        help='Path to calibration file to calibrate with.')
    parser.add_argument('--calibrate_signal_estimate',
                        dest='calibrate_signal_estimate', required=False,
                        default=False, action='store_true', help='Calibrate '
                        'the signal estimate using linear regression.')
    parser.add_argument('--ringdb', required=True, help='Ring DB file')
    parser.add_argument('--odfirst', required=False, default=None,
                        type=int, help='First OD to use')
    parser.add_argument('--odlast', required=False, default=None,
                        type=int, help='Last OD to use')
    parser.add_argument('--ringfirst', required=False, default=None,
                        type=int, help='First ring to use')
    parser.add_argument('--ringlast', required=False, default=None,
                        type=int, help='Last ring to use')
    parser.add_argument('--obtfirst', required=False, default=None,
                        type=float, help='First OBT to use')
    parser.add_argument('--obtlast', required=False, default=None,
                        type=float, help='Last OBT to use')
    parser.add_argument('--out', required=False, default='.',
                        help='Output directory')
    parser.add_argument('--nbin_psd', required=False, default=1000,
                        type=int, help='Number of logarithmically '
                        'distributed spectral bins to write.')
    parser.add_argument('--lagmax', required=False, default=100000,
                        type=int, help='Maximum lag to evaluate for the '
                        'autocovariance function [samples].')
    parser.add_argument('--stationary_period', required=False, default=86400.,
                        type=float,
                        help='Length of a stationary interval [seconds].')
    # Dipole parameters
    dipogroup = parser.add_mutually_exclusive_group()
    dipogroup.add_argument('--dipole', dest='dipole', required=False,
                           default=False, action='store_true',
                           help='Simulate dipole')
    dipogroup.add_argument('--solsys_dipole', dest='solsys_dipole',
                           required=False, default=False, action='store_true',
                           help='Simulate solar system dipole')
    dipogroup.add_argument('--orbital_dipole', dest='orbital_dipole',
                           required=False, default=False, action='store_true',
                           help='Simulate orbital dipole')
    # Extra filter
    parser.add_argument('--filterfile', required=False,
                        help='Extra filter file.')

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if comm.comm_world.rank == 0:
        print('All parameters:')
        print(args, flush=True)

    timer = Timer()
    timer.start()

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
        detectors = re.split(',', args.dets)

    if args.nosingle and len(detectors) != 2:
        raise RuntimeError('You cannot skip the single detectors PSDs '
                           'without multiple detectors.')

    # This is the distributed data, consisting of one or
    # more observations, each distributed over a communicator.
    data = toast.Data(comm)

    # Make output directory

    if not os.path.isdir(args.out) and comm.comm_world.rank == 0:
        os.mkdir(args.out)

    # create the TOD for this observation

    tod = tp.Exchange(
        comm=comm.comm_group,
        detectors=detectors,
        ringdb=args.ringdb,
        effdir_in=args.effdir,
        effdir_pntg=args.effdir_pntg,
        obt_range=obtrange,
        ring_range=ringrange,
        od_range=odrange,
        freq=args.freq,
        RIMO=args.rimo,
        obtmask=args.obtmask,
        flagmask=args.flagmask,
        do_eff_cache=False,
    )

    rimo = tod.rimo

    ob = {}
    ob['name'] = 'mission'
    ob['id'] = 0
    ob['tod'] = tod
    ob['intervals'] = tod.valid_intervals
    ob['baselines'] = None
    ob['noise'] = tod.noise

    data.obs.append(ob)

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Metadata queries")

    # Read the signal

    tod_name = 'signal'
    flags_name = 'flags'

    reader = tp.OpInputPlanck(signal_name=tod_name, flags_name=flags_name)
    if comm.comm_world.rank == 0:
        print('Reading input signal from {}'.format(args.effdir),
              flush=True)
    reader.exec(data)
    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Reading")

    if args.calibrate is not None:
        fn = args.calibrate
        if comm.comm_world.rank == 0:
            print('Calibrating with {}'.format(fn), flush=True)
        calibrator = tp.OpCalibPlanck(
            signal_in=tod_name, signal_out=tod_name, file_gain=fn)
        calibrator.exec(data)
        comm.comm_world.barrier()
        if comm.comm_world.rank == 0:
            timer.report_clear("Calibrate")

    # Optionally subtract the dipole

    do_dipole = (args.dipole or args.solsys_dipole or args.orbital_dipole)

    if do_dipole:
        if args.dipole:
            dipomode = 'total'
        elif args.solsys_dipole:
            dipomode = 'solsys'
        else:
            dipomode = 'orbital'

        dipo = tp.OpDipolePlanck(args.freq, mode=dipomode, output='dipole',
                                 keep_quats=True)
        dipo.exec(data)

        comm.comm_world.barrier()
        if comm.comm_world.rank == 0:
            timer.report_clear("Dipole")

        subtractor = tp.OpCacheMath(in1=tod_name, in2='dipole',
                                    subtract=True, out=tod_name)
        if comm.comm_world.rank == 0:
            print('Subtracting dipole', flush=True)
        subtractor.exec(data)

        comm.comm_world.barrier()
        if comm.comm_world.rank == 0:
            timer.report_clear("Dipole subtraction")

    # Optionally filter the signal

    apply_filter(args, data)
    timer.clear()

    # Estimate noise

    noise_estimator = tp.OpNoiseEstim(
        signal=tod_name, flags=flags_name,
        detmask=args.flagmask, commonmask=args.obtmask, maskfile=args.skymask,
        mapfile=args.skymap, out=args.out, rimo=rimo, pol=args.skypol,
        nbin_psd=args.nbin_psd, lagmax=args.lagmax,
        stationary_period=args.stationary_period, nosingle=args.nosingle,
        no_spin_harmonics=args.no_spin_harmonics,
        calibrate_signal_estimate=args.calibrate_signal_estimate)

    noise_estimator.exec(data)

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Noise estimation")

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


if __name__ == '__main__':
    try:
        mpiworld, procs, rank, comm = get_comm()
        main()
    except Exception as e:
        print('Exception occurred: "{}"'.format(e), flush=True)
        if procs == 1:
            raise
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
        mpiworld.Abort()
