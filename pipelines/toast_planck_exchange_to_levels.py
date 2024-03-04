#!/usr/bin/env python

# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import datetime
import os
import re
import sys
import traceback

import numpy as np

from toast import Comm, Data, distribute_discrete
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm

import toast_planck as tp


def create_observations(args, comm):
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

    if args.noisefile != 'RIMO':
        do_eff_cache = True
    else:
        do_eff_cache = False

    tods = []

    for obtrange, ringrange, odrange in zip(obtranges, ringranges, odranges):
        tods.append(tp.Exchange(
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
            pntflagmask=args.pntflagmask,
            do_eff_cache=do_eff_cache,
            coord=args.coord,
        ))

    rimo = tods[0].rimo

    # Make output directory

    if comm.world_rank == 0:
        os.makedirs(args.out, exist_ok=True)

    if args.noisefile != 'RIMO':
        # We split MPI_COMM_WORLD into single process groups, each of
        # which is assigned one or more observations (rings)
        comm = Comm(groupsize=1)

    # This is the distributed data, consisting of one or
    # more observations, each distributed over a communicator.
    data = Data(comm)

    for iobs, tod in enumerate(tods):
        if args.noisefile != 'RIMO':
            # Use a toast helper method to optimally distribute rings between
            # processes.
            dist = distribute_discrete(tod.ringsizes, comm.world_size)
            my_first_ring, my_n_ring = dist[comm.world_rank]

            for my_ring in range(my_first_ring, my_first_ring + my_n_ring):
                ringtod = tp.Exchange.from_tod(
                    tod, my_ring, comm.comm_group, noisefile=args.noisefile)
                ob = {}
                ob['name'] = 'ring{:05}'.format(ringtod.globalfirst_ring)
                ob['id'] = ringtod.globalfirst_ring
                ob['tod'] = ringtod
                ob['intervals'] = ringtod.valid_intervals
                ob['baselines'] = None
                ob['noise'] = ringtod.noise
                data.obs.append(ob)
        else:
            ob = {}
            ob['name'] = 'observation{:04}'.format(iobs)
            ob['id'] = 0
            ob['tod'] = tod
            ob['intervals'] = tod.valid_intervals
            ob['baselines'] = None
            ob['noise'] = tod.noise

            data.obs.append(ob)

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Metadata queries")
    return data


# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)
def main():
    timer0 = Timer()
    timer0.start()

    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_planck_reduce (total)")

    mpiworld, procs, rank, comm = get_comm()
    memreport("At start of pipeline", mpiworld)

    if comm.world_rank == 0:
        print("Running with {} processes at {}".format(
            procs, str(datetime.datetime.now())))

    parser = argparse.ArgumentParser(description='Convert EFF data to Level-S',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--freq', required=True, type=int, help='Frequency')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true',
                        help='Write data distribution info to file')
    parser.add_argument('--dets', required=False, default=None,
                        help='Detector list (comma separated)')
    parser.add_argument('--effdir', required=True,
                        help='Input Exchange Format File directory')
    parser.add_argument('--effdir_pntg', required=False,
                        help='Input Exchange Format File directory '
                        'for pointing')
    parser.add_argument('--coord', default='G',
                        help='Coordinate system, "G", "E" or "C"')
    parser.add_argument('--obtmask', required=False, default=1, type=int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=2, type=int,
                        help='Quality flag mask')
    parser.add_argument('--pntflagmask', required=False, default=0, type=int,
                        help='Pointing flag mask')
    parser.add_argument('--bad_intervals', required=False,
                        help='Path to bad interval file.')
    parser.add_argument('--ringdb', required=True, help='Ring DB file')
    parser.add_argument('--odfirst', required=False, default=None, type=int,
                        help='First OD to use')
    parser.add_argument('--odlast', required=False, default=None, type=int,
                        help='Last OD to use')
    parser.add_argument('--ringfirst', required=False, default=None,
                        help='First ring to use (can be a list)')
    parser.add_argument('--ringlast', required=False, default=None,
                        help='Last ring to use (can be a list)')
    parser.add_argument('--obtfirst', required=False, default=None,
                        type=float, help='First OBT to use')
    parser.add_argument('--obtlast', required=False, default=None,
                        type=float, help='Last OBT to use')
    parser.add_argument('--out', required=False, default='.',
                        help='Output directory')
    # noise parameters
    parser.add_argument('--noisefile', required=False, default='RIMO',
                        help='Path to noise PSD files for noise filter. '
                        'Tag DETECTOR will be replaced with detector name.')
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

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if comm.world_rank == 0:
        print('All parameters:')
        print(args, flush=True)

    data = create_observations(args, comm)
    rimo = data.obs[0]["tod"].rimo

    memreport("After create observations", mpiworld)

    # Read in the signal

    timer = Timer()
    timer.start()

    reader = tp.OpInputPlanck(signal_name='signal', flags_name='flags')
    if comm.world_rank == 0:
        print('Reading input signal from {}'.format(args.effdir),
              flush=True)
    reader.exec(data)
    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Reading")
    tod_name = 'signal'
    flags_name = 'flags'

    memreport("After read", mpiworld)

    # Optionally flag bad intervals

    if args.bad_intervals is not None:
        flagger = tp.OpBadIntervals(path=args.bad_intervals)
        flagger.exec(data)
        if comm.world_rank == 0:
            timer.report_clear("Applying {}".format(args.bad_intervals))

    do_dipole = (args.dipole or args.solsys_dipole or args.orbital_dipole)

    # make a planck Healpix pointing matrix
    pointing = tp.OpPointingPlanck(
        nside=1024, mode='IQU', RIMO=rimo,
        margin=0, apply_flags=False, keep_vel=do_dipole,
        keep_pos=False, keep_phase=True, keep_quats=True)

    pointing.exec(data)

    memreport("After pointing", mpiworld)

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Pointing Matrix")

    # Optionally subtract the dipole

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
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Dipole")
        subtractor = tp.OpCacheMath(in1=tod_name, in2='dipole',
                                    subtract=True, out=tod_name)
        if comm.comm_world.rank == 0:
            print('Subtracting dipole', flush=True)
        subtractor.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Dipole subtraction")

        memreport("After dipole", mpiworld)

    to_levels = tp.OpToLevelS(
        rimo,
        mpiworld,
        common_flag_mask=args.obtmask,
        flag_mask=args.flagmask,
        out=args.out,
    )
    to_levels.exec(data)

    memreport("After Level-S conversion", mpiworld)

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Level-S conversion")

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()
    timer = Timer()
    timer.start()
    alltimers = gather_timers(comm=mpiworld)
    if comm.world_rank == 0:
        out = os.path.join(args.out, "timing")
        dump_timing(alltimers, out)
        timer.report_clear("Gather and dump timing info")
        timer0.report_clear("Full pipeline")
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
