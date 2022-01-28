#!/usr/bin/env python

# Copyright (c) 2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import datetime
import os
import re
import sys
from toast_planck.utilities import to_radiometer
import traceback

import numpy as np
import toast
from toast.mpi import MPI
import toast_planck as tp
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm


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

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.world_rank == 0:
        print("Running with {} processes at {}".format(
                procs, str(datetime.datetime.now())))

    parser = argparse.ArgumentParser(
        description='Accumulate polarization moments',
        fromfile_prefix_chars='@')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--freq', required=True, type=int, help='Frequency')
    parser.add_argument('--nside', required=False, type=int,
                        default=512, help='Map resolution')
    parser.add_argument('--smax', required=False, type=int,
                        default=6, help='Highest moment')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true',
                        help='Write data distribution info to file')
    parser.add_argument('--dets', required=False, default=None,
                        help='Detector list (comma separated)')
    parser.add_argument('--effdir', required=True,
                        help='Input Exchange Format File directory')
    parser.add_argument('--effdir_in_diode0', required=False, default=None,
                        help='Input Exchange Format File directory, '
                        'LFI diode 0')
    parser.add_argument('--effdir_in_diode1', required=False, default=None,
                        help='Input Exchange Format File directory, '
                        'LFI diode 1')
    parser.add_argument('--effdir_pntg', required=False,
                        help='Input Exchange Format File directory '
                        'for pointing')
    parser.add_argument('--obtmask', required=False, default=1, type=int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=1, type=int,
                        help='Quality flag mask')
    parser.add_argument('--pntflagmask', required=False, default=0, type=int,
                        help='Pointing flag mask')
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
    parser.add_argument('--prefix', required=False, default='spins',
                        help='map prefix')

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if comm.world_rank == 0:
        print('All parameters:')
        print(args, flush=True)

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

    tods = []

    for obtrange, ringrange, odrange in zip(obtranges, ringranges, odranges):
        tods.append(tp.Exchange(
            comm=comm.comm_group, detectors=detectors, ringdb=args.ringdb,
            effdir_in=args.effdir, effdir_in_diode0=args.effdir_in_diode0,
            effdir_in_diode1=args.effdir_in_diode1,
            effdir_pntg=args.effdir_pntg, obt_range=obtrange,
            ring_range=ringrange, od_range=odrange, freq=args.freq,
            RIMO=args.rimo, obtmask=args.obtmask, flagmask=args.flagmask,
            pntflagmask=args.pntflagmask, do_eff_cache=False,
            noisefile='RIMO'))

    rimo = tods[0].rimo

    # Make output directory

    if not os.path.isdir(args.out) and comm.comm_world.rank == 0:
        os.makedirs(args.out)

    # This is the distributed data, consisting of one or
    # more observations, each distributed over a communicator.
    data = toast.Data(comm)

    for iobs, tod in enumerate(tods):
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
    if comm.comm_world.rank == 0:
        timer.report_clear("Metadata queries")

    # Accumulate and save the moment maps
    polmoments = tp.OpPolMomentsPlanck(
        nside=args.nside, RIMO=rimo, margin=0, keep_vel=False,
        keep_pos=False, keep_phase=False, keep_quats=False,
        smax=args.smax, prefix=os.path.join(args.out, args.prefix))

    polmoments.exec(data)

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Accumulate moment maps")

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
