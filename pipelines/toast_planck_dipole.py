#!/usr/bin/env python

# Simple pipeline to dump the estimated dipole TOD

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import datetime
import os
import re
import sys
import traceback

from toast import Comm, Data, distribute_discrete, qarray
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

    parser = argparse.ArgumentParser(description='Simple dipole pipeline',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--freq', required=True, type=np.int, help='Frequency')
    parser.add_argument('--dets', required=False, default=None,
                        help='Detector list (comma separated)')
    parser.add_argument('--effdir', required=True,
                        help='Input Exchange Format File directory')
    parser.add_argument('--effdir_pntg', required=False,
                        help='Input Exchange Format File directory for '
                        'pointing')
    parser.add_argument('--obtmask', required=False, default=1, type=np.int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=1, type=np.int,
                        help='Quality flag mask')
    parser.add_argument('--pntflagmask', required=False, default=0, type=np.int,
                        help='Which OBT flag bits to raise for HCM maneuvers')
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
    parser.add_argument('--out', required=False, default='.',
                        help='Output directory')
    # Dipole parameters
    dipogroup = parser.add_mutually_exclusive_group()
    dipogroup.add_argument(
        '--dipole', dest='dipole', required=False,
        default=False, action='store_true',
        help='Simulate dipole')
    dipogroup.add_argument(
        '--solsys-dipole',
        required=False, default=False, action='store_true',
        help='Simulate solar system dipole')
    dipogroup.add_argument(
        '--orbital-dipole',
        required=False, default=False, action='store_true',
        help='Simulate orbital dipole')
    dipo_parameters_group = parser.add_argument_group('dipole_parameters')
    dipo_parameters_group.add_argument(
        '--solsys_speed', required=False, type=np.float,
        default=DEFAULT_PARAMETERS["solsys_speed"],
        help='Solar system speed wrt. CMB rest frame in km/s. Default is '
        'Planck 2015 best fit value')
    dipo_parameters_group.add_argument(
        '--solsys-glon', required=False, type=np.float,
        default=DEFAULT_PARAMETERS["solsys_glon"],
        help='Solar system velocity direction longitude in degrees')
    dipo_parameters_group.add_argument(
        '--solsys-glat', required=False, type=np.float,
        default=DEFAULT_PARAMETERS["solsys_glat"],
        help='Solar system velocity direction latitude in degrees')

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if comm.world_rank == 0:
        print('All parameters:')
        print(args, flush=True)

    timer = Timer()
    timer.start()

    do_dipole = args.dipole or args.solsys_dipole or args.orbital_dipole
    if not do_dipole:
        raise RuntimeError("You have to set dipole, solsys-dipole or orbital-dipole")

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
        # create the TOD for this observation
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
            do_eff_cache=False))

    # Make output directory

    if not os.path.isdir(args.out) and comm.world_rank == 0:
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

    """
    # Clear the signal

    eraser = tp.OpCacheMath(in1=tod_name, in2=0, multiply=True,
                            out=tod_name)
    if comm.world_rank == 0:
        print('Erasing TOD', flush=True)
    eraser.exec(data)
    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Erase")
    """

    # make a planck Healpix pointing matrix

    mode = 'IQU'
    nside = 512

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
        mode=dipomode, output='dipole', keep_quats=False, npipe_mode=True)
    dipo.exec(data)
    dipo = tp.OpDipolePlanck(
        args.freq, solsys_speed=args.solsys_speed,
        solsys_glon=args.solsys_glon, solsys_glat=args.solsys_glat,
        mode=dipomode, output='dipole4pi', keep_quats=False, npipe_mode=False, lfi_mode=False)
    dipo.exec(data)
    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Dipole")

    # Write out the values in ASCII
    for iobs, obs in enumerate(data.obs):
        tod = obs["tod"]
        times = tod.local_times()
        velocity = tod.local_velocity()
        for det in tod.local_dets:
            quat = tod.local_pointing(det)
            angles = np.vstack(qarray.to_angles(quat)).T
            signal = tod.local_signal(det)
            dipole = tod.local_signal(det, "dipole")
            dipole4pi = tod.local_signal(det, "dipole4pi")
            fname_out = os.path.join(args.out, "{}_dipole.{}.{}.{}.txt".format(dipomode, comm.world_rank, iobs, det))
            with open(fname_out, "w") as fout:
                for t, ang, vel, sig, dipo, dipo4pi in zip(times, angles, velocity, signal, dipole, dipole4pi):
                    fout.write((10 * " {}" + "\n").format(t, *ang, *vel, sig, dipo, dipo4pi))
            print("{} : Wrote {}".format(comm.world_rank, fname_out))

    if comm.world_rank == 0:
        timer.report_clear("Write dipole")

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
