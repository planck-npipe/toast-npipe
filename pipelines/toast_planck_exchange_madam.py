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
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm

import toast_planck as tp
from toast_planck.utilities import to_radiometer


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

    memreport("at beginning of main", mpiworld)

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.world_rank == 0:
        print("Running with {} processes at {}".format(
                procs, str(datetime.datetime.now())))

    parser = argparse.ArgumentParser(description='Simple MADAM Mapmaking',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--freq', required=True, type=int, help='Frequency')
    parser.add_argument('--nside', required=False, type=int,
                        default=512, help='Map resolution')
    parser.add_argument('--nside_cross', required=False, type=int,
                        default=512, help='Destriping resolution')
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
    parser.add_argument('--effdir_out', required=False,
                        help='Output directory for destriped TOD')
    parser.add_argument('--effdir_out_diode0', required=False,
                        help='Output directory for destriped TOD, LFI diode 0')
    parser.add_argument('--effdir_out_diode1', required=False,
                        help='Output directory for destriped TOD, LFI diode 1')
    parser.add_argument('--obtmask', required=False, default=1, type=int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=1, type=int,
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
    parser.add_argument('--madampar', required=False, default=None,
                        help='Madam parameter file')
    parser.add_argument('--out', required=False, default='.',
                        help='Output directory')
    parser.add_argument('--madam_prefix', required=False, help='map prefix')
    parser.add_argument('--split_mask', required=False, default=None,
                        help='Intensity mask, non-zero pixels are not split.')
    parser.add_argument('--save_leakage_matrices', dest='save_leakage_matrices',
                        default=False, action='store_true',
                        help='Compile and write out the leakage projection '
                        'matrices.')
    # noise parameters
    parser.add_argument('--noisefile', required=False, default='RIMO',
                        help='Path to noise PSD files for noise filter. '
                        'Tag DETECTOR will be replaced with detector name.')
    parser.add_argument('--static_noise', dest='static_noise',
                        required=False, default=False, action='store_true',
                        help='Assume constant noise PSD')
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

    if args.noisefile != 'RIMO' and not args.static_noise:
        do_eff_cache = True
    else:
        do_eff_cache = False

    tods = []

    if args.static_noise:
        noisefile = args.noisefile
    else:
        noisefile = 'RIMO'

    for obtrange, ringrange, odrange in zip(obtranges, ringranges, odranges):
        tods.append(tp.Exchange(
            comm=comm.comm_group, detectors=detectors, ringdb=args.ringdb,
            effdir_in=args.effdir, effdir_in_diode0=args.effdir_in_diode0,
            effdir_in_diode1=args.effdir_in_diode1,
            effdir_pntg=args.effdir_pntg, obt_range=obtrange,
            ring_range=ringrange, od_range=odrange, freq=args.freq,
            RIMO=args.rimo, obtmask=args.obtmask, flagmask=args.flagmask,
            pntflagmask=args.pntflagmask, do_eff_cache=do_eff_cache,
            noisefile=noisefile))

    rimo = tods[0].rimo

    # Make output directory

    if not os.path.isdir(args.out) and comm.comm_world.rank == 0:
        os.makedirs(args.out)

    # Read in madam parameter file
    # Allow more than one entry, gather into a list
    repeated_keys = ['detset', 'detset_nopol', 'survey']
    pars = {}

    if comm.comm_world.rank == 0:
        pars['kfirst'] = False
        pars['temperature_only'] = True
        pars['base_first'] = 60.0
        pars['nside_map'] = args.nside
        pars['nside_cross'] = min(args.nside, args.nside_cross)
        pars['nside_submap'] = 16
        pars['write_map'] = False
        pars['write_binmap'] = True
        pars['write_matrix'] = False
        pars['write_wcov'] = False
        pars['write_hits'] = True
        pars['kfilter'] = False
        pars['info'] = 3
        pars['pixlim_map'] = 1e-3
        pars['pixlim_cross'] = 1e-3
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

        if args.save_leakage_matrices:
            pars['write_leakmatrix'] = True

    pars = comm.comm_world.bcast(pars, root=0)

    if args.noisefile != 'RIMO':
        # We split MPI_COMM_WORLD into single process groups, each of
        # which is assigned one or more observations (rings)
        comm = toast.Comm(groupsize=1)

    # This is the distributed data, consisting of one or
    # more observations, each distributed over a communicator.
    data = toast.Data(comm)

    for iobs, tod in enumerate(tods):
        if args.noisefile != 'RIMO' and not args.static_noise:
            # Use a toast helper method to optimally distribute rings between
            # processes.
            dist = toast.distribute_discrete(tod.ringsizes, comm.world_size)
            my_first_ring, my_n_ring = dist[comm.comm_world.rank]

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

    comm.comm_world.barrier()
    timer.stop()
    if comm.comm_world.rank == 0:
        timer.report("Metadata queries")

    if args.effdir_out is not None or (
        args.effdir_out_diode0 is not None
            and args.effdir_out_diode1 is not None):
        do_output = True
    else:
        do_output = False

    # Read in the signal

    timer.clear()
    timer.start()
    reader = tp.OpInputPlanck(signal_name='signal', flags_name='flags')
    if comm.comm_world.rank == 0:
        print('Reading input signal from {}'.format(args.effdir),
              flush=True)
    reader.exec(data)
    comm.comm_world.barrier()
    timer.stop()
    if comm.comm_world.rank == 0:
        timer.report("Read")
    tod_name = 'signal'
    flags_name = 'flags'

    # Optionally filter the signal

    apply_filter(args, data)

    # Optionally flag bad intervals

    if args.bad_intervals is not None:
        timer = Timer()
        timer.start()
        flagger = tp.OpBadIntervals(path=args.bad_intervals)
        flagger.exec(data)
        timer.stop()
        if comm.comm_world.rank == 0:
            timer.report("Apply {}".format(args.bad_intervals))

    # make a planck Healpix pointing matrix
    timer.clear()
    timer.start()
    mode = 'IQU'
    if pars['temperature_only'] == 'T':
        mode = 'I'
    nside = int(pars['nside_map'])
    pointing = tp.OpPointingPlanck(
        nside=nside, mode=mode, RIMO=rimo,
        margin=0, apply_flags=(not do_output), keep_vel=False,
        keep_pos=False, keep_phase=False, keep_quats=False)

    pointing.exec(data)

    comm.comm_world.barrier()
    timer.stop()
    if comm.comm_world.rank == 0:
        timer.report("Pointing Matrix, mode = {}".format(mode))

    for obs in data.obs:
        obs['tod'].purge_eff_cache()

    # for now, we pass in the noise weights from the RIMO.
    detweights = {}
    for d in tod.detectors:
        if d[-1] in '01' and d[-2] != '-':
            det = to_radiometer(d)
        else:
            det = d
        net = tod.rimo[det].net
        fsample = tod.rimo[det].fsample
        detweights[d] = 1.0 / (fsample * net * net)

    if do_output:
        name_out = 'madam_tod'
    else:
        name_out = None

    timer.clear()
    timer.start()
    try:
        madam = toast.todmap.OpMadam(
            name=tod_name, flag_name=flags_name, apply_flags=do_output,
            params=pars, detweights=detweights, purge=True, name_out=name_out,
            translate_timestamps=False)
    except Exception as e:
        raise Exception('{:4} : ERROR: failed to initialize Madam: {}'.format(
                comm.comm_world.rank, e))
    madam.exec(data)

    comm.comm_world.barrier()
    timer.stop()
    if comm.comm_world.rank == 0:
        timer.report("Madam")

    if do_output:
        timer = Timer()
        timer.start()
        writer = tp.OpOutputPlanck(
            signal_name='madam_tod', flags_name=None, commonflags_name=None,
            effdir_out=args.effdir_out,
            effdir_out_diode0=args.effdir_out_diode0,
            effdir_out_diode1=args.effdir_out_diode1)

        writer.exec(data)

        comm.comm_world.barrier()
        timer.stop()
        if comm.comm_world.rank == 0:
            timer.report("Madam output")

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
