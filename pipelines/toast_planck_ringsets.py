#!/usr/bin/env python

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import datetime
import os
import re
import sys
from toast_planck.utilities import DEFAULT_PARAMETERS
import traceback

from toast import Comm, Data, distribute_discrete
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm

import numpy as np
import toast.tod as tt
import toast_planck as tp


def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_planck_reduce (total)")

    mpiworld, procs, rank, comm = get_comm()

    if comm.comm_world.rank == 0:
        print("Running with {} processes at {}".format(
            procs, str(datetime.datetime.now())))

    parser = argparse.ArgumentParser(
        description='Planck Ringset making', fromfile_prefix_chars='@')
    parser.add_argument('--rimo', required=True, help='RIMO file')
    parser.add_argument('--freq', required=True, type=int, help='Frequency')
    parser.add_argument('--dets', required=False, default=None,
                        help='Detector list (comma separated)')
    parser.add_argument('--effdir', required=True,
                        help='Input Exchange Format File directory')
    parser.add_argument('--read_eff', dest='read_eff', default=False,
                        action='store_true',
                        help='Read and co-add the signal from effdir')
    parser.add_argument('--effdir_pntg', required=False,
                        help='Input Exchange Format File directory for '
                        'pointing')
    parser.add_argument('--obtmask', required=False, default=1, type=int,
                        help='OBT flag mask')
    parser.add_argument('--flagmask', required=False, default=1, type=int,
                        help='Quality flag mask')
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
    # dipole parameters
    dipo_parameters_group = parser.add_argument_group('dipole_parameters')
    dipo_parameters_group.add_argument(
        '--solsys_speed', dest='solsys_speed', required=False,
        type=float, default=DEFAULT_PARAMETERS["solsys_speed"],
        help='Solar system speed wrt. CMB rest frame in km/s. '
        'Default is Planck 2015 best fit value')
    dipo_parameters_group.add_argument(
        '--solsys_glon', dest='solsys_glon', required=False, type=float,
        default=DEFAULT_PARAMETERS["solsys_glon"],
        help='Solar system velocity direction longitude in degrees')
    dipo_parameters_group.add_argument(
        '--solsys_glat', dest='solsys_glat', required=False, type=float,
        default=DEFAULT_PARAMETERS["solsys_glat"],
        help='Solar system velocity direction latitude in degrees')

    # libconviqt parameters
    parser.add_argument('--lmax', required=False, default=1024, type=int,
                        help='Simulation lmax')
    parser.add_argument('--fwhm', required=False, default=0.0, type=float,
                        help='Sky fwhm [arcmin] to deconvolve')
    parser.add_argument('--beammmax', required=False, default=None,
                        type=int, help='Beam mmax')
    parser.add_argument('--order', required=False, default=11, type=int,
                        help='Iteration order')
    parser.add_argument('--pxx', required=False, default=False,
                        action='store_true',
                        help='Beams are in Pxx frame, not Dxx')
    parser.add_argument('--skyfile', required=False, default=None,
                        help='Path to sky alm files. Tag DETECTOR will be '
                        'replaced with detector name.')
    parser.add_argument('--beamfile', required=False, default=None,
                        help='Path to beam alm files. Tag DETECTOR will be '
                        'replaced with detector name.')
    parser.add_argument('--nopol', dest='nopol', default=False,
                        action='store_true',
                        help='Sky and beam should be treated unpolarized')
    # noise simulation parameters
    parser.add_argument('--add_noise', dest='add_noise', default=False,
                        action='store_true', help='Simulate noise')
    parser.add_argument('--noisefile', required=False, default='RIMO',
                        help='Path to noise PSD files for noise filter. '
                        'Tag DETECTOR will be replaced with detector name.')
    parser.add_argument('--noisefile_simu', required=False, default='RIMO',
                        help='Path to noise PSD files for noise simulation. '
                        'Tag DETECTOR will be replaced with detector name.')
    parser.add_argument('--mc', required=False, default=0, type=int,
                        help='Noise realization')
    # ringset parameters
    parser.add_argument('--nside_ring', required=False, default=128,
                        type=int, help='Ringset resolution')
    parser.add_argument('--ring_root', required=False, default='ringset',
                        help='Root filename for ringsets (setting to empty '
                        'disables ringset output).')

    args = parser.parse_args()

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

    # Make output directory

    if comm.world_rank == 0:
        os.makedirs(args.out, exist_ok=True)

    do_dipole = args.dipole or args.solsys_dipole or args.orbital_dipole
    do_convolve = args.skyfile is not None and args.beamfile is not None
    do_noise = args.add_noise
    if not do_noise and args.noisefile_simu != 'RIMO':
        raise RuntimeError('Did you mean to simulate noise? add_noise = {} '
                           'but noisefile_simu = {}'.format(
                               args.add_noise, args.noisefile_simu))

    if comm.world_rank == 0:
        print('read_eff = {}'.format(args.read_eff))
        print('do_dipole = {}'.format(do_dipole))
        print('solsys_speed = {}'.format(args.solsys_speed))
        print('solsys_glon = {}'.format(args.solsys_glon))
        print('solsys_glat = {}'.format(args.solsys_glat))
        print('do_convolve = {}'.format(do_convolve))
        print('do_noise = {}'.format(do_noise), flush=True)

    # create the TOD for the whole data span (loads ring database and caches
    # directory contents)

    if do_noise and args.noisefile_simu != 'RIMO':
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
            do_eff_cache=do_eff_cache))

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Metadata queries")

    if args.noisefile != 'RIMO' or args.noisefile_simu != 'RIMO':
        # We split MPI_COMM_WORLD into single process groups, each of
        # which is assigned one or more observations (rings)
        comm = Comm(groupsize=1)

    # This is the distributed data, consisting of one or
    # more observations, each distributed over a communicator.
    data = Data(comm)

    for tod in tods:
        if args.noisefile != 'RIMO' or args.noisefile_simu != 'RIMO':
            # Use a toast helper method to optimally distribute rings between
            # processes.
            dist = distribute_discrete(tod.ringsizes, procs)
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
            # This is the distributed data, consisting of one or
            # more observations, each distributed over a communicator.
            ob = {}
            ob['name'] = 'mission'
            ob['id'] = 0
            ob['tod'] = tod
            ob['intervals'] = tod.valid_intervals
            ob['baselines'] = None
            ob['noise'] = tod.noise

            data.obs.append(ob)

    rimo = tods[0].rimo
    fsample = rimo[detectors[0]].fsample

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Create observations")

    # make a planck Healpix pointing matrix
    mode = 'IQU'

    pointing = tp.OpPointingPlanck(
        nside=args.nside_ring, mode=mode, RIMO=rimo, margin=0,
        apply_flags=False, keep_vel=do_dipole, keep_pos=False,
        keep_phase=False, keep_quats=(do_dipole or do_convolve))

    pointing.exec(data)

    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Pointing Matrix")

    # Always read the signal because we always need the flags

    reader = tp.OpInputPlanck(signal_name='tod')
    if comm.world_rank == 0:
        print('Reading input signal from {}'.format(args.effdir),
              flush=True)
    reader.exec(data)
    comm.comm_world.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Read")
    tod_name = 'tod'

    # Clear the signal if we don't need it

    if not args.read_eff:
        eraser = tp.OpCacheMath(in1='tod', in2=0, multiply=True, out='tod')
        if comm.comm_world.rank == 0:
            print('Erasing TOD', flush=True)
        eraser.exec(data)
        comm.comm_world.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Erase")

    if do_convolve:
        # simulate the TOD by convolving the sky with the beams
        detectordata = []
        for det in tod.detectors:
            skyfile = args.skyfile.replace('DETECTOR', det)
            beamfile = args.beamfile.replace('DETECTOR', det)
            epsilon = rimo[det].epsilon
            # Getting the right polarization angle can be a sensitive matter.
            # Dxx beams are always defined without psi_uv or psi_pol rotation
            # but some Pxx beams may require psi_pol to be removed and psi_uv
            # left in.
            if args.pxx:
                # Beam is in the polarization basis.
                # No extra rotations are needed
                psipol = np.radians(rimo[det].psi_pol)
            else:
                # Beam is in the detector basis. Convolver needs to remove
                # the last rotation into the polarization sensitive frame.
                psipol = np.radians(rimo[det].psi_uv + rimo[det].psi_pol)
            detectordata.append((det, skyfile, beamfile, epsilon, psipol))

        # always construct conviqt with dxx=True and modify the psipol
        # to produce the desired rotation.

        conviqt = tt.OpSimConviqt(
            args.lmax, args.beammmax, detectordata, pol=(not args.nopol),
            fwhm=args.fwhm, order=args.order, calibrate=True, dxx=True,
            out='tod', quat_name='quats', apply_flags=False)
        conviqt.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Convolution")
        tod_name = 'tod'

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
            mode=dipomode, output='tod', add_to_existing=args.read_eff)
        dipo.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Dipole")
        tod_name = 'tod'

    if do_noise:
        nse = tt.OpSimNoise(out='tod', realization=args.mc, component=0,
                            noise='noise_simu', rate=fsample)
        nse.exec(data)
        if mpiworld is not None:
            mpiworld.barrier()
        if comm.world_rank == 0:
            timer.report_clear("Noise simulation")
        tod_name = 'tod'

    # for now, we pass in the noise weights from the RIMO.
    detweights = {}
    for d in tod.detectors:
        net = tod.rimo[d].net
        fsample = tod.rimo[d].fsample
        detweights[d] = 1.0 / (fsample * net * net)

    # Make rings

    ringmaker = tp.OpRingMaker(
        args.nside_ring, args.nside_ring, signal=tod_name,
        fileroot=args.ring_root, out=args.out, detmask=args.flagmask,
        commonmask=args.obtmask)
    ringmaker.exec(data)
    if mpiworld is not None:
        mpiworld.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Ringmaking")

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
        raise e
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
