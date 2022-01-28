#!/usr/bin/env python

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re

import argparse
import toast
from toast.mpi import MPI

import numpy as np
import toast_planck as tp

parser = argparse.ArgumentParser(description='Find pixel hit times',
                                 fromfile_prefix_chars='@')
parser.add_argument('--rimo', required=True, help='RIMO file')
parser.add_argument('--freq', required=True, type=int,
                    help='Frequency')
parser.add_argument('--dets', required=False, default=None,
                    help='Detector list (comma separated)')
parser.add_argument('--pixels', required=True, default=None,
                    help='Pixel list (comma separated)')
parser.add_argument('--effdir', required=True,
                    help='Input Exchange Format File directory')
parser.add_argument('--effdir_pntg', required=False,
                    help='Input Exchange Format File directory for pointing')
parser.add_argument('--nside', required=True, type=int,
                    help='Healpix resolution')
parser.add_argument('--obtmask', required=False, default=1, type=int,
                    help='OBT flag mask')
parser.add_argument('--flagmask', required=False, default=1, type=int,
                    help='Quality flag mask')
parser.add_argument('--ringdb', required=True, help='Ring DB file')
parser.add_argument('--odfirst', required=False, default=None, type=int,
                    help='First OD to use')
parser.add_argument('--odlast', required=False, default=None, type=int,
                    help='Last OD to use')
parser.add_argument('--ringfirst', required=False, default=None, type=int,
                    help='First ring to use')
parser.add_argument('--ringlast', required=False, default=None, type=int,
                    help='Last ring to use')
parser.add_argument('--obtfirst', required=False, default=None, type=float,
                    help='First OBT to use')
parser.add_argument('--obtlast', required=False, default=None, type=float,
                    help='Last OBT to use')

args = parser.parse_args()

start = MPI.Wtime()

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

if args.pixels is not None:
    target_pixels = np.array(re.split(',', args.pixels)).astype(int)

# This is the 2-level toast communicator.  By default,
# there is just one group which spans MPI_COMM_WORLD.
comm = toast.Comm()

# This is the distributed data, consisting of one or
# more observations, each distributed over a communicator.
data = toast.Data(comm)

# Make output directory

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

# normally we would get the intervals from somewhere else, but since
# the Exchange TOD already had to get that information, we can
# get it from there.

ob = {}
ob['name'] = 'mission'
ob['id'] = 0
ob['tod'] = tod
ob['intervals'] = tod.valid_intervals
ob['baselines'] = None
ob['noise'] = tod.noise

data.obs.append(ob)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Metadata queries took {:.3f} s".format(elapsed), flush=True)
start = stop

# make a planck Healpix pointing matrix
mode = 'I'
pointing = tp.OpPointingPlanck(
    nside=args.nside, mode=mode, RIMO=rimo, margin=0, apply_flags=False,
    keep_vel=False, keep_pos=False, keep_phase=False, keep_quats=False)
pointing.exec(data)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Pointing Matrix took {:.3f} s, mode = {}".format(elapsed, mode),
          flush=True)
start = stop

nsamp = tod.local_samples[1]

intervals = None
if 'intervals' in data.obs[0].keys():
    intervals = data.obs[0]['intervals']
if intervals is None:
    intervals = [Interval(start=0.0, stop=0.0, first=0, last=(tod.total_samples - 1))]

intervals = tod.local_intervals()
local_starts = [ival.first for ival in intervals]
local_stops = [ival.last + 1 for ival in intervals]

ring_offset = tod.globalfirst_ring
for interval in intervals:
    if interval.last < tod.local_samples[0]:
        ring_offset += 1

timestamps = tod.read_times()

# Galactic center at Nside1024 = pixel 4631180 (Nested order) OD 125
# Jupiter crossing on Survey1 with 100-1a: 9028180 OD 168
# Galactic point source near Crab: 6719438 OD 132

for det in detectors:
    detflags, commonflags = tod.read_flags(detector=det)

    flags = np.logical_or(
        (detflags & args.flagmask) != 0, (commonflags & args.obtmask) != 0)

    pixelsname = "{}_{}".format('pixels', det)
    pixels = tod.cache.reference(pixelsname)

    ring_number = ring_offset - 1

    for iring, (ring_start, ring_stop) in enumerate(zip(local_starts, local_stops)):

        ring_number += 1

        startsample = ring_start + tod.globalfirst + tod.local_samples[0]
        stopsample = startsample + (ring_stop - ring_start)

        ind = slice(ring_start, ring_stop)

        pix = pixels[ind]
        tme = timestamps[ind]
        flg = flags[ind]

        for p in target_pixels:

            hits = pixels[ind] == p

            flagged_hits = hits.copy()
            flagged_hits[ flg == 0 ] = False

            nhit = np.sum(hits)
            nhit_flag = np.sum(flagged_hits)

            if nhit > 0:
                print('Detector {:8} hits pixel {:8} on ring {:5} ({:12.1f} - {:12.1f}) {:5} times. {:5} of the hits are flagged.'.format(
                        det, p, ring_number, tme[0], tme[-1], nhit, nhit_flag
                        ), flush=True)
