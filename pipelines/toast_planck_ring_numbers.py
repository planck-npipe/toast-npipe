#!/usr/bin/env python

# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import os

from toast import Comm, Data, distribute_discrete
from toast.utils import Logger, Environment, memreport
from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing
from toast.pipeline_tools import get_comm

import toast.tod
import toast.map
import toast_planck as tp
from toast_planck.utilities import to_radiometer, ringdb_table_name

import numpy as np
import re
import argparse

import time
import pickle

parser = argparse.ArgumentParser( description="Build toast/DPC ring number dictionaries.", fromfile_prefix_chars="@" )
parser.add_argument( "--rimo", required=True, help="RIMO file" )
parser.add_argument( "--freq", required=True, type=np.int, help="Frequency" )
parser.add_argument( "--dets", required=False, default=None, help="Detector list (comma separated)" )
parser.add_argument( "--effdir", required=True, help="Input Exchange Format File directory" )
parser.add_argument( "--obtmask", required=False, default=1, type=np.int, help="OBT flag mask" )
parser.add_argument( "--flagmask", required=False, default=1, type=np.int, help="Quality flag mask" )
parser.add_argument( "--ringdb", required=True, help="Ring DB file" )
parser.add_argument( "--odfirst", required=False, default=None, type=np.int, help="First OD to use" )
parser.add_argument( "--odlast", required=False, default=None, type=np.int, help="Last OD to use" )
parser.add_argument( "--ringfirst", required=False, default=None, type=np.int, help="First ring to use" )
parser.add_argument( "--ringlast", required=False, default=None, type=np.int, help="Last ring to use" )
parser.add_argument( "--obtfirst", required=False, default=None, type=np.float, help="First OBT to use" )
parser.add_argument( "--obtlast", required=False, default=None, type=np.float, help="Last OBT to use" )

args = parser.parse_args()

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
    detectors = re.split(",", args.dets)

comm = Comm()

data = Data(comm)

# create the TOD for this observation

tod = tp.Exchange(
    comm=comm.comm_group, 
    detectors=detectors,
    ringdb=args.ringdb,
    effdir_in=args.effdir,
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
ob["name"] = "mission"
ob["id"] = 0
ob["tod"] = tod
ob["intervals"] = tod.valid_intervals
ob["baselines"] = None
ob["noise"] = tod.noise

data.obs.append(ob)

if comm.world_rank == 0:
    timer.report_clear("Metadata queries")

ring_starts = [ t.first for t in tod.valid_intervals ]
ring_times = [ t.start for t in tod.valid_intervals ]
ring_lens = [ (t.last-t.first) for t in tod.valid_intervals ]

ring_offset = tod.globalfirst_ring
for interval in tod.valid_intervals:
    if interval.last  < tod.local_samples[0]:
        ring_offset += 1

ring_number = ring_offset - 1

globalfirst = tod.globalfirst

ringtable = ringdb_table_name(args.freq)
ringdb = tod.ringdb

toast_to_lfi = {}
lfi_to_toast = {}
toast_to_hfi = {}
hfi_to_toast = {}

for iring, (ring_start, ring_time, ring_len) in enumerate(
        zip(ring_starts, ring_times, ring_lens)):

    ring_number += 1

    global_start = globalfirst + ring_start
    global_stop = globalfirst + ring_start + ring_len

    cmd = "select pointID_unique from {} where start_index <= {}" \
        " and stop_index >= {}".format(ringtable, global_stop, global_start)

    acms_ids = ringdb.execute(cmd).fetchall()

    pids = []
    for acms_id in acms_ids:
        pids.append(acms_id[0].split("-")[0])
    pids = set(pids)

    if len(pids) == 0:
        esa_id = -1
        lfi_id = -1
        hfi_id = -1
    else:    
        if len(pids) > 1:
            raise Exception("This query {} corresponds to these PIDs: {}".format(cmd, pids))
        for pid in pids: pass

        esa_id = np.int( pid )

        cmd = "select LFI_ID, HFI_ID from rings where ESA_ID == {}".format(esa_id)

        dpc_ids = ringdb.execute( cmd ).fetchall()

        lfi_id, hfi_id = dpc_ids[0]

    print("TOAST ring_number = {:4} is Planck PID {:12} = LFI ID = {:4} = "
          "HFI ID = {:4}. starts at {}".format(
              ring_number, pid, lfi_id, hfi_id, ring_time), flush=True)

    toast_to_lfi[ring_number] = (lfi_id, ring_time)
    
    toast_to_hfi[ring_number] = (hfi_id, ring_time)
    
    if lfi_id in lfi_to_toast:
        lfi_to_toast[lfi_id].append( ring_number )
    else:
        lfi_to_toast[lfi_id] = [ring_number]
        
    if hfi_id in hfi_to_toast:
        hfi_to_toast[hfi_id].append( ring_number )
    else:
        hfi_to_toast[hfi_id] = [ring_number]

pickle.dump( toast_to_lfi, open( "toast_to_lfi.pck", "wb" ), protocol=2 )
pickle.dump( toast_to_hfi, open( "toast_to_hfi.pck", "wb" ), protocol=2 )
pickle.dump( lfi_to_toast, open( "lfi_to_toast.pck", "wb" ), protocol=2 )
pickle.dump( hfi_to_toast, open( "hfi_to_toast.pck", "wb" ), protocol=2 )

