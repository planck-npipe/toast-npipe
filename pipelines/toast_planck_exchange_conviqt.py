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
from toast.todmap import OpMadam, OpSimConviqt


import toast_planck as tp


def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_planck_reduce (total)")

    mpiworld, procs, rank, comm = get_comm()

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        print(
            "Running with {} processes at {}".format(
                procs, str(datetime.datetime.now())
            )
        )

    parser = argparse.ArgumentParser(
        description="Simple on-the-fly signal convolution + MADAM Mapmaking",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--lmax", required=True, type=np.int, help="Simulation lmax")
    parser.add_argument(
        "--fwhm", required=True, type=np.float, help="Sky fwhm [arcmin] to deconvolve"
    )
    parser.add_argument("--beammmax", required=True, type=np.int, help="Beam mmax")
    parser.add_argument("--order", default=11, type=np.int, help="Iteration order")
    parser.add_argument(
        "--pxx",
        required=False,
        default=False,
        action="store_true",
        help="Beams are in Pxx frame, not Dxx",
    )
    parser.add_argument(
        "--normalize",
        required=False,
        default=False,
        action="store_true",
        help="Normalize the beams",
    )
    parser.add_argument(
        "--skyfile",
        required=True,
        help="Path to sky alm files. Tag DETECTOR will be "
        "replaced with detector name.",
    )
    parser.add_argument(
        "--remove_monopole",
        required=False,
        default=False,
        action="store_true",
        help="Remove the sky monopole before convolution",
    )
    parser.add_argument(
        "--remove_dipole",
        required=False,
        default=False,
        action="store_true",
        help="Remove the sky dipole before convolution",
    )
    parser.add_argument(
        "--beamfile",
        required=True,
        help="Path to beam alm files. Tag DETECTOR will be "
        "replaced with detector name.",
    )
    parser.add_argument("--rimo", required=True, help="RIMO file")
    parser.add_argument("--freq", required=True, type=np.int, help="Frequency")
    parser.add_argument(
        "--dets", required=False, default=None, help="Detector list (comma separated)"
    )
    parser.add_argument(
        "--effdir", required=True, help="Input Exchange Format File directory"
    )
    parser.add_argument(
        "--effdir_pntg",
        required=False,
        help="Input Exchange Format File directory " "for pointing",
    )
    parser.add_argument(
        "--effdir_out", required=False, help="Output directory for convolved TOD"
    )
    parser.add_argument(
        "--obtmask", required=False, default=1, type=np.int, help="OBT flag mask"
    )
    parser.add_argument(
        "--flagmask", required=False, default=1, type=np.int, help="Quality flag mask"
    )
    parser.add_argument("--ringdb", required=True, help="Ring DB file")
    parser.add_argument(
        "--odfirst", required=False, default=None, type=np.int, help="First OD to use"
    )
    parser.add_argument(
        "--odlast", required=False, default=None, type=np.int, help="Last OD to use"
    )
    parser.add_argument(
        "--ringfirst",
        required=False,
        default=None,
        type=np.int,
        help="First ring to use",
    )
    parser.add_argument(
        "--ringlast", required=False, default=None, type=np.int, help="Last ring to use"
    )
    parser.add_argument(
        "--obtfirst",
        required=False,
        default=None,
        type=np.float,
        help="First OBT to use",
    )
    parser.add_argument(
        "--obtlast", required=False, default=None, type=np.float, help="Last OBT to use"
    )
    parser.add_argument("--madam_prefix", required=False, help="map prefix")
    parser.add_argument(
        "--madampar", required=False, default=None, help="Madam parameter file"
    )
    parser.add_argument(
        "--obtmask_madam", required=False, type=np.int, help="OBT flag mask for Madam"
    )
    parser.add_argument(
        "--flagmask_madam",
        required=False,
        type=np.int,
        help="Quality flag mask for Madam",
    )
    parser.add_argument(
        "--skip_madam",
        required=False,
        default=False,
        action="store_true",
        help="Do not run Madam on the convolved timelines",
    )
    parser.add_argument("--out", required=False, default=".", help="Output directory")

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

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

    # This is the distributed data, consisting of one or
    # more observations, each distributed over a communicator.
    data = toast.Data(comm)

    # Ensure output directory exists

    if not os.path.isdir(args.out) and comm.comm_world.rank == 0:
        os.makedirs(args.out)

    # Read in madam parameter file

    # Allow more than one entry, gather into a list
    repeated_keys = ["detset", "detset_nopol", "survey"]
    pars = {}

    if comm.comm_world.rank == 0:
        pars["kfirst"] = False
        pars["temperature_only"] = True
        pars["base_first"] = 60.0
        pars["nside_map"] = 512
        pars["nside_cross"] = 512
        pars["nside_submap"] = 16
        pars["write_map"] = False
        pars["write_binmap"] = True
        pars["write_matrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = True
        pars["kfilter"] = False
        pars["info"] = 3
        if args.madampar:
            pat = re.compile(r"\s*(\S+)\s*=\s*(\S+(\s+\S+)*)\s*")
            comment = re.compile(r"^#.*")
            with open(args.madampar, "r") as f:
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
        if "file_root" not in pars:
            pars["file_root"] = "madam"
        if args.madam_prefix is not None:
            pars["file_root"] = args.madam_prefix
        sfreq = "{:03}".format(args.freq)
        if sfreq not in pars["file_root"]:
            pars["file_root"] += "_" + sfreq
        try:
            fsample = {30: 32.51, 44: 46.55, 70: 78.77}[args.freq]
        except Exception:
            fsample = 180.3737
        pars["fsample"] = fsample
        pars["path_output"] = args.out

        print("All parameters:")
        print(args, flush=True)

    pars = comm.comm_world.bcast(pars, root=0)

    memreport("after parameters", MPI.COMM_WORLD)

    # madam only supports a single observation.  Normally
    # we would have multiple observations with some subset
    # assigned to each process group.

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

    # Add the bare minimum focal plane information for the conviqt operator
    focalplane = {}
    for det in tod.detectors:
        if args.pxx:
            # Beam is in the polarization basis.
            # No extra rotations are needed
            psipol = tod.rimo[det].psi_pol
        else:
            # Beam is in the detector basis. Convolver needs to remove
            # the last rotation into the polarization sensitive frame.
            psipol = tod.rimo[det].psi_uv + tod.rimo[det].psi_pol
        focalplane[det] = {
            "pol_leakage" : tod.rimo[det].epsilon,
            "pol_angle_deg" : psipol,
        }
    ob["focalplane"] = focalplane

    data.obs.append(ob)

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Metadata queries")

    loader = tp.OpInputPlanck(
        commonflags_name="common_flags", flags_name="flags", margin=0
    )

    loader.exec(data)

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Data read and cache")
        tod.cache.report()

    memreport("after loading", mpiworld)

    # make a planck Healpix pointing matrix
    mode = "IQU"
    if pars["temperature_only"] == "T":
        mode = "I"
    nside = int(pars["nside_map"])
    pointing = tp.OpPointingPlanck(
        nside=nside,
        mode=mode,
        RIMO=tod.RIMO,
        margin=0,
        apply_flags=False,
        keep_vel=False,
        keep_pos=False,
        keep_phase=False,
        keep_quats=True,
    )
    pointing.exec(data)

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Pointing Matrix took, mode = {}".format(mode))

    memreport("after pointing", mpiworld)

    # simulate the TOD by convolving the sky with the beams

    if comm.comm_world.rank == 0:
        print("Convolving TOD", flush=True)

    for pattern in args.beamfile.split(","):
        skyfiles = {}
        beamfiles = {}
        for det in tod.detectors:
            freq = "{:03}".format(tp.utilities.det2freq(det))
            if "LFI" in det:
                psmdet = "{}_{}".format(freq, det[3:])
                if det.endswith("M"):
                    arm = "y"
                else:
                    arm = "x"
                graspdet = "{}_{}_{}".format(freq[1:], det[3:5], arm)
            else:
                psmdet = det.replace("-", "_")
                graspdet = det
            skyfile = (
                args.skyfile.replace("FREQ", freq)
                .replace("PSMDETECTOR", psmdet)
                .replace("DETECTOR", det)
            )
            skyfiles[det] = skyfile
            beamfile = pattern.replace("GRASPDETECTOR", graspdet).replace(
                "DETECTOR", det
            )
            beamfiles[det] = beamfile
            if comm.comm_world.rank == 0:
                print("Convolving {} with {}".format(skyfile, beamfile), flush=True)

        conviqt = OpSimConviqt(
            comm.comm_world,
            skyfiles,
            beamfiles,
            lmax=args.lmax,
            beammmax=args.beammmax,
            pol=True,
            fwhm=args.fwhm,
            order=args.order,
            calibrate=True,
            dxx=True,
            out="conviqt_tod",
            apply_flags=False,
            remove_monopole=args.remove_monopole,
            remove_dipole=args.remove_dipole,
            verbosity=1,
            normalize_beam=args.normalize,
        )
        conviqt.exec(data)

    comm.comm_world.barrier()
    if comm.comm_world.rank == 0:
        timer.report_clear("Convolution")

    memreport("after conviqt", mpiworld)

    if args.effdir_out is not None:
        if comm.comm_world.rank == 0:
            print("Writing TOD", flush=True)

        tod.set_effdir_out(args.effdir_out, None)
        writer = tp.OpOutputPlanck(
            signal_name="conviqt_tod",
            flags_name="flags",
            commonflags_name="common_flags",
        )
        writer.exec(data)

        comm.comm_world.barrier()
        if comm.comm_world.rank == 0:
            timer.report_clear("Conviqt output")

        memreport("after writing", mpiworld)

    # for now, we pass in the noise weights from the RIMO.
    detweights = {}
    for d in tod.detectors:
        net = tod.rimo[d].net
        fsample = tod.rimo[d].fsample
        detweights[d] = 1.0 / (fsample * net * net)

    if not args.skip_madam:
        if comm.comm_world.rank == 0:
            print("Calling Madam", flush=True)

        try:
            if args.obtmask_madam is None:
                obtmask = args.obtmask
            else:
                obtmask = args.obtmask_madam
            if args.flagmask_madam is None:
                flagmask = args.flagmask
            else:
                flagmask = args.flagmask_madam
            madam = OpMadam(
                params=pars,
                detweights=detweights,
                name="conviqt_tod",
                flag_name="flags",
                purge=True,
                name_out="madam_tod",
                common_flag_mask=obtmask,
                flag_mask=flagmask,
            )
        except Exception as e:
            raise Exception(
                "{:4} : ERROR: failed to initialize Madam: {}".format(
                    comm.comm_world.rank, e
                )
            )
        madam.exec(data)

        comm.comm_world.barrier()
        if comm.comm_world.rank == 0:
            timer.report_clear("Madam took {:.3f} s")

        memreport("after madam", mpiworld)

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
        if MPI.COMM_WORLD.size == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("*** print_exception:")
        traceback.print_exception(
            exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout
        )
        print("*** print_exc:")
        traceback.print_exc()
        print("*** format_exc, first and last line:")
        formatted_lines = traceback.format_exc().splitlines()
        print(formatted_lines[0])
        print(formatted_lines[-1])
        print("*** format_exception:")
        print(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        print("*** extract_tb:")
        print(repr(traceback.extract_tb(exc_traceback)))
        print("*** format_tb:")
        print(repr(traceback.format_tb(exc_traceback)))
        print("*** tb_lineno:", exc_traceback.tb_lineno, flush=True)
        MPI.COMM_WORLD.Abort()
