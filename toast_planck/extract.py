# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

from collections import OrderedDict
import os
from toast_planck.preproc_modules import Pnt2Planeter, MapSampler

import toast.healpix
import toast.qarray

import astropy.io.fits as pf
import healpy as hp
import numpy as np

PLANETS = ["mars", "jupiter", "saturn", "uranus", "neptune"]


class Target(object):

    def __init__(self, name, lon, lat, radius, info):
        self.name = name
        self.lon = lon
        self.lat = lat
        self.theta = np.radians(90 - self.lat)
        self.phi = np.radians(self.lon)
        self.vec = toast.healpix.ang2vec(
            np.array([self.theta]), np.array([self.phi]))
        self.radius = radius
        self.info = info


XAXIS, YAXIS, ZAXIS = np.eye(3)


# import warnings
# warnings.filterwarnings("error")
class OpExtractPlanck(toast.Operator):
    """
    Extract TOD in the vicinity of point sources and other coordinates
    of interest.

    Args:
        RIMO -- Reduced instrument model, used for noise and position angle
        catalog -- catalog of targets
        radius -- search radius around the target [arc min]
    """

    def __init__(
            self, rimo, catalog, radius, comm, out=".", common_flag_mask=255,
            flag_mask=255, pnt_mask=2, sso_mask=2, maskfile=None, bg=None,
            full_rings=False, recalibrate_bg=False):
        self.comm = comm
        if comm is None:
            self.rank = 0
        else:
            self.rank = self.comm.rank
        self.rimo = rimo
        self.out = out
        self.catalog = catalog
        # parse the catalog
        self.parse_catalog()
        self.radius = radius
        self.common_flag_mask = common_flag_mask
        self.flag_mask = flag_mask
        self.pnt_mask = pnt_mask
        self.sso_mask = sso_mask
        self.masksampler = None
        if maskfile is not None:
            self.masksampler = MapSampler(maskfile, pol=False, comm=self.comm)
        self.mapsampler = None
        if bg is not None:
            self.mapsampler = MapSampler(bg, pol=True, comm=self.comm)
        self.full_rings = full_rings
        self.recalibrate_bg = recalibrate_bg

    def parse_catalog(self):
        if self.rank == 0:
            header = True
            self.targets = OrderedDict()
            for line in open(self.catalog, "r"):
                if line.startswith("#"):
                    continue
                parts = line.split(",")
                if header:
                    self.target_coord = parts[0].strip()
                    self.ntarget = np.int(parts[1])
                    header = False
                else:
                    name = parts[0].strip()
                    lon = np.float(parts[1])
                    lat = np.float(parts[2])
                    radius = np.float(parts[3])
                    info = parts[4].strip()
                    self.targets[name] = Target(name, lon, lat, radius, info)
        else:
            self.target_coord = None
            self.targets = None
        if self.comm is not None:
            self.target_coord = self.comm.bcast(self.target_coord, root=0)
            self.targets = self.comm.bcast(self.targets, root=0)

    def collect_detector_data(
            self, target, det, timevec, signalvec, thetavec, phivec, psivec,
            dthetavec, dphivec, ringnumbervec, pntflagvec, qwvec, uwvec,
            phasevec):
        """Collect and write out data for this detector
        """
        if self.rank == 0:
            print("  gathering {} data".format(det), flush=True)
            cols = []

        column_specs = [
            ("time", timevec, np.float64, "second"),
            ("signal", signalvec, np.float32, "K_CMB"),
            ("theta", thetavec, np.float32, "radians"),
            ("phi", phivec, np.float32, "radians"),
            ("psi", psivec, np.float32, "radians"),
            ("dtheta", dthetavec, np.float32, "arc min"),
            ("dphi", dphivec, np.float32, "arc min"),
            ("ring", ringnumbervec, np.int32, "ring number"),
            ("pntflag", pntflagvec, np.int8, "pointing flag"),
            ("qweight", qwvec, np.float32, "Stokes Q weight"),
            ("uweight", uwvec, np.float32, "Stokes U weight"),
            ("phase", phasevec, np.float32, "radian"),
        ]

        is_sorted = None
        ind = None
        for name, vec, dtype, unit in column_specs:
            if len(vec) == 0:
                vec = np.array([], dtype=dtype)
            if self.comm is not None:
                self.comm.Barrier()
                vec = self.comm.gather(vec)
            else:
                vec = [vec]
            if self.rank != 0:
                continue
            vec = np.hstack(vec).astype(dtype)
            if is_sorted is None:
                # First vector is time
                is_sorted = np.all(np.diff(vec) >= 0)
                if not is_sorted:
                    # The times are not sorted if multiple ring
                    # ranges were given
                    ind = np.argsort(vec)
            if ind is not None:
                vec = vec[ind]
            vec = vec.reshape([1, -1])
            if dtype == np.float32:
                form = "{}E".format(vec.size)
            elif dtype == np.float64:
                form = "{}D".format(vec.size)
            elif dtype == np.int32:
                form = "{}I".format(vec.size)
            elif dtype == np.int8:
                form = "{}B".format(vec.size)
            else:
                raise RuntimeError(
                    "Unknown datatype {}".format(dtype))
            cols.append(
                pf.Column(name=name, format=form, array=vec, unit=unit))
            
        if self.rank == 0 and vec.size > 0:
            hdulist = [pf.PrimaryHDU()]
            hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols))
            hdu.header["extname"] = det
            hdu.header["detector"] = det
            hdu.header["fsample"] = (self.rimo[det].fsample, "sampling rate")
            hdu.header["psi"] = (
                self.rimo[det].psi_uv + self.rimo[det].psi_pol,
                "polarization angle")
            eps = self.rimo[det].epsilon
            eta = (1 - eps) / (1 + eps)
            hdu.header["eta"] = (eta, "polarization efficiency")
            hdu.header["target"] = (target.name, "target name")
            hdu.header["info"] = (target.info, "target info")
            hdu.header["lon"] = (target.lon, "target longitude")
            hdu.header["lat"] = (target.lat, "target latitude")
            hdu.header["coord"] = (self.target_coord, "Coordinate system")
            hdu.header["radius"] = (target.radius + self.radius,
                                    "search radius")
            hdulist.append(hdu)
            print("  gathered {} samples".format(vec.size), flush=True)
            filename = os.path.join(
                self.out, "small_dataset_{}_{}.fits".format(target.name, det))
            pf.HDUList(hdulist).writeto(filename, overwrite=True)
            print("small dataset saved to {}".format(filename), flush=True)
        if self.comm is not None:
            self.comm.barrier()

        return

    def process_ring(
            self, istart, istop, ring_offset, iring, tod, det, planetmode,
            planeter, target, cos_lim, psidet, timevec, signalvec, thetavec,
            phivec, psivec, dthetavec, dphivec, ringnumbervec, pntflagvec,
            qwvec, uwvec, phasevec):
        """ Collect samples that fall within the search radius

        """
        ind = slice(istart, istop)
        ring_number = (np.zeros(istop - istart, dtype=np.int) +
                       ring_offset + iring)
        times = tod.local_times()[ind]
        common_flags = (tod.local_common_flags()[ind] &
                        self.common_flag_mask)
        pnt_flags = (tod.local_common_flags()[ind] &
                     self.pnt_mask)
        phase = tod.local_phase()[ind]
        signal = tod.local_signal(det)[ind]
        flags = tod.local_flags(det)[ind] & self.flag_mask
        flags |= common_flags
        quat = tod.local_pointing(det)[ind]
        iquweights = tod.local_weights(det)[ind]
        # Which samples are within the search radius?
        vec = toast.qarray.rotate(quat, ZAXIS)
        # Check if the TOD coordinate system matches the catalog
        if self.target_coord.upper() != tod.coord.upper():
            coord_matrix = hp.rotator.get_coordconv_matrix(
                [tod.coord, self.target_coord])[0]
            coord_quat = toast.qarray.from_rotmat(coord_matrix)
            vec = toast.qarray.rotate(coord_quat, vec)
        if planetmode:
            dp, planetvec = planeter.cosdist_vec(
                vec.T, times, full_output=True)
        else:
            dp = np.dot(vec, target.vec.T).ravel()
        good = dp > cos_lim
        good[flags != 0] = False
        ngood = np.sum(good)
        if ngood > 0:
            if self.full_rings:
                good = flags == 0
            # Rotate these samples into a frame where the
            # detector looks along the X axis.
            # print("{} hits {} {} times".format(
            #    det, target.name, ngood))
            theta, phi, psi = toast.qarray.to_angles(
                quat[good])
            if self.masksampler is not None:
                not_masked = self.masksampler.at(theta, phi) > .5
                theta = theta[not_masked]
                phi = phi[not_masked]
                psi = psi[not_masked]
                good[good] = not_masked
            bg = 0
            if self.mapsampler is not None:
                bg = self.mapsampler.atpol(theta, phi,
                                           iquweights[good])
                if self.recalibrate_bg:
                    ind_fit = (
                        tod.local_flags(det)[ind][good] & self.sso_mask == 0)
                    templates = np.vstack([
                        np.ones(np.sum(ind_fit)),
                        bg[ind_fit]])
                    invcov = np.dot(templates, templates.T)
                    cov = np.linalg.inv(invcov)
                    proj = np.dot(templates, signal[good])
                    offset, gain = np.dot(cov, proj)
                    bg = gain * bg + offset
            psirot = toast.qarray.rotation(vec[good], -psi + psidet)
            thetarot = toast.qarray.rotation(YAXIS, np.pi / 2 - theta)
            phirot = toast.qarray.rotation(ZAXIS, -phi)
            rot = toast.qarray.mult(
                thetarot, toast.qarray.mult(phirot, psirot))
            if planetmode:
                tvec = toast.qarray.rotate(rot, planetvec[:, good].T)
            else:
                tvec = toast.qarray.rotate(rot, target.vec)
            ttheta, tphi = toast.healpix.vec2ang(tvec)
            thetavec.append(theta)
            phivec.append(phi)
            psivec.append(psi)
            # invert the source position into PSF
            dthetavec.append(ttheta - np.pi / 2)
            dphivec.append(-tphi)
            timevec.append(times[good])
            signalvec.append(signal[good] - bg)
            ringnumbervec.append(ring_number[good])
            pntflagvec.append(pnt_flags[good])
            _, qw, uw = iquweights[good].T
            qwvec.append(qw)
            uwvec.append(uw)
            phasevec.append(phase[good])
        return

    # @profile
    def exec(self, data):
        dets = data.obs[0]["tod"].local_dets
        for target in self.targets.values():
            planetmode = target.name.lower() in PLANETS
            # Collect samples around the target on every process
            if self.rank == 0:
                print("Collecting data for {}. planetmode = {}".format(
                    target.name, planetmode), flush=True)
            if planetmode:
                planeter = Pnt2Planeter(target.name.lower())
            else:
                planeter = None
            # Avoid taking arc cosines. Measure the radius in dot
            # product instead.
            cos_lim = np.cos(np.radians((self.radius + target.radius) / 60))
            for det in dets:
                # We only want the position angle without the extra
                # polarization angle
                psidet = np.radians(
                    self.rimo[det].psi_uv + self.rimo[det].psi_pol - 90)
                timevec = []
                thetavec = []
                phivec = []
                psivec = []
                dthetavec = []
                dphivec = []
                signalvec = []
                ringnumbervec = []
                pntflagvec = []
                qwvec = []
                uwvec = []
                phasevec = []
                for obs in data.obs:
                    tod = obs["tod"]

                    if "intervals" not in obs:
                        raise RuntimeError(
                            "observation must specify intervals")
                    intervals = tod.local_intervals(obs["intervals"])
                    local_starts = [ival.first for ival in intervals]
                    local_stops = [ival.last + 1 for ival in intervals]
                    ring_offset = tod.globalfirst_ring
                    for interval in obs["intervals"]:
                        if interval.last < tod.local_samples[0]:
                            ring_offset += 1

                    for iring, (istart, istop) in enumerate(zip(local_starts,
                                                                local_stops)):
                        self.process_ring(
                            istart, istop, ring_offset, iring, tod, det,
                            planetmode, planeter, target, cos_lim, psidet,
                            timevec, signalvec, thetavec, phivec, psivec,
                            dthetavec, dphivec, ringnumbervec, pntflagvec,
                            qwvec, uwvec, phasevec)
                if len(timevec) > 0:
                    timevec = np.hstack(timevec)
                    signalvec = np.hstack(signalvec)
                    thetavec = np.hstack(thetavec)
                    phivec = np.hstack(phivec)
                    psivec = np.hstack(psivec)
                    dthetavec = np.hstack(dthetavec)
                    dphivec = np.hstack(dphivec)
                    dphivec[dphivec < -np.pi] += 2 * np.pi
                    dphivec[dphivec > np.pi] -= 2 * np.pi
                    dthetavec *= 180 / np.pi * 60.
                    dphivec *= 180 / np.pi * 60.
                    ringnumbervec = np.hstack(ringnumbervec)
                    pntflagvec = np.hstack(pntflagvec)
                    qwvec = np.hstack(qwvec)
                    uwvec = np.hstack(uwvec)
                    phasevec = np.hstack(phasevec)
                self.collect_detector_data(
                    target, det, timevec, signalvec, thetavec,
                    phivec, psivec, dthetavec, dphivec, ringnumbervec,
                    pntflagvec, qwvec, uwvec, phasevec)
            """
            if self.rank != 0:
                continue
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(np.degrees(phivec), 90-np.degrees(thetavec), ".")
            plt.figure()
            plt.plot(np.degrees(dphivec)*60, np.degrees(dthetavec)*60, ".")
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(dphivec, dthetavec, signalvec, c="r", marker=".")
            plt.show()
            import pdb
            pdb.set_trace()
            """
        return
