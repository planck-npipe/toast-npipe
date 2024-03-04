# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from memory_profiler import profile

from collections import OrderedDict
import os

import toast.healpix
import toast.qarray

import astropy.io.fits as pf
import healpy as hp
import numpy as np

XAXIS, YAXIS, ZAXIS = np.eye(3)


# import warnings
# warnings.filterwarnings("error")
class OpToLevelS(toast.Operator):
    """
    Convert the data into a file format that is compatible with the
    Planck Level-S software

    The files will contain tables with column names
    theta, phi, psi and skyLoad

    Args:
        RIMO -- Reduced instrument model, used for noise and position angle
    """

    def __init__(
            self, rimo, comm, out=".", common_flag_mask=255, flag_mask=255):
        self.comm = comm
        if comm is None:
            self.rank = 0
        else:
            self.rank = self.comm.rank
        self.rimo = rimo
        self.out = out
        self.common_flag_mask = common_flag_mask
        self.flag_mask = flag_mask
        self.masksampler = None

    # @profile
    def exec(self, data):
        dets = data.obs[0]["tod"].local_dets
        for det in dets:
            # We only want the position angle without the extra
            # polarization angle
            psidet = np.radians(
                self.rimo[det].psi_uv + self.rimo[det].psi_pol - 90)
            thetavec = []
            phivec = []
            psivec = []
            signalvec = []
            flagvec = []
            for obs in data.obs:
                tod = obs["tod"]
                common_flags = tod.local_common_flags() & self.common_flag_mask
                signal = tod.local_signal(det)
                flags = tod.local_flags(det) & self.flag_mask
                flags |= common_flags
                quat = tod.local_pointing(det)
                theta, phi, psi = toast.qarray.to_angles(quat)
                psi -= psidet  # To position angle

                thetavec.append(theta)
                phivec.append(phi)
                psivec.append(psi)
                signalvec.append(signal)
                flagvec.append(flags)

            if len(signalvec) == 0:
                continue
            
            flagvec = np.hstack(flagvec)
            good = flagvec == 0
            # flagvec will be all zeros.
            # ArtDeco todtobin2 code reads the flags but does not apply them...
            flagvec = flagvec[good]

            signalvec = np.hstack(signalvec)[good]
            thetavec = np.hstack(thetavec)[good]
            phivec = np.hstack(phivec)[good]
            psivec = np.hstack(psivec)[good]


            cols = []
            nwrite = len(signalvec)
            column_specs = [
                ("theta", thetavec, np.float32, "radians"),
                ("phi", phivec, np.float32, "radians"),
                ("psi", psivec, np.float32, "radians"),
                ("skyLoad", signalvec, np.float32, "K_CMB"),
                ("qualityFlag", flagvec, np.int32, "flag"),
            ]

            for name, vec, dtype, unit in column_specs:
                if dtype == np.float32:
                    form = f"{nwrite}E"
                elif dtype == np.float64:
                    form = f"{nwrite}D"
                elif dtype == np.int16:
                    form = f"{nwrite}I"
                elif dtype == np.int32:
                    form = f"{nwrite}J"
                elif dtype == np.uint8:
                    form = f"{nwrite}B"
                else:
                    raise RuntimeError(f"Unknown datatype {dtype}")
                cols.append(pf.Column(
                    name=name, format=form, array=vec.reshape(1, -1), unit=unit
                ))
            
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
            hdulist.append(hdu)
            filename = os.path.join(
                self.out, f"levels_{det}_{self.rank:04}.fits"
            )
            print(f"Writing {nwrite} samples to {filename}", flush=True)
            pf.HDUList(hdulist).writeto(filename, overwrite=True)
            print(f"Wrote {filename}", flush=True)

        return
