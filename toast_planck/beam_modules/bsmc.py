# Copyright (c) 2016-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# MAIN CALLS
# -- IMPORTS -- ------------------------------------------------------
import numpy as np

from . import bsutil


# ------------- ------------------------------------------------------
# -- BSDEC -- --------------------------------------------------------
def bsdec(bpar, cache):
    freq, freqi, bolo, bolonum = bsutil.id2fbn(bpar['boloID'])
    knots, knostep, extsplinenum = bsutil.createkgrid(bpar, freqi)
    x, y, z = bsutil.smpsel(bpar, cache, knots, freq)
    bscoeff, arcsamp = bsutil.bscoefeval(bpar, x, y, z,
                                         knots, extsplinenum)
    xs, ys, arc = bsutil.sqcoords(bpar, knots, extsplinenum)
    return bscoeff


# ----------- --------------------------------------------------------
# -- TFBS -- ---------------------------------------------------------
def tfbs(bpar, cache):
    freq, freqi, bolo, bolonum = bsutil.id2fbn(bpar['boloID'])
    knx, kny, kstpx, kstpy, xsplx, xsply = bsutil.createtfkgrid(bpar, freqi)
    x, y, z = bsutil.smpsel(bpar, cache, knx, freq, knotsy=kny)
    bscoeff, arc = bsutil.bscoefeval(bpar, x, y, z, knx, xsplx,
                                     knotsy=kny, extsplinenumy=xsply)
    xs, ys, arc = bsutil.sqcoords(bpar, knx, xsplx,
                                  knotsy=kny, extsplinenumy=xsply)
    return bscoeff


# ---------- ---------------------------------------------------------
# -- HYBRID -- -------------------------------------------------------
def hybrid(bpar, bscoeffs, tfcoeffs, cache):
    freq, freqi, bolo, bolonum = bsutil.id2fbn(bpar['boloID'])
    knx, kny, kstpx, kstpy, xsplx, xsply = bsutil.createtfkgrid(bpar, freqi)
    tfxs, tfys, tfarc = bsutil.sqcoords(bpar, knx, xsplx,
                                        knotsy=kny, extsplinenumy=xsply)
    poltfxs, poltfys, poltfarc = bsutil.sqcoords(
        bpar, knx, xsplx, knotsy=kny, extsplinenumy=xsply, polar=True)
    knots, knostep, extsplinenum = bsutil.createkgrid(bpar, freqi)
    mbxs, mbys, mbarc = bsutil.sqcoords(bpar, knots, extsplinenum)
    polmbxs, polmbys, polmbarc = bsutil.sqcoords(bpar, knots, extsplinenum,
                                                 polar=True)
    mainbeam = np.mat(mbarc).T * np.mat(bscoeffs)
    polmb = np.mat(polmbarc).T * np.mat(bscoeffs)
    tfres = np.mat(tfarc).T * np.mat(tfcoeffs)
    poltfres = np.mat(poltfarc).T * np.mat(tfcoeffs)
    poltmp, r, theta = bsutil.poltmp(bpar, freqi, polmb, polmbxs, polmbys)
    hmap, xhmap, yhmap = bsutil.hmapfill(bpar, mainbeam, mbxs, mbys,
                                         tfres, tfxs, tfys,
                                         poltmp, r, theta,
                                         freq, freqi)
    polhmap, polxhmap, polyhmap = bsutil.hmapfill(bpar, polmb, polmbxs, polmbys,
                                                  poltfres, poltfxs, poltfys,
                                                  poltmp, r, theta,
                                                  freq, freqi, polar=True)
    hbeam = {}
    hbeam['SQUARE_X'] = xhmap
    hbeam['SQUARE_Y'] = yhmap
    hbeam['SQUARE'] = hmap
    hbeam['POLAR_X'] = polxhmap
    hbeam['POLAR_Y'] = polyhmap
    hbeam['POLAR'] = polhmap
    return hbeam


# ------------ -------------------------------------------------------
# -- BSOVERSAMPLING -- -----------------------------------------------
def bsoversampling(bpar, cache):
    raise RuntimeError('bsoversampling is not implemented.')
# -------------------- -----------------------------------------------
