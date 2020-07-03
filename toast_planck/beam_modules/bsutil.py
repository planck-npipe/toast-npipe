# Copyright (c) 2016-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# UTILS
# -- IMPORTS -- ------------------------------------------------------
import numpy as np
import scipy.interpolate as itp
import scipy.linalg as lin
import scipy.optimize as opt


# ------------- ------------------------------------------------------
# -- FREQ 2 INDEX -- -------------------------------------------------
def freq2i(freq):
    switcher = {'100': 0, '143': 1, '217': 2,
                '353': 3, '545': 4, '857': 5}
    return switcher.get(freq, np.nan)


# ------------------ -------------------------------------------------
# -- BOLOID 2 FREQ BOLO NUM -- ---------------------------------------
def id2fbn(boloID):
    name = boloID.split('_')
    bolonum = name[0]
    freq = name[1]
    bolo = '{}-{}'.format(name[1], name[2])
    freqi = freq2i(freq)
    if not np.isfinite(freqi):
        raise RuntimeError(__name__ + ' : UNKNOWN FREQUENCY :' + boloID)
    return freq, freqi, bolo, bolonum


# ---------------------------- ---------------------------------------
# -- CREATE KNOTS GRID -- --------------------------------------------
def createkgrid(bpar, freqi):
    knotextframe = int(bpar['knotextframe'])
    knotstep = np.array(bpar['knotstep'])
    bsorder = int(bpar['bsorder'])
    splnum = 1 + int(knotextframe / float(knotstep[freqi]))
    extsplinenum = splnum + int(bsorder) - 2
    knots = (np.arange(splnum) * float(knotstep[freqi])) \
        - int(knotextframe) / 2.
    knotstep = knots[1] - knots[0]
    knots = np.concatenate((np.zeros(int(bsorder) - 1) + min(knots),
                            knots,
                            np.zeros(int(bsorder) - 1) + max(knots)))
    return knots, knotstep, extsplinenum


# ----------------------- --------------------------------------------
# -- CREATE TF KNOTS GRID -- -----------------------------------------
def createtfkgrid(bpar, freqi):
    knotextframex = int(bpar['knotextframex'])
    knotextframey = int(bpar['knotextframey'])
    knotstep = np.array(bpar['knotstep'])
    bsorder = int(bpar['bsorder'])
    splnumx = 1 + int(knotextframex / float(knotstep[freqi]))
    splnumy = 1 + int(knotextframey / float(knotstep[freqi]))
    extsplinenumx = splnumx + int(bsorder) - 2
    extsplinenumy = splnumy + int(bsorder) - 2
    knotsx = (np.arange(splnumx) *
              float(knotstep[freqi])) - int(knotextframex) / 2.
    knotsy = (np.arange(splnumy) * float(knotstep[freqi]))
    knotstepx = knotsx[1] - knotsx[0]
    knotstepy = knotsy[1] - knotsy[0]
    knotsx = np.concatenate((np.zeros(int(bsorder) - 1) + min(knotsx),
                             knotsx,
                             np.zeros(int(bsorder) - 1) + max(knotsx)))
    knotsy = np.concatenate((np.zeros(int(bsorder) - 1) + min(knotsy),
                             knotsy,
                             np.zeros(int(bsorder) - 1) + max(knotsy)))
    return knotsx, knotsy, knotstepx, knotstepy, extsplinenumx, extsplinenumy


# ----------------------- --------------------------------------------
# -- SAMPLE SELECTION -- ---------------------------------------------
def smpsel(bpar, cache, knotsx, freq, knotsy=None):
    if knotsy is None:
        knotsy = knotsx
    x = []
    y = []
    z = []
    for target in cache:
        xtx = cache[target]['xdxx']
        ytx = cache[target]['ydxx']
        ztx = cache[target]['toi']
        x.extend(list(xtx))
        y.extend(list(ytx))
        z.extend(list(ztx))
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    select = np.logical_and((x >= min(knotsx)), (x <= max(knotsx)))
    x = x[select]
    y = y[select]
    z = z[select]
    if (np.size(x) == 0):
        if bpar['bsverbose']:
            print('!!! OUTSIDE GRID XLIMITS !!!')
    select = np.logical_and((y >= min(knotsy)), (y <= max(knotsy)))
    x = x[select]
    y = y[select]
    z = z[select]
    if (np.size(x) == 0):
        if bpar['bsverbose']:
            print('!!! OUTSIDE GRID YLIMITS !!!')
    return x, y, z


# ---------------------- ---------------------------------------------
# -- BS SPLINE EVALUATION -- -----------------------------------------
def bsplineev(order, knots, ns, xval):
    """
    Computes BSpline basis function of given order, index and knots
    at given locations using De Boor algorithm.
    ===============================================================
    order [I]: scalar integer, BSpline order
    knots [I]: vector float, knots location
    ns    [I]: scalar integer, BSpline function index
    xval  [I]: vector float, sample location
    yval  [O]: vector float, BSpline value at xval
    [I] Input
    [O] Output
    [K] Keyword
    ---------------------------------------------------------------
    HISTORY:
    2015-09-06: Python port, G. Roudier,
    adapted from E. Hivon IDL BSPLINEEVALVECT routine
    2014-02-06: version 1.0, E. Hivon,
    adapted from G. Roudier BSPLINEEVALVECT routine
    with significant speed-up
    2011-07-06: version alpha, G. Roudier,
    allows redundant knots
    """
    nvect = np.size(xval)
    nmat = np.zeros((nvect, order, order))
    bsbuffer = np.zeros((nvect, order + 1))
    critsum = knots[ns + 1] + knots[ns + 2] + knots[ns + 3] + knots[ns + 4]
    fourmax = max(knots) * 4.
    for i in np.arange(order + 1):
        bsbuffer[:, i] = xval - knots[ns + i]
    for i in np.arange(order):
        select = ((bsbuffer[:, i] * bsbuffer[:, i + 1]) < 0.)
        if np.size(nmat[select, 0, i]) != 0:
            nmat[select, 0, i] = 1.
    for i in np.arange(order + 1):
        select = (bsbuffer[:, i] == 0.)
        if critsum == fourmax:
            nmat[select, 0, i - 1] = 1.
        if i < 4:
            nmat[select, 0, i] = 1.
    for i in (np.arange(order - 1) + 1):
        for j in np.arange(order - i):
            d1 = knots[ns + i + j] - knots[ns + j]
            d2 = knots[ns + i + j + 1] - knots[ns + j + 1]
            if d1 == 0:
                left = xval * 0.
            else:
                left = bsbuffer[:, j] / d1
            if d2 == 0:
                right = xval * 0.
            else:
                right = -bsbuffer[:, i + j + 1] / d2
            nmat[:, i, j] = left * nmat[:, i - 1, j] \
                + right * nmat[:, i - 1, j + 1]
    yval = nmat[:, order - 1, 0]
    return yval


# -------------------------- -----------------------------------------
# -- BUILDASQ -- -----------------------------------------------------
def buildasq(bsorder, knotsx, extsplinenumx, x, y, z,
             knotsy=None, extsplinenumy=None):
    if knotsy is None:
        knotsy = knotsx
    if extsplinenumy is None:
        extsplinenumy = extsplinenumx
    zsq = np.zeros(extsplinenumx * extsplinenumy)
    nz = np.size(z)
    leftmat = np.zeros((nz, extsplinenumx))
    rightmat = np.zeros((nz, extsplinenumy))
    for j in np.arange(extsplinenumx):
        leftmat[:, j] = bsplineev(bsorder, knotsx, j, x)
    for j in np.arange(extsplinenumy):
        rightmat[:, j] = bsplineev(bsorder, knotsy, j, y)
    asq = np.zeros((extsplinenumx * extsplinenumy, nz))
    for lx in np.arange(extsplinenumx):
        for ly in np.arange(extsplinenumy):
            lindice = lx * extsplinenumy + ly
            prod = leftmat[:, lx] * rightmat[:, ly]
            value = np.mat(prod) * np.transpose(np.mat(z))
            zsq[lindice] = value[0, 0]
            asq[lindice, :] = prod
    arc = np.copy(asq)
    asq = (np.mat(asq)) * np.transpose(np.mat(asq))
    zsq = np.mat(zsq)
    return asq, zsq, arc


# -------------- -----------------------------------------------------
# -- BS COEFF -- -----------------------------------------------------
def bscoefeval(bpar, x, y, z, knotsx, extsplinenumx,
               knotsy=None, extsplinenumy=None):
    asq, zsq, arc = buildasq(int(bpar['bsorder']), knotsx, extsplinenumx,
                             x, y, z,
                             knotsy=knotsy, extsplinenumy=extsplinenumy)
    bscoeff = lin.solve(asq, np.transpose(zsq))
    return bscoeff, arc


# -------------- -----------------------------------------------------
# -- SQUARE COORDINATES -- -------------------------------------------
def sqcoords(bpar, knotsx, extsplinenumx,
             knotsy=None, extsplinenumy=None, polar=False):
    if knotsy is None:
        mysidex = int(1 + 2 * (int(bpar['sqmaphpix']) - 10))
        mysidey = int(1 + 2 * (int(bpar['sqmaphpix']) - 10))
        mycx = int(bpar['sqmaphpix']) - 10
        mycy = int(bpar['sqmaphpix']) - 10
    else:
        mysidex = int(1 + 2 * (int(bpar['rectmaphpixx']) - 10))
        mysidey = int(1 + 2 * (int(bpar['rectmaphpixy']) - 10))
        mycx = int(bpar['rectmaphpixx']) - 10
        mycy = 0
    if polar:
        semisidepix = int((mysidey - 1) / 2.)
        phiradmax = semisidepix * float(bpar['pixsize']) / 60.
        phimax = 180. / np.pi * 60 * np.arcsin(phiradmax / 60.*np.pi / 180.)
        nphi = 1 + int(60.*phimax / float(bpar['pixsize']))
        phi = np.arange(nphi) * float(bpar['pixsize']) / 60.
        theta = np.arange(720) * np.pi / 180. / 2.
        xpol = np.array(np.mat(np.cos(theta)).T *
                        np.mat(np.sin(phi * np.pi / 60. / 180.)))
        xpol = np.reshape(xpol, phi.size * theta.size)
        ypol = np.array(np.mat(np.sin(theta)).T *
                        np.mat(np.sin(phi * np.pi / 60. / 180.)))
        ypol = np.reshape(ypol, phi.size * theta.size)
        xs = xpol * 3600.*180. / np.pi
        ys = ypol * 3600.*180. / np.pi
        if mycy == 0:
            selectx = np.abs(xs) <= (int(bpar['rectmaphpixx'] - 10) *
                                     float(bpar['pixsize']) / 60.)
            selecty = ys >= 0.
            select = np.logical_and(selectx, selecty)
            xs = xs[select]
            ys = ys[select]
    else:
        xs, ys = np.meshgrid(range(mysidex), range(mysidey))
        xs = xs * float(bpar['pixsize'])
        ys = ys * float(bpar['pixsize'])
        xs -= mycx * float(bpar['pixsize'])
        ys -= mycy * float(bpar['pixsize'])
        xs.astype(float)
        ys.astype(float)
    xs = xs / float(60)
    ys = ys / float(60)
    xs = np.reshape(xs, xs.size)
    ys = np.reshape(ys, ys.size)
    zs = np.zeros(xs.size)
    asq, zsq, arc = buildasq(int(bpar['bsorder']), knotsx, extsplinenumx,
                             xs, ys, zs,
                             knotsy=knotsy, extsplinenumy=extsplinenumy)
    return xs, ys, arc


# ------------------------ -------------------------------------------
# -- SQUARE SELECTION -- ---------------------------------------------
def sqsel(bpar, data, mask, x, y, t, xc, yc):
    x = np.array(x)
    y = np.array(y)
    z = np.array(data)
    select = np.logical_and(
        (np.abs(x - xc) < bpar['datarad']),
        (np.abs(y - yc) < bpar['datarad'])
    )
    select[mask] = False
    y = y[select]
    x = x[select]
    z = z[select]
    t = t[select]
    return x, y, z, t


# ---------------------- ---------------------------------------------
# -- CIRCLE DESTRIPING -- --------------------------------------------
def cdestripe(bpar, myd, myx, myy, myt):
    myt = np.array(myt)
    myx = np.array(myx)
    myy = np.array(myy)
    myd = np.array(myd)
    ref = np.diff(myy)[0]
    if ref > 0:
        ref = 1.
    else:
        ref = -1.
    circles = np.zeros(myd.size)
    for i in np.arange(myy.size):
        if i == 0:
            circles[0] = 0
        else:
            if ref * (myy[i] - myy[i - 1]) < 0:
                circles[i:] = circles[i:] + 1
    dstr = []
    dstrtime = []
    flagdstr = []
    for ic in set(circles):
        vf = True
        select = circles == ic
        cmyx = myx[select]
        cmyy = myy[select]
        cmyz = myd[select]
        dstrtime.append(np.mean(myt[select]))
        valid = cmyz[(cmyx ** 2 + cmyy ** 2) > (bpar['dstrrad']) ** 2]
        dstr.append(np.median(valid))
        if np.abs(float(valid[valid < 0].size) /
                  float(cmyz.size) - 0.5) > bpar['dstrtol']:
            if vf:
                flagdstr.append(True)
                vf = False
            if bpar['bsdebug']:
                print('!!! Unbalanced Circle !!!')
        if valid.size < 10:
            if vf:
                flagdstr.append(True)
                vf = False
            if bpar['bsdebug']:
                print('!!! Not Enough Valid Data !!!')
        if vf:
            flagdstr.append(False)
    dstr = np.array(dstr)
    dstrtime = np.array(dstrtime)
    flagdstr = np.array(flagdstr)
    return dstr, dstrtime, flagdstr, circles


# ----------------------- --------------------------------------------
# -- DEFINE TRANSIT -- -----------------------------------------------
def transit(bpar, mycache):
    mynewcache = {}
    for target in mycache:
        mynewcache[target] = []
        visits = []
        trings = []
        for ringdata in mycache[target]:
            trings.append(np.mean(ringdata['c_time']))
            visits.append(0)
        workwithme = np.argsort(trings)
        trings = np.array(trings)[workwithme]
        diff = list(np.concatenate((np.array([0]), np.diff(trings))))
        for dtring in diff:
            if (dtring > 24.*60.*60.):
                visits[diff.index(dtring):] = visits[diff.index(dtring):] + 1
        visits = np.array(visits)[workwithme]
        for itransit in set(visits):
            # CACHE FORMAT
            thisvisit = {}
            thisvisit['toi'] = []
            thisvisit['xdxx'] = []
            thisvisit['ydxx'] = []
            thisvisit['time'] = []
            thisvisit['circles'] = []
            thisvisit['c_offset'] = []
            thisvisit['c_time'] = []
            thisvisit['c_flags'] = []
            select = np.where(visits == itransit)[0]
            for i in select:
                mydict = mycache[target][i]
                thisvisit['toi'].extend(list(mydict['toi']))
                thisvisit['xdxx'].extend(list(mydict['xdxx']))
                thisvisit['ydxx'].extend(list(mydict['ydxx']))
                thisvisit['time'].extend(list(mydict['time']))
                thisvisit['circles'].extend(list(mydict['circles']))
                thisvisit['c_offset'].extend(list(mydict['c_offset']))
                thisvisit['c_time'].extend(list(mydict['c_time']))
                thisvisit['c_flags'].extend(list(mydict['c_flags']))
            mynewcache[target].append(thisvisit)
    if bpar['bsverbose']:
        for target in mynewcache:
            print('--< TARGET: NUMBER OF VISIT(S) >--')
            print(target + ': ' + str(len(mynewcache[target])))
            print('----------------------------------')
    return mynewcache


# -------------------- -----------------------------------------------
# -- VISIT MERGING -- ------------------------------------------------
def visitmerge(bpar, cache):
    mynewcache = {}
    freq, freqi, bolo, bolonum = id2fbn(bpar['boloID'])
    jupthr = np.array(bpar['jupthr'])
    for target in cache:
        # CACHE FORMAT
        thistarget = {}
        thistarget['toi'] = []
        thistarget['xdxx'] = []
        thistarget['ydxx'] = []
        for visit in cache[target]:
            ivisit = cache[target].index(visit)
            zextend = np.array(visit['toi'])
            xextend = np.array(visit['xdxx'])
            yextend = np.array(visit['ydxx'])
            selrms = np.logical_and(
                ((xextend ** 2 + yextend ** 2) > float(bpar['radrms']) ** 2),
                (np.abs(xextend) > float(bpar['xtfthr']))
                )
            thisrms = np.std(zextend[selrms])
            select = np.arange(zextend.size)
            satfactor = float(jupthr[freqi])
            if target == 'MARS':
                if ('JUPITER' in cache) or ('SATURN' in cache):
                    select = zextend > 5.*thisrms
            if target == 'SATURN':
                if 'JUPITER' in cache:
                    select = zextend > 9.*thisrms
                if 'MARS' in cache:
                    select = zextend < 1. / satfactor
            if target == 'JUPITER':
                if ('MARS' in cache) or ('SATURN' in cache):
                    select = zextend < 1. / satfactor
            if target == 'URANUS':
                if ('JUPITER' in cache) or ('SATURN' in cache):
                    select = zextend > 5.*thisrms
            if target == 'NEPTUNE':
                if ('JUPITER' in cache) or ('SATURN' in cache):
                    select = zextend > 5.*thisrms
            zextend = zextend[select]
            yextend = yextend[select]
            xextend = xextend[select]
            if bpar['bsverbose']:
                print(target + str(ivisit) + ' rms: ' + str(thisrms))
            thistarget['toi'].extend(list(zextend))
            thistarget['xdxx'].extend(list(xextend))
            thistarget['ydxx'].extend(list(yextend))
        mynewcache[target] = thistarget
    return mynewcache


# ------------------- ------------------------------------------------
# -- DESTRIPE -- -----------------------------------------------------
def destripe(bpar, mycache):
    for target in mycache:
        for visit in np.arange(len(mycache[target])):
            time = np.array(mycache[target][visit]['c_time'])
            flags = np.array(mycache[target][visit]['c_flags'])
            baseline = np.array(mycache[target][visit]['c_offset'])
            tord = np.argsort(time)
            bs = itp.UnivariateSpline(time[tord][flags == 0],
                                      baseline[tord][flags == 0])
            circles = np.array(mycache[target][visit]['circles'])
            toi = mycache[target][visit]['toi']
            dstrtoi = []
            newcirc = np.concatenate((np.array([0]), np.abs(np.diff(circles))))
            j = 0
            for i in np.arange(newcirc.size):
                if (newcirc[i] > 0.5):
                    j = j + 1
                dstrtoi.append(toi[i] - bs(time)[tord][j])
            mycache[target][visit]['toi'] = dstrtoi
    return mycache


# -------------- -----------------------------------------------------
# -- EXTRAFLAG -- ----------------------------------------------------
def extraflag(bpar, template, mycache):
    return mycache


# --------------- ----------------------------------------------------
# -- RECENTER -- -----------------------------------------------------
def recenter(bpar, template, mycache):
    for target in mycache:
        for visit in np.arange(len(mycache[target])):
            x = np.array(mycache[target][visit]['xdxx'])
            y = np.array(mycache[target][visit]['ydxx'])
            z = np.array(mycache[target][visit]['toi'])
            if template is None:
                p0 = np.array([np.max(z), 0., 0., 3., 3., 0., 0.])
                out, cov, info, mesg, ier = opt.leastsq(
                    bivargauss, p0, args=(x, y, z), full_output=True,
                    maxfev=10000)
                if ier not in range(5):
                    print('Gaussian fitting failed for {} on visit {}: "{}". '
                          'Discarding.'.format(target, visit, mesg))
                    mycache[target][visit]['toi'] = np.array([])
                    mycache[target][visit]['xdxx'] = np.array([])
                    mycache[target][visit]['ydxx'] = np.array([])
                    continue
                dx = out[1]
                dy = out[2]
                dz = out[0]
            mycache[target][visit]['toi'] = z / dz
            mycache[target][visit]['xdxx'] = x - dx
            mycache[target][visit]['ydxx'] = y - dy
    return mycache


# -------------- -----------------------------------------------------
# -- BI GAUSS -- -----------------------------------------------------
def bivargauss(p, x, y, zz):
    norm, xo, yo, sigma_x, sigma_y, theta, offset = p
    a = ((np.cos(theta) ** 2) / (2 * sigma_x ** 2) +
         (np.sin(theta) ** 2) / (2 * sigma_y ** 2))
    b = ((np.sin(2 * theta)) / (4 * sigma_y ** 2) -
         (np.sin(2 * theta)) / (4 * sigma_x ** 2))
    c = ((np.sin(theta) ** 2) / (2 * sigma_x ** 2) +
         (np.cos(theta) ** 2) / (2 * sigma_y ** 2))
    z = offset + norm * np.exp(-(a * ((x - xo) ** 2) +
                               2 * b * (x - xo) * (y - yo) +
                               c * ((y - yo) ** 2)))
    return zz - z.ravel()


# -------------- -----------------------------------------------------
# -- POLAR BEAM TEMPLATES -- -----------------------------------------
def poltmp(bpar, freqi, mainbeam, x, y):
    nslices = int(bpar['nslices'][freqi])
    phiradmax = (int(bpar['sqmaphpix']) - 10) * float(bpar['pixsize']) / 60
    radmax = 180 / np.pi * 60 * np.arcsin(phiradmax / 60 * np.pi / 180)
    nradius = 1 + int(60 * radmax / float(bpar['pixsize']))
    radii = np.sin(np.arange(nradius) * float(
        bpar['pixsize']) / 3600 * np.pi / 180)
    radii = radii * 60 * 180 / np.pi
    datatheta = [np.angle(np.complex(xx, yy)) for xx, yy in zip(x, y)]
    datatheta = np.array(datatheta)
    postheta = np.copy(datatheta)
    postheta[postheta < 0] = postheta[postheta < 0] + 2 * np.pi
    dataradius = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
    angles = np.arange(nslices) * 2 * np.pi / nslices
    deltaangle = np.pi / nslices
    poltemplates = []
    for mytheta in angles:
        selecttheta = np.abs(datatheta - mytheta) < deltaangle
        if mytheta >= (np.pi - deltaangle):
            selecttheta = np.abs(postheta - mytheta) < deltaangle
        azimav = []
        for iradii in np.arange(radii.size):
            myrad = radii[iradii]
            if myrad == 0:
                azimav.append(np.mean(mainbeam[dataradius == 0.]))
            else:
                if myrad == np.max(radii):
                    selectrad = (myrad - dataradius <=
                                 (radii[iradii] - radii[iradii - 1]) / 2.)
                else:
                    selectradp = (dataradius - myrad <=
                                  (radii[iradii + 1] - radii[iradii]) / 2.)
                    selectradn = (myrad - dataradius <
                                  (radii[iradii] - radii[iradii - 1]) / 2.)
                    selectrad = np.logical_and(selectradp, selectradn)
                select = np.logical_and(selectrad, selecttheta)
                if np.all(select == 0):
                    azimav.append(np.nan)
                else:
                    azimav.append(np.mean(mainbeam[select]))
        poltemplates.append(azimav)
    ntf = int(bpar['ntf'][freqi])
    poltemplates = np.transpose(np.array(poltemplates))
    indtf = np.where(np.abs(angles - np.pi / 2.) < deltaangle)
    indtf = indtf[0][0]
    poltemplates = np.concatenate((poltemplates[:, 0:indtf - int(ntf / 2.)],
                                   poltemplates[:, indtf + int(ntf / 2.) + 1:]),
                                  axis=1)
    angles = np.concatenate((angles[0:indtf - int(ntf / 2.)],
                             angles[indtf + int(ntf / 2.) + 1:]))
    return poltemplates, radii, angles


# -------------------------- -----------------------------------------
# -- HYBRID MAP -- ---------------------------------------------------
def hmapfill(bpar, mainbeam, mbxs, mbys, tfres, tfxs, tfys,
             poltmp, polrad, theta, freq, freqi, polar=False):
    hybthd = float(bpar['hybthd'][freqi])
    nslices = float(bpar['nslices'][freqi])
    dangle = np.pi / nslices
    rmb = np.sqrt(np.array(mbxs) ** 2 + np.array(mbys) ** 2)
    tmb = [np.angle(np.complex(xx, yy)) for xx, yy in zip(mbxs, mbys)]
    tmb = np.array(tmb)
    tmb[tmb < 0] = tmb[tmb < 0] + 2.*np.pi
    intbeam = np.copy(mainbeam)
    selrms = np.logical_and(
        (((mbxs ** 2 + mbys ** 2) > 20.**2),
         (np.abs(mbxs) > float(bpar['xtfthr'])))
    )
    thisrms = np.std(mainbeam[selrms])
    if bpar['bsverbose']:
        print('Estimated beam RMS: ' + str(thisrms))
    nslices = poltmp.shape[1]
    indice = 0
    # AZIMUTHAL TEMPLATES
    for myrad, mytheta in zip(rmb, tmb):
        template = []
        ttemplate = []
        ctemplate = []
        value = np.nan
        if myrad <= np.max(polrad):
            ir = np.where(np.abs(polrad - myrad) < (float(bpar['pixsize'])
                                                    / 60 / 2))
            ir = ir[0][0]
            irc = None
            if myrad - polrad[ir] < 0:
                irc = ir - 1
            if myrad - polrad[ir] > 0:
                irc = ir + 1
            template.extend(poltmp[ir, :])
            template.extend(poltmp[ir, :])
            template.extend(poltmp[ir, :])
            template = np.reshape(np.array(template),
                                  np.array(template).size)
            ttemplate.extend(np.array(theta) - 2.*np.pi)
            ttemplate.extend(theta)
            ttemplate.extend(np.array(theta) + 2.*np.pi)
            ttemplate = np.array(ttemplate)
            bs = itp.UnivariateSpline(ttemplate, template, s=0)
            value = bs(mytheta)
            if irc is not None:
                ctemplate.extend(poltmp[irc, :])
                ctemplate.extend(poltmp[irc, :])
                ctemplate.extend(poltmp[irc, :])
                ctemplate = np.reshape(np.array(ctemplate),
                                       np.array(ctemplate).size)
                bsc = itp.UnivariateSpline(ttemplate, ctemplate, s=0)
                cvalue = bsc(mytheta)
                weight = np.abs(myrad - polrad[ir]) / (float(bpar['pixsize'])
                                                       / 60)
                value = (1. - weight) * value + weight * cvalue
        intbeam[indice] = value
        indice = indice + 1
    azav = [np.mean(poltmp[irad, :]) for irad in np.arange(polrad.size)]
    azav = np.array(azav)
    rlim = np.max(polrad[azav > hybthd * thisrms])
    # INTERMEDIATE BEAM
    mainbeam[rmb > (rlim + float(bpar['trans']))] = intbeam[
        rmb > (rlim + float(bpar['trans']))]
    # INTERMEDIATE BEAM TRANSITION
    iribw = np.where(
        np.logical_and((rmb >= rlim), (rmb <= rlim + float(bpar['trans'])))
    )[0]
    ribw = rmb[iribw]
    if ribw.size > 0:
        myrrange = float(bpar['trans'])
        for myr, myind in zip(ribw, iribw):
            weight = (myr - rlim) / myrrange
            mainbeam[myind] = ((1. - weight) * mainbeam[myind] +
                               weight * intbeam[myind])
    powerint, powerskirt, noisefactor, edgetapcorr = freq2dpp(freq)
    edgetapcoef = 6. - edgetapcorr
    yxskirt = 10.**(-edgetapcoef)
    xxskirt = 60.
    estnoise = np.std(azav[polrad > 30.])
    plateau = np.mean(azav[polrad > 30.])
    mythr = (noisefactor * estnoise) + plateau
    maxrmb = np.max(polrad[azav >= mythr])
    maxmbpol = np.max(polrad)
    yxpt = azav[polrad == maxrmb]
    xxpt = polrad[polrad == maxrmb]
    skirt = np.copy(polrad)
    skirt[0] = 1.
    skirt = (xxskirt ** (-powerskirt)) * yxskirt * (skirt ** powerskirt)
    skirt[0] = 1.
    select = azav < mythr
    redpolrad = polrad[select]
    redazav = azav[select]
    redskirt = skirt[select]
    selrmin = redpolrad[
        np.abs(redazav - redskirt) == np.min(np.abs(redazav - redskirt))]
    selrmin = selrmin[0]
    if bpar['bsverbose']:
        print('Crossing diffraction pattern at radius = ' + str(selrmin))
    # HYBRID MAP INIT
    if polar:
        phimax = 180 / np.pi * 60 * np.arcsin(
            int(bpar['hrmax']) / 60 * np.pi / 180)
        nphi = 1 + int(60.*phimax / float(bpar['pixsize']))
        phi = np.arange(nphi) * float(bpar['pixsize']) / 60.
        theta = np.arange(720) * np.pi / 180. / 2.
        xpol = np.array(np.mat(np.cos(theta)).T *
                        np.mat(np.sin(phi * np.pi / 60. / 180.)))
        xpol = np.reshape(xpol, phi.size * theta.size)
        ypol = np.array(np.mat(np.sin(theta)).T *
                        np.mat(np.sin(phi * np.pi / 60. / 180.)))
        ypol = np.reshape(ypol, phi.size * theta.size)
        xs = xpol * 3600.*180. / np.pi
        ys = ypol * 3600.*180. / np.pi
    else:
        myc = int(float(bpar['hrmax']) * 60. / float(bpar['pixsize']))
        myside = int(1 + 2 * myc)
        xs, ys = np.meshgrid(range(myside), range(myside))
        xs = xs * float(bpar['pixsize'])
        ys = ys * float(bpar['pixsize'])
        xs -= myc * float(bpar['pixsize'])
        ys -= myc * float(bpar['pixsize'])
        xs.astype(float)
        ys.astype(float)
    xs = xs / 60.
    ys = ys / 60.
    xhmap = np.reshape(xs, xs.size)
    yhmap = np.reshape(ys, ys.size)
    rhmap = np.sqrt(xhmap ** 2 + yhmap ** 2)
    hmap = np.zeros(xs.size)
    # MAIN + INTERMEDIATE BEAM STITCHING (TOO LONG)
    for xin, yin, zin in zip(mbxs, mbys, mainbeam):
        hmap[np.logical_and((xhmap == xin), (yhmap == yin))] = zin
    # DIFFRACTION PATTERN
    diffps = rhmap >= maxmbpol
    hmap[diffps] = ((xxskirt ** (-powerskirt)) * yxskirt *
                    (rhmap[diffps] ** (powerskirt)))
    # DIFFRACTION PATTERN TRANSITION
    iribw = np.where(
        np.logical_and((rhmap >= selrmin), (rhmap <= maxmbpol))
    )[0]
    ribw = rhmap[iribw]
    if ribw.size > 0:
        myrrange = maxmbpol - selrmin
        for myr, myri in zip(ribw, iribw):
            weight = (myr - selrmin) / myrrange
            diffpat = (xxskirt ** (-powerskirt)) * yxskirt \
                * (myr ** (powerskirt))
            hmap[myri] = ((1. - weight) * hmap[myri] + weight * diffpat)
    fullhb = np.copy(hmap)
    # TRANSFER FUNCTION RESIDUALS
    for addx, addy, addz in zip(tfxs, tfys, tfres):
        if (addy <= float(bpar['hrmax'])) and (addz < hybthd * thisrms):
            hmap[np.logical_and((xhmap == addx), (yhmap == addy))] = addz
    # TRANSFER FUNCTION RESIDUALS TRANSITIONS
    lefttf = -(float(bpar['rectmaphpixx']) - 10) * float(bpar['pixsize']) / 60
    righttf = (float(bpar['rectmaphpixx']) - 10) * float(bpar['pixsize']) / 60
    selecttl = np.abs(xhmap - lefttf) <= 2.*float(bpar['trans'])
    selecttl = np.logical_and(selecttl, (yhmap > 0))
    selecttf = np.abs(xhmap - righttf) <= 2.*float(bpar['trans'])
    selecttf = np.logical_and(selecttf, (yhmap > 0))
    # LEFT
    for myxs, myys in zip(xhmap[selecttl], yhmap[selecttl]):
        dist = (myxs - lefttf) / float(bpar['trans']) / 2
        if dist >= 0:
            weight = (np.sin(dist * np.pi / 2)) ** 4
            seltr = np.logical_and((xhmap == myxs), (yhmap == myys))
            value = ((1. - weight) * fullhb[seltr] + weight * hmap[seltr])
            hmap[seltr] = value
    # RIGHT
    for myxs, myys in zip(xhmap[selecttf], yhmap[selecttf]):
        dist = (myxs - righttf) / float(bpar['trans']) / 2
        if dist <= 0:
            weight = (np.sin(dist * np.pi / 2)) ** 4
            seltr = np.logical_and((xhmap == myxs), (yhmap == myys))
            value = ((1. - weight) * fullhb[seltr] + weight * hmap[seltr])
            hmap[seltr] = value
    if bpar['optical']:
        hmap = fullhb
    return hmap, xhmap, yhmap


# ---------------- ---------------------------------------------------
# -- FREQ TO DIFFRACTION PATTERN PARAMS ------------------------------
def freq2dpp(freq):
    if freq == '100':
        powerint = -7.
        powerskirt = -3.
        noisefactor = 9.
        edgetapcorr = 0.064
    elif freq == '143':
        powerint = 666.
        powerskirt = -3.
        noisefactor = 5.
        edgetapcorr = -0.11
    elif freq == '217':
        powerint = 666.
        powerskirt = -3.
        noisefactor = 3.3
        edgetapcorr = -0.458
    elif freq == '353':
        powerint = 666
        powerskirt = -3.
        noisefactor = 3.3
        edgetapcorr = -0.27
    elif freq == '545':
        powerint = 666.
        powerskirt = -3.
        noisefactor = 4.
        edgetapcorr = -0.27
    elif freq == '857':
        powerint = 666
        powerskirt = -3.
        noisefactor = 4.
        edgetapcorr = -0.27
    else:
        raise RuntimeError(
            'bsutil.freq2dpp: unknown frequency: "{}"'.format(freq))
    return powerint, powerskirt, noisefactor, edgetapcorr
# ------------------------------------- ------------------------------
