# Copyright (c) 2016-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# -- IMPORTS -- ------------------------------------------------------
import os
import pickle
import time

import scipy.signal

import astropy.io.fits as pyfits
import numpy as np

from . import bsmc
from . import bsutil


# ------------- ------------------------------------------------------
# -- BEAM CLASS -- ---------------------------------------------------
class Beam_Reconstructor():

    def __init__(self,
                 boloID, beampar, mpicomm=None, brep=None, biter=0):
        # BEAM OBJECT FORMAT
        self.boloID = boloID
        self.brep = brep
        self.biter = biter
        self.bpar = beampar
        self.beam = None
        self.bscoeffs = None
        self.tfcoeffs = None
        self.cache = {}
        self.scache = {}
        if mpicomm is not None:
            itask = mpicomm.Get_rank()
        else:
            itask = 0
        if self.bpar['bsverbose'] and itask == 0:
            print('--< INIT BEAM OBJECT >--')
            print('Channel ' + self.boloID)
            print('Iteration ' + str(self.biter))
            print('Parameter File ' + str(type(self.bpar)))
            print('Beam Prior Rec ' + str(type(self.beam)))
            print('Representation ' + str(self.brep))
            print('BS Coefficients ' + str(type(self.bscoeffs)))
            print('BS TF Coefficients ' + str(type(self.tfcoeffs)))
            print('------------------------', flush=True)

    def preproc(self, tsamp, toi, flag, offset_az, offset_el, targetname):

        # PLANET DATA PREPROCESSING
        mydata = toi  # bsutil.caldata(self.bpar, tsamp, toi, dipole, bg)
        mymask = flag  # bsutil.planetflag(self.bpar, toi.mask)
        myx, myy, myd, myt = bsutil.sqsel(self.bpar,
                                          mydata, mymask,
                                          offset_az, offset_el, tsamp, 0., 0.)
        if np.size(myy) > 0:
            dstr, dstrt, dstrf, crcl = bsutil.cdestripe(
                self.bpar, myd, myx, myy, myt)
            if targetname not in self.cache:
                self.cache[targetname] = []
            # CACHE FORMAT
            thisring = {}
            thisring['toi'] = myd
            thisring['xdxx'] = myx
            thisring['ydxx'] = myy
            thisring['time'] = myt
            thisring['circles'] = crcl
            thisring['c_offset'] = dstr
            thisring['c_time'] = dstrt
            thisring['c_flags'] = dstrf
            self.cache[targetname].append(thisring)
        return

    def recache(self):
        # REFORMAT CACHE AFTER GATHERING
        mynewcache = {}
        targetlist = []
        for ncache in self.cache:
            for ntarget in ncache:
                for planet in ntarget:
                    if planet not in targetlist:
                        targetlist.extend(ntarget)
        for target in targetlist:
            if target not in mynewcache:
                mynewcache[target] = []
            for ncache in self.cache:
                for ntarget in ncache:
                    if target in ntarget:
                        mynewcache[target].extend(ntarget[target])
        self.cache = mynewcache
        return

    def mergedata(self):
        # PLANET DATA MERGING
        self.cache = bsutil.transit(self.bpar, self.cache)
        self.cache = bsutil.destripe(self.bpar, self.cache)
        self.cache = bsutil.recenter(self.bpar, self.beam, self.cache)
        self.cache = bsutil.extraflag(self.bpar, self.beam, self.cache)
        self.cache = bsutil.visitmerge(self.bpar, self.cache)
        return

    def reconstruct(self):
        # BUILD HYBRID BEAM MAP IN SELF.BEAM
        self.bscoeffs = bsmc.bsdec(self.bpar, self.cache)
        self.tfcoeffs = bsmc.tfbs(self.bpar, self.cache)
        if self.bpar['savebeamobj']:
            self.save()
        self.beam = bsmc.hybrid(self.bpar,
                                self.bscoeffs, self.tfcoeffs, self.cache)
        return

    def update(self, niter):
        # UPDATE BEAM OBJECT
        self.biter = niter + 1
        self.brep = 'HYBRID'
        if self.bpar['bsverbose']:
            print('--< BEAM OBJECT >--')
            print('Channel ' + self.boloID)
            print('Iteration ' + str(self.biter))
            print('Parameter File ' + str(type(self.bpar)))
            print('Beam Prior Rec ' + str(type(self.beam)))
            print('Representation ' + str(self.brep))
            print('BS Coefficients ' + str(type(self.bscoeffs)))
            print('BS TF Coefficients ' + str(type(self.tfcoeffs)))
            print('-------------------')
        return

    def save(self):
        # SAVE BEAM OBJECT
        whereto = self.bpar['savepath']
        inwhat = self.bpar['savedir']
        prefix = self.bpar['prefix']
        if whereto == 'myscratch':
            whereto = os.environ.get('SCRATCH', '.')
        if prefix == 'today':
            prefix = time.strftime('%Y_%m_%d')
        whereto = os.path.join(whereto, inwhat)
        if not(os.path.isdir(whereto)):
            os.mkdir(whereto)
        myfile = '-'.join((prefix, self.bpar['boloID'], 'bclass.pickle'))
        myfile = os.path.join(whereto, myfile)
        with open(myfile, 'wb') as myfptr:
            pickle.dump(self, myfptr, protocol=2)
            myfptr.close()
        return

    def emptycache(self):
        # EMPTY BEAM OBJECT CACHE
        self.cache = {}
        self.scache = {}

    def hmapsave(self, filename, polar=False):
        '''
        This helper function saves a beam to a fits file.
        The default is to save a rectangular-coordinate beam map to a
        binary table extension.

        Args:
            filename -- file in which to save the data.
            polar -- Save in a FEBECOP-compatible polar-coordinate format.
                default is False.
        '''
        # SAVE HYBRID BEAM MAP (self.beam) INTO FITS
        # FORMAT:
        # self.beam['SQUARE_X']: -hrmax, +hrmax in arcmin per
        #                        pixsize arcsec steps - 1D vector
        # self.beam['SQUARE_Y']
        # self.beam['SQUARE']: associated beam values - 1D vector
        # self.beam['POLAR_X']: carthesian coordinates of spherical phi, theta
        #                       for phi in 0, hrmax per pixsize arcsec steps
        #                       and theta in 0, 359.5 per half a degree steps
        # self.beam['POLAR_Y']
        # self.beam['POLAR']: associated beam values - 1D vector
        fits_keyword_data = {}
        if polar:
            # set up fits keywords
            fits_keyword_data['Mintheta'] = [
                np.min(self.beam['POLAR_Y']) * np.pi / 180,
                'Min polar angle [rad]']
            fits_keyword_data['Maxtheta'] = [
                np.max(self.beam['POLAR_Y']) * np.pi / 180,
                'Max polar angle [rad]']
            fits_keyword_data['Nphi'] = [np.size(self.beam['POLAR_X']),
                                         'Number of points in azimuth angle.']
            fits_keyword_data['Ntheta'] = [np.size(self.beam['POLAR_Y']),
                                           'Number of points in polar angle.']
            # Add a bunch of zeros to the polarized beam since it doesn't exist
            nulldata = np.zeros(np.shape(np.ravel(self.beam['POLAR'])))
            tbhdu = pyfits.BinTableHDU.from_columns(
                [pyfits.Column(name='BEAMDATA', unit='', format='E',
                               array=np.ravel(self.beam['POLAR'])),
                 pyfits.Column(name='BEAMDATAQ', unit='', format='E',
                               array=nulldata),
                 pyfits.Column(name='BEAMDATAU', unit='', format='E',
                               array=nulldata),
                 pyfits.Column(name='BEAMDATAV', unit='', format='E',
                               array=nulldata)])
        else:
            fits_keyword_data['NX'] = [np.size(self.beam['SQUARE_X']),
                                       'Grid X size']
            fits_keyword_data['NY'] = [np.size(self.beam['SQUARE_Y']),
                                       'Grid Y size']
            fits_keyword_data['XDELTA'] = [
                (self.beam['SQUARE_X'][1] - self.beam['SQUARE_X'][0])
                * np.pi / 180 / 60,
                'Grid X step [radians]']
            fits_keyword_data['YDELTA'] = [
                (self.beam['SQUARE_Y'][1] - self.beam['SQUARE_Y'][0])
                * np.pi / 180 / 60,
                'Grid Y step [radians]']
            fits_keyword_data['XCENTRE'] = [np.size(self.beam['SQUARE_X']) / 2,
                                            'Center location (X index)']
            fits_keyword_data['YCENTRE'] = [np.size(self.beam['SQUARE_Y']) / 2,
                                            'Center location (Y index)']
            tbhdu = pyfits.BinTableHDU.from_columns([
                pyfits.Column(name='BEAMDATA', unit='', format='E',
                              array=np.ravel(self.beam['SQUARE']))])

        for kk in fits_keyword_data.keys():
            tbhdu.header.set(kk, fits_keyword_data[kk][0],
                             fits_keyword_data[kk][1])

        tbhdu.writeto(filename)
        return

# ---------------- ---------------------------------------------------
