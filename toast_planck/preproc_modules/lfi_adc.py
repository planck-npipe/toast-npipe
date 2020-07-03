# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os
import pickle

import numpy as np
import toast.timing as timing

from toast_planck.utilities import to_diodes, det2freq

__path__ = os.path.dirname(__file__)
NL_PATH = 'lfi_adc_data'

class LFINLCorrector():

    def __init__(self, radiometer, comm=None):
        self.radiometer = radiometer
        self.diodes = to_diodes(self.radiometer)
        self.freq = det2freq(self.radiometer)

        if comm is not None:
            self.comm = comm
            self.rank = comm.rank
        else:
            self.comm = None
            self.rank = 0

        self.adcs = []
        self.diodefiles = []
        if self.rank == 0:
            for diode in range(2):
                pattern = '{}/*{}*pic'.format(
                    os.path.join(__path__, NL_PATH),
                    self.diodes[diode].replace('LFI', ''))
                filelist = glob.glob(pattern)
                if len(filelist) == 0:
                    raise Exception(
                        'No ADC file matches pattern {}'.format(pattern))
                self.diodefiles.append(filelist)

                self.adcs.append([])
                for fn in self.diodefiles[-1]:
                    if 'FLAT' in fn.upper():
                        continue
                    with open(fn, 'rb') as f:
                        adc = pickle.load(f, encoding='latin-1')
                        self.adcs[-1].append(adc)
        self.diodefiles = comm.bcast(self.diodefiles, root=0)
        self.adcs = comm.bcast(self.adcs, root=0)

    def correct(self, signal, timestamps):
        """
        Correct the ADC nonlinearity in raw LFI TOI.

        signal is assumed to be calibrated into VOLTS and to be an
        [nsamp, 4] array where the columns are
          sky0
          load0
          sky1
          load1

        """
        tstart = timestamps[0]
        tstop = timestamps[-1]

        sig = signal.T.copy()

        for diode in range(2):
            if len(self.adcs[diode]) == 0:
                # Flat correction, don't touch the data
                continue
            if len(self.adcs[diode]) == 1:
                adc = self.adcs[diode][0]
            else:
                found = False
                failed = []
                for adc in self.adcs[diode]:
                    if adc.keys['startOBT']*2**-16 > tstop \
                       and adc.keys['startOBT']*2**-16 > 1.7e9:
                        failed.append(
                            [adc.keys['startOBT']*2**-16,
                             adc.keys['endOBT']*2**-16])
                    else:
                        found = True
                        break

                if not found:
                    raise Exception(
                        'There is no ADC NL correction available for {} '
                        'between {} and {}. Available ranges were: {}'.format(
                            self.radiometer, tstart, tstop, failed))

            sky_in = adc.sky_volt_in
            sky_out = adc.sky_volt_out
            load_in = adc.load_volt_in
            load_out = adc.load_volt_out

            good = np.logical_and(sig[2*diode] >= sky_out[0],
                                  sig[2*diode] <= sky_out[-1])
            ngood = np.sum(good)
            nbad = len(good) - ngood
            if nbad != 0:
                print('WARNING: {:.2f} % of the raw sky TOD ({} - {}) is '
                      'outside of the ADC correction range ({} - {}) for '
                      '{}{}{} during {} - {}'.format(
                          nbad*100./len(good),
                          sig[2*diode].min(), sig[2*diode].max(),
                          sky_out.min(), sky_out.max(),
                          adc.keys['horn'],adc.keys['radiometer'], diode,
                          timestamps[0], timestamps[-1]))
            sig[2*diode][good] = np.interp(sig[2*diode][good], sky_out, sky_in)

            good = np.logical_and(sig[2*diode+1] >= load_out[0],
                                  sig[2*diode+1] <= load_out[-1])
            ngood = np.sum(good)
            nbad = len(good) - ngood
            if nbad != 0:
                print('WARNING: {:.2f} % of the raw load TOD ({} - {}) is '
                      'outside of the ADC correction range ({} - {}) for '
                      '{}{}{} during {} - {}'.format(
                          nbad*100./len(good),
                          sig[2*diode+1].min(), sig[2*diode+1].max(),
                          load_out.min(), load_out.max(),
                          adc.keys['horn'],adc.keys['radiometer'], diode,
                          timestamps[0], timestamps[-1]))
            sig[2*diode+1][good] = np.interp(sig[2*diode+1][good],
                                             load_out, load_in)

        return sig.T.copy()
