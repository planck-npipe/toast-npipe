# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import pickle

import numpy as np

import toast
import toast.tod as tt

from toast_planck import shdet

from toast_planck.preproc_modules import Transf1, GainCorrector

from toast_planck.utilities import bolo_to_pnt, read_gains

from toast_planck.imo import IMO

from toast_planck.shdet import SHDet


class OpSimSHDET(toast.Operator):
    """
    Operator which takes parameters and a timestream of optical power and
    pass this information into SHDET to simulate detector response.

    Args:
        params (dicts): SHDET parameter dictionary.
        optical (str): if None, read TOD otherwise the cache name to use to
            get the optical power in Watts.
        out (str): if None, write the output to the TOD, otherwise the cache
            name to use.
    """

    def __init__(self, dets = None, imofile=None, adc_table=None, params=None,
                 margin=0, calfile=None, tffile=None, read_no_signal=False,
                 offset_file=None):
        self._imo = IMO(imofile)
        self._margin = margin
        self._calfile = calfile
        self._read_no_signal = read_no_signal

        # these are dictionaries
        self._adc_table = adc_table
        self._tffile = tffile
        self._params = params

        self._offset_file = offset_file

        self._shdet = {}
        self._n = {}
        self._nadc = {}
        self._nparam = {}
        self._sentinel = {}
        self._nparam2 = {}
        if self._offset_file is not None:
            self._offsets = {}
        else:
            self._offsets = None

        self._base_seed = {}

        for det in dets:
            self._shdet[det] = SHDet(parameters = params[det])
            self._n[det] = self._shdet[det].get_n()
            self._nadc[det] = self._shdet[det].get_nadc()
            self._nparam[det] = self._shdet[det].get_nparam()
            self._sentinel[det] = self._shdet[det].get_sentinel()
            self._nparam2[det] = self._shdet[det].get_nparam2()
            if 'seed' in params[det].keys():
                self._base_seed[det] = params[det]['seed']
            else:
                self._base_seed[det] = None
            if self._offset_file is not None:
                self._offsets[det] = pickle.load(open(
                    self._offset_file.replace('DETECTOR', det), 'rb'))


        # The margin is required in preproc. SHDET outputs should have the
        # margin allocated and filled to be able to replace reading flight
        # data off disk.

        # placeholders for the SHDet transfer function
        self._tf_freq = {}
        self._TF = {}

        super().__init__()

    def get_TF(self):
        return self._tf_freq, self._TF

    def exec(self, data):
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world

        # Measure SHDet transfer function to set up tau deconvolver

        for det in self._shdet.keys():
            if self._tffile is None:
                tf_freq, tf_real, tf_imag \
                    = self._shdet[det].measure_transfer_function(comm=comm)
                TF = tf_real + 1j * tf_imag
            else:
                # load TF from file
                input_tf = np.genfromtxt(self._tffile[det]).T
                tf_freq = input_tf[0]
                TF = input_tf[1] + 1j * input_tf[2]

            # remove the mean
            TF /= np.mean(TF[:5])

            # store the transfer function until later
            self._tf_freq[det] = tf_freq
            self._TF[det] = TF


        for obs in data.obs:
            tod = obs['tod']
            nsamp = tod.local_samples[1]

            intervals = tod.local_intervals(obs['intervals'])
            local_starts = [ival.first for ival in intervals]
            local_stops = [ival.last+1 for ival in intervals]

            ring_offset = tod.globalfirst_ring
            for interval in obs['intervals']:
                if interval.last  < tod.local_samples[0]:
                    ring_offset += 1

            timestamps = tod.local_timestamps(margin=self._margin)

            for det in tod.local_dets:
                #print(det) # DEBUG
                bolo_id = bolo_to_pnt(det)
                bc = np.int(bolo_id[:2]) # belt code

                transf1 = Transf1()
                gaincorrector = GainCorrector(self._imo, bolo_id, linear=True)

                if self._calfile is not None:
                    gains = read_gains(self._calfile, det, timestamps[0],
                                       timestamps[-1], cworld)
                else:
                    gains = None

                # get the optical power in Kelvin

                signal = tod.local_signal(det)
                if len(signal) != nsamp + 2*self._margin:
                    raise Exception('Cached signal does not include margins.')

                # DEBUG: filter the input signal to remove noise
                #from scipy.signal import fftconvolve
                #kernel = np.ones(101)/101.
                #signal = fftconvolve( signal, kernel, mode='same' )
                # DEBUG end

                # DEBUG: just zero the signal
                if self._read_no_signal:
                    signal = np.zeros(np.shape(signal))

                if ((self._adc_table is not None)):
                    # Load the ADC nonlinearity table. If the path to the
                    # table contains the string "DETECTOR", it will be
                    # replaced with the correct detector name
                    path = self._adc_table[det]
                    if 'DETECTOR' in path:
                        path = path.replace('DETECTOR', det)
                    try:
                        adc_table = np.genfromtxt(path)
                    except:
                        raise RuntimeError(
                            'Warning: cannot read ADC table from {}'
                            ''.format(path))
                        #adc_table = np.arange(self._nadc[det], dtype=np.float)
                else:
                    # Linear ADC table
                    adc_table = np.arange(self._nadc[det], dtype=np.float)

                ring_number = ring_offset - 1

                # Loop over rings, call shdet for every ring separately

                for ring_start, ring_stop in zip(local_starts, local_stops):
                    ring_number += 1

                    # This slice does not have the margins so that every
                    # shdet call is disjoint
                    ind = slice(ring_start+self._margin, ring_stop+self._margin)

                    sig = signal[ind] # a memory view to one ring of data.
                    tme = timestamps[ind]

                    if self._offsets is not None:
                        offsets = self._offsets[det]
                        ioffset = np.argmin(np.abs(offsets[0] - tme[0]))
                        optical_offset = offsets[4][ioffset] # K_CMB
                        raw_offset = offsets[1][ioffset] # digitized units
                    else:
                        optical_offset = None
                        raw_offset = None

                    # This is the time-dependent scaling between
                    # integer-valued 180Hz data and KCMB. One value per
                    # TOI sample.

                    dsp2cmb = np.ones(len(sig), dtype=np.float64)
                    dsp2cmb = transf1.convert(dsp2cmb, tme, det)
                    dsp2cmb = gaincorrector.correct(dsp2cmb, np.isnan(dsp2cmb))
                    dsp2cmb = tt.calibrate(tme, dsp2cmb, *gains)

                    # update the random number seed for noise generation
                    if self._base_seed[det] is not None:
                        noise_seed = 30000*bc + ring_number \
                                     + self._base_seed[det]*3000000
                    else:
                        noise_seed = None

                    # There are no quality flags: every sample is assumed
                    # to have a reasonable optical power value. This will
                    # be guaranteed at the pipeline level.

                    sig_shdet = self._shdet[det].simulate(
                        sig, noise_seed=noise_seed,
                        optical_offset=optical_offset, raw_offset=raw_offset,
                        adc_table=adc_table)

                    signal[ind] = sig_shdet[:len(sig)]


                # Return the simulated timeline

                # FIXME: This is where the margins need to be communicated
                # between processes

                tod.local_signal(det)[:] = signal

        return
