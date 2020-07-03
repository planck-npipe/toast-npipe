#cython: language_level=3, boundscheck=True, wraparound=True, embedsignature=True, cdivision=True

# Copyright (c) 2015-2018 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, fabs, sin, cos, M_PI

import toast.timing as timing


cdef extern from "cshdet.h":
    bint cSHDet_f(double *rng_arry, double *adc_ctp, double *prm_rnng,
                  double *prm_inst, double *prm_rdout, double *prm_opt,
                  double *prm_blmtr)

    
cdef extern from "shdet_types.h":
    cdef long l_rng_ttl
    cdef long n_prm_shdet
    cdef long mx_blmtr_mdl
    cdef long n_smpl_prd
    cdef long n_ctp
    cdef long n_vr_shdet
    cdef long shdet_sntnl


shdet_internal_parameter_positions = {
    'rnng':{'T_sink':1, 'adc_on':4,'raw_model':35, 'noise_dsn':36, 'seed':37,
            'raw_constant_offset_dsn':38, 'switch_raw_offset': 39},
    'inst':{},
    'rdout':{'cstray': 17, 'bdac':20},
    'opt':{'optical_load_watts':0, 'gain_w_per_kcmb':1,
           'optical_load_offset_kcmb':30, 'switch_optical_offset': 31},
    'blmtr':{}
}


cdef class SHDet:

    cdef long n
    cdef long nadc
    cdef long nparam
    cdef long sentinel
    cdef long nparam2

    cdef dict shdet_parameters

    def __cinit__(self, parameters=None):

        self.n = l_rng_ttl - 4
        self.nadc = n_ctp
        self.nparam = n_prm_shdet
        self.sentinel = shdet_sntnl
        self.nparam2 = mx_blmtr_mdl

        self.shdet_parameters = {}
        for category in ['rnng', 'inst', 'rdout', 'opt']:
            self.shdet_parameters[category] = np.zeros(self.nparam) \
                                              + self.sentinel

        self.shdet_parameters['blmtr'] = np.zeros([self.nparam, self.nparam2]) \
                                         + self.sentinel

        if parameters is not None:
            for key, value in parameters.items():
                self.set_parameter(key, value)

    def set_parameter(self, key, value):
        for category in self.shdet_parameters.keys():
            if key in shdet_internal_parameter_positions[category]:
                self.shdet_parameters[category][
                    shdet_internal_parameter_positions[category][key]] = value

    def get_parameter_value(self, key):
        return_value = None
        for category in self.shdet_parameters.keys():
            if key in shdet_internal_parameter_positions[category]:
                return_value = self.shdet_parameters[category][
                    shdet_internal_parameter_positions[category][key]]

        return return_value

    def get_n(self):
        return self.n

    def get_nadc(self):
        return self.nadc

    def get_nparam(self):
        return self.nparam

    def get_sentinel(self):
        return self.sentinel

    def get_nparam2(self):
        return self.nparam2

    def simulate(self, signal, generate_parity=False, noise_seed=None,
                 optical_offset=None, raw_offset=None, adc_table= None):
        """
        Simulate the response of the HFI instrument to optical signal
        using SHDet.  returns raw modulated output signal, either in
        volts or ADC digits.

        signal is an array of sky samples in K_CMB (up to the maximum
            allowed by SHDet) that are used as input into the simulator.
        generate_parity: generate parity vector and return a tuple of
            signal, parity (default is False)
        noise_seed is an updated random number seed.  If None (default),
            optical_offset is an offset in K CMB that is added to the
            optical load
        raw_offset is an offset in digital units of science samples
            (0 to 2^16*40-1, or DSN units) that is added to all the terms
            of the raw constant

        """
        if len(signal) + 4 > l_rng_ttl:
            raise RuntimeError(
                'Signal argument to simulate cannot be greater '
                'than {} elements long, including pad.'.format(l_rng_ttl))

        # set the parameter containing the number of samples to the size
        # of the input signal first array element
        self.shdet_parameters['rnng'][50] = len(signal) + 4

        # reset noise seed if requested
        if noise_seed is not None:
            self.shdet_parameters['rnng'][37] = noise_seed

        # reset optical offset if requested (recalibrated in W)
        if optical_offset is not None:
            self.shdet_parameters['opt'][30] = optical_offset \
                                               * self.shdet_parameters['opt'][1]

        # reset raw offset if requested (recalibrated in volts at the fast
        # sample rate and removing the 'zero' level of the measured offsets)
        if raw_offset is not None:
            self.shdet_parameters['rnng'][38] = raw_offset/40*10.2/2**16 - 5.1

        signal = np.concatenate(
            (np.array([signal[0]]),
             signal,
             np.array([signal[-1], signal[-1], signal[-1]])))

        cdef double[:] adc_copy = np.arange(self.nadc, dtype=np.float)
        if adc_table is not None:
            adc_copy = adc_table.copy()

        cdef double[:] signal_ = np.ascontiguousarray(signal, dtype=np.float)
        cdef double[:] adc_ = np.ascontiguousarray(adc_copy, dtype=np.float)
        cdef double[:] param_rnng_ = np.ascontiguousarray(
            self.shdet_parameters['rnng'], dtype=np.float)
        cdef double[:] param_inst_ = np.ascontiguousarray(
            self.shdet_parameters['inst'], dtype=np.float)
        cdef double[:] param_rdout_ = np.ascontiguousarray(
            self.shdet_parameters['rdout'], dtype=np.float)
        cdef double[:] param_opt_ = np.ascontiguousarray(
            self.shdet_parameters['opt'], dtype=np.float)
        cdef double[:] param_blmtr_ = np.ascontiguousarray(
            self.shdet_parameters['blmtr'].ravel(), dtype=np.float)
        result = cSHDet_f(&signal_[0], &adc_[0], &param_rnng_[0],
                          &param_inst_[0], &param_rdout_[0], &param_opt_[0],
                          &param_blmtr_[0])

        if generate_parity:
            parity = np.int((-1) * np.arange(np.size(signal[:-4])))
            return signal[:-4], parity
        else:
            return signal[:-4]
      
    def measure_transfer_function(
            self, signal_DC=0.0, signal_amplitude=1.0e-5, debug=False,
            minimum_frequency=1e-2, maximum_frequency=1e2, n_frequencies=100,
            n_samp=900000, comm=None):
        """
        Measures the time response complex transfer function of the
        SHDet instance.

        Returns a 2D array: column 0 is frequency, column 1 is real
        response, and column 2 is imaginary response.

        The time response function is measured by feeding a series of
        tones (logarithmically spaced in frequency)
        into the the SHDet instance and using a lock-in to get the real
        and imaginary response.

        signal_DC: additional DC background to add in K_CMB (default is 0.0)
        signal_amplitude: amplitude of test signal (default is 1e-5 K_CMB)

        debug : show more output if set to True (default is False)

        minimum_frequency: minimum frequency to simulate in Hz (default is 1e-2 Hz)
        maximum_frequency: maximum frequency to simulate in Hz (default is 100 Hz)
        n_frequencies: number of frequencies to simulate (default is 100)
        n_samp: number of samples to simulate at each frequency step

        comm: optional MPI communicator that will allow embarassingly
        parallel splitting of the job

        """
        
        if comm is None:
            rank = 0
            ntask = 1
        else:
            ntask = comm.comm_world.size
            rank = comm.comm_world.rank

        # generate list of frequencies
        frequency_list = np.logspace(np.log10(minimum_frequency),
                                     np.log10(maximum_frequency),
                                     num=n_frequencies)

        # figure out which frequencies to do locally
        frequency_list_local = []
        ii = rank
        while ii < len(frequency_list):
            frequency_list_local.append(frequency_list[ii])
            ii += ntask

        # set up timelines
        fsamp = 180.37518
        tsamp = 1 / fsamp

        # make a time array
        isamp = np.arange(n_samp)
        t = isamp * tsamp

        igood = np.where(np.logical_and(t > 0.25*t[-1], t < t[-5]))

        # make a parity array
        parity = (-1) ** isamp

        # set up empty dictionaries to contain output transfer function
        results = {}

        # loop over frequencies
        for f in frequency_list_local:

            # generate sine and cosine carriers
            mod_sin = np.sin(2 * np.pi * f * t)
            mod_cos = np.cos(2 * np.pi * f * t)

            # run SHdet simulation
            sig = signal_amplitude * mod_cos + signal_DC

            sig = self.simulate(sig)

            # demodulate
            sig -= np.mean(sig)
            sig *= parity

            # do lock-in

            X = np.mean(mod_cos[igood] * sig[igood])
            Y = np.mean(mod_sin[igood] * sig[igood])

            results[f] = [X, Y]
            if debug:
                print("{} Hz: real: {} imag: {}".format(f, X, Y))

        # gather all results on to task 0
        if ntask > 1:
            comm.barrier()
            all_results = comm.gather(results, root = 0)
        else:
            all_results = [results]

        # sort these results
        tf_real = []
        tf_imag = []
        if rank == 0:
            for f in frequency_list:
                for result in all_results:
                    if f in result.keys():
                        tf_real.append(result[f][0])
                        tf_imag.append(result[f][1])
            tf_real = np.array(tf_real)
            tf_imag = np.array(tf_imag)

        if ntask > 1:
            tf_real = comm.bcast(tf_real, root=0)
            tf_imag = comm.bcast(tf_imag, root=0)

        return frequency_list, tf_real, tf_imag
