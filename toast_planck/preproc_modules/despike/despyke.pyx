# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import toast.timing as timing

cimport numpy as np

cdef extern from "despike_func.h":
    cdef int GLITCHINFO_NTERM
    ctypedef struct GlitchInfoStruct:
        double * longtau
        double * longamp
        double * shorttau
        double * shortamp
        double * slowtau
        double * slowamp

    ctypedef struct DespikeStruct:
        double critcut
        double factth
        double ppa
        double ppb
        double selcrit
        double crit
        double critrem
        double selsnail
        long ringsize
        long long * xcosm
        long n_event_expected
        double * amplspike
        double * ampltemplate
        double * ampltemplateSH
        double * allhsig
        double * allchi2

    DespikeStruct * alloc_despike_struct(long n_event_expected,
                                         long ringsize,
                                         GlitchInfoStruct glitchinfo,
                                         long kernel_size,
                                         double * kernel,
                                         bint do_snail)
    void free_despike_struct(DespikeStruct *)
    bint despike_func(DespikeStruct * dsp,
                      long ring_number,
                      long signal_size,
                      #  signal_size
                      double * in_signal,
                      # signal_size
                      unsigned char * zeroflg,
                      unsigned char * out_flag,
                      # signal_size
                      double * out_residual,
                      # signal_size
                      double * out_glitch,
                      int verbose
                      )

# Default values for Despike for bolometers
DESPIKE_BOLO_PARAMS = {
    # pixname ppa ppb cutcl snailrm
    "100-1a": [               0,            0.7,   0.50, True],
    "100-1b": [   4.2565296e-07,     0.92304962,   0.50, False],
    "100-2a": [    0.0001469912,     0.85131524,   0.50, True],
    "100-2b": [   3.1265773e-05,      0.8887139,   0.80, False],
    "100-3a": [   0.00013752925,     0.88681273,   0.50, True],
    "100-3b": [   5.3438519e-05,     0.88682691,   0.60, False],
    "100-4a": [         0.00014,            0.8,   0.75, True],
    "100-4b": [      3.3142e-05,         0.5012,   0.75, False],
    "143-1a": [     0.000135117,              1,   0.45, True],
    "143-1b": [   8.2008007e-05,      0.8614482,   0.55, False],
    "143-2a": [   0.00013511677,     0.79599222,   0.50, True],
    "143-2b": [           7e-05,           0.74,   0.60, False],
    "143-3a": [   0.00014902236,     0.78729126,   0.50, True],
    "143-3b": [    4.459689e-05,     0.89404228,   0.50, False],
    "143-4a": [    0.0001230547,     0.75566726,   0.40, True],
    "143-4b": [   5.5618534e-05,     0.86602321,   0.60, False],
    "143-5": [           5e-05,            0.9,   0.80, False],
    "143-6": [   5.5999346e-05,     0.84994839,   0.50, False],
    "143-7": [   6.2898673e-05,     0.85406074,   0.50, False],
    "143-8": [               0,              1,   0.50, False],
    "217-1": [   5.2847104e-05,      0.8738982,   0.50, False],
    "217-2" : [   7.3277026e-05,     0.86908392,   0.55, False],
    "217-3" : [   0.00010447213,     0.81192593,   0.60, False],
    "217-4" : [   7.2621928e-05,     0.86441814,   0.50, False],
    "217-5a": [   0.00013596426,     0.79010597,   0.50, True],
    "217-5b": [   3.5221842e-05,     0.91090116,   0.50, False],
    "217-6a": [   0.00019069927,     0.74162541,   0.55, True],
    "217-6b": [   8.4709273e-05,     0.81678555,   0.65, False],
    "217-7a": [         0.00022,           0.66,   0.50, True],
    "217-7b": [   6.0806958e-05,     0.86657709,   0.55, False],
    "217-8a": [   0.00010746567,     0.81759747,   0.45, True],
    "217-8b": [   4.0905752e-05,     0.88760745,   0.50, False],
    "353-1" : [   0.00010381128,     0.83445838,   0.60, False],
    "353-2" : [     8.10697e-05,     0.86996625,   0.55, False],
    "353-3a": [   0.00013065466,     0.79387245,   0.50, True],
    "353-3b": [   5.6199435e-05,     0.86920244,   0.60, False],
    "353-4a": [   0.00011643114,     0.78688226,   0.45, True],
    "353-4b": [           2e-05,           0.84,   0.60, False],
    "353-5a": [    0.0001466299,     0.81172709,   0.45, True],
    "353-5b": [               0,              1,   0.50, False],
    "353-6a": [   8.1571858e-05,     0.79066092,   0.40, True],
    "353-6b": [   2.5197166e-05,     0.94681598,   0.50, False],
    "353-7" : [   4.7133373e-05,     0.89023969,   0.50, False],
    "353-8" : [   6.1264473e-05,     0.88599897,   0.50, False],
    "545-1" : [   5.2798643e-06,     0.88171833,   0.50, False],
    "545-2" : [   4.9397096e-06,     0.93916627,   0.50, False],
    "545-3" : [               0,     0.88549605,   0.50, False],
    "545-4" : [   2.5185109e-06,      0.9490894,   0.50, False],
    "857-1" : [   6.3737453e-06,     0.92701424,   0.50, False],
    "857-2" : [   2.2300093e-06,     0.95197433,   0.50, False],
    "857-3" : [   8.6793087e-06,     0.91084806,   0.55, False],
    "857-4" : [   2.4097572e-05,      0.8203073,   0.60, False],
    "Dark1" : [   3.1306768e-05,     0.94351875,   0.50, False],
    "Dark2" : [   2.9383944e-05,     0.92420468,   0.50, False],
    "Dark-1" : [   3.1306768e-05,     0.94351875,   0.50, False],
    "Dark-2" : [   2.9383944e-05,     0.92420468,   0.50, False],
    "default": [               0,              1,   0.50, False]}

GLITCHINFOPATH = ("IMO:HFI:DET:Phot_Pixel Name='{boloid}':"
                  "NoiseAndSyst:NoiseTechData:ParticlesEffect:"
                  "{gtype}GlitchPattern:{info}%d")
GLITCHTYPES = ('Long', 'Short', 'Slow')
GLITCHINFOTYPES = ('Amplitude', 'Tau')


def getglitchinfo(imo, gtype, info, boloid, infopath=GLITCHINFOPATH):
    path = infopath.format(gtype=gtype, info=info, boloid=boloid)
    ginfo = []
    i = 1
    while True:
        pathi = path % i
        try:
            ginfo.append(imo.get(path % i, np.float64))
            i += 1
        except Exception:
            break
    return ginfo


def init_glitch_info(imo, boloid, acq_freq=1):

    glitch_info = GlitchInfo(
        getglitchinfo(imo, "Long", "Tau", boloid),
        getglitchinfo(imo, "Long", "Amplitude", boloid),
        getglitchinfo(imo, "Short", "Tau", boloid),
        getglitchinfo(imo, "Short", "Amplitude", boloid),
        getglitchinfo(imo, "Slow", "Tau", boloid),
        getglitchinfo(imo, "Slow", "Amplitude", boloid)
    )
    glitch_info.longtau *= acq_freq
    glitch_info.shorttau *= acq_freq
    glitch_info.slowtau *= acq_freq
    return glitch_info


cdef copy_to_array(alist, double * arr):
    cdef int i
    for i in range(min(GLITCHINFO_NTERM, len(alist))):
        arr[i] = alist[i]
    for i in range(min(GLITCHINFO_NTERM, len(alist)), GLITCHINFO_NTERM):
        arr[i] = 0.0


cdef list_from_array(double * arr):
    cdef int i
    return np.asarray([arr[i] for i in range(GLITCHINFO_NTERM)])


cdef class GlitchInfo:
    cdef GlitchInfoStruct glitch_info_struct

    def __init__(self, longtau=[], longamp=[], shorttau=[],
                 shortamp=[], slowtau=[], slowamp=[]):
        self.longtau = longtau
        self.longamp = longamp
        self.shorttau = shorttau
        self.shortamp = shortamp
        self.slowtau = slowtau
        self.slowamp = slowamp

    property longtau:
        def __get__(self):
            return list_from_array(self.glitch_info_struct.longtau)

        def __set__(self, alist):
            copy_to_array(alist, self.glitch_info_struct.longtau)

    property shorttau:
        def __get__(self):
            return list_from_array(self.glitch_info_struct.shorttau)

        def __set__(self, alist):
            copy_to_array(alist, self.glitch_info_struct.shorttau)

    property slowtau:
        def __get__(self):
            return list_from_array(self.glitch_info_struct.slowtau)

        def __set__(self, alist):
            copy_to_array(alist, self.glitch_info_struct.slowtau)

    property longamp:
        def __get__(self):
            return list_from_array(self.glitch_info_struct.longamp)

        def __set__(self, alist):
            copy_to_array(alist, self.glitch_info_struct.longamp)

    property shortamp:
        def __get__(self):
            return list_from_array(self.glitch_info_struct.shortamp)

        def __set__(self, alist):
            copy_to_array(alist, self.glitch_info_struct.shortamp)

    property slowamp:
        def __get__(self):
            return list_from_array(self.glitch_info_struct.slowamp)

        def __set__(self, alist):
            copy_to_array(alist, self.glitch_info_struct.slowamp)

    def __str__(self):
        res = []
        res.append('longtau=%s' % (self.longtau))
        res.append('longamp=%s' % (self.longamp))
        res.append('shorttau=%s' % (self.shorttau))
        res.append('shortamp=%s' % (self.shortamp))
        res.append('slowtau=%s' % (self.slowtau))
        res.append('slowamp=%s' % (self.slowamp))
        return 'GlitchInfo(' + (',\n' + ' ' * 11).join(res) + ')'


cdef class Despiker:
    cdef DespikeStruct * despike_struct
    cdef int verbose

    def __cinit__(self, imo, bolometer, boloid, n_event_expected=1000000,
                  selcrit=None, factth=None, critcut=None, verbose=False):

        cdef GlitchInfo glitchinfo = init_glitch_info(imo, boloid,
                                                      acq_freq=180.374)

        ppa, ppb, cutcl, do_snail = DESPIKE_BOLO_PARAMS[bolometer]

        kernel = np.array([.25, .50, .25])

        ringsize = 14345

        cdef double[:] kernel_ = np.ascontiguousarray(kernel, dtype=float)

        self.despike_struct = alloc_despike_struct(
            n_event_expected, ringsize, glitchinfo.glitch_info_struct,
            kernel_.size, &kernel_[0], do_snail)
        self.ppa = ppa
        self.ppb = ppb
        self.selcrit = selcrit
        self.factth = factth
        self.critcut = critcut
        self.verbose = verbose

    def __dealloc__(self):
        free_despike_struct(self.despike_struct)

    property critcut:
        def __get__(self):
            return self.despike_struct.critcut

        def __set__(self, value):
            if value is not None:
                self.despike_struct.critcut = value

    property factth:
        def __get__(self):
          return self.despike_struct.factth

        def __set__(self, value):
            if value is not None:
                self.despike_struct.factth = value

    property ppa:
        def __get__(self):
            return self.despike_struct.ppa

        def __set__(self, value):
            if value is not None:
                self.despike_struct.ppa = value

    property ppb:
        def __get__(self):
            return self.despike_struct.ppb

        def __set__(self, value):
            if value is not None:
                self.despike_struct.ppb = value

    property selcrit:
        def __get__(self):
            return self.despike_struct.selcrit

        def __set__(self, value):
            if value is not None:
                self.despike_struct.selcrit = value

    property crit:
        def __get__(self):
            return self.despike_struct.crit

        def __set__(self, value):
            if value is not None:
                self.despike_struct.crit = value

    property critrem:
        def __get__(self):
            return self.despike_struct.critrem

        def __set__(self, value):
            if value is not None:
                self.despike_struct.critrem = value

    property selsnail:
        def __get__(self):
            return self.despike_struct.selsnail

        def __set__(self, value):
            if value is not None:
                self.despike_struct.selsnail = value

    property xcosm:
        def __get__(self):
            return np.array(<long long[:self.despike_struct.n_event_expected]>
                            self.despike_struct.xcosm, copy=True)

    property amplspike:
        def __get__(self):
            return np.array(<double[:self.despike_struct.n_event_expected]>
                            self.despike_struct.amplspike, copy=True)

    property ampltemplate:
        def __get__(self):
            return np.array(<double[:self.despike_struct.n_event_expected]>
                            self.despike_struct.ampltemplate, copy=True)

    property ampltemplateSH:
        def __get__(self):
            return np.array(<double[:self.despike_struct.n_event_expected]>
                            self.despike_struct.ampltemplateSH, copy=True)

    property allhsig:
        def __get__(self):
            return np.array(<double[:self.despike_struct.n_event_expected]>
                            self.despike_struct.allhsig, copy=True)

    property allchi2:
        def __get__(self):
            return np.array(<double[:self.despike_struct.n_event_expected]>
                                    self.despike_struct.allchi2, copy=True)

    def despike(self, ring_number, signal, zeroflg, start_sample=0):
        """Remove spikes from given signal.
        Parameters:
        -----------
        - ring_number: the DPC ring number
        - signal: the signal to despike (TOI)
        - zeroflg: the unavailable data flag
        - start_sample: the sample number of the first sample in the ring
        """
        nan = np.isnan(signal)
        nnan = np.sum(nan)
        if nnan != 0:
            raise RuntimeError(
                'Ring {:4} : signal passed to despike contains {} NaNs'
                ''.format(ring_number, nnan))

        #cdef double[:] signal_ = np.ascontiguousarray(signal, dtype=float)
        cdef double[:] signal_ = np.ascontiguousarray(
            np.convolve(signal, [.25, .50, .25], mode='same'), dtype=float)
        cdef np.uint8_t[:] zeroflg_ = np.ascontiguousarray(zeroflg, dtype=np.uint8)

        # Check the input sizes
        signal_size = signal_.size
        # print 'signal_size=%d' % signal_size

        if zeroflg_.size != signal_size:
            raise ValueError('Wrong size: zeroflg (%d, expected %d)' %
                             (zeroflg_.size, signal_size))

        cdef np.ndarray[np.uint8_t, ndim=1] out_flag = np.zeros(
            signal_size, dtype=np.uint8)
        cdef np.ndarray[np.float64_t, ndim=1] out_residual = np.zeros(
            signal_size, dtype=float)
        cdef np.ndarray[np.float64_t, ndim=1] out_glitch = np.zeros(
            signal_size, dtype=float)

        result = despike_func(self.despike_struct,
                              ring_number,
                              signal_size,
                              &signal_[0],
                              &zeroflg_[0],
                              &out_flag[0],
                              &out_residual[0],
                              &out_glitch[0],
                              self.verbose)
        if result or np.all(out_flag != 0):  # nothing_to_do
            return None, None

        return signal - out_glitch, out_flag
