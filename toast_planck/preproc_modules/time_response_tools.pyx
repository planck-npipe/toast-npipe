# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
cimport numpy as np

import toast.timing as timing

DTYPE = np.double

cdef double fmod = 90.1875901876
cdef double fsamp = 2.0 * fmod


def kernel_four_tau(double f, np.ndarray par):
  ''' sum of four lowpass filters '''
  assert par.dtype == DTYPE
  w = np.complex(0.0,2.0*np.pi*f)
  return (par[0] / (1.0+w*par[3])
          + par[1] / (1.0+w*par[4])
          + par[2] / (1.0+w*par[5])
          + par[8] / (1.0+w*par[9]))


def lowpass(double omega, double tau):
  return 1 / np.complex(1.0, omega*tau)


def TFelect(np.ndarray f, np.ndarray par, tauhp=51.e3*1e-6, avg_srh=False):
  ''' HFI electronics transfer function '''
  assert par.dtype == DTYPE

  cdef double tau1 = 1.e3*100.e-9
  cdef double tau3 = 10.e3*10.e-9
  cdef double tau4 = tauhp
  cdef double zz3 = 510.e3
  cdef double zz4 = 1e-6
  cdef double zx1 = 18.7e3
  cdef double zx2 = 37.4e3
  cdef double fangmod = fmod * np.pi * 2
  cdef int nn = 5
  cdef double tau0 = par[6]
  cdef double sphase = par[7]
  cdef np.ndarray zout = np.ones(f.shape[0], dtype=complex)
  cdef complex norm = np.complex(1.0, 0.0)
  cdef int i,i1,signe
  cdef double ff, omega, omegamod, omegap, omegam
  cdef double zden3, zden2, zden1i, zden1r, arg
  cdef complex tf, zfelp, zfelm, zf1plu, zfimin, zSKplu, zSKmin, arg_2
  # <SRH
  cdef t_rw = 0
  if avg_srh:
     t_rw = 1.0 / 80 / fmod
  # SRH>

  for i in range(f.shape[0]):
      ff = f[i]
      omega = 2 * np.pi * ff
      tf =  0
      signe = -1
      for i1 in [1, 2, 3, 4, 5]:
        signe *= -1
        omegamod = (2.0*i1 - 1.0) * fangmod
        omegap = omega + omegamod
        omegam = - (omegamod - omega)

        # resonance low pass
        zfelp = lowpass(omegap,tau0)
        zfelm = lowpass(omegam,tau0)

        # electronic rejection filter
        zf1plu = np.complex(1.0, 0.5*omegap*tau1) / np.complex(1.0, omegap*tau1)
        zf1min = np.complex(1.0, 0.5*omegam*tau1) / np.complex(1.0, omegam*tau1)
        zfelp = zfelp * zf1plu
        zfelm = zfelm * zf1min

        # Sallen-Key high pass
        zSKplu = np.complex(0.0, tau4*omegap) / np.complex(1.0, omegap*tau4)
        zSKmin = np.complex(0.0, tau4*omegam) / np.complex(1.0, omegam*tau4)

        zfelp = zfelp * zSKplu * zSKplu
        zfelm = zfelm * zSKmin * zSKmin

        # sign reverse and gain
        zfelp *= -5.1
        zfelm *= -5.1

        # lowpass
        zfelp = zfelp * 1.5 *lowpass(omegap,tau3)
        zfelm = zfelm * 1.5 *lowpass(omegam,tau3)

        # third order equation

        zden3 = -1.0 * omegap*omegap*omegap * zx1*zx1*zz3*zx2*zx2*1.0e-16*zz4
        zden2 = -1.0 * omegap*omegap*(zx1*zx2*zx2*zz3*1.e-16
                                      + zx1*zx1*zx2*zx2*1.e-16
                                      + zx1*zx2*zx2*zz3*zz4*1.e-8)
        zden1i = omegap * (zx1*zx2*zx2*1.e-8+zx2*zz3*zx1*zz4) + zden3
        zden1r = zx2*zx1 + zden2

        zfelp = zfelp * np.complex(0.0, 2.0*zx2*zx1*zz3*zz4*omegap) \
                / np.complex(zden1r, zden1i)

        zden3 = -1.0 * omegam*omegam*omegam * zx1*zx1*zz3*zx2*zx2*1.0e-16*zz4
        zden2 = -1.0 * omegam*omegam*(zx1*zx2*zx2*zz3*1.e-16
                                      + zx1*zx1*zx2*zx2*1.e-16
                                      + zx1*zx2*zx2*zz3*zz4*1.e-8)
        zden1i = omegam * (zx1*zx2*zx2*1.e-8+zx2*zz3*zx1*zz4) + zden3
        zden1r = zx2*zx1 + zden2

        zfelm = zfelm * np.complex(0.0,2.0*zx2*zx1*zz3*zz4*omegam) \
                / np.complex(zden1r, zden1i)

        # averaging effect: original JH version
        #arg = np.pi * omegap / (2.0*fangmod)
        #zfelp = zfelp * (-1.0)*np.sin(arg)/arg
        #arg = np.pi * omegam / (2.0*fangmod)
        #zfelm = zfelm * (-1.0)*np.sin(arg)/arg

        # averaging effect: SRH version
        arg = np.pi * omegap / (2*fangmod)
        # <SRH
        if avg_srh:
           arg_2 = np.complex(np.cos(omegap*t_rw), np.sin(omegap*t_rw)) - 1
        else:
           arg_2 = arg
        zfelp = zfelp * (-1) * np.sin(arg) / arg_2
        # SRH>
        arg = np.pi * omegam / (2*fangmod)
        # <SRH
        if avg_srh:
           arg_2 = np.complex(np.cos(omegam*t_rw), np.sin(omegam*t_rw)) - 1
        else:
           arg_2 = arg
        zfelm = zfelm * (-1) * np.sin(arg) / arg_2

        zfelp = zfelp * np.complex(np.cos(sphase*omegap), np.sin(sphase*omegap))
        zfelm = zfelm * np.complex(np.cos(sphase*omegam), np.sin(sphase*omegam))
        tf = tf + (signe/(2.0*i1-1))*(zfelp+zfelm)
      if ff == 0:
        norm = tf
      zout[i] = tf / norm

  return zout


def LFER4(np.ndarray f, np.ndarray par):
  ''' LFER4 transfer function '''
  assert par.dtype == DTYPE

  cdef double tau1 = 1.e3*100.e-9
  cdef double tau3 = 10.e3*10.e-9
  cdef double tau4 = 51.e3*1.e-6
  cdef double zz3 = 510.e3
  cdef double zz4 = 1e-6
  cdef double zx1 = 18.7e3
  cdef double zx2 = 37.4e3
  cdef double fangmod = fmod * np.pi * 2
  cdef int nn = 5
  cdef double tau0 = par[6]
  cdef double sphase = par[7]
  cdef np.ndarray zout = np.ones(f.shape[0], dtype=complex)
  cdef complex norm = np.complex(1.0, 0.0)
  cdef int i, i1, signe
  cdef double ff, omega, omegamod, omegap, omegam, zden3
  cdef double zden2, zden1i, zden1r, arg
  cdef complex tf, zfelp, zfelm, zf1plu, zfimin, zSKplu, zSKmin, zbolo

  for i in range(f.shape[0]):
      ff = f[i]
      zbolo = kernel_four_tau(ff,par)
      omega = 2 * np.pi * ff
      tf =  0
      signe = -1
      for i1 in np.arange(1, nn+1):
        signe *= -1
        omegamod = (2.0*i1 - 1.0) * fangmod
        omegap = omega + omegamod
        omegam = - (omegamod - omega)

        # resonance low pass
        zfelp = lowpass(omegap, tau0)
        zfelm = lowpass(omegam, tau0)

        # electronic rejection filter
        zf1plu = np.complex(1.0, 0.5*omegap*tau1) / np.complex(1.0, omegap*tau1)
        zf1min = np.complex(1.0, 0.5*omegam*tau1) / np.complex(1.0, omegam*tau1)
        zfelp = zfelp * zf1plu
        zfelm = zfelm * zf1min

        # Sallen-Key high pass
        zSKplu = np.complex(0.0, tau4*omegap) / np.complex(1.0, omegap*tau4)
        zSKmin = np.complex(0.0, tau4*omegam) / np.complex(1.0, omegam*tau4)

        zfelp = zfelp * zSKplu * zSKplu
        zfelm = zfelm * zSKmin * zSKmin

        # sign reverse and gain
        zfelp *= -5.1
        zfelm *= -5.1

        # lowpass
        zfelp = zfelp * 1.5 * lowpass(omegap, tau3)
        zfelm = zfelm * 1.5 * lowpass(omegam, tau3)

        # third order equation

        zden3 = -1.0 * omegap**3 * zx1*zx1*zz3*zx2*zx2*1.0e-16*zz4
        zden2 = -1.0 * omegap*omegap*(zx1*zx2*zx2*zz3*1.e-16
                                      + zx1*zx1*zx2*zx2*1.e-16
                                      + zx1*zx2*zx2*zz3*zz4*1.e-8)
        zden1i = omegap*(zx1*zx2*zx2*1.e-8+zx2*zz3*zx1*zz4) + zden3
        zden1r = zx2*zx1 + zden2

        zfelp = zfelp * np.complex(0.0, 2.0*zx2*zx1*zz3*zz4*omegap) \
                / np.complex(zden1r, zden1i)

        zden3 = -1.0 * omegam**3 * zx1*zx1*zz3*zx2*zx2*1.0e-16*zz4
        zden2 = -1.0 * omegam*omegam*(zx1*zx2*zx2*zz3*1.e-16
                                      + zx1*zx1*zx2*zx2*1.e-16
                                      + zx1*zx2*zx2*zz3*zz4*1.e-8)
        zden1i = omegam*(zx1*zx2*zx2*1.e-8+zx2*zz3*zx1*zz4) + zden3
        zden1r = zx2*zx1 + zden2

        zfelm = zfelm*np.complex(0.0, 2.0*zx2*zx1*zz3*zz4*omegam) \
                / np.complex(zden1r, zden1i)

        # averaging effect
        arg = np.pi * omegap / (2*fangmod)
        zfelp = zfelp * (-1.0) * np.sin(arg) / arg
        arg = np.pi * omegam / (2*fangmod)
        zfelm = zfelm * (-1.0) * np.sin(arg) / arg

        zfelp = zfelp * np.complex(np.cos(sphase*omegap), np.sin(sphase*omegap))
        zfelm = zfelm * np.complex(np.cos(sphase*omegam), np.sin(sphase*omegam))
        tf = tf + (signe/(2.0*i1-1))*(zfelp + zfelm)
      if ff == 0:
        norm = tf
      zout[i] = tf*zbolo/norm

  return zout


def filter_function(np.ndarray f):
   ''' for input frequency array f,
   return the standard cos-Gauss lowpass filter function
   '''
   assert f.dtype == DTYPE
   cdef double fc = 80
   cdef double factor = 0.9
   cdef double gauss_f = 65

   cdef double fmax = fc + factor*(fsamp/2-fc)
   cdef np.ndarray kernel = np.ones(f.shape[0])
   cdef int i = 0
   cdef double ftest
   for i in range(f.shape[0]):
       ftest = np.abs(f[i])
       if ftest >= fmax:
          kernel[i] = 0
       if (ftest>fc) and (ftest<fmax):
          x = (ftest-fc) / (fmax-fc)
          kernel[i] = (np.cos(np.pi/2 * x))**2
       kernel[i] *= np.exp(-0.5 * (f[i]/gauss_f)**2)
   return kernel


def LFERn(np.ndarray f, int n,dict pars,avg_srh=False):
    ''' This function builds the transfer function for n single-pole
    thermal time constants and the HFI time response function.
    '''
    cdef int i, j
    cdef double a, tau
    # fake some parameters
    par = np.zeros(8)
    par[7] = pars['Sphase']
    par[6] = pars['tau_stray']
    if 'tauhp1' in pars.keys():
       kelect = TFelect(f, par, tauhp=pars['tauhp1'], avg_srh=avg_srh)
    else:
       kelect = TFelect(f,par,avg_srh=avg_srh)
    kbolo = np.zeros(f.shape[0],dtype=complex)

    for j in range(n):
       a = pars['a%d'%(j+1)]
       tau = pars['tau%d'%(j+1)]
       for i in range(f.shape[0]):
          ff = f[i]
          kbolo[i] += a * lowpass(2*np.pi*ff, tau)

    return kelect * kbolo
