#include <string.h>
#include <cstdio>
#include <cstdlib>
#include "todprocess_planck.h"
#include "despike_func.h"

DespikeStruct *alloc_despike_struct(const long n_event_expected,
    const long ringsize, GlitchInfoStruct glitchinfo, const long kernel_size,
    const double *kernel, /* kernel_size */
    bool do_snail) {
  DespikeStruct *self;

  self = new DespikeStruct;
  if (!self) {
    return NULL;
  }

  //(set 1.5 for 217 and above, set 2 for 143 GHz, set 3.0 for 100GHz)
  self->critcut = 3.2;
  // threshold of match filter with respect to critcut and noise sig
  self->crit = 1.0 / 13.92;
  self->critrem = 4;
  self->factth = 1.0; // default for 143 GHz
  // 0.5 for V33 and V34   // 0.25 is better for 143_2a (if no ppa and ppb)
  self->selcrit = 0.5;
  self->selsnail = 4;

  self->ppa = 0;
  self->ppb = 1;

  self->n_event_expected = n_event_expected;
  self->ringsize = ringsize;

  self->kernel_size = kernel_size;
  self->kernel = new double[kernel_size];
  for (long i = 0; i < kernel_size; i++) {
    self->kernel[i] = kernel[i];
  }

  self->ampl2cr = new double[GLITCHINFO_NTERM * n_event_expected];
  self->ampl2crtmp = new double[GLITCHINFO_NTERM * n_event_expected];
  self->xerror = new int[n_event_expected];
  self->pos = new int[n_event_expected];
  self->xcosm = new long long[n_event_expected];
  self->xcosmfit = new int[n_event_expected];
  self->xspike = new int[n_event_expected];
  self->chisqs = new double[n_event_expected];
  self->amplspike = new double[n_event_expected];
  self->allhsig = new double[n_event_expected];
  self->allchi2 = new double[n_event_expected];
  self->posp = new int[n_event_expected];
  self->amplp = new double[n_event_expected];
  self->ampltemplate = new double[n_event_expected];
  self->ampltemplateSH = new double[n_event_expected];
  self->allhsigp = new double[n_event_expected];

  self->pos2 = new int[n_event_expected];
  self->xcosm2 = new long long[n_event_expected];
  self->xspike2 = new int[n_event_expected];
  self->chisqs2 = new double[n_event_expected];
  self->amplspike2 = new double[n_event_expected];
  self->allhsig2 = new double[n_event_expected];
  self->posp2 = new int[n_event_expected];
  self->amplp2 = new double[n_event_expected];
  self->allhsigp2 = new double[n_event_expected];

  self->glitchinfo = glitchinfo;
  self->template_size = 5000;

  self->do_snail = do_snail;
  if (do_snail) {
    self->number_of_templates = 3;
    self->snailtempl = 1;
  } else {
    self->number_of_templates = 2;
    self->snailtempl = 0;
  }
  self->ntempl = self->number_of_templates; // Useful shorthand
  self->templg = new double[self->template_size * self->ntempl];

  for (long ii = 0; ii < self->template_size; ii++) {
    for (long jj = 0; jj < self->number_of_templates; jj++) {
      self->templg[ii * self->ntempl + jj] = 0;
    }
  }

  for (long ii = 0; ii < GLITCHINFO_NTERM; ii++) {
    for (long jj = 0; jj < self->template_size; jj++) {
      if (glitchinfo.longtau[ii] > 0) {
        self->templg[jj * self->ntempl + 0] += glitchinfo.longamp[ii]
            * exp(-jj / glitchinfo.longtau[ii]);
      }
      if ((self->number_of_templates >= 2) && (glitchinfo.shorttau[ii] > 0)) {
        self->templg[jj * self->ntempl + 1] += glitchinfo.shortamp[ii]
            * exp(-jj / glitchinfo.shorttau[ii]);
      }
      if (self->snailtempl) {
        self->templg[jj * self->ntempl + 2] += glitchinfo.slowamp[ii]
            * exp(-jj / glitchinfo.slowtau[ii]);
      }
    }
  }

  return self;
}

void free_despike_struct(DespikeStruct *dsp) {
  if (dsp == NULL) {
    return;
  }
  delete[] dsp->ampl2cr;
  delete[] dsp->ampl2crtmp;
  delete[] dsp->templg;

  delete[] dsp->kernel;

  delete[] dsp->xerror;
  delete[] dsp->pos;
  delete[] dsp->xcosm;
  delete[] dsp->xcosmfit;
  delete[] dsp->xspike;
  delete[] dsp->chisqs;
  delete[] dsp->amplspike;
  delete[] dsp->allhsig;
  delete[] dsp->allchi2;
  delete[] dsp->posp;
  delete[] dsp->amplp;
  delete[] dsp->ampltemplate;
  delete[] dsp->ampltemplateSH;
  delete[] dsp->allhsigp;

  delete[] dsp->pos2;
  delete[] dsp->xcosm2;
  delete[] dsp->xspike2;
  delete[] dsp->chisqs2;
  delete[] dsp->amplspike2;
  delete[] dsp->allhsig2;
  delete[] dsp->posp2;
  delete[] dsp->amplp2;
  delete[] dsp->allhsigp2;

  delete dsp;
}

void print_despike_struct(DespikeStruct *dsp) {
  fprintf(stderr,
      "critcut=%lg   factth=%lg   ppa=%lg   ppb=%lg\n",
      dsp->critcut, dsp->factth, dsp->ppa, dsp->ppb);
  fprintf(stderr, "selcrit=%lg   crit=%lg   critrem=%lg   selsnail=%lg\n",
      dsp->selcrit, dsp->factth, dsp->critrem, dsp->selsnail);
  fprintf(stderr, "n_event_expected=%ld   ringsize=%ld   "
      "do_snail=%d\n", dsp->n_event_expected, dsp->ringsize,
      (int) dsp->do_snail);
  print_glitchinfo_struct(&dsp->glitchinfo);
}

void print_glitchinfo_struct(GlitchInfoStruct *glitchinfo) {
  int i;

  fprintf(stderr, "LongTau = [");
  for (i = 0; i < GLITCHINFO_NTERM; i++) {
    fprintf(stderr, "%lg", glitchinfo->longtau[i]);
    if (i < GLITCHINFO_NTERM - 1) {
      fprintf(stderr, ", ");
    } else {
      fprintf(stderr, "]\n");
    }
  }
  fprintf(stderr, "LongAmp = [");
  for (i = 0; i < GLITCHINFO_NTERM; i++) {
    fprintf(stderr, "%lg", glitchinfo->longamp[i]);
    if (i < GLITCHINFO_NTERM - 1) {
      fprintf(stderr, ", ");
    } else {
      fprintf(stderr, "]\n");
    }
  }
  fprintf(stderr, "ShortTau = [");
  for (i = 0; i < GLITCHINFO_NTERM; i++) {
    fprintf(stderr, "%lg", glitchinfo->shorttau[i]);
    if (i < GLITCHINFO_NTERM - 1) {
      fprintf(stderr, ", ");
    } else {
      fprintf(stderr, "]\n");
    }
  }
  fprintf(stderr, "ShortAmp = [");
  for (i = 0; i < GLITCHINFO_NTERM; i++) {
    fprintf(stderr, "%lg", glitchinfo->shortamp[i]);
    if (i < GLITCHINFO_NTERM - 1) {
      fprintf(stderr, ", ");
    } else {
      fprintf(stderr, "]\n");
    }
  }
  fprintf(stderr, "SlowTau = [");
  for (i = 0; i < GLITCHINFO_NTERM; i++) {
    fprintf(stderr, "%lg", glitchinfo->slowtau[i]);
    if (i < GLITCHINFO_NTERM - 1) {
      fprintf(stderr, ", ");
    } else {
      fprintf(stderr, "]\n");
    }
  }
  fprintf(stderr, "SlowAmp = [");
  for (i = 0; i < GLITCHINFO_NTERM; i++) {
    fprintf(stderr, "%lg", glitchinfo->slowamp[i]);
    if (i < GLITCHINFO_NTERM - 1) {
      fprintf(stderr, ", ");
    } else {
      fprintf(stderr, "]\n");
    }
  }
}

bool despike_func(DespikeStruct *dsp, const long ring_number,
    const long signal_size,
    double *in_signal, /* signal_size */
    unsigned char *masked, /* signal_size */
    unsigned char *out_flag, /* signal size */
    double *out_residual, /* signal_size */
    double *out_glitch, /* signal_size*/
    int verbose) {

  double t_tail = 0, t_fillgaps = 0, t_butterworth = 0, t_findspike_clean = 0,
      t_remove_longCR = 0, t_fft = 0, t_remove_poly = 0;

  VERBOSE = verbose;

  long iter = 0;

  unsigned char *flag2 = new unsigned char[signal_size];
  unsigned char *pflag = new unsigned char[signal_size];
  double *esigdet = new double[signal_size];
  double *ecritcut_mf = new double[signal_size];

  double *bfilter = new double[signal_size];
  double *dtest = new double[signal_size];
  double *data_out = new double[signal_size];
  double *datadesp = new double[signal_size];

  double *data = out_residual; /* used as working buffer */
  double *datatailmod = out_glitch; /* used as working buffer */

  double f_lppix;
  double mhsig = 0;

  int rrunn = 0;

  ticker time_start = tic();
  double time_elapsed = 0;

  for (long ii = 0; ii < signal_size; ii++) {
    if (masked[ii] == 0) {
      rrunn = 1;
    }
  }

  bool nothing_to_do = false;
  if (rrunn == 0) {
    if (VERBOSE) {
      fprintf(stderr, "NO VALID DATA, GIVING UP FOR THIS RING\n");
    }
    nothing_to_do = true;
  }

  if (nothing_to_do) {
    return nothing_to_do;
  }

  if (VERBOSE) {
    fprintf(stderr, "===*** DEBUG ***===\n");
    print_despike_struct(dsp); // DEBUG only
    double sum_sig = 0;
    for (long i = 0; i < signal_size; i++) {
      sum_sig += in_signal[i];
    }
    fprintf(stderr, "sum(in_signal)=%.17lg   signal_size=%ld", sum_sig,
        signal_size);
    fprintf(stderr, "------fin debug----\n");
  }

  /*==============================================
   Remove data masked
   ==============================================*/

  long tau3 = 1.2 * 180.18;

  long testnan = 0;

  for (long i = 0; i < signal_size; i++) {
    if (masked[i] == 1) {
      in_signal[i] = 0;
    } else if (!(in_signal[i] < 1e100) || !(in_signal[i] > -1e100)) {
      in_signal[i] = 0;
      masked[i] = 1;
      testnan++;
    }
  }

  if (testnan && VERBOSE) {
    fprintf(stderr, "%ld, unexpected NAN in input data "
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", testnan);
  }

  // Initialize random number

  ticker tck;

  tck = tic();

  if (VERBOSE) {
    fprintf(stderr, "Filling gaps\n");
  }

  fillgaps(in_signal, signal_size, data, masked, 0, 0);

  t_fillgaps += toc(tck);

  memcpy(in_signal, data, (signal_size) * sizeof(double));

  double sign0 = -1;

  /*===============================
   DEBUT ALGO
   ===============================*/

  long itermax = 4;
  long itt;
  int terr;
  long iic, alpha;
  int errr, nokl;
  long orderpoly = 5;
  double fsamp = 180.3737;
  double f_lp = 1.0 / 30.0;
  long napod = 2000;

  int nball = 0;

  // initialize random number generator with seed from ring_number and signal
  // (so it will depend on bolometer...)
  {
    double mean = 0;
    double maximum = fabs(in_signal[0]);
    long n = 0;
    for (long i = 0; i < signal_size; i++) {
      mean += fabs(in_signal[i]);
      if (fabs(in_signal[i]) > maximum)
        maximum = fabs(in_signal[i]);
      n++;
    }
    mean /= n;
    unsigned long seed = (unsigned long) ((maximum / mean) * 1e9) + ring_number;
    srand(seed);
    if (VERBOSE) {
      fprintf(stderr, "Setting seed (ring %d): %lu\n", int(ring_number), seed);
    }
  }

  for (itt = 0; itt < 2; itt++) {

    nball++;

    nokl = 1;

    if (itt > 0) {
      itermax = 9;
      nokl = terr;
    } else {
      itermax = 4;
    }

    if (nokl) {
      iter = 0;
      errr = 0;

      if (VERBOSE) {
        fprintf(stderr, "STARTING RING NB %ld\n", ring_number);
        fprintf(stderr, "nball = %d\n", nball);
      }

      f_lppix = f_lp / fsamp * signal_size;

      for (long ii = 0; ii < signal_size; ii++) {
        esigdet[ii] = 0;
      }

      /// Main loop over iterations
      while (iter < itermax && errr != 2) {

        iter = itermax - 1;
        for (long ii = 0; ii < signal_size; ii++) {
          datadesp[ii] = in_signal[ii];
        }

        //(re)set flag to 0;
        for (long ii = 0; ii < signal_size; ii++) {
          pflag[ii] = 0;
          out_flag[ii] = 0;
          if (masked[ii]) {
            out_flag[ii] = 1;
          }
          flag2[ii] = 0;
          datatailmod[ii] = 0;
        }

        alpha = itermax - iter - 2;
        if (alpha < 0) {
          alpha = 0;
        }

        //finspike
        if (VERBOSE) {
          long count = 0;
          for (long i = 0; i < signal_size; i++) {
            if (out_flag[i]) {
              count++;
            }
          }
          fprintf(stderr, "before findspike_clean: count(flag)=%ld "
              "signal_size=%ld", count, signal_size);
        }
        tck = tic();
        findspike_clean(datadesp, signal_size, datatailmod, out_flag, pflag,
            dsp->xcosm, dsp->pos, dsp->xspike, dsp->amplspike, dsp->allhsig,
            dsp->allchi2, dsp->posp, dsp->amplp, dsp->allhsigp,
            dsp->critcut * pow(1.7, alpha), sign0, esigdet, dsp->kernel,
            dsp->kernel_size, dsp->glitchinfo.longtau, dsp->glitchinfo.longamp,
            dsp->glitchinfo.shorttau, dsp->glitchinfo.shortamp,
            dsp->glitchinfo.slowtau, dsp->glitchinfo.slowamp,
            GLITCHINFO_NTERM, dsp->n_event_expected, dsp->ampl2cr, dsp->templg,
            dsp->number_of_templates, dsp->template_size, dsp->critrem,
            dsp->ampltemplate, dsp->ampltemplateSH, dsp->selcrit, dsp->selsnail,
            dsp->ppa, dsp->ppb, dsp->do_snail, dsp->snailtempl);
        t_findspike_clean += toc(tck);
        if (VERBOSE) {
          long count = 0;
          for (long i = 0; i < signal_size; i++) {
            if (out_flag[i]) {
              count++;
            }
          }
          fprintf(stderr, "after findspike_clean: count(flag)=%ld "
              "signal_size=%ld\n", count, signal_size);
        }

        iic = 0;
        mhsig = 0;
        while (dsp->xcosm[iic] > 0) {
          mhsig += dsp->allhsig[iic];
          iic++;
        }
        mhsig /= double(iic);
        sign0 = mhsig;
        if (VERBOSE) {
          fprintf(stderr, "sign0=%.17g, iic=%ld\n", sign0, (long) iic);
        }

        for (long ii = 0; ii < signal_size; ii++) {
          datadesp[ii] = datadesp[ii] - datatailmod[ii];
        }

        /// cut spikes from data and fill with line
        tck = tic();
        if (VERBOSE) {
          fprintf(stderr, "filling gaps\n");
        }
        fillgaps(datadesp, signal_size, data, out_flag, mhsig * 0.8, 0);
        t_fillgaps += toc(tck);

        /// second iter despike on map despiker
        for (long ii = 0; ii < signal_size; ii++) {
          datadesp[ii] = data[ii];
        }

        // findspike second iter

        for (long ii = 0; ii < signal_size; ii++) {
          flag2[ii] = out_flag[ii];
        }

        tck = tic();
        if (VERBOSE) {
          fprintf(stderr, "calling findspike_clean\n");
        }
        findspike_clean(datadesp, signal_size, data, flag2, pflag, dsp->xcosm2,
            dsp->pos2, dsp->xspike2, dsp->amplspike2, dsp->allhsig2,
            dsp->chisqs2, dsp->posp2, dsp->amplp2, dsp->allhsigp2,
            dsp->critcut * pow(1.7, alpha) * 1.1, sign0, esigdet, dsp->kernel,
            dsp->kernel_size, dsp->glitchinfo.longtau, dsp->glitchinfo.longamp,
            dsp->glitchinfo.shorttau, dsp->glitchinfo.shortamp,
            dsp->glitchinfo.slowtau, dsp->glitchinfo.slowamp,
            GLITCHINFO_NTERM, dsp->n_event_expected, dsp->ampl2crtmp, NULL, 0,
            0, 0, NULL, NULL, dsp->selcrit, dsp->selsnail, dsp->ppa, dsp->ppb,
            dsp->do_snail, dsp->snailtempl);
        t_findspike_clean += toc(tck);

        if (VERBOSE) {
          fprintf(stderr, "Merging flags\n");
        }
        for (long ii = 0; ii < signal_size; ii++) {
          out_flag[ii] = out_flag[ii] | (flag2[ii] & 1);
        }

        //detect excess CR tails
        tck = tic();
        if (VERBOSE) {
          fprintf(stderr, "Detecting CR tails\n");
        }
#pragma omp parallel for
        for (long ii = 0; ii < signal_size; ii++) {
          long jjmin = ii - 2 * tau3;
          if (jjmin < 0) {
            jjmin = 0;
          }
          long jjmax = ii + 2 * tau3;
          if (jjmax >= signal_size) {
            jjmax = signal_size;
          }
          double tmpd = 0;
          for (long jj = jjmin; jj < jjmax; jj++) {
            if (esigdet[jj] > tmpd) {
              tmpd = esigdet[jj];
            }
          }
          ecritcut_mf[ii] = sqrt(tmpd * tmpd + mhsig * mhsig);
          ecritcut_mf[ii] *= dsp->crit * pow(1.7, alpha) * dsp->critcut
              * dsp->factth;
        }
        t_tail += toc(tck);

        for (long ii = 0; ii < 2; ii++) {
          if (VERBOSE) {
            fprintf(stderr, "filling gaps\n");
          }
          fillgaps(datadesp, signal_size, data, out_flag, 0, 0);
          t_fillgaps += toc(tck);

          if (VERBOSE) {
            fprintf(stderr, "removing polynomial\n");
          }
          remove_poly(data, signal_size, orderpoly, data_out, 0);
          t_remove_poly += toc(tck);

          if (VERBOSE) {
            fprintf(stderr, "applying butterworth\n");
          }
          butterworth(data_out, signal_size, 0.0001 / fsamp * (signal_size), 8,
              data, bfilter, 1, napod, 0, t_fft);
          t_butterworth += toc(tck);

          if (VERBOSE) {
            fprintf(stderr, "removing long CR\n");
          }
          remove_longCR(data, signal_size, tau3, out_flag, ecritcut_mf, t_fft,
              dtest);
          t_remove_longCR += toc(tck);
        }

        if (iter == itermax - 1) {

          /// cut spikes from data and fill with line
          fillgaps(datadesp, signal_size, data, out_flag, mhsig * 0.8, 0);
          t_fillgaps += toc(tck);

          /// second iter despike on map despiker
          for (long ii = 0; ii < signal_size; ii++) {
            datadesp[ii] = data[ii];
          }

          for (long ii = 0; ii < signal_size; ii++) {
            flag2[ii] = out_flag[ii];
          }

          tck = tic();
          // findspike second iter
          findspike_clean(datadesp, signal_size, data, flag2, pflag,
              dsp->xcosm2, dsp->pos2, dsp->xspike2, dsp->amplspike2,
              dsp->allhsig2, dsp->chisqs2, dsp->posp2, dsp->amplp2,
              dsp->allhsigp2, dsp->critcut * pow(1.7, alpha) * 1.1, sign0,
              esigdet, dsp->kernel, dsp->kernel_size, dsp->glitchinfo.longtau,
              dsp->glitchinfo.longamp, dsp->glitchinfo.shorttau,
              dsp->glitchinfo.shortamp, dsp->glitchinfo.slowtau,
              dsp->glitchinfo.slowamp, GLITCHINFO_NTERM, dsp->n_event_expected,
              dsp->ampl2crtmp, NULL, 0, 0, 0, NULL, NULL, dsp->selcrit,
              dsp->selsnail, dsp->ppa, dsp->ppb, dsp->do_snail,
              dsp->snailtempl);
          t_findspike_clean += toc(tck);

          for (long ii = 0; ii < signal_size; ii++) {
            out_flag[ii] = out_flag[ii] | (flag2[ii] & 1);
          }
        }

        for (long ii = 0; ii < signal_size; ii++) {
          data_out[ii] = in_signal[ii] - datatailmod[ii];
        }

        tck = tic();
        fillgaps(data_out, signal_size, data, out_flag, 0, 0);
        t_fillgaps += toc(tck);

        // filter data
        remove_poly(data, signal_size, orderpoly, data_out, 0);
        t_remove_poly += toc(tck);
        butterworth(data_out, signal_size, f_lppix / 5, 8, data, bfilter, 1,
            napod, 0, t_fft);
        t_butterworth += toc(tck);
        iter++;
      }
    }

    tck = tic();

    time_elapsed += toc(time_start);
    if (VERBOSE) {
      fprintf(stderr, "   Time elapsed: %.2f s\n", time_elapsed);
    }
  }

  if (!nothing_to_do) {

    // final flagging of template corrections above 5 sigma of the noise
    // datatailmod is an alias for out_glitch
    for (long ii = 0; ii < signal_size; ii++) {
      if (datatailmod[ii] > 5 * mhsig) {
        out_flag[ii] = out_flag[ii] | 3;
      }
    }

    // data is an alias for out_residual.
    if (mhsig > 1e-60 && mhsig < 1e60) {
      fillgaps(datadesp, signal_size, data, out_flag, mhsig / 0.61237244, 0);
    }

    for (long ii = 1; ii < signal_size - 1; ii++) {
      datadesp[ii] = data[ii];
      if (out_flag[ii] & 1) {
        data[ii] = 0.25 * data[ii + 1] + 0.5 * datadesp[ii]
            + 0.25 * datadesp[ii - 1];
      }
    }
  }

  delete[] flag2;
  delete[] pflag;
  delete[] esigdet;
  delete[] ecritcut_mf;

  delete[] bfilter;
  delete[] dtest;
  delete[] data_out;
  delete[] datadesp;

  /*
   printf("            Fill gaps %8.2f s\n", t_fillgaps);
   printf("          Remove poly %8.2f s\n", t_remove_poly);
   printf("          Butterworth %8.2f s\n", t_butterworth);
   printf("      Findspike_clean %8.2f s\n", t_findspike_clean);
   printf("        Remove_longCR %8.2f s\n", t_remove_longCR);
   printf("                - FFT %8.2f s\n", t_fft);
   printf(" Detect excess tails: %8.2f s\n", t_tail);
   */

  return nothing_to_do;
    }
