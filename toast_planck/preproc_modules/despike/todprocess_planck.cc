#include <iostream>
#include <fstream>
#include <math.h>
#include "todprocess_planck.h"
#include <fftw3.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>
#include <string.h>
#include <omp.h>

typedef struct timeval ticker;

ticker tic(void) {
  ticker t;
  gettimeofday(&t, 0);
  return t;
}

double toc(ticker &t_old) {
  ticker t;
  gettimeofday(&t, 0);
  double elapsed = t.tv_sec - t_old.tv_sec
    + (t.tv_usec - t_old.tv_usec) * 1e-6;
  t_old = t;
  return elapsed;
}

using namespace std;

int VERBOSE = 0;

extern "C" {
void dposv_(char* uplo, int* n, int* nrhs, double* a, int* lda, double* b,
    int* ldb, int* info);
/*
void dgesv_(int *n, int *nn, double *m, int *n2, int *ipiv, double *x, int *n3,
    int *info);
void dgecon_(const char *norm, int *n, double *m, int *n2, double *anorm,
    double *rcond, double *work, int *iwork, int *info);
double dlange_(const char *norm, int *n, int *n2, double *m, int *n3,
 double *work);*/
}

void minmax(double* data, int ndata, double *min, double *max, int *posmin,
    int *posmax, unsigned char *flag) {
  int k;

  *min = data[0];
  *max = data[0];
  *posmin = 0;
  *posmax = 0;
  k = 0;
  if (flag != NULL) {
    while (flag[k] != 0 && k < ndata - 1) {
      *min = data[k + 1];
      *max = data[k + 1];
      *posmin = k + 1;
      *posmax = k + 1;
      k++;
    }
  }

  for (k = 1; k < ndata; k++) {
    if ((flag == NULL || flag[k] == 0) && (isnan(data[k]) == 0)) {
      if (data[k] < *min) {
        *min = data[k];
        *posmin = k;
      } else if (data[k] > *max) {
        *max = data[k];
        *posmax = k;
      }
    }
  }
}

void dpolyfit(double x[], double y[], int ndata, int norder, double *a) {
  /*
   *  Fit (x, y) with a polynomial of order `norder` and return the
   *  polynomial coefficients in `a`.
   */

  int ntemplate = norder + 1;
  double *invcov = new double[ntemplate * ntemplate];
  double *proj = a;

  // Initialize
  for (long row = 0; row < ntemplate; ++row) {
    proj[row] = 0;
    double *pcov = invcov + ntemplate * row;
    for (long col = 0; col < ntemplate; ++col) {
      pcov[col] = 0;
    }
  }

  // Compute the dot products.
  double *templates = new double[ntemplate];
  for (long samp = 0; samp < ndata; ++samp) {
    // Evaluate the polynomials at `x[samp]`
    templates[0] = 1;
    for (long itemplate = 0; itemplate < ntemplate - 1; ++itemplate) {
      templates[itemplate + 1] = templates[itemplate] * x[samp];
    }
    // Accumulate dot products between the templates and `y`
    for (long row = 0; row < ntemplate; ++row) {
      proj[row] += templates[row] * y[samp];
      double *pcov = invcov + row * ntemplate;
      for (long col = row; col < ntemplate; ++col) {
        pcov[col] += templates[row] * templates[col];
      }
    }
  }
  delete[] templates;

  // Solve polynomial coefficients from invcov x coeff = proj
  int n = ntemplate;
  while (n > 0) {
    int one = 1;
    int info = 0;
    char uplo = 'L';
    dposv_(&uplo, &n, &one, invcov, &ntemplate, proj, &ntemplate, &info);
    if (info) {
      // Solution failed, reduce the order of the polynomial and run again
      --n;
      proj[n] = 0;
    } else {
      break;
    }
  }

  // Finalize
  delete[] invcov;
}


void remove_poly(double y[], int ndata, int norder, double* yout,
    unsigned char* flag) {
  double *sx = new double[ndata];
  double *sy = new double[ndata];
  double *a = new double[norder + 1];

  long ndint = 0;
  for (long i = 0; i < ndata; i++) {
    if (flag != NULL && flag[i]) {
      continue;
    }
    sx[ndint] = i;
    sy[ndint] = y[i];
    ndint++;
  }

  dpolyfit(sx, sy, ndint, norder, a);

  //remove best fit poly
  for (long i = 0; i < ndata; i++) {
    yout[i] = y[i];
  }
#pragma omp parallel for
  for (long i = 0; i < ndata; i++) {
    double x = 1;
    for (long j = 0; j <= norder; j++) {
      yout[i] -= a[j] * x;
      x *= i;
    }
  }

  delete[] sx;
  delete[] sy;
  delete[] a;
}


void butterworth(double y[], int ndata, double f_lp, int orderB, double *yout,
    double *bfilter, bool apodize, int napod, bool overwrite, double &t_fft) {
  ticker tck;

  fftw_init_threads();
  fftw_plan_with_nthreads (omp_get_max_threads());

  double *tdata = (double *) fftw_malloc(sizeof(double) * ndata);
  fftw_complex *fdata = (fftw_complex *) fftw_malloc(
      sizeof(fftw_complex) * (ndata / 2 + 1));

  memcpy(tdata, y, sizeof(double) * ndata);

  // Apodize if asked, and define plan for fft

  if (apodize) {
    double *apodwind = apodwindow(ndata, napod);
    for (long ii = 0; ii < ndata; ii++) {
      tdata[ii] *= apodwind[ii];
    }
    delete[] apodwind;
  }
  tck = tic();
  fftw_plan fftplan = fftw_plan_dft_r2c_1d(ndata, tdata, fdata, FFTW_ESTIMATE);
  t_fft += toc(tck);

  long nnan = 0;
  for (long ii = 0; ii < ndata; ii++) {
    if (!(tdata[ii] < 1e100) || !(tdata[ii] > -1e100)) {
      ++nnan;
    }
  }
  if (nnan) {
    fprintf(stderr, "butterworth tdata has %ld NaNs\n", nnan);
  }

  // FFT

  tck = tic();
  fftw_execute(fftplan);
  fftw_destroy_plan(fftplan);
  t_fft += toc(tck);

  nnan = 0;
  for (int ii = 0; ii < ndata / 2 + 1; ii++) {
    if (!(fdata[ii][0] < 1e100) || !(fdata[ii][0] > -1e100)) {
      ++nnan;
    }
    if (!(fdata[ii][1] < 1e100) || !(fdata[ii][1] > -1e100)) {
      ++nnan;
    }
  }
  if (nnan) {
    fprintf(stderr, "butterworth fdata has %ld NaNs\n", nnan);
  }

  // Filter

  double ndata_inv = 1. / ndata;
  double f_lp_inv = 1. / f_lp;
#pragma omp for
  for (long ii = 0; ii < ndata / 2 + 1; ii++) {
    double pw = pow(double(ii) * f_lp_inv, 2 * orderB);
    pw /= 1 + pw;
    bfilter[ii] = pw;
    pw *= ndata_inv;
    fdata[ii][0] *= pw;
    fdata[ii][1] *= pw;
  }

  // FFT back

  tck = tic();
  fftplan = fftw_plan_dft_c2r_1d(ndata, fdata, tdata, FFTW_ESTIMATE);
  fftw_execute(fftplan);
  fftw_destroy_plan(fftplan);
  t_fft += toc(tck);

  if (overwrite) {
    memcpy(y, tdata, sizeof(double) * ndata);
  } else {
    memcpy(yout, tdata, sizeof(double) * ndata);
  }

  fftw_free(tdata);
  fftw_free(fdata);
}


double* apodwindow(int ns, int nn) {
  double *apodis = new double[ns];

  for (long ii = 0; ii < ns; ii++) {
    apodis[ii] = 1;
  }

  if (nn) {
    for (long ii = 0; ii < nn; ii++) {
      apodis[ii] = (sin(double(ii) / (nn - 1) * M_PI - M_PI * .5) + 1) * .5;
    }
    for (long ii = ns - nn; ii < ns; ii++) {
      apodis[ii] = (sin(double(ns - ii - 1) / (nn - 1) * M_PI - M_PI * .5) + 1)
          * .5;
    }
  }

  return apodis;
}


//*************************** linear Prediction ****************************//


int compare_doubles(const void *a, const void *b) {
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}

void cutdata(double y[], int indm, int indp, double *yout) {
  for (long i = indm; i <= indp; i++) {
    yout[i - indm] = y[i];
  }
}

void cutdata(unsigned char y[], int indm, int indp, unsigned char *yout) {
  for (long i = indm; i <= indp; i++) {
    yout[i - indm] = y[i];
  }
}

void mergedata(double y1[], int ndata1, double y2[], int ndata2, double *yout) {
  for (long i = 0; i < ndata1; i++) {
    yout[i] = y1[i];
  }
  for (long i = 0; i < ndata2; i++) {
    yout[i + ndata1] = y2[i];
  }
}

void sort(double y[], int nn, double *yout, int *nrel, unsigned char *flag) {
  if (flag != NULL) {
    *nrel = 0;
    for (long i = 0; i < nn; i++) {
      if (flag[i] == 0) {
        yout[*nrel] = y[i];
        *nrel = *nrel + 1;
      }
    }
  } else {
    *nrel = nn;
    for (long i = 0; i < nn; i++) {
      yout[i] = y[i];
    }
  }

  for (long i = 0; i < *nrel; i++) {
    for (long j = 1; j < *nrel; j++) {
      if (yout[j - 1] > yout[j]) {
        double temp = yout[j];
        yout[j] = yout[j - 1];
        yout[j - 1] = temp;
      }
    }
  }
}

double fitMultiTemplates(double *smdata, unsigned char *smflag,
    const double hmean, const double mhsig, const long nn, long *pos,
    const long npos, double *ShortTau, double *LongTau, double *ShortAmp,
    double *LongAmp, double *SnailTau, double *SnailAmp, const long nterm,
    const double ppa, const double ppb, const long Nbtemp, double *aa,
    unsigned char *sntpl, const bool snail, const bool snailtempl,
    const bool verb) {
  int err = 0, err2 = 0;
  int ip;
  double ffactcorr, cond1, tmp1;

  unsigned char tototmp = 0;

  bool rrun, cond2;
  int iter;

  int Nbtempfit = 2;
  if (Nbtemp == 1) {
    Nbtempfit = 1;
  }

  int ldMat = Nbtempfit * npos + 1; // leading dimension for Mat
  double *Ptd = new double[ldMat];
  double *p = new double[ldMat];
  double *moddata = new double[nn];
  double *smdata_mod = new double[nn];
  double *Mat = new double[ldMat * ldMat];
  double *Alltplates = new double[nn * ldMat];

  double *aat = new double[ldMat];
  long *we = new long[ldMat];
  unsigned char *ff = new unsigned char[ldMat];

  double mincritshort = 60;
  double critlongmin = 20;
  double critsnail = 6;
  if (snail == 0) {
    mincritshort = critlongmin;
  }

  ip = 0;
  for (long ii = 0; ii < Nbtempfit * npos + 1; ii++) {
    aa[ii] = 0;
    ff[ii] = 1;
    sntpl[ii] = 0;
    if ((ii / npos == 0) || (ii / npos == 2)
        || (smdata[pos[ii % npos]] - hmean > mincritshort * mhsig)
        || (ii == Nbtempfit * npos)) {
      ff[ii] = 0;
      we[ip] = ii;
      ip++;
    }
  }

  rrun = 1;
  iter = 0;
  while (rrun) {

    // set templates
    for (long jj = 0; jj < Nbtempfit * npos + 1; jj++) {
      for (long ii = 0; ii < nn; ii++) {
        Alltplates[ii * ldMat + jj] = 0;
      }
    }

    // set templates for all events
    for (long jj = 0; jj < Nbtempfit * npos; jj++) {
      long kk = jj / npos;
      long ll = jj % npos;
      ffactcorr = ((smdata[pos[ll]] - hmean) / mhsig) * ppa + ppb;
      // to change? must include ffactcorr here??
      cond1 = aa[jj] / (smdata[pos[ll]] - hmean);
      cond2 = (smdata[pos[ll]] - hmean > mincritshort * mhsig);
      for (long ii = pos[ll]; ii < nn; ii++) {
        if (smflag[ii] == 0) {
          tmp1 = 0;
          for (long qq = 0; qq < nterm; qq++) { /// !!!!! fix the number of exp
            if (kk == 0) {
              if (snailtempl == 0) {
                if (qq == 0) {
                  //don't go here if snail detected after iter = 0
                  if (iter == 0 || cond1 < 4 || snail == 0) {
                    tmp1 += (LongAmp[qq] * exp(-(ii - pos[ll]) / LongTau[qq]))
                        / ffactcorr;
                  }
                } else {
                  if (LongTau[qq] > 0) {
                    tmp1 += LongAmp[qq] * exp(-(ii - pos[ll]) / LongTau[qq]);
                  }
                }
              } else {  // in case snail template is defined
                if (iter == 0 || snail == 0
                    || (((cond1 < 4.0 && sntpl[jj] == 0)
                        || (cond1 < 0.5 && sntpl[jj] == 1))
                        && (ff[jj] == 0 || sntpl[jj] == 0))) {
                  // Long template. In principle the second condition
                  // should not be necessary here.
                  // -Remove very last condition??-
                  if (qq == 0) {
                    tmp1 += (LongAmp[qq] * exp(-(ii - pos[ll]) / LongTau[qq]))
                        / ffactcorr;
                  } else if (LongTau[qq] > 0) {
                    tmp1 += LongAmp[qq] * exp(-(ii - pos[ll]) / LongTau[qq]);
                  }
                  tototmp = 0;
                } else { // Snails
                  if ((pos[ll] - 3 > 1) & (smflag[pos[ll] - 4] == 0)) {
                    //  new line added for snails
                    smflag[pos[ll] - 3] = 1;
                  }
                  if (SnailTau[qq] > 0) {
                    tmp1 += SnailAmp[qq] * exp(-(ii - pos[ll]) / SnailTau[qq]);
                  }
                  tototmp = 1;
                }
              }
            }
            // last condition is just to gain time
            if (kk == 1 && (ShortTau[qq] > 0) && cond2) {
              if (ShortTau[qq] > 0) {
                tmp1 += ShortAmp[qq] * exp(-(ii - pos[ll]) / ShortTau[qq]);
              }
            }
          }
          Alltplates[ii * ldMat + jj] = tmp1;
        }
      }
      if (kk == 0 && snailtempl) {
        sntpl[jj] = tototmp;
      }
    }

    for (long ii = 0; ii < nn; ii++) {
      for (long jj = 0; jj < Nbtempfit * npos + 1; jj++) {
        if (smflag[ii] == 1) {
          Alltplates[ii * ldMat + jj] = 0;
        }
      }
    }

    for (long ii = 0; ii < nn; ii++) {
      if (smflag[ii] == 0) {
        Alltplates[ii * ldMat + Nbtempfit * npos] = 1;
      }
    }

    // compute Pt d
    for (long ii = 0; ii < nn; ii++) {
      smdata_mod[ii] = 0;
    }
    for (long jj = 0; jj < Nbtempfit * npos + 1; jj++) {
      if (ff[jj]) {
        for (long ii = 0; ii < nn; ii++) {
          smdata_mod[ii] -= aa[jj] * Alltplates[ii * ldMat + jj];
        }
      }
    }

    ip = 0;
    for (long jj = 0; jj < Nbtempfit * npos + 1; jj++) {
      if (ff[jj] == 0) {
        Ptd[ip] = 0;
        for (long ii = 0; ii < nn; ii++) {
          Ptd[ip] += Alltplates[ii * ldMat + jj]
              * (smdata[ii] - smdata_mod[ii]);
        }
        ip++;
      }
    }

    // compute PtP
    for (long ii = 0; ii < ip; ii++) {
      for (long jj = 0; jj < ip; jj++) {
        Mat[ii * ldMat + jj] = 0;
        for (long kk = 0; kk < nn; kk++) {
          Mat[ii * ldMat + jj] += Alltplates[kk * ldMat + we[jj]]
              * Alltplates[kk * ldMat + we[ii]];
        }
      }
    }

    // invert matrix and solve system
    int one = 1;
    int info = 0;
    char uplo = 'L';
    for (long i = 0; i < ip; ++i) {
      aat[i] = Ptd[i];
    }
    dposv_(&uplo, &ip, &one, Mat, &ldMat, aat, &ldMat, &info);

    if (info) {
      // compute PtP + C-1 (for conditioning)
      for (long ii = 0; ii < ip; ii++) {
        for (long jj = 0; jj < ip; jj++) {
          Mat[ii * ldMat + jj] = 0;
          for (long kk = 0; kk < nn; kk++) {
            Mat[ii * ldMat + jj] += Alltplates[kk * ldMat + we[jj]]
                * Alltplates[kk * ldMat + we[ii]];
          }
        }
        // This cannot be right... pos is only populated up to npos.
        //double x = 1.0 / (smdata[pos[nn/2]] + 1000.0) / 10.0);
        double x = 1.0 / (smdata[nn / 2] + 1000.0) / 10.0;
        Mat[ii * ldMat + ii] += x * x;
      }
      dposv_(&uplo, &ip, &one, Mat, &ip, aat, &ip, &info);
      if (info && VERBOSE) {
        printf("Error in dposv: matrix is probably not positive definite\n");
      }
    }

    rrun = 0;

    if ((iter == 0) && (Nbtempfit > 1)) {
      for (long ii = 0; ii < ip; ii++) {
        aa[we[ii]] = aat[ii];
      }
      for (long ii = 0; ii < npos; ii++) {
        if ((aa[ii + npos] > 4 * (smdata[pos[ii]] - aa[Nbtempfit * npos]))
            && ((smdata[pos[ii]] - aa[Nbtempfit * npos]) < 8000 * mhsig)) {
          rrun = 1;
        }
      }
      ip = 0;
      for (long ii = 0; ii < Nbtempfit * npos; ii++) {
        long kk = ii / npos;
        long ll = ii % npos;
        if (kk == 0) { //// Long templates
          if (smdata[pos[ll]] - aa[Nbtempfit * npos] >= critlongmin * mhsig) {
            we[ip] = ii;
            aat[ip] = aa[ii];
            ip++;
            ff[ii] = 0;
          } else { //// For long template, small events, detected snails.
            if (snailtempl == 0) {
              if (snail
                  && aa[ii]
                      > critsnail * (smdata[pos[ll]] - aa[Nbtempfit * npos])
                          * (((smdata[pos[ll]] - hmean) / mhsig) * ppa + ppb)) {
                aa[ii] = 7 * (smdata[pos[ll]] - aa[Nbtempfit * npos])
                    * (((smdata[pos[ll]] - hmean) / mhsig) * ppa + ppb);
              } else {
                aa[ii] = (smdata[pos[ll]] - aa[Nbtempfit * npos])
                    * (((smdata[pos[ll]] - hmean) / mhsig) * ppa + ppb);
              }
              ff[ii] = 1;
            } else {
              if (aa[ii]
                  < critsnail * (smdata[pos[ll]] - aa[Nbtempfit * npos])
                      * (((smdata[pos[ll]] - hmean) / mhsig) * ppa + ppb)) {
                aa[ii] = (smdata[pos[ll]] - aa[Nbtempfit * npos])
                    * (((smdata[pos[ll]] - hmean) / mhsig) * ppa + ppb);
                ff[ii] = 1;
              } else {
                we[ip] = ii;
                aat[ip] = aa[ii];
                ip++;
                ff[ii] = 0;
              }
            }
          }
        } else {  //// Short templates
          if (((aa[ii] <= 4 * (smdata[pos[ll]] - aa[Nbtempfit * npos]))
              || ((smdata[pos[ll]] - aa[Nbtempfit * npos]) >= 8000 * mhsig))
              && (smdata[pos[ll]] - hmean > mincritshort * mhsig)) {
            we[ip] = ii;
            aat[ip] = aa[ii];
            ip++;
            ff[ii] = 0;
          } else {
            aa[ii] = 0;
            ff[ii] = 1;
          }
        }
      }
      /// for the average
      we[ip] = Nbtempfit * npos;
      aat[ip] = aa[Nbtempfit * npos];
      ip++;
      ff[Nbtempfit * npos] = 0;
    }

    if ((iter > 0) || (rrun == 0)) {
      /// only if ampl short glitch outliers have been sorted out
      for (long ii = 0; ii < ip - 1; ii++) {
        if (aat[ii] <= 0) {
          rrun = 1;
        }
      }

      for (long ii = 0; ii < ip; ii++) {
        aa[we[ii]] = aat[ii];
      }

      ip = 0;
      for (long ii = 0; ii < Nbtempfit * npos + 1; ii++) {
        if (ff[ii] == 0) {
          if (((aa[ii] > 0)
              && ((ii / npos == 0)
                  || ((aa[ii]
                      <= 4 * (smdata[pos[ii % npos]] - aa[Nbtempfit * npos]))
                      && (smdata[pos[ii % npos]] - hmean > mincritshort * mhsig))
                  || ((smdata[pos[ii % npos]] - aa[Nbtempfit * npos])
                      >= 8000 * mhsig))) || (ii == Nbtempfit * npos)) {
            we[ip] = ii;
            aat[ip] = aa[ii];
            ip++;
          } else {
            aa[ii] = 0;
            ff[ii] = 1;
          }
        }
      }
    }

    if (ip == 0) {
      rrun = 0;
    }
    if (iter == 0) {
      rrun = 1;
    }
    iter++;

    /// TO DO: rerun if no stability of the results (might happen with the snails)
  }

  if (err && VERBOSE) {
    fprintf(stderr, "WARNING : PROBLEM IN GLITCH TEMPLATE MULTI-FIT, "
            "MATRIX IS ILL-CONDITIONED \n");
    if (err2) {
      fprintf(stderr,
              "NOT REPARED BY CONDITIONING!!!!!!!!!!!!!!!!!!! \n");
    } else {
      fprintf(stderr, "REPARED BY CONDITIONING... \n");
    }
  }

  // Compute chi2
  //Combine templates
  for (long ii = 0; ii < nn; ii++) {
    moddata[ii] = aa[Nbtempfit * npos];
  }

  for (long jj = 0; jj < Nbtempfit * npos; jj++) {
    for (long ii = 0; ii < nn; ii++) {
      moddata[ii] += aa[jj] * Alltplates[ii * ldMat + jj];
    }
  }

  double chi2;
  if (err && err2) {
    chi2 = -1;
  } else {
    chi2 = 0;
    long npchi2 = 0;
    for (long ii = 0; ii < nn; ii++) {
      if (smflag[ii] == 0) {
        chi2 += pow((smdata[ii] - moddata[ii]), 2);
        npchi2++;
      }
    }
    chi2 /= (mhsig * mhsig);
    chi2 /= npchi2;
  }

  delete[] we;
  delete[] ff;
  delete[] aat;
  delete[] Ptd;
  delete[] p;
  delete[] moddata;
  delete[] smdata_mod;
  delete[] Alltplates;
  delete[] Mat;

  return chi2;
}

void findspike_clean(double y[], const int ndata, double* yout,
    unsigned char* flag, unsigned char* pflag, long long *xcosm, int *pos,
    int *xspike, double *amplspike, double *allhsig, double *allchi2, int *posp,
    double *amplp, double *allhsigp, const double criterion, const double sign0,
    double *esigdet, const double *kernel, const int sker, double *taulg,
    double *atplg, double *taush, double *atpsh, double *tausn, double *atpsn,
    const long nterm, const long nevent, double *ampl2cr, double *templates,
    const long Nbtemp, const long ndtemp, const double critrem,
    double *ampltemplate, double *ampltemplateSH, const double selcrit,
    const double selsnail, const double ppa, const double ppb, const bool snail,
    const bool snailtempl) {
  // data are assumed to vary linearly in every window

  bool ttest;

  int iter, xorig, count, countspike, icosm, countp, pcount = 0, tmpc, stage0,
      stage1;
  int posmin, posmax, posmm, nrel, margcutmod_p, margcutmod_pr, boundflagfit;
  int temp;
  int pkermax;
  double chi2;

  const int npb = 1000;
  const double sigbeam = 5; /// just used for the edge of the window here

  const double critspike = 20;
  const double critneg = 5;

  const double tailston = 0.5;

  //// interval to cut when an event is detected
  const int margcut_m = 3;
  const int margcut_p = 6;
  const int margcutpos_m = 2;
  const int margcutpos_p = 2;

  const int nmaxiter = 10;  //// maximum number of iteration in every window
  const int nwindowsig = 50;

  bool balt = 0;
  bool notp = 0;

  double valmin, valmax, hmean, hsig = 0, hsigeff, hsig_, tap, selcritdyn;
  long shl, limg, tpp;

  if (VERBOSE) {
    double somme_y = 0;
    for (long i = 0; i < ndata; i++) {
      somme_y += y[i];
    }
    fprintf(stderr, "findspike_clean: sum(y)=%.17g  (ndata=%d) "
            "sign0=%.17g\n", somme_y, ndata, sign0);
  }

  double *tte = new double[npb];
  double *tts = new double[npb];

  double *hsigall = new double[nwindowsig + 1];
  double *strsig = new double[nwindowsig + 1];

  double *y2 = new double[ndata];
  double *smdata = new double[npb];
  double *dsmdata = new double[npb];
  double *tempdata = new double[npb];
  double *strval = new double[npb];
  long *ppp = new long[npb];
  long *ppptemp = new long[npb];
  unsigned char *flag22 = new unsigned char[ndata + 3 * npb + 1];
  unsigned char *smflag = new unsigned char[npb];
  unsigned char *tflag = new unsigned char[npb];
  unsigned char *flderiv = new unsigned char[npb];

  double *a = new double[2];

  string str1, str2;

  double *aa = new double[nterm];
  double *amptps = new double[Nbtemp * npb];
  unsigned char *sntpl = new unsigned char[Nbtemp * npb];

  pkermax = 0;
  if (kernel != NULL) {
    for (long i = 0; i < sker; i++) {
      if (abs(kernel[i]) > abs(kernel[pkermax]))
        pkermax = i;
    }
  }

  xorig = 0;
  count = 0;
  countspike = 0;
  icosm = 0;
  countp = 0;
  posmm = 0;

  ////copy data
  for (long i = 0; i < ndata; i++) {
    y2[i] = y[i];
    flag22[i] = 0;
  }

  iter = 0;

  if (sign0 <= 0) {
    for (long i = 0; i < nwindowsig; i++) {
      hsigall[i] = 64.28;
    }
  } else {
    for (long i = 0; i < nwindowsig; i++) {
      hsigall[i] = sign0;
    }
  }

  while ((xorig + npb + npb / 2) < ndata) {

    cutdata(y2, xorig, xorig + npb - 1, tempdata);
    cutdata(flag, xorig, xorig + npb - 1, smflag);
    remove_poly(tempdata, npb, 0, smdata, smflag);

    minmax(smdata, npb, &valmin, &valmax, &posmin, &posmax, smflag);

    if (abs(valmin) > abs(valmax)) {
      posmm = posmin;
    }
    if (abs(valmin) <= abs(valmax)) {
      posmm = posmax;
    }

    //center on the extremum if well located in the window
    if ((xorig > npb / 2 - posmm) && (posmm < npb - int(4 * sigbeam))) {

      xorig = xorig - npb / 2 + posmm;

      cutdata(y2, xorig, xorig + npb - 1, tempdata);
      cutdata(flag, xorig, xorig + npb - 1, smflag);
      remove_poly(tempdata, npb, 0, smdata, smflag);

      // estimate mean and standard deviation of the data in the
      // small interval (may be improved)

      nrel = 0;
      for (long i = 0; i < npb; i++)
        if (smflag[i] == 0) {
          strval[nrel] = smdata[i];
          nrel++;
        }
      qsort(strval, nrel, sizeof(double), compare_doubles);

      hmean = strval[nrel / 2];
      hsig = strval[int((double) nrel * 0.84)] - hmean;

      hsig_ = hsig;
      hsigall[nwindowsig] = hsig;
      sort(hsigall, nwindowsig + 1, strsig, &temp);
      if (sign0 > 0 && strsig[(nwindowsig + 1) / 2] != 64.28)
        hsig = strsig[(nwindowsig + 1) / 2];

      hsigeff = sqrt(
          hsig * hsig + esigdet[xorig + npb / 2] * esigdet[xorig + npb / 2]);
      if (esigdet[xorig + npb / 2] > 4 * hsig) {
        notp = 1;
      } else {
        notp = 0;
      }

      /// replace flag data

      //// 1st test: spike larger than threshold
      if (abs(smdata[npb / 2] - hmean) > criterion * hsigeff
          && iter < nmaxiter) {

        //// identify single corrupted pixels
        if (abs(smdata[npb / 2] - hmean) > critspike * hsigeff
            && abs(smdata[npb / 2 + 1] - hmean) < 4 * hsigeff
            && abs(smdata[npb / 2 - 1] - hmean) < 4 * hsigeff) {
          xspike[countspike] = xorig + npb / 2;
          countspike++;

          if ((flag[xorig + npb / 2] & 1) == 0) {
            flag[xorig + npb / 2] += 1;
            pflag[xorig + npb / 2] += 1;
          }
          if ((flag[xorig + npb / 2] & 2) == 0) {
            flag[xorig + npb / 2] += 2;
            pflag[xorig + npb / 2] += 2;
          }
        } else {
          ////////// group of pixels ////////////

          pos[count] = xorig + npb / 2;
          count++;
          //// 3rd test (for cosmic rays detection): positive amplitude
          if ((smdata[npb / 2] - hmean) > 0) {
            chi2 = 1;
            if ((ndtemp && smdata[npb / 2] - hmean > critrem * hsigeff)
                && (notp == 0)) {
              //mm = 50; // no events taken into account above npb-mm
              // compute derivative (or something similar) to make sure all
              // event is identified for multi-fit of large events
              dsmdata[0] = 0;
              dsmdata[npb - 1] = 0;
              for (long ii = 0; ii < npb - 2; ii++) {
                dsmdata[ii + 1] = -0.5 * smdata[ii] + smdata[ii + 1]
                    - 0.5 * smdata[ii + 2];
              }
              for (long ii = 0; ii < npb; ii++) {
                flderiv[ii] = 0;
              }
              for (long ii = 0; ii < npb; ii++) {
                // max must be basically the interval of fit
                // times a factor which should be around 1/2 to
                // first order given our data
                if ((dsmdata[ii] > criterion * hsigeff) && (smflag[ii] == 0)) {
                  for (long i = 0; i < 5; i++) {
                    if ((ii + i - 2 >= 0) && (ii + i - 2 < npb)) {
                      flderiv[ii + i - 2] = 1;
                    }
                  }
                }
              }

              // find maxima in windows
              // maximum number of event possible is npb
              for (long ii = 0; ii < npb; ii++) {
                ppp[ii] = -1;
              }
              pcount = 0;
              for (long ii = 0; ii < npb;) {
                long i = 0;
                tap = smdata[ii];
                tpp = ii;
                while ((ii + i < npb) && (flderiv[ii + i] == 1)) {
                  if (i == 0) {
                    pcount++;
                  }
                  if (tap < smdata[ii + i]) {
                    tap = smdata[ii + i];
                    tpp = ii + i;
                  }
                  i++;
                  ppp[pcount - 1] = tpp;
                }
                ii += i + 1;
              }
              for (long ii = 0; ii < npb; ii++) {
                tflag[ii] = 0;
                if (smflag[ii] != 0) {
                  tflag[ii] = 1;
                }
              }

              for (long i = 0; i < pcount; i++) {
                boundflagfit = 8;
                if (smdata[ppp[i]] - hmean < 30.0 * hsig) {
                  boundflagfit = 6;
                }
                if (!snail) {
                  if (smdata[ppp[i]] - hmean < 300.0 * hsig) {
                    boundflagfit = 3;
                  } else {
                    boundflagfit = 5;
                  }
                }
                for (long ii = -2; ii < boundflagfit; ii++) {
                  if ((ppp[i] + ii >= 0) && (ppp[i] + ii < npb)) {
                    tflag[ppp[i] + ii] = 1;
                  }
                }
              }

              // remove last events if not constrained by the last point in the
              // window, and add central event if missing, this is a correct
              // approach
              if (pcount) {
                for (long i = 0; i < pcount;) {
                  tmpc = 0;
                  for (long ii = ppp[pcount - 1]; ii < npb; ii++) {
                    if (tflag[ii] == 0) {
                      tmpc++;
                    }
                  }
                  if (tmpc < pcount - i) {
                    ppp[pcount - 1] = -1;
                    pcount--;
                    i--;
                  }
                  i++;
                }
              }

              ttest = 0;
              for (long ii = 0; ii < npb; ii++) {
                if (ppp[ii] == npb / 2) {
                  ttest = 1;
                }
              }
              if (!ttest) {
                pcount++;
                ppp[pcount - 1] = npb / 2;
                boundflagfit = 8;
                if (smdata[npb / 2] - hmean < 30.0 * hsig) {
                  boundflagfit = 6;
                }
                if (!snail) {
                  boundflagfit = 3;
                }
                for (long ii = -2; ii < boundflagfit; ii++) {
                  tflag[npb / 2 + ii] = 1;
                }
              }

              /////////////// remove events at the edges
              stage1 = 0;
              stage0 = 0;
              long ii = npb - 1;
              while ((ii > 0) && ((stage1 == 0) || (tflag[ii] == 1))) {
                if ((stage1 == 0) && (tflag[ii] == 0)) {
                  stage0++;
                }
                if (tflag[ii] == 1) {
                  stage1++;
                }
                ii--;
              }
              stage1 = 0;
              for (long i = 0; i < pcount; i++) {
                if (ppp[i] >= ii) {
                  stage1++;
                }
              }

              if (2 * stage1 > stage0) {
                for (long i = ii; i < npb; i++) {
                  tflag[i] = 1;
                }

                long j = 0;
                for (long i = 0; i < pcount; i++) {
                  if ((ppp[i] < ii) || (ppp[i] == npb / 2)) {
                    ppptemp[j] = ppp[i];
                    j++;
                  }
                }
                if (VERBOSE) {
                  fprintf(stderr, "%ld EVENTS NOT FITTED, j=%ld\n", pcount - j,
                      j);
                }
                pcount = j;
                for (long i = 0; i < pcount; i++) {
                  ppp[i] = ppptemp[i];
                }
              }

              for (long ii = 0; ii < Nbtemp * pcount + 1; ii++) {
                amptps[ii] = 0;
                sntpl[ii] = 0;
              }

              ///// Do the multi-fit of templates

              int verb = 0;

              chi2 = fitMultiTemplates(smdata, tflag, hmean, hsig, npb, ppp,
                  pcount, taush, taulg, atpsh, atplg, tausn, atpsn, nterm, ppa,
                  ppb, Nbtemp, amptps, sntpl, snail, snailtempl, verb);

              if (chi2 > 0) {
                if (Nbtemp >= 2) {
                  hmean = amptps[pcount * 2];
                } else {
                  hmean = amptps[pcount];
                }
              }
            }

            //***** fit TC amplitudes to define the data interval to cut  *****

            margcutmod_p = margcut_p;
            margcutmod_pr = margcut_p;

            // Refine future versions: must do something special if hsigeff
            // is significantly different from hsig
            // Also: must do something about snails below hsig

            selcritdyn = selcrit;
            if ((smdata[npb / 2] - hmean > 800 * hsig) && (chi2 < 3) && snail) {
              selcritdyn = 0.2;
            }

            // must define here interval to cut in case of template

            // identify template amplitude for central event and extend
            // flags if other events are in the interval
            if ((ndtemp && smdata[npb / 2] - hmean > critrem * hsigeff)
                && (notp == 0)) {
              long i = 0;
              while ((i < pcount) && (ppp[i] != npb / 2)) {
                i++;
              }

              if ((Nbtemp == 1)
                  && (amptps[i] / (smdata[npb / 2] - hmean)
                      / (ppa * (smdata[npb / 2] - hmean) / hsig + ppb)
                      > selcritdyn)) {

                margcutmod_pr = log(
                                    (smdata[npb / 2] - hmean) / hsig
                                    / tailston) * taulg[0];
                for (long ic = 1; ic < nterm; ic++) {
                  if ((taulg[ic] > 0.18) & (taulg[ic] < 18.0)) {
                    double x = log(atplg[ic] * amptps[i] / hsig / tailston)
                        * taulg[ic];
                    if (margcutmod_pr < x) {
                      margcutmod_pr = x;
                    }
                  }
                }

                for (long i = 0; i < pcount; i++) {
                  if ((ppp[i] > npb / 2)
                      && (margcutmod_pr + npb / 2 > ppp[i])) {
                    if (amptps[i] > 0) {
                      double x = log(atplg[1] * amptps[i] / hsig / tailston)
                          * taulg[1] - npb / 2 + ppp[i];
                      if (margcutmod_pr < x) {
                        margcutmod_pr = x;
                      }
                      for (long ic = 2; ic < nterm; ic++) {
                        if ((taulg[ic] > 0.18) & (taulg[ic] < 18.0)) {
                          double x = log(
                              atplg[ic] * amptps[i] / hsig / tailston)
                              * taulg[ic] - npb / 2 + ppp[i];
                          if (margcutmod_pr < x) {
                            margcutmod_pr = x;
                          }
                        }
                      }
                    }
                    double x = log((smdata[ppp[i]] - hmean) / hsig / tailston)
                        * taulg[0] - npb / 2 + ppp[i];
                    if (margcutmod_pr < x) {
                      margcutmod_pr = x;
                    }
                  }
                }
                margcutmod_p = margcutmod_pr;
              }

              if (Nbtemp >= 2) {
                if (amptps[i] / (smdata[npb / 2] - hmean)
                    / (ppa * (smdata[npb / 2] - hmean) / hsig + ppb)
                    > selcritdyn) {
                  // comment this here with new version
                  if (snailtempl == 0 || sntpl[i] == 0) {
                    margcutmod_pr = log(
                        (smdata[npb / 2] - hmean) / hsig / tailston) * taulg[0];
                    for (long ic = 1; ic < nterm; ic++) {
                      if ((taulg[ic] > 0.18) & (taulg[ic] < 180)) {
                        double x = log(atplg[ic] * amptps[i] / hsig / tailston)
                            * taulg[ic];
                        if (margcutmod_pr < x) {
                          margcutmod_pr = x;
                        }
                      }
                    }
                  } else {
                    margcutmod_pr = log(atpsn[0] * amptps[i] / hsig / tailston)
                        * tausn[0];
                    for (long ic = 1; ic < nterm; ic++) {
                      if ((tausn[ic] > 0.18) & (tausn[ic] < 180)) {
                        double x = log(atpsn[ic] * amptps[i] / hsig / tailston)
                            * tausn[ic];
                        if (margcutmod_pr < x) {
                          margcutmod_pr = x;
                        }
                      }
                    }
                  }
                }
                if (amptps[i] / (smdata[npb / 2] - hmean)
                    / (ppa * (smdata[npb / 2] - hmean) / hsig + ppb)
                    < selcritdyn) {
                  margcutmod_pr = log(
                      (smdata[npb / 2] - hmean) / hsig / tailston * 5)
                      * taulg[0];
                  for (long ic = 1; ic < nterm; ic++) {
                    if ((taulg[ic] > 0.18) & (taulg[ic] < 180000)) {
                      double x = log(
                          atplg[ic] * amptps[i] / hsig / tailston * 5)
                          * taulg[ic];
                      if (margcutmod_pr < x) {
                        margcutmod_pr = x;
                      }
                    }
                  }
                  if ((xorig + npb / 2 > 216126 + 2000)
                      && (xorig + npb / 2 < 216130 + 2000)) {
                    if (VERBOSE) {
                      fprintf(stderr,
                              "Ampl event = %10.15g, amptps[i] = "
                              "%10.15g, amptps[i+pcount] = %10.15g,"
                              "  margcutmod_pr = %d\n",
                              (smdata[npb / 2] - hmean),
                              amptps[i],
                              amptps[i + pcount],
                              margcutmod_pr);
                    }
                  }

                  if (smdata[npb / 2] - hmean > 10000 * hsig) {
                    double x = log(
                        (smdata[npb / 2] - hmean) / hsig / tailston * 5)
                        * taush[0];
                    if (margcutmod_pr < x) {
                      margcutmod_pr = x;
                    }
                    for (long ic = 1; ic < nterm; ic++) {
                      if (taush[ic] > 0.18) {
                        double x = log(
                            atpsh[ic] * amptps[i + pcount] / hsig / tailston
                                * 5) * taush[ic];
                        if (margcutmod_pr < x) {
                          margcutmod_pr = x;
                        }
                      }
                    }
                  } else {
                    double x = log(
                        (smdata[npb / 2] - hmean) / hsig / tailston * 5)
                        * taush[0];
                    if (margcutmod_pr < x) {
                      margcutmod_pr = x;
                    }
                    for (long ic = 1; ic < nterm; ic++) {
                      if (taush[ic] > 0.18) {
                        double x = log(
                            atpsh[ic] * (smdata[ppp[i]] - hmean) / hsig
                                / tailston * 5) * taush[ic];
                        if (margcutmod_pr < x) {
                          margcutmod_pr = x;
                        }
                      }
                    }
                  }
                }
                for (long i = 0; i < pcount; i++) {
                  if ((ppp[i] >= npb / 2 - margcut_m)
                      && (margcutmod_pr + npb / 2 > ppp[i])) {
                    if (amptps[i] > 0) {
                      double x = margcut_p - npb / 2 + ppp[i];
                      if (margcutmod_pr < x) {
                        margcutmod_pr = x;
                      }
                      for (long ic = 1; ic < nterm; ic++) {
                        if ((taulg[ic] > 0.18) & (taulg[ic] < 18)) {
                          double x = log(
                              atplg[ic] * amptps[i] / hsig / tailston)
                              * taulg[ic] - npb / 2 + ppp[i];
                          if (margcutmod_pr < x) {
                            margcutmod_pr = x;
                          }
                        }
                      }
                    }
                    if ((amptps[i] / (smdata[ppp[i]] - hmean)
                        / (ppa * (smdata[ppp[i]] - hmean) / hsig + ppb)
                        < selcritdyn) && (amptps[i + pcount] > 0)
                        && (smdata[ppp[i]] - hmean > critrem * hsigeff)) {
                      for (long ic = 1; ic < nterm; ic++) {
                        if (taush[ic] > 0.18) {
                          double x = log(
                              atpsh[ic] * (smdata[ppp[i]] - hmean) / hsig
                                  / tailston * 5) * taush[ic] - npb / 2
                              + ppp[i];
                          if (margcutmod_pr < x) {
                            margcutmod_pr = x;
                          }
                        }
                      }
                    }
                    double x = log((smdata[ppp[i]] - hmean) / hsig / tailston)
                        * taulg[0] - npb / 2 + ppp[i];
                    if (margcutmod_pr < x) {
                      margcutmod_pr = x;
                    }
                  }
                }
                margcutmod_p = margcutmod_pr;
              }
            }

            for (long i = xorig + npb / 2 - margcut_m;
                i < xorig + npb / 2 + margcutmod_p && i < ndata + 3 * npb + 1;
                i++) {
              flag22[i] = flag22[i] | 3;
            }

            for (long i = xorig + npb / 2 - margcut_m;
                i < xorig + npb / 2 + margcutmod_pr; i++) {
              if (i < ndata) {
                flag[i] = flag[i] | 3;
              }
            }

            if (!ndtemp || smdata[npb / 2] - hmean <= critrem * hsigeff) {
              pflag[xorig + npb / 2] = pflag[xorig + npb / 2] | 5;
              flag[xorig + npb / 2] = flag[xorig + npb / 2] | 5;
            }

            //****************************************************************
            //***************    Part concerning removing templates    *******
            //****************************************************************

            // subtract templates for the central event and events mixed

            if ((ndtemp && smdata[npb / 2] - hmean > critrem * hsigeff)
                && (notp == 0)) {
              // select only events larger than a certain value

              for (long ii = 0; ii < pcount; ii++) {
                if ((ppp[ii] > npb / 2 - margcut_m)
                    && (ppp[ii] < npb / 2 + margcutmod_pr)) {

                  if ((amptps[ii] / (smdata[ppp[ii]] - hmean)
                      / (ppa * (smdata[ppp[ii]] - hmean) / hsig + ppb)
                      > selcritdyn)
                      && (smdata[ppp[ii]] - hmean > critrem * hsigeff)) {

                    //Subtract only templates above thresholds
                    // (for max event and for template itself)

                    //soustraction part
                    shl = 5; //does'nt matter much
                    // to define depending on amplitude of the glitch
                    limg = long(
                        2 * 180.18
                            * log(
                                (smdata[ppp[ii]] - hmean) * 2 / hsig * 2
                                    * sqrt(180.18)));
                    if (limg >= ndtemp) {
                      limg = ndtemp;
                    }
                    if (limg + ppp[ii] + xorig >= ndata) {
                      limg = ndata - (ppp[ii] + xorig) - 1;
                    }

                    long itpl = 0;
                    if (snailtempl && (sntpl[ii] == 1)) {
                      itpl = 2;
                    }

                    if (xorig + npb / 2 == 338264 && VERBOSE) {
                      fprintf(stderr,
                              "SNTPL = %d, amptps = %10.15g\n",
                              sntpl[ii], amptps[ii]);
                    }

                    for (long i = shl; i < limg; i++) {
                      y2[i + xorig + ppp[ii]] = y2[i + xorig + ppp[ii]]
                          - amptps[ii] * templates[i * Nbtemp + itpl];
                      yout[i + xorig + ppp[ii]] += amptps[ii]
                          * templates[i * Nbtemp + itpl];
                    }

                    pflag[xorig + ppp[ii]] = pflag[xorig + ppp[ii]] | 9;
                    flag[xorig + ppp[ii]] = flag[xorig + ppp[ii]] | 9;
                    if (itpl == 0) {
                      pflag[xorig + ppp[ii]] = pflag[xorig + ppp[ii]] | 5;
                      flag[xorig + ppp[ii]] = flag[xorig + ppp[ii]] | 5;
                    }

                  } else {
                    pflag[xorig + ppp[ii]] = pflag[xorig + ppp[ii]] | 5;
                    flag[xorig + ppp[ii]] = flag[xorig + ppp[ii]] | 5;
                  }
                  ampltemplate[icosm] = amptps[ii];
                  if (snailtempl && sntpl[ii]) {
                    ampltemplate[icosm] = -amptps[ii];
                  }
                  if (Nbtemp >= 2) {
                    ampltemplateSH[icosm] = amptps[ii + pcount];
                  }
                  xcosm[icosm] = xorig + ppp[ii];
                  amplspike[icosm] = smdata[ppp[ii]] - hmean;
                  allhsig[icosm] = hsig;
                  allchi2[icosm] = chi2;
                  if (ppp[ii] == npb / 2) {
                    for (long i = 0; i < nterm; i++) {
                      ampl2cr[i * nevent + icosm] = aa[i];
                    }
                  } else {
                    for (long i = 0; i < nterm; i++) {
                      ampl2cr[i * nevent + icosm] = 0;
                    }
                  }
                  icosm++;
                }
              }
            } else {
              if (ndtemp) {
                ampltemplate[icosm] = 0;
                if (Nbtemp >= 2) {
                  ampltemplateSH[icosm] = 0;
                }
              }
              xcosm[icosm] = xorig + npb / 2;
              amplspike[icosm] = smdata[npb / 2] - hmean;
              allhsig[icosm] = hsig;
              allchi2[icosm] = chi2;
              for (long ii = 0; ii < nterm; ii++) {
                ampl2cr[ii * nevent + icosm] = aa[ii];
              }
              icosm++;
            }
          } else {
            /// negative for some reason
            posp[countp] = xorig + npb / 2;
            amplp[countp] = smdata[npb / 2] - hmean;
            allhsigp[countp] = hsig;
            countp++;
            for (long i = xorig + npb / 2 - margcutpos_m;
                 i < xorig + npb / 2 + margcutpos_p; i++) {
              flag[i] = flag[i] | 64;
              /// flag large negative data point
              if (abs(smdata[npb / 2] - hmean) > critneg * hsigeff
                  && (flag[i] & 1) == 0) {
                flag[i]++;
              }
            }
            pflag[xorig + npb / 2] = pflag[xorig + npb / 2] | 64;

            /// flag large negative data point
            if (abs(smdata[npb / 2] - hmean) > critneg * hsigeff
                && (pflag[xorig + npb / 2] & 1) == 0) {
              pflag[xorig + npb / 2]++;
            }
          }
        }
        xorig += npb / 2 - posmm;
        iter++;
        if (balt) {
          xorig -= npb / 2 - int(4 * sigbeam);
        }
        balt = 0;
      } else {
        xorig += npb - posmm;
        iter = 0;
        if (balt) {
          xorig -= npb / 2 - int(4 * sigbeam);
        }
        balt = 0;

        for (long i = 0; i < nwindowsig - 1; i++) {
          hsigall[i] = hsigall[i + 1];
        }
        hsigall[nwindowsig - 1] = hsig_;
      }
    } else {
      xorig += npb / 2 - int(4 * sigbeam);
      balt = 1;
    }
  }

  for (long i = 0; i < ndata; i++) {
    flag[i] = flag22[i] | flag[i];
  }

  // add extra flag to account for various filtering
  if (kernel != NULL) {
    for (long i = 0; i < icosm; i++) {
      long ii = 0;
      while ((abs(kernel[ii] / kernel[pkermax]) * amplspike[i] < hsig / 10)
          && ii < pkermax) {
        ii++;
      }
      for (long iii = ii - 1; iii < pkermax; iii++) {
        if (xcosm[i] + iii - pkermax >= 0) {
          flag[xcosm[i] + iii - pkermax] = flag[xcosm[i] + iii - pkermax] | 3;
        }
      }
    }
  }

  pos[count] = -1;
  xcosm[icosm] = -1;
  xspike[countspike] = -1;
  amplspike[icosm] = -1;
  allhsig[icosm] = -1;
  allchi2[icosm] = -1;
  posp[countp] = -1;
  amplp[countp] = -1;
  allhsigp[countp] = -1;

  delete[] dsmdata;
  delete[] ppp;
  delete[] ppptemp;
  delete[] tflag;
  delete[] flderiv;
  delete[] hsigall;
  delete[] strsig;
  delete[] y2;
  delete[] smdata;
  delete[] tempdata;
  delete[] strval;
  delete[] smflag;
  delete[] flag22;
  delete[] amptps;
  delete[] sntpl;
  delete[] a;
  delete[] aa;
  delete[] tte;
  delete[] tts;
}

void remove_longCR(double *data, long ndata, double tau, unsigned char *flag,
    double *crit, double &t_fft, double *filtdata) {
  long pos, count;
  double sigma, mm;

  fftw_init_threads();
  fftw_plan_with_nthreads (omp_get_max_threads());

fftw_complex  *fdata = new fftw_complex[ndata / 2 + 1];
  fftw_complex *ftail = new fftw_complex[ndata / 2 + 1];
  fftw_complex *fres = new fftw_complex[ndata / 2 + 1];

  double *tail = new double[ndata];
  double *res = new double[ndata];

  double tottail = 0;
  double tau_inv = 1. / tau;
#pragma omp parallel
  {
#pragma omp for reduction (+:tottail)
    for (long ii = 0; ii < ndata; ii++) {
      tail[ii] = exp(-(ndata - ii - 1) * tau_inv);
      tottail += tail[ii];
    }
    double tottail_inv = 1. / tottail;
#pragma omp for
    for (long ii = 0; ii < ndata; ii++) {
      tail[ii] *= tottail_inv;
    }
  }

  ticker tck = tic();

  //compute ffts
  fftw_plan fftplan = fftw_plan_dft_r2c_1d(ndata, data, fdata, FFTW_ESTIMATE);
  fftw_execute(fftplan);
  fftw_destroy_plan(fftplan);

  fftplan = fftw_plan_dft_r2c_1d(ndata, tail, ftail, FFTW_ESTIMATE);
  fftw_execute(fftplan);
  fftw_destroy_plan(fftplan);

  t_fft += toc(tck);

  //compute convolution
  double ndata_inv = 1. / ndata;
#pragma omp parallel
  {
#pragma omp for
    for (long ii = 0; ii < ndata / 2 + 1; ii++) {
      fres[ii][0] = (fdata[ii][0] * ftail[ii][0]
                     - fdata[ii][1] * ftail[ii][1]) * ndata_inv;
      fres[ii][1] = (fdata[ii][0] * ftail[ii][1]
                     + fdata[ii][1] * ftail[ii][0]) * ndata_inv;
    }
#pragma omp for
    for (long ii = 1; ii < ndata / 2 + 1; ii++) {
      double fact = 1.
          / (1 + pow(0.095 / 180.18 * double(ndata) / double(ii), 1.4));
      fres[ii][0] *= fact;
      fres[ii][1] *= fact;
    }
  }

  fftplan = fftw_plan_dft_c2r_1d(ndata, fres, res, FFTW_ESTIMATE);
  fftw_execute(fftplan);
  fftw_destroy_plan(fftplan);

  //define criterion
  sigma = 0;
#pragma omp parallel for reduction (+:sigma)
  for (long ii = 0; ii < ndata; ii++) {
    sigma += res[ii] * res[ii] * ndata_inv;
  }
  sigma = sqrt(sigma);

  if (VERBOSE) {
    fprintf(stderr, "mf crit p=%21.17g, m=%21.17g\n", crit[0], -2 * crit[0]);
  }

  count = 0;
  pos = 0;
  for (long ii = 0; ii < ndata;) {
    if (res[ii] > crit[ii]) {
      count = 0;
      mm = 0;
      while ((ii + count < ndata) && (res[ii + count] > crit[ii + count])) {
        if (mm < res[ii + count]) {
          pos = count;
          mm = res[ii + count];
        }
        count++;
      }
      for (long jj = ii + pos - 150;
          jj < ii + pos + long(tau * (log(mm / sigma) + log(3.0))); jj++)
        if (jj >= 0 && jj < ndata)
          flag[jj] = flag[jj] | 33;
      ii += count;
    } else {
      ii++;
    }
  }

  count = 0;
  pos = 0;
  for (long ii = 0; ii < ndata;) {
    if (-res[ii] > 2 * crit[ii]) {
      count = 0;
      mm = 0;
      while ((ii + count < ndata) && (-res[ii + count] > 2 * crit[ii + count])) {
        if (mm < -res[ii + count]) {
          pos = count;
          mm = -res[ii + count];
        }
        count++;
      }
      for (long jj = ii + pos - 150;
          jj < ii + pos + long(tau * (log(mm / sigma) + log(3.0))); jj++) {
        if (jj >= 0 && jj < ndata) {
          flag[jj] = flag[jj] | 33;
        }
      }
      ii += count;
    } else {
      ii++;
    }
  }

  if (filtdata != NULL) {
    for (long ii = 0; ii < ndata; ii++) {
      filtdata[ii] = res[ii];
    }
  }

  delete[] res;
  delete[] tail;
  delete[] fdata;
  delete[] ftail;
  delete[] fres;
}


void fillgaps(double y[], int ndata, double* yout, unsigned char* flag,
    double sign, bool oneside) {
  // data are assumed to vary linearly in every window

  int count, countp, countm;
  bool sp;

  int tteest = 0;
  for (long j = 0; j < ndata; j++) {
    if (!(y[j] < 1e100) || !(y[j] > -1e100)) {
      tteest++;
    }
  }
  if (tteest > 0 && VERBOSE) {
    fprintf(stderr,
            "NAN BEFORE FILLGAPS: %d UNVALID POINTS!!!!!!!!!!!!!!!!!\n",
            tteest);
  }
  const int margfit = 200;

  double *seriem = NULL, *tempdata1 = NULL, *xx2;
  double valtemp;

  double *a = new double[2];

  double *xx = new double[2 * margfit];
  double *yy = new double[2 * margfit];
  double *seriep = new double[margfit];
  double *tempdata2 = new double[margfit];

  ////copy data
  for (long i = 0; i < ndata; i++) {
    yout[i] = y[i];
  }

  count = 0;
  sp = 0;
  countm = 0;
  while ((countm < margfit) && ((flag[countm] & 1) == 0)) {
    countm++;
  }

  for (long i = 0; i < ndata; i++) {
    if (flag[i] & 1) {
      count++;
      sp = 0;
      if (i == ndata - 1)
        sp = 1;
    } else {
      sp = 1;
    }

    if (sp && count) {
      countp = 0;
      for (long j = 0; (countp < margfit) && (j + i < ndata - 1) && (!oneside);
          ) {
        if ((flag[i + j] & 1) == 0) {
          seriep[countp] = j;
          tempdata2[countp] = yout[i + j];
          countp++;
        }
        j++;
      }

      xx2 = new double[count];

      if (countm > 0) {
        seriem = new double[countm];
        tempdata1 = new double[countm];
        for (long ii = 0; ii < countm; ++ii) {
          seriem[ii] = ii;
        }
        cutdata(yout, i - count - countm, i - count - 1, tempdata1);
      }

      if (countm && countp) {
        mergedata(seriem, countm, seriep, countp, xx);
        mergedata(tempdata1, countm, tempdata2, countp, yy);
      } else {
        if (countm) {
          for (long j = 0; j < countm; j++) {
            xx[j] = seriem[j];
            yy[j] = tempdata1[j];
          }
        }
        if (countp) {
          for (long j = 0; j < countp; j++) {
            xx[j] = seriep[j];
            yy[j] = tempdata2[j];
          }
        }
      }

      if (countp) {
        for (long j = 0; j < countp; j++) {
          xx[countm + j] += double(countm + count);
        }
      }

      dpolyfit(xx, yy, countp + countm, 1, a);

      for (long ii = 0; ii < count; ++ii) {
        xx2[ii] = ii + countm;
      }
      // delete initialization
      for (long j = 0; j < count; j++) {
        valtemp = randg();
        yout[i + j - count] = a[0] + a[1] * xx2[j] + sign * valtemp;
      }

      if (countm) {
        if (seriem) {
          delete[] seriem;
        }
        if (tempdata1) {
          delete[] tempdata1;
        }
        seriem = NULL;
        tempdata1 = NULL;
      }
      delete[] xx2;

      if (i - count > margfit) {
        countm = margfit;
      }
      countp = 0;
      count = 0;
      sp = 0;
    }
  }

  int teest = 0;
  for (long j = 0; j < ndata; j++) {
    if (!(yout[j] < 1e100) || !(yout[j] > -1e100)) {
      yout[j] = y[j];
      teest++;
    }
  }
  if (teest > 0 && VERBOSE) {
    fprintf(stderr,
            "NAN IN FILLGAPS: %d UNVALID POINTS!!!!!!!!!!!!!!!!!!\n",
            teest);
  }

  delete[] a;
  delete[] xx;
  delete[] yy;
  delete[] seriep;
  delete[] tempdata2;
}

double randg() {
  double nombre_hasard;
  double t1 = (double(rand()) / RAND_MAX);
  double t2 = (double(rand()) / RAND_MAX);

  nombre_hasard = sqrt(-2 * log(t1)) * cos(2 * M_PI * t2);

  return nombre_hasard;
}
