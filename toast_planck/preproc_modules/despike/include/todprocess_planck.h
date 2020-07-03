#ifndef TODPROCESS
#define TODPROCESS

#include <iostream>
#include <cmath>
#include <string>
#include <fcntl.h>

extern int VERBOSE;

using namespace std;

#include <sys/time.h>

typedef struct timeval ticker;

ticker tic(void);
double toc(ticker &tck);

void dpolyfit(double x[], double y[], int ndata, int norder, double *a);
void remove_poly(double y[], int ndata, int norder, double* yout,
    unsigned char *flag = NULL);

void cutdata(double y[], int indm, int indp, double *yout);
void cutdata(unsigned char y[], int indm, int indp, unsigned char *yout);
void mergedata(double y1[], int ndata1, double y2[], int ndata2, double *yout);
void sort(double y[], int nn, double *yout, int *nrel, unsigned char* flag = NULL);

double fitMultiTemplates(double *smdata, unsigned char *smflag, double hmean,
    double mhsig, long nn, long *pos, long npos, double *ShortTau,
    double *LongTau, double *ShortAmp, double *LongAmp, double *SnailTau,
    double *SnailAmp, double ppa, double ppb, long nterm, double *aa,
    unsigned char *sntpl, bool snail, bool snailtempl, bool verb);

void findspike_clean(double y[], int ndata, double* yout, unsigned char* flag,
    unsigned char* pflag, long long *xcosm, int *pos, int *xspike,
    double *amplspike, double *allhsig, double *allchi2, int *posp,
    double *amplp, double *allhsigp, double criterion, double sign0,
    double *esigdet, const double *kernel, const int sker, double *taulg,
    double *atplg, double *taush, double *atpsh, double *tausn, double *atpsn,
    long nterm, long nevent, double *ampl2cr, double *templates, long Nbtemp,
    long ndtemp, double critrem, double *ampltemplate, double *ampltemplateSH,
    double selcrit, double selsnail, double ppa, double ppb, bool snail,
    bool snailtempl);

void remove_longCR(double *data, long ndata, double tau, unsigned char *flag,
    double *crit, double &t_fft, double* filtdata = NULL);

void fillgaps(double y[], int ndata, double* yout, unsigned char* flag,
    double sign, bool oneside);

void butterworth(double y[], int ndata, double f_lp, int orderB, double *yout,
    double *bfilter, bool apodize, int napod, bool overwrite, double &t_fft);
double* apodwindow(int ns, int nn);

int compare_doubles(const void *a, const void *b);

double randg();

void minmax(double* data, int ndata, double *min, double *max, int *posmin,
    int *posmax, unsigned char *flag = NULL);

#endif
