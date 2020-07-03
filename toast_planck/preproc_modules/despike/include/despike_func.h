#ifndef __DESPIKE_FUNC_H__
#define __DESPIKE_FUNC_H__

#define GLITCHINFO_NTERM 8

typedef struct {
  double longtau[GLITCHINFO_NTERM], longamp[GLITCHINFO_NTERM];
  double shorttau[GLITCHINFO_NTERM], shortamp[GLITCHINFO_NTERM];
  double slowtau[GLITCHINFO_NTERM], slowamp[GLITCHINFO_NTERM];
} GlitchInfoStruct;


typedef struct {
  double critcut;
  double factth;
  double ppa;
  double ppb;
  double selcrit;

  double crit;
  double critrem;
  double selsnail;

  long n_event_expected;
  long ringsize;
  GlitchInfoStruct glitchinfo;
  bool do_snail;

  double *kernel;
  long kernel_size;

  double *ampl2cr;
  double *ampl2crtmp;
  int *xerror;
  int *pos;

  long long *xcosm;
  int *xcosmfit;
  int *xspike;
  double *chisqs;
  double *amplspike;
  double *allhsig;
  double *allchi2;
  int *posp;
  double *amplp;
  double *ampltemplate;
  double *ampltemplateSH;
  double *allhsigp;

  int *pos2;
  long long *xcosm2;
  int *xspike2;
  double *chisqs2;
  double *amplspike2;
  double *allhsig2;
  int *posp2;
  double *amplp2;
  double *allhsigp2;

  int snailtempl;
  int template_size;
  int number_of_templates;
  int ntempl;
  double *templg;

} DespikeStruct;

DespikeStruct *alloc_despike_struct(const long n_event_expected,
    const long ringsize, GlitchInfoStruct glitchinfo, const long kernel_size,
    const double *kernel, /* kernel_size */
    bool do_snail);

void free_despike_struct(DespikeStruct *dsp);

void print_despike_struct(DespikeStruct *dsp);

void print_glitchinfo_struct(GlitchInfoStruct *glitchinfo);

bool despike_func(DespikeStruct *dsp,
    const long ring_number,
    const long signal_size,
    double *in_signal, /* signal_size */
    unsigned char *masked, /* signal_size */
    unsigned char *out_flag, /* signal size */
    double *out_residual, /* signal_size */
    double *out_glitch, /* signal_size*/
    int verbose = 0 /* verbosity */
    );


#endif
