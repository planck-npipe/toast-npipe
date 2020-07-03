#include <string.h>
#include <stdio.h>
#include "medianmap.h"

typedef  struct {
  long rk;
  double val;
} datadouble;


int compardouble(const void *a, const void *b)
{
  datadouble *pa = (datadouble *) a;
  datadouble *pb = (datadouble *) b;

  if (pa->val>pb->val) return(1);
  if (pb->val>pa->val) return(-1);
  return((int) 0);
}

int median_compardouble(const void *a, const void *b)
{
  datadoubleph *pa = (datadoubleph *) a;
  datadoubleph *pb = (datadoubleph *) b;

  if (pa->val>pb->val) return(1);
  if (pb->val>pa->val) return(-1);
  return((int) 0);
}


void medianmap(double *signal, unsigned char *flg, double *ph, double *ring_out,
               double *outphase, long ndata, long nring)
{
  datadoubleph **Tri_Casier = (datadoubleph **) malloc(sizeof(datadoubleph *)
                                                       * nring);
  long *N_Tri_Casier = (long *) malloc(sizeof(long)*nring);
  long i;

  int ttest=0;

  memset(N_Tri_Casier, 0, nring*sizeof(long));

  ring_out[0] = 0;
  ring_out[nring-1] = 0;

  // Arrange the signal into bins

  for (i=0; i<ndata; i++) if (flg[i] == 0) {
    int idx=((int) ph[i])%nring;
    if (N_Tri_Casier[idx] == 0) {
      Tri_Casier[idx] = (datadoubleph *) malloc(sizeof(datadoubleph));
    }
    else {
      Tri_Casier[idx] = (datadoubleph *) realloc(Tri_Casier[idx],
                                                 sizeof(datadoubleph)
                                                 * (N_Tri_Casier[idx]+1));
    }
    Tri_Casier[idx][N_Tri_Casier[idx]].val = signal[i];
    Tri_Casier[idx][N_Tri_Casier[idx]].ph = ph[i];
    N_Tri_Casier[idx]++;
  }

  // Measure the median in each hit bin

  for (i=0; i<nring; i++) {
    if (N_Tri_Casier[i] > 0) {
      qsort(Tri_Casier[i], N_Tri_Casier[i], sizeof(datadoubleph),
            median_compardouble);
      ring_out[i] = Tri_Casier[i][N_Tri_Casier[i]/2].val;
      outphase[i] = Tri_Casier[i][N_Tri_Casier[i]/2].ph;
    }
  }

  // Fill empty bins with the mean of the nearest valid bins

  for (i=1; i<nring-1; i++) {
    int ii, j1=0, j2=0;
    int in_gap=0;
    if (N_Tri_Casier[i] == 0) {
      if (!in_gap) {
	// New gap, find the edges
	for (ii=0, j1=i; ii<nring && N_Tri_Casier[j1]==0; ++ii) {
	  j1--;
	  if (j1 == -1) j1 = nring-1; // Wrap around
	}
	for (ii=0, j2=i; ii<nring && N_Tri_Casier[j2]==0; ++ii) {
	  j2++;
	  if (j2 == nring) j2 = 0; // Wrap around
	}
      }
      if (N_Tri_Casier[j1]==0 || N_Tri_Casier[j2]==0) {
	if (!ttest)
	  fprintf(stderr, "WARNING : Not enough valid samples median map may "
                  "be wrong\n");
	ring_out[i] = 0;
	ttest = 1;
      } else {      
	ring_out[i] = 0.5*(ring_out[j1]+ring_out[j2]);
	outphase[i] = 0.5*(outphase[j1]+outphase[j2]);
      }
      in_gap = 1;
    } else {
      free(Tri_Casier[i]);
      in_gap = 0;
    }
  }
  if (N_Tri_Casier[0]!=0)
    free(Tri_Casier[0]);
  if (N_Tri_Casier[nring-1]!=0)
    free(Tri_Casier[nring-1]);

  free(Tri_Casier);
  free(N_Tri_Casier);
}
