#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef  struct {
  double ph;
  double val;
} datadoubleph;

int median_compardouble(const void *a, const void *b);
void medianmap(double *signal, unsigned char *flg, double *ph, double *ring_out,
               double *outphase, long ndata, long nring);

#ifdef __cplusplus
}
#endif
