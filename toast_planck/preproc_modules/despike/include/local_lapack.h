#ifdef SCIPY_MKL_H
#  include "mkl_lapack.h"
#else
#  ifdef SCIPY_OSX
#    include <Accelerate/Accelerate.h>
#  else
//#    include "clapack.h"
#  endif
#endif
