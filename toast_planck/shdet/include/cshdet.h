// C interface for calling libshdet

#ifndef CSHDET_H
#define CSHDET_H

#include "shdet_types.h"

ulng l_rng_ttl_c = l_rng_ttl;
uint n_prm_shdet_c = n_prm_shdet;
uint mx_blmtr_mdl_c = mx_blmtr_mdl;
uint n_smpl_prd_c = n_smpl_prd;
uint n_ctp_c = n_ctp;
uint n_vr_shdet_c = n_vr_shdet;
int shdet_sntnl_c = shdet_sntnl;

//#ifdef __cplusplus
//extern "C" {
//#endif
  
int cSHDet_f(double *, double *, double *, double *, double *, double *, double * );

//#ifdef __cplusplus
//}
//#endif

#endif /* CSHDET_H */

