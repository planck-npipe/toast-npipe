//#include "cshdet.h"
#include "shdet.hpp"

int cSHDet_f(double *rng_arry, double *adc_ctp, double *prm_rnng, double *prm_inst, double *prm_rdout, double *prm_opt, double *prm_blmtr ) {

  SHDet_f(
	  reinterpret_cast<flt(&)[l_rng_ttl]>(*rng_arry),
	  reinterpret_cast<flt(&)[n_ctp]>(*adc_ctp),
	  reinterpret_cast<flt(&)[n_prm_shdet]>(*prm_rnng),
	  reinterpret_cast<flt(&)[n_prm_shdet]>(*prm_inst),
	  reinterpret_cast<flt(&)[n_prm_shdet]>(*prm_rdout),
	  reinterpret_cast<flt(&)[n_prm_shdet]>(*prm_opt),
	  reinterpret_cast<flt(&)[n_prm_shdet][mx_blmtr_mdl]>(*prm_blmtr)
	  );
  
  return 0;
}
