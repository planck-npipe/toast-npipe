#ifndef SHDET_HPP
#define SHDET_HPP

//- STL: I/O
#ifndef IOSTRM_H
#define IOSTRM_H
#include <iostream>
#endif
//- STL: MATH
#ifndef MTH_H
#define MTH_H
#include <cmath>
#endif
//- C STDLIB
#ifndef STDLB_H
#define STDLB_H
#include <stdlib.h>
#endif

#include <stdexcept>

#include "shdet_types.h"

// Function declarations
void inpt_prmtr_f ( flt ( &) [ n_prm_shdet ], flt ( & ) [ n_prm_shdet ], flt ( & ) [ n_prm_shdet ], flt ( & ) [ n_prm_shdet ], flt ( & )  [ n_prm_shdet ] [ mx_blmtr_mdl ] ) ;
void SHDet_f (
	      flt ( & ) [ l_rng_ttl ],
	      flt ( & ) [ n_ctp ],
	      flt ( & ) [ n_prm_shdet ],
	      flt ( & ) [ n_prm_shdet ],
	      flt ( & ) [ n_prm_shdet ],
	      flt ( & ) [ n_prm_shdet ],
	      flt ( & ) [ n_prm_shdet ] [ mx_blmtr_mdl ] ) ;
void SHDet_smltn_f( flt ( & ) [ n_prm_shdet ], flt ( & ) [ n_prm_shdet ], flt ( & ) [ n_prm_shdet ], flt ( & )  [ n_prm_shdet ] [ mx_blmtr_mdl ], c_uint &, c_flt &, flt ( & ) [ mx_blmtr_mdl ], \
                    flt ( & ) [ n_vr_shdet ], flt &, c_flt & ) ;

#endif /* SHDET_HPP */
