#ifndef SHDET_TYPES_H
#define SHDET_TYPES_H

// Miscellaneous
// Single/Double precision
typedef const int c_int ;
typedef unsigned int uint ;
typedef const unsigned int c_uint ;
//typedef unsigned long ulng ;
//typedef const unsigned long c_ulng ;
typedef unsigned int ulng ;
typedef const unsigned int c_ulng ;
typedef double flt ;
typedef const double c_flt ;

// Some constant, global variables
//- Length of the input array
#define l_rng_ttl 2000004
#define n_prm_shdet 100
#define mx_blmtr_mdl 10
#define n_smpl_prd 80 
#define n_ctp 65536 // 2 << 15
#define n_vr_shdet 14
//-- A sentinel value for the parameters I/O (Q&D)
#define shdet_sntnl -12345

#endif /* SHDET_TYPES_H */
