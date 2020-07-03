#include "shdet.hpp"

//------------------
// SUB-FUNCTIONS
//------------------

// Parameters I/O (default. Significant reduction of the full functionality of SHDet. Several are not explicitly used in the simulation code here, but can be incorporated as needed)
void inpt_prmtr_f ( flt ( & prm_rnng_out ) [ n_prm_shdet ],  flt ( & prm_inst_out ) [ n_prm_shdet ], flt ( & prm_rdout_out ) [ n_prm_shdet ], flt  ( & prm_opt_out ) [ n_prm_shdet ], flt ( & prm_blmtr_out ) [ n_prm_shdet ] [ mx_blmtr_mdl ] )
{ 
/*************************/
/* A) Running parameters */
/************************/
/* 1. Number of modules of the bolometer */
( prm_rnng_out ) [ 0 ] = 1 ;
/* 2. Sink temperature if constant */
( prm_rnng_out ) [ 1 ] = 0.1 ; // K
/* 3. frequency channel */
( prm_rnng_out ) [ 2 ] = 100 ;
/* 4. Granularity of the integration step */
( prm_rnng_out ) [ 3 ] = 10 ;
/* 5. ADC OFF/ON=0/1 */
( prm_rnng_out ) [ 4 ] = 0 ;
/* 6. BIT number ADC (16 is the flight case) */
( prm_rnng_out ) [ 5 ] = 16 ;
/* 7. V_MAX (V) */
( prm_rnng_out ) [ 6 ] = 12 ;
/* 8. V_MIN (V, it should be -V_MAX) */
( prm_rnng_out ) [ 7 ] = -12 ;
/* 9. S_PHASE: value in sample units */
( prm_rnng_out ) [ 8 ] = 0 ;
/* 10 "Burn in time" */
( prm_rnng_out ) [ 9 ] = 902 ; 
/* Full sampling frequency
(exact value from the flight configuration: NSAMPLE = 40, NBLANK = 0, MPHASE = 0, FDIV = 124, TJFET = 125.00000->FMOD = 90.187590) */
( prm_rnng_out ) [ 10 ] = ( ( flt ) 40 ) * ( ( flt ) 180.37518 ) ;
/* Integration step: dlt_t */
( prm_rnng_out ) [ 11 ] = ( ( flt ) 1 ) / ( ( ( prm_rnng_out ) [ 10 ] ) * ( ( prm_rnng_out ) [ 3 ] ) ) ;
/* Non-optical input power at each module (glitches, or else. See mdl_pwr_vct_f) */
( prm_rnng_out ) [ 12 ] = 0 ;
( prm_rnng_out ) [ 13 ] = 0 ; 
( prm_rnng_out ) [ 14 ] = 0 ;
( prm_rnng_out ) [ 15 ] = 0 ;
( prm_rnng_out ) [ 16 ] = 0 ; 
/* Range of the ADC voltage conversion. +/- maximum value (V) */
( prm_rnng_out ) [ 17 ] = 5.1 ;
/* Amplitude of the 4K lines
(all 3 harmonics, or more and shift within the 18 DSN samples
will be incorporated later on -easy. Now it is only for some checks) */
( prm_rnng_out ) [ 18 ] = 0 ; // 4K lines on/off
( prm_rnng_out ) [ 19 ] = 5e-17 ; // Watts
( prm_rnng_out ) [ 20 ] = 1e-17 ; // Watts
/* Rise/Lower time of the square wave: on/off -> 1/0 */
( prm_rnng_out ) [ 30 ] = 0 ;
/* Static raw sample model ON/OFF (1/0)*/
( prm_rnng_out ) [ 35 ] = 0 ;
/* WN RMS at the raw sample level in V (NS_DSN/sqrt(40)*V_ADC_RNG/2^N_BIT (100 DSN->0.00246). OFF: 0 */
( prm_rnng_out ) [ 36 ] = 0 ;
/* Seed used to activate the Marsaglia random generator */
( prm_rnng_out ) [ 37 ] = 1 ;
/* Offset to be applied to the raw constant term */
( prm_rnng_out ) [ 38 ] = 0 ;
/* Switch on/off the offset of the raw constant term. on=1 (default), off=0 */
( prm_rnng_out ) [ 39 ] = 1 ;
/* NB: parameter #50 is reserved for passing the number of samples to be simulated (default l_rng_ttl) */

/***************************************************************/
/* B) Instrumental parameters (check/set limit of # in shdet.h)*/
/***************************************************************/
( prm_inst_out ) [ 0 ] = ( flt ) 200 ;
flt r_c_1 = ( flt ) 1e4, r_c_2 = ( flt ) 1e4 ; // Ohm
flt r_c_3 = ( flt ) 1e3, r_c_4 = ( flt ) 1e3, c_c = ( flt ) 1e-7 ; // Ohm, Faraday
flt tau_c_3 = r_c_3 * c_c ; //s
flt tau_c_4 = r_c_4 * c_c ;
( prm_inst_out ) [ 1 ] = ( prm_rnng_out ) [ 11 ] / tau_c_3 ;
( prm_inst_out ) [ 2 ] = ( flt ) 1 - ( prm_rnng_out ) [ 11 ] / tau_c_4 ;
flt lmbd_c_1_12 = r_c_1 / ( r_c_1 + r_c_2 ) ;
( prm_inst_out ) [ 3 ] = ( ( flt ) 1 ) - lmbd_c_1_12 ;
flt r_d_1 = ( flt ) 51e3, c_d_1 = ( flt ) 1e-6 ; // Ohm, Faraday
flt r_d_2 = ( flt ) 51e3, c_d_2 = ( flt ) 1e-6 ;
flt tau_d_1 = r_d_1 * c_d_1 ; // s
flt tau_d_2 = r_d_2 * c_d_2 ;
flt dlt_d_t = ( r_d_2 - r_d_1 ) * c_d_2 ;
flt lmbd_d_1 = ( prm_rnng_out ) [ 11 ] / tau_d_1 ;
( prm_inst_out ) [ 4 ] = ( prm_rnng_out ) [ 11 ] / tau_d_2 ;
flt lmbd_d_2_2 = dlt_d_t / tau_d_2 ;
( prm_inst_out ) [ 5 ] = ( ( flt ) 1 ) - ( prm_inst_out ) [ 4 ] ;
flt one_mns_lmbd_d_2_2 = ( ( flt ) 1 ) - lmbd_d_2_2 ;
( prm_inst_out ) [ 6 ] =  ( ( flt ) 1 ) - lmbd_d_1 * one_mns_lmbd_d_2_2 ;
( prm_inst_out ) [ 7 ] = lmbd_d_1 * lmbd_d_2_2 ;
( prm_inst_out ) [ 8 ]= lmbd_d_1 * one_mns_lmbd_d_2_2 ;
flt r_e_1 = ( flt ) 1e4, r_e_2 = ( flt ) 5.1e4 ; // Ohm
( prm_inst_out ) [ 9 ] = - r_e_2 / r_e_1 ;
flt r_f_1 = ( flt ) 1e4, c_f_1 = ( flt ) 1e-8 ; // Ohm, Faraday
flt r_f_2 = ( flt ) 2e4, r_f_3 = ( flt ) 1e4 ;
flt tau_f_1 = r_f_1 * c_f_1 ; // s
( prm_inst_out ) [ 10 ] = ( ( prm_rnng_out ) [ 11 ] ) / tau_f_1 ;
( prm_inst_out ) [ 11 ] = -( ( ( flt ) 1 ) + r_f_3 / r_f_2 ) ; // NB_1
( prm_inst_out ) [ 12 ] = ( ( flt ) 1 ) - ( prm_inst_out ) [ 10 ] ;
flt r_h_1 = ( flt ) 5.1e5, c_h_1 = ( flt ) 1e-6 ; // Ohm, Faraday
flt r_h_2 = ( flt ) 1.87e4, c_h_2 = ( flt ) 1e-8 ;
flt r_h_3 = ( flt ) 3.74e4, c_h_3 = ( flt ) 1e-8 ;
flt r_h_4 = ( flt ) 1e3, r_h_5 = ( flt ) 1e3 ;
flt tau_h_11 = r_h_1 * c_h_1 ; // s
flt tau_h_21 = r_h_2 * c_h_1 ;
flt tau_h_22 = r_h_2 * c_h_2 ;
flt tau_h_33 = r_h_3 * c_h_3 ;
( prm_inst_out ) [ 13 ] = ( flt ) 1 ;
( prm_inst_out ) [ 14 ] = ( ( prm_rnng_out ) [ 11 ] ) * ( ( ( flt ) 1 ) / tau_h_11  + ( ( flt ) 1 ) / tau_h_21 ) ;
( prm_inst_out ) [ 15 ] = ( ( flt ) 1 ) - ( ( prm_inst_out ) [ 14 ] ) ;
( prm_inst_out ) [ 16 ] = - ( ( prm_rnng_out ) [ 11 ] ) / tau_h_21 ;
( prm_inst_out ) [ 17 ] = - ( ( prm_rnng_out ) [ 11 ] ) / tau_h_21 * ( ( ( flt ) 1 ) + r_h_4 / r_h_5 ) ;
( prm_inst_out ) [ 18 ] = ( ( prm_rnng_out ) [ 11 ] ) / tau_h_22 ;
( prm_inst_out ) [ 19 ] = - ( prm_inst_out ) [ 18 ] ;
( prm_inst_out ) [ 20 ] = ( ( flt ) 1 ) - ( ( prm_rnng_out ) [ 11 ] ) * \
                            ( ( ( flt ) 1 ) / tau_h_22 + ( ( flt ) 1 ) / tau_h_33 ) ;
( prm_inst_out ) [ 21 ] =  -( ( prm_rnng_out ) [ 11 ] ) / tau_h_22 * \
                             ( ( ( flt ) 1 ) + r_h_5 / r_h_4 * ( ( ( flt ) 1 ) + r_h_2 / r_h_3 ) ) ;
( prm_inst_out ) [ 22 ] = ( ( prm_rnng_out ) [ 11 ] ) / tau_h_33 ;
( prm_inst_out ) [ 23 ]  = ( ( flt ) 1 ) +  ( ( prm_rnng_out ) [ 11 ] ) / tau_h_33 * r_h_5 / r_h_4 ;
( prm_inst_out ) [ 24 ]  =  ( ( flt ) 1 ) + r_h_5 / r_h_4 ;
/************************************************/
/* C) Readout parameters + bolometer resistance */
/************************************************/
flt fct_seb = ( flt ) 4096 / ( flt ) 4095 ;
flt fct_seb_tri = ( flt ) 0.926994383335113525390625000000 ;
flt fct_seb_tran = ( flt ) 0.913063108921051025390625000000 ;
( prm_rdout_out ) [ 20 ] = ( flt ) 408.98 ;
( prm_rdout_out ) [ 20 ] *= fct_seb ;
( prm_rdout_out ) [ 21 ] = ( flt ) 288.5 ;
( prm_rdout_out ) [ 21 ] *= ( fct_seb * fct_seb_tri ) ;
( prm_rdout_out ) [ 22 ] = ( flt ) 1691.69265 ;
( prm_rdout_out ) [ 22 ] *= ( fct_seb * fct_seb_tran ) ;
/* r_str */
( prm_rdout_out ) [ 0 ] = ( flt ) 60.020000 ; // 69.75 ; // Ohm FFP7: 100 ; 69.75 (SEB, 10_143_5) // Ohm
/* t_g */
( prm_rdout_out ) [ 1 ] = ( flt ) 18.320000 ; // 19.21 ; // K FFP7: 17 ; // K
/* exp_m */
( prm_rdout_out ) [ 2 ] = ( flt ) 0.5 ;
/* Parameters from the transient signal and the square signal */
/* Rise/Lower time for the square wave below */
/* R_ZAP */
( prm_rdout_out ) [ 4 ] = ( flt ) 4e30 ; // Ohm
flt v_in_bias = ( flt ) 2.5 ; // V
flt r_3 = ( flt ) 1e4 ; flt r_4 = ( flt ) 5.6e3 ; // Ohm
flt r_1    = ( flt ) 1e4 ;  // Ohm
flt r_2    = ( flt ) 2e4 ;   // Ohm
flt r_trng = ( flt ) 240e3 ; // Ohm
flt c_trng = ( flt ) 20e-9 ; // F
flt tau_trng = r_trng * c_trng ; // s
flt vbal_mx = ( ( ( flt ) 1 ) + r_3 / r_4 ) * v_in_bias ;
/*  The correct factor is 4096 (specifications of DAC8143), but SEB uses 4095 (so be careful if comparing) */
flt vtrans_mx = -( prm_rdout_out [ 20 ] ) / ( ( flt ) 4096 ) * vbal_mx ;
/* vtrans_mx_eff */
( prm_rdout_out ) [ 5 ] = lmbd_c_1_12 * vtrans_mx ;
flt c_stry = ( flt ) 1.16500e-10 ; // 1.48500e-10
/* mu_bias */
flt mu_bias =  ( ( flt ) 1 ) / ( ( ( flt ) 1 ) + r_1 / r_2 ) * ( vbal_mx ) / tau_trng * ( prm_rdout_out [ 21 ] ) / ( ( flt ) 4096 ) ;
/* mu_trans */
flt mu_trans = ( vbal_mx ) * ( prm_rdout_out [ 20 ] ) / ( ( flt ) 4096 ) * ( prm_rdout_out [ 22 ] ) / ( ( flt ) 4096 ) * \
               ( ( flt ) 1 ) / ( ( ( flt ) 1 ) + r_2 / r_1 ) ;
/* C_BIAS_1 and C_BIAS_2 go to the equations. If they are equal, one can
simplify the equations and the parameters: 'mu_bias' and 'mu_trans'
are the ones that go to the code (see shdet v5 or earlier versions) */
/* C_BIAS_1 */
( prm_rdout_out ) [ 6 ] = ( flt ) 4.9691252e-12 ; // 4.9691252e-12; 5.0413221e-12 ; // FFP7: 4.7e-12 ; // F
/* C_BIAS_2 */
( prm_rdout_out ) [ 7 ] = ( flt )  4.9691252e-12 ; // 4.9691252e-12 ; 4.7035415e-12 ; // 4.7035415e-12 ; // 4.8927415e-12 ; // FFP7: 4.7e-12 ; // F
/* 'DET_M': Determinant of the equations for V_stray and V_CBIAS_1 */
( prm_rdout_out ) [ 8 ] = c_stry * ( ( ( prm_rdout_out ) [ 6 ] ) + ( ( prm_rdout_out ) [ 7 ] ) ) + ( ( prm_rdout_out ) [ 6 ] ) * ( ( prm_rdout_out ) [ 7 ] ) ;
/* ALPHA_2 = ( c_bias_2 - c_bias_1 ) / r_zap / det_m */
flt alph_2 = ( ( prm_rdout_out ) [ 7 ] - ( prm_rdout_out ) [ 6 ] ) / \
             ( ( prm_rdout_out ) [ 4 ] ) / ( ( prm_rdout_out ) [ 8 ] ) ;
/* LAMBDA_2 = alpha_2 * dlt_t */
( prm_rdout_out ) [ 9 ] = alph_2 * ( ( prm_rnng_out ) [ 11 ] ) ;
/* ALPHA_3 = 2 * c_bias_1 * c_bias_2 / det_m */
flt alph_3 = ( ( flt ) 2 ) * ( ( prm_rdout_out ) [ 6 ] ) * ( ( prm_rdout_out ) [ 7 ] ) / \
             ( ( prm_rdout_out ) [ 8 ] ) ;
/* LAMBDA_3 = mu_trans * alph_3 */
( prm_rdout_out ) [ 10 ] = mu_trans * alph_3 ;
/* LAMBDA_4 =  (mu_trans * alph_2 + mu_bias * alph_3) * dlt_t */
( prm_rdout_out ) [ 11 ] = ( mu_trans * alph_2 + mu_bias * alph_3 ) * ( ( prm_rnng_out ) [ 11 ] ) ;
/* LAMBDA_5 = mu_bias * alph_2 * dlt_t^2 */
( prm_rdout_out ) [ 12 ] = mu_bias * alph_2 * ( ( prm_rnng_out ) [ 11 ] ) * ( ( prm_rnng_out ) [ 11 ] ) ;
/* V_C_BIAS_1 */
/* BETA_2 =  ( 2 * c_stry + c_bias_2 ) / r_zap / det_m */
flt bt_2 = ( ( ( flt ) 2 ) * c_stry + ( ( prm_rdout_out ) [ 7 ] ) ) / \
           ( ( prm_rdout_out ) [ 4 ] ) / ( ( prm_rdout_out ) [ 8 ] ) ;
/* GAMMA_2 = 1 - beta_2 * dlt_t */
( prm_rdout_out ) [ 13 ] = ( ( flt ) 1 ) - bt_2 * ( ( prm_rnng_out ) [ 11 ] ) ;
/* BETA_3 = 2 * c_stray * c_bias_2 / det_m */
flt bt_3 = ( ( flt ) 2 ) * c_stry * ( ( prm_rdout_out ) [ 7 ] ) / ( ( prm_rdout_out ) [ 8 ] ) ;
/* GAMMA_3 = ( mu_trans * beta_2 + mu_bias * beta_3 ) * dlt_t */
( prm_rdout_out ) [ 14 ] = ( mu_trans * bt_2 + mu_bias * bt_3 ) * ( ( prm_rnng_out ) [ 11 ] ) ;
/* GAMMA_4 = mu_trans * beta_3 */
( prm_rdout_out ) [ 15 ] = mu_trans * bt_3 ;
/* GAMMA_5 = mu_bias * beta_2 * dlt_t^2 */
( prm_rdout_out ) [ 16 ] = mu_bias * bt_2 * ( ( prm_rnng_out ) [ 11 ] ) * ( ( prm_rnng_out ) [ 11 ] ) ;

/* C_STRAY is needed in the code as well */
( prm_rdout_out ) [ 17 ] = c_stry ;

/* Rise/Lower time of the square wave in very fast samples (on/off setting */
flt t_u = 0.0e-6 ; // second. Rise ('up') time. Check lab measurements for values depending on each detector. Usually, around 50 micros.
flt t_d = 0.0e-6 ; // second. Lower ('down') time. Check lab measurements for values depending on each detector
( prm_rdout_out ) [ 18 ] = t_u / ( prm_rnng_out ) [ 11 ] ;
( prm_rdout_out ) [ 19 ] = t_d / ( prm_rnng_out ) [ 11 ] ;


/***************************************************************************/
/* D) Default optical power parameters:  */
/***************************************************************************/
( prm_opt_out ) [ 0 ] = 2.9e-13 ; // constant optical load W 
( prm_opt_out ) [ 1 ] = 2.1e-13 ; // gain W/K_CMB
( prm_opt_out ) [ 2 ] = 18 ; // Period in units of the sample time. From opt_pwr_vct_f: 1 / ( ( prm_opt_arg ) [ 2 ] ) / dlt_t_dsn_arg ; // Hz
/* Intentional blank in case other parameters for basic response are considered */
/* Planets: teen numbers */
( prm_opt_out ) [ 10 ] = 1e-13 ; // W. Peak amplitude of the planet
( prm_opt_out ) [ 11 ] = 7.0 ; // arcmin. FWHM beam + planet
( prm_opt_out ) [ 12 ] = 40 ; // # planet consecutive crossings
( prm_opt_out ) [ 13 ] = 0 ; // optional shift for the peak position of the planet
( prm_opt_out ) [ 20 ] = 240 ;
( prm_opt_out ) [ 30 ] = 0 ; // Offset of the optical load in W
( prm_opt_out ) [ 31 ] = 1 ; // Switch on/off offset of the optical load. on=1 (default), off=0.

/***************************************************************************/
/* E) Bolometer module parameters: Set in shdet.h */
/***************************************************************************/
/* Typical values. A mixture of different sources. */
/* Thermal conductance at T_0: gs, j->0 */
( prm_blmtr_out ) [ 0 ] [ 0 ] = ( flt ) 4.7599998e-11 ; // 7.0800003e-11 ; // FFP7: 6e-11 ;
( prm_blmtr_out ) [ 0 ] [ 1 ] = ( flt ) 6e-12 ;
( prm_blmtr_out ) [ 0 ] [ 2 ] = ( flt ) 6e-12 ;
( prm_blmtr_out ) [ 0 ] [ 3 ] = ( flt ) 6e-12 ;
( prm_blmtr_out ) [ 0 ] [ 4 ] = ( flt ) 6e-12 ;

/* Thermal conductance at T_0 (NB): gs, j->j+1
NB: Assuming they were measured at T_0, although they indeed depend on the module temperature
NB2: Notice that there are values for up to 5 bolometer modules (0 to 4). If one is going to consider
more modules, one should extent the list of parameters. Probably, it would mean that one needs a _different_ bolometer model
rather than more modules of this particular LFER-like model */
( prm_blmtr_out ) [ 1 ] [ 0 ] = ( flt ) 1e-11 ;
( prm_blmtr_out ) [ 1 ] [ 1 ] = ( flt ) 1e-11 ;
( prm_blmtr_out ) [ 1 ] [ 2 ] = ( flt ) 1e-12 ;
( prm_blmtr_out ) [ 1 ] [ 3 ] = ( flt ) 4e-12;
/* exp_bt, j->0 */
( prm_blmtr_out ) [ 2 ] [ 0 ] = ( flt ) 1.2800000 ; // 1.51 ; // FFP7: 1.50
( prm_blmtr_out ) [ 2 ] [ 1 ] = ( flt ) 1.5 ;
( prm_blmtr_out ) [ 2 ] [ 2 ] = ( flt ) 1.5 ;
( prm_blmtr_out ) [ 2 ] [ 3 ] = ( flt ) 1.5 ;
( prm_blmtr_out ) [ 2 ] [ 4 ] = ( flt ) 1.5 ;
/* exp_bt, j->j+1 */
( prm_blmtr_out ) [ 3 ] [ 0 ] = ( flt ) 1.5 ;
( prm_blmtr_out ) [ 3 ] [ 1 ] = ( flt ) 1.5 ;
( prm_blmtr_out ) [ 3 ] [ 2 ] = ( flt ) 1.5 ;
( prm_blmtr_out ) [ 3 ] [ 3 ] = ( flt ) 1.5 ;
/* C_0,j / dlt_t */
( prm_blmtr_out ) [ 4 ] [ 0 ] = ( ( flt ) 5e-12 ) / ( ( prm_rnng_out ) [ 11 ] ) ; // 4.76e-13, Alexandre:  2.151e-11
( prm_blmtr_out ) [ 4 ] [ 1 ] = ( ( flt ) 2.5e-11 ) / ( ( prm_rnng_out ) [ 11 ] ) ;
( prm_blmtr_out ) [ 4 ] [ 2 ] = ( ( flt ) 2.5e-11 ) / ( ( prm_rnng_out ) [ 11 ] ) ;
( prm_blmtr_out ) [ 4 ] [ 3 ] = ( ( flt ) 2.5e-11 ) / ( ( prm_rnng_out ) [ 11 ] ) ;
( prm_blmtr_out ) [ 4 ] [ 4 ] = ( ( flt ) 2.5e-11 ) / ( ( prm_rnng_out ) [ 11 ] ) ;
/* exp_gmm, j */
( prm_blmtr_out ) [ 5 ] [ 0 ] = ( flt ) 1.35 ; // 1.61 ; // FFP7: 1.60
( prm_blmtr_out ) [ 5 ] [ 1 ] = ( flt ) 1.6 ;
( prm_blmtr_out ) [ 5 ] [ 2 ] = ( flt ) 1.6 ;
( prm_blmtr_out ) [ 5 ] [ 3 ] = ( flt ) 1.6 ;
( prm_blmtr_out ) [ 5 ] [ 4 ] = ( flt ) 1.6 ;
/* (exp_bt, j->0) + 1 */
( prm_blmtr_out ) [ 6 ] [ 0 ] = ( prm_blmtr_out ) [ 2 ] [ 0 ] + ( flt ) 1 ;
( prm_blmtr_out ) [ 6 ] [ 1 ] = ( prm_blmtr_out ) [ 2 ] [ 1 ] + ( flt ) 1 ;
( prm_blmtr_out ) [ 6 ] [ 2 ] = ( prm_blmtr_out ) [ 2 ] [ 2 ] + ( flt ) 1 ;
( prm_blmtr_out ) [ 6 ] [ 3 ] = ( prm_blmtr_out ) [ 2 ] [ 3 ] + ( flt ) 1 ;
( prm_blmtr_out ) [ 6 ] [ 4 ] = ( prm_blmtr_out ) [ 2 ] [ 4 ] + ( flt ) 1 ;
/* (exp_bt, j->j+1) + 1 */
( prm_blmtr_out ) [ 7 ] [ 0 ] = ( prm_blmtr_out ) [ 3 ] [ 0 ] + ( flt ) 1 ;
( prm_blmtr_out ) [ 7 ] [ 1 ] = ( prm_blmtr_out ) [ 3 ] [ 1 ] + ( flt ) 1 ;
( prm_blmtr_out ) [ 7 ] [ 2 ] = ( prm_blmtr_out ) [ 3 ] [ 2 ] + ( flt ) 1 ;
( prm_blmtr_out ) [ 7 ] [ 3 ] = ( prm_blmtr_out ) [ 3 ] [ 3 ] + ( flt ) 1 ;
/* END */
} // inpt_prmtr_f

//- SHDet_f 
void SHDet_f( flt ( & rng_arry ) [ l_rng_ttl ], flt ( & adc_ctp ) [ n_ctp ], flt ( & prm_rnng_in ) [ n_prm_shdet ],  flt ( & prm_inst_in ) [ n_prm_shdet ], flt ( & prm_rdout_in ) [ n_prm_shdet ], flt ( & prm_opt_in ) [ n_prm_shdet ], flt ( & prm_blmtr_in ) [ n_prm_shdet ] [ mx_blmtr_mdl ] )
{ 
  // 1.- Parameters: Getting default values
  // Local parameter groups
  flt prm_rnng [ n_prm_shdet ] ;
  flt prm_inst [ n_prm_shdet ] ;
  flt prm_rdout [ n_prm_shdet ] ;
  flt prm_opt [ n_prm_shdet ] ;
  flt prm_blmtr [ n_prm_shdet ] [ mx_blmtr_mdl ] ;
  inpt_prmtr_f ( prm_rnng, prm_inst, prm_rdout, prm_opt, prm_blmtr ) ;
  //-- Checking for changes in the default parameters (Q&D)
  uint n_prm_cngd = 0 ;
    for ( uint i_prm = 0; i_prm != n_prm_shdet ; i_prm++ )
    {   if ( prm_rnng_in [ i_prm ] != shdet_sntnl ) { prm_rnng [ i_prm ] = prm_rnng_in [ i_prm ] ; ++ n_prm_cngd ; }
        if ( prm_inst_in [ i_prm ] != shdet_sntnl ) { prm_inst [ i_prm ] = prm_inst_in [ i_prm ] ; ++ n_prm_cngd ; }
        if ( prm_rdout_in [ i_prm ] != shdet_sntnl ) { prm_rdout [ i_prm ] = prm_rdout_in [ i_prm ] ; ++ n_prm_cngd ; }
        if ( prm_opt_in [ i_prm ] != shdet_sntnl ) { prm_opt [ i_prm ] = prm_opt_in [ i_prm ] ; ++ n_prm_cngd ; }
        for ( uint i_mdl = 0; i_mdl != mx_blmtr_mdl; i_mdl++ ) 
        {   if ( prm_blmtr_in [ i_prm ] [ i_mdl ] != shdet_sntnl ) 
            { prm_blmtr [ i_prm ] [ i_mdl ] = prm_blmtr_in [ i_prm ] [ i_mdl ] ;  
              ++ n_prm_cngd ; 
            }
        }
    }

  //-- If the Stray capacitance is modified, with its associated modified BDAC value, there are a set of parameters that need be updated, too

    if ( ( prm_rdout_in [ 17 ] != shdet_sntnl ) || ( prm_rdout_in [ 20 ] != shdet_sntnl ) )
    { flt fct_seb = ( flt ) 4096 / ( flt ) 4095 ;
      ( prm_rdout ) [ 20 ] *= fct_seb ;
      flt v_in_bias = ( flt ) 2.5 ; // V
      flt r_3 = ( flt ) 1e4 ; flt r_4 = ( flt ) 5.6e3 ; // Ohm
      flt r_1    = ( flt ) 1e4 ;  // Ohm
      flt r_2    = ( flt ) 2e4 ;   // Ohm
      flt r_trng = ( flt ) 240e3 ; // Ohm
      flt c_trng = ( flt ) 20e-9 ; // F
      flt tau_trng = r_trng * c_trng ; // s
      flt vbal_mx = ( ( ( flt ) 1 ) + r_3 / r_4 ) * v_in_bias ;
      flt vtrans_mx = -( prm_rdout [ 20 ] ) / ( ( flt ) 4096 ) * vbal_mx ;
      flt r_c_1 = ( flt ) 1e4, r_c_2 = ( flt ) 1e4 ; // Ohm
      flt lmbd_c_1_12 = r_c_1 / ( r_c_1 + r_c_2 ) ;
      ( prm_rdout ) [ 5 ] = lmbd_c_1_12 * vtrans_mx ;
      flt c_stry = prm_rdout [ 17 ] ;
      flt mu_bias =  ( ( flt ) 1 ) / ( ( ( flt ) 1 ) + r_1 / r_2 ) * ( vbal_mx ) / tau_trng * ( prm_rdout [ 21 ] ) / ( ( flt ) 4096 ) ;
      flt mu_trans = ( vbal_mx ) * ( prm_rdout [ 20 ] ) / ( ( flt ) 4096 ) * ( prm_rdout [ 22 ] ) / ( ( flt ) 4096 ) * \
               ( ( flt ) 1 ) / ( ( ( flt ) 1 ) + r_2 / r_1 ) ;
      ( prm_rdout ) [ 8 ] = c_stry * ( ( ( prm_rdout ) [ 6 ] ) + ( ( prm_rdout ) [ 7 ] ) ) + ( ( prm_rdout ) [ 6 ] ) * ( ( prm_rdout ) [ 7 ] ) ;
      flt alph_2 = ( ( prm_rdout ) [ 7 ] - ( prm_rdout ) [ 6 ] ) / \
                   ( ( prm_rdout ) [ 4 ] ) / ( ( prm_rdout ) [ 8 ] ) ;
      ( prm_rdout ) [ 9 ] = alph_2 * ( ( prm_rnng ) [ 11 ] ) ;
      flt alph_3 = ( ( flt ) 2 ) * ( ( prm_rdout ) [ 6 ] ) * ( ( prm_rdout ) [ 7 ] ) / \
                   ( ( prm_rdout ) [ 8 ] ) ;

      ( prm_rdout ) [ 10 ] = mu_trans * alph_3 ;
      ( prm_rdout ) [ 11 ] = ( mu_trans * alph_2 + mu_bias * alph_3 ) * ( ( prm_rnng ) [ 11 ] ) ;
      ( prm_rdout ) [ 12 ] = mu_bias * alph_2 * ( ( prm_rnng ) [ 11 ] ) * ( ( prm_rnng ) [ 11 ] ) ;
      flt bt_2 = ( ( ( flt ) 2 ) * c_stry + ( ( prm_rdout ) [ 7 ] ) ) / \
             ( ( prm_rdout ) [ 4 ] ) / ( ( prm_rdout ) [ 8 ] ) ;
      ( prm_rdout ) [ 13 ] = ( ( flt ) 1 ) - bt_2 * ( ( prm_rnng ) [ 11 ] ) ;
      flt bt_3 = ( ( flt ) 2 ) * c_stry * ( ( prm_rdout ) [ 7 ] ) / ( ( prm_rdout ) [ 8 ] ) ;
      ( prm_rdout ) [ 14 ] = ( mu_trans * bt_2 + mu_bias * bt_3 ) * ( ( prm_rnng ) [ 11 ] ) ;
      ( prm_rdout ) [ 15 ] = mu_trans * bt_3 ;
      ( prm_rdout ) [ 16 ] = mu_bias * bt_2 * ( ( prm_rnng ) [ 11 ] ) * ( ( prm_rnng ) [ 11 ] ) ;
    }
    

  // Minimum check (remove it if annoying)
  //  if ( n_prm_cngd != 0 ) std::cout << "(SHDet_f) " << n_prm_cngd << " default parameters changed\n" ; 

  //-- Setting the number of samples to be simulated (std::to_string( ) fails to compile with the provided tools)
  uint l_rng_lp = 3 ;
    if ( ( prm_rnng_in [ 50 ] == shdet_sntnl ) || ( prm_rnng_in [ 50 ] < 3 ) ) throw std::runtime_error( "(shdet_func.cpp): The number of samples to be simulated has to be set to a value greater than two samples." ) ;
    if ( prm_rnng_in [ 50 ] > l_rng_ttl ) throw std::runtime_error( "(shdet_func.cpp): The number of samples to be simulated has to be less than l_rng_ttl set in shdet/include/shdet_types.h." ) ;
  l_rng_lp = prm_rnng_in [ 50 ] ;

  //-- Global gain factor in the electronics
  c_uint n_blmtr_mdl = prm_rnng [ 0 ] ;
    if ( prm_inst [ 3 ] > 500 ) { ( prm_inst ) [ 13 ] /= ( flt ) 3 ; }

  // Calibration of the input array from K CMB to W

    for ( uint i_tmp = 0; i_tmp != l_rng_lp; i_tmp++ ) { rng_arry [ i_tmp ] *= prm_opt [ 1 ] ; rng_arry [ i_tmp ] += prm_opt [ 0 ] ; }

  // Addition of an optional offset to the oprtical power (already calibrated in W)
    if ( prm_opt [ 31 ] )
    {  for ( uint i_tmp = 0; i_tmp != l_rng_lp; i_tmp++ ) { rng_arry [ i_tmp ] += prm_opt [ 30 ] ; }
      //std::cout << "(SHDET_FUNC) OPTICAL OFFSET=" << prm_opt [ 30 ] << " W, or " << prm_opt [ 30 ] / prm_opt [ 1 ] << " K CMB \n" ;
    }
  

//std::cout << __LINE__ << ": prm_opt [ 1 ], prm_opt [ 0 ]=" << prm_opt [ 1 ] << ", " << prm_opt [ 0 ] << std::endl ;
//exit( -1 ) ;
 
  // 2.- Local variables and initialization
  //- Generic variables for SHDet that should never change
  c_ulng n_sm_smpl_pr_fst_smpl = ( ulng ) ( prm_rnng ) [ 3 ] ; // The number of simulated samples per fast sample
  c_uint n_smpl_sm_prd = ( n_smpl_prd / 2 ) ; // N_SAMP: # samples per semiperiod
  c_uint n_brn_in = 902 ;
  c_uint n_spr_fst_smpl_sm_prd = n_smpl_prd / 2 * n_sm_smpl_pr_fst_smpl ;
  //- ADC conversion
  c_flt adc_rng = ( flt ) ( n_ctp - 1 ) ;
  c_flt v_adc_mx = 5.1 ; // V
  c_flt v_adc_mn = -v_adc_mx ;
  c_flt v_adc_rng = ( flt ) 2 * v_adc_mx ; // V
  c_flt adc_cnv_fct = v_adc_rng / ( adc_rng + 1 ) ;
  c_flt adc_cnv_fct_inv = ( adc_rng + 1 ) / v_adc_rng ;
  flt d_adc = 0, sgn_d_adc = 0 ;
    for ( uint i_ctp = 0; i_ctp < n_ctp; i_ctp++ ) { adc_ctp [ i_ctp ] = v_adc_mn + adc_cnv_fct * ( adc_ctp [ i_ctp ] ) ; }

  flt rw_cnst [ n_smpl_prd ] ;
  flt rw_gn [ n_smpl_prd ] ;

  //rw_cnst [ 0 ] = 0.5719535236100221; rw_cnst [ 1 ] = 0.8949717228309323; rw_cnst [ 2 ] = 1.143279327185221; rw_cnst [ 3 ] = 1.251587984568856; rw_cnst [ 4 ] = 1.233189274167315; rw_cnst [ 5 ] = 1.126924735513685; rw_cnst [ 6 ] = 0.9722914108091523; rw_cnst [ 7 ] = 0.8000611611838464; rw_cnst [ 8 ] = 0.6304179368953371; rw_cnst [ 9 ] = 0.4744071342403546; rw_cnst [ 10 ] = 0.3364789941333826; rw_cnst [ 11 ] = 0.2169982684454146; rw_cnst [ 12 ] = 0.1142329404601276; rw_cnst [ 13 ] = 0.02570444126330832; rw_cnst [ 14 ] = -0.05102838350138193; rw_cnst [ 15 ] = -0.1180058422555225; rw_cnst [ 16 ] = -0.1767731532067905; rw_cnst [ 17 ] = -0.2284311684763292; rw_cnst [ 18 ] = -0.2737450485677849; rw_cnst [ 19 ] = -0.313258513853382; rw_cnst [ 20 ] = -0.3473879375457007; rw_cnst [ 21 ] = -0.3764881720292102; rw_cnst [ 22 ] = -0.4008919107474228; rw_cnst [ 23 ] = -0.4209287062916814; rw_cnst [ 24 ] = -0.4369305937039915; rw_cnst [ 25 ] = -0.4492302848028518; rw_cnst [ 26 ] = -0.4581562536930202; rw_cnst [ 27 ] = -0.4640273988810033; rw_cnst [ 28 ] = -0.467148659829778; rw_cnst [ 29 ] = -0.4678080706419335; rw_cnst [ 30 ] = -0.4662752101775179; rw_cnst [ 31 ] = -0.4628007639983359; rw_cnst [ 32 ] = -0.4576168504837009; rw_cnst [ 33 ] = -0.4509377990458183; rw_cnst [ 34 ] = -0.4429611442281546; rw_cnst [ 35 ] = -0.43386867999312; rw_cnst [ 36 ] = -0.4238274858509568; rw_cnst [ 37 ] = -0.4129908847174925; rw_cnst [ 38 ] = -0.4014993223921712; rw_cnst [ 39 ] = -0.3894811742569484; rw_cnst [ 40 ] = -0.5719535236100189; rw_cnst [ 41 ] = -0.8949717228309292; rw_cnst [ 42 ] = -1.143279327185218; rw_cnst [ 43 ] = -1.251587984568852; rw_cnst [ 44 ] = -1.233189274167311; rw_cnst [ 45 ] = -1.12692473551368; rw_cnst [ 46 ] = -0.9722914108091476; rw_cnst [ 47 ] = -0.8000611611838417; rw_cnst [ 48 ] = -0.6304179368953327; rw_cnst [ 49 ] = -0.4744071342403505; rw_cnst [ 50 ] = -0.3364789941333788; rw_cnst [ 51 ] = -0.2169982684454111; rw_cnst [ 52 ] = -0.1142329404601242; rw_cnst [ 53 ] = -0.0257044412633051; rw_cnst [ 54 ] = 0.05102838350138508; rw_cnst [ 55 ] = 0.1180058422555256; rw_cnst [ 56 ] = 0.1767731532067935; rw_cnst [ 57 ] = 0.2284311684763322; rw_cnst [ 58 ] = 0.273745048567788; rw_cnst [ 59 ] = 0.313258513853385; rw_cnst [ 60 ] = 0.3473879375457039; rw_cnst [ 61 ] = 0.3764881720292134; rw_cnst [ 62 ] = 0.4008919107474259; rw_cnst [ 63 ] = 0.4209287062916845; rw_cnst [ 64 ] = 0.4369305937039947; rw_cnst [ 65 ] = 0.4492302848028551; rw_cnst [ 66 ] = 0.4581562536930233; rw_cnst [ 67 ] = 0.4640273988810065; rw_cnst [ 68 ] = 0.4671486598297813; rw_cnst [ 69 ] = 0.4678080706419368; rw_cnst [ 70 ] = 0.466275210177521; rw_cnst [ 71 ] = 0.462800763998339; rw_cnst [ 72 ] = 0.457616850483704; rw_cnst [ 73 ] = 0.4509377990458213; rw_cnst [ 74 ] = 0.4429611442281575; rw_cnst [ 75 ] = 0.4338686799931228; rw_cnst [ 76 ] = 0.4238274858509596; rw_cnst [ 77 ] = 0.4129908847174955; rw_cnst [ 78 ] = 0.4014993223921742; rw_cnst [ 79 ] = 0.3894811742569515;

//- Identity tests
  rw_cnst [ 0 ] = 0; rw_cnst [ 1 ] = 0;  rw_cnst [ 2 ] = 0;  rw_cnst [ 3 ] = 0;  rw_cnst [ 4 ] = 0;  rw_cnst [ 5 ] = 0;  rw_cnst [ 6 ] = 0;  rw_cnst [ 7 ] = 0;  rw_cnst [ 8 ] = 0;  rw_cnst [ 9 ] = 0;  rw_cnst [ 10 ] = 0;  rw_cnst [ 11 ] = 0;  rw_cnst [ 12 ] = 0;  rw_cnst [ 13 ] = 0;  rw_cnst [ 14 ] = 0;  rw_cnst [ 15 ] = 0;  rw_cnst [ 16 ] = 0;  rw_cnst [ 17 ] = 0;  rw_cnst [ 18 ] = 0;  rw_cnst [ 19 ] = 0;  rw_cnst [ 20 ] = 0;  rw_cnst [ 21 ] = 0;  rw_cnst [ 22 ] = 0;  rw_cnst [ 23 ] = 0;  rw_cnst [ 24 ] = 0;  rw_cnst [ 25 ] = 0;  rw_cnst [ 26 ] = 0;  rw_cnst [ 27 ] = 0;  rw_cnst [ 28 ] = 0;  rw_cnst [ 29 ] = 0;  rw_cnst [ 30 ] = 0;  rw_cnst [ 31 ] = 0;  rw_cnst [ 32 ] = 0;  rw_cnst [ 33 ] = 0;  rw_cnst [ 34 ] = 0;  rw_cnst [ 35 ] = 0;  rw_cnst [ 36 ] = 0;  rw_cnst [ 37 ] = 0;  rw_cnst [ 38 ] = 0;  rw_cnst [ 39 ] = 0;  rw_cnst [ 40 ] = 0;  rw_cnst [ 41 ] = 0;  rw_cnst [ 42 ] = 0;  rw_cnst [ 43 ] = 0;  rw_cnst [ 44 ] = 0;  rw_cnst [ 45 ] = 0;  rw_cnst [ 46 ] = 0;  rw_cnst [ 47 ] = 0;  rw_cnst [ 48 ] = 0;  rw_cnst [ 49 ] = 0;  rw_cnst [ 50 ] = 0;  rw_cnst [ 51 ] = 0;  rw_cnst [ 52 ] = 0;  rw_cnst [ 53 ] = 0;  rw_cnst [ 54 ] = 0;  rw_cnst [ 55 ] = 0;  rw_cnst [ 56 ] = 0;  rw_cnst [ 57 ] = 0;  rw_cnst [ 58 ] = 0;  rw_cnst [ 59 ] = 0;  rw_cnst [ 60 ] = 0;  rw_cnst [ 61 ] = 0;  rw_cnst [ 62 ] = 0;  rw_cnst [ 63 ] = 0;  rw_cnst [ 64 ] = 0;  rw_cnst [ 65 ] = 0;  rw_cnst [ 66 ] = 0;  rw_cnst [ 67 ] = 0;  rw_cnst [ 68 ] = 0;  rw_cnst [ 69 ] = 0;  rw_cnst [ 70 ] = 0;  rw_cnst [ 71 ] = 0;  rw_cnst [ 72 ] = 0;  rw_cnst [ 73 ] = 0;  rw_cnst [ 74 ] = 0;  rw_cnst [ 75 ] = 0;  rw_cnst [ 76 ] = 0;  rw_cnst [ 77 ] = 0;  rw_cnst [ 78 ] = 0;  rw_cnst [ 79 ] = 0;

// Additional offset of the raw constant to allow time-varying values over pointing period, for instance. UNITS: V
  if ( prm_rnng [ 39 ] )
  {   for ( uint i_tmp = 0; i_tmp < n_smpl_prd; i_tmp++ ) rw_cnst [ i_tmp ] += prm_rnng [ 38 ] ; 
    //std::cout << "(SHDET_FUNC) RAW CONSTANT OFFSET=" << prm_rnng [ 38 ] << " V, or " << prm_rnng [ 38 ] * adc_cnv_fct_inv << " fast digital units" ;
  }


//-NB: First semi-period starts with negative sign. Otherwise, map will be globally x(-1)

//  rw_gn [ 0 ] = 0.02817025245158668; rw_gn [ 1 ] = 0.02717152792368956; rw_gn [ 2 ] = 0.02519214680028788; rw_gn [ 3 ] = 0.02205288005247853; rw_gn [ 4 ] = 0.01791192668629114; rw_gn [ 5 ] = 0.01310490563751459; rw_gn [ 6 ] = 0.00799962383969819; rw_gn [ 7 ] = 0.00290955970561284; rw_gn [ 8 ] = -0.001940311587678469; rw_gn [ 9 ] = -0.006413867448324828; rw_gn [ 10 ] = -0.01044733314512824; rw_gn [ 11 ] = -0.01402771669483281; rw_gn [ 12 ] = -0.0171728965597599; rw_gn [ 13 ] = -0.01991626796601795; rw_gn [ 14 ] = -0.02229645816725964; rw_gn [ 15 ] = -0.02435137965636213; rw_gn [ 16 ] = -0.02611544259928692; rw_gn [ 17 ] = -0.0276187643417582; rw_gn [ 18 ] = -0.02888744718087761; rw_gn [ 19 ] = -0.02994428595855549; rw_gn [ 20 ] = -0.03080952766186403; rw_gn [ 21 ] = -0.03150150090069352; rw_gn [ 22 ] = -0.03203705995354742; rw_gn [ 23 ] = -0.03243185766628727; rw_gn [ 24 ] = -0.03270049056625765; rw_gn [ 25 ] = -0.03285656394601855; rw_gn [ 26 ] = -0.03291271668439528; rw_gn [ 27 ] = -0.03288063343572912; rw_gn [ 28 ] = -0.0327710602172376; rw_gn [ 29 ] = -0.03259383045175903; rw_gn [ 30 ] = -0.03235790263633589; rw_gn [ 31 ] = -0.03207140760275785; rw_gn [ 32 ] = -0.03174170208398867; rw_gn [ 33 ] = -0.03137542526146024; rw_gn [ 34 ] = -0.03097855554533932; rw_gn [ 35 ] = -0.03055646562217799; rw_gn [ 36 ] = -0.03011397455080588; rw_gn [ 37 ] = -0.02965539628556128; rw_gn [ 38 ] = -0.02918458442803669; rw_gn [ 39 ] = -0.02870497327201024; rw_gn [ 40 ] = -0.02817025245153242; rw_gn [ 41 ] = -0.02717152792364608; rw_gn [ 42 ] = -0.02519214680024386; rw_gn [ 43 ] = -0.02205288005244695; rw_gn [ 44 ] = -0.01791192668626804; rw_gn [ 45 ] = -0.01310490563752095; rw_gn [ 46 ] = -0.007999623839729894; rw_gn [ 47 ] = -0.002909559705661132; rw_gn [ 48 ] = 0.001940311587615185; rw_gn [ 49 ] = 0.006413867448250139; rw_gn [ 50 ] = 0.01044733314504732; rw_gn [ 51 ] = 0.01402771669474877; rw_gn [ 52 ] = 0.01717289655967955; rw_gn [ 53 ] = 0.01991626796594655; rw_gn [ 54 ] = 0.0222964581671945; rw_gn [ 55 ] = 0.02435137965629551; rw_gn [ 56 ] = 0.02611544259921731; rw_gn [ 57 ] = 0.0276187643416963; rw_gn [ 58 ] = 0.02888744718083137; rw_gn [ 59 ] = 0.02994428595852903; rw_gn [ 60 ] = 0.03080952766185828; rw_gn [ 61 ] = 0.03150150090070069; rw_gn [ 62 ] = 0.0320370599535689; rw_gn [ 63 ] = 0.03243185766632885; rw_gn [ 64 ] = 0.0327004905663132; rw_gn [ 65 ] = 0.03285656394608528; rw_gn [ 66 ] = 0.03291271668446839; rw_gn [ 67 ] = 0.03288063343580211; rw_gn [ 68 ] = 0.03277106021730525; rw_gn [ 69 ] = 0.0325938304518214; rw_gn [ 70 ] = 0.0323579026363939; rw_gn [ 71 ] = 0.03207140760281355; rw_gn [ 72 ] = 0.03174170208404134; rw_gn [ 73 ] = 0.03137542526151215; rw_gn [ 74 ] = 0.0309785555453873; rw_gn [ 75 ] = 0.03055646562222209; rw_gn [ 76 ] = 0.03011397455084712; rw_gn [ 77 ] = 0.02965539628560367; rw_gn [ 78 ] = 0.02918458442808479; rw_gn [ 79 ] = 0.02870497327206363 ;


//-Identity tests
rw_gn [ 0 ] = -0.025; rw_gn [ 1 ] = -0.025;  rw_gn [ 2 ] = -0.025;  rw_gn [ 3 ] = -0.025;  rw_gn [ 4 ] = -0.025;  rw_gn [ 5 ] = -0.025;  rw_gn [ 6 ] = -0.025;  rw_gn [ 7 ] = -0.025;  rw_gn [ 8 ] = -0.025;  rw_gn [ 9 ] = -0.025;  rw_gn [ 10 ] = -0.025;  rw_gn [ 11 ] = -0.025;  rw_gn [ 12 ] = -0.025;  rw_gn [ 13 ] = -0.025;  rw_gn [ 14 ] = -0.025;  rw_gn [ 15 ] = -0.025;  rw_gn [ 16 ] = -0.025;  rw_gn [ 17 ] = -0.025;  rw_gn [ 18 ] = -0.025;  rw_gn [ 19 ] = -0.025;  rw_gn [ 20 ] = -0.025;  rw_gn [ 21 ] = -0.025;  rw_gn [ 22 ] = -0.025;  rw_gn [ 23 ] = -0.025;  rw_gn [ 24 ] = -0.025;  rw_gn [ 25 ] = -0.025;  rw_gn [ 26 ] = -0.025;  rw_gn [ 27 ] = -0.025;  rw_gn [ 28 ] = -0.025;  rw_gn [ 29 ] = -0.025;  rw_gn [ 30 ] = -0.025;  rw_gn [ 31 ] = -0.025;  rw_gn [ 32 ] = -0.025;  rw_gn [ 33 ] = -0.025;  rw_gn [ 34 ] = -0.025;  rw_gn [ 35 ] = -0.025;  rw_gn [ 36 ] = -0.025;  rw_gn [ 37 ] = -0.025;  rw_gn [ 38 ] = -0.025;  rw_gn [ 39 ] = -0.025;  rw_gn[ 40 ] = 0.025;  rw_gn [ 41 ] = 0.025; rw_gn [ 42 ] = 0.025; rw_gn [ 43 ] = 0.025; rw_gn [ 44 ] = 0.025; rw_gn [ 45 ] = 0.025; rw_gn [ 46 ] = 0.025; rw_gn [ 47 ] = 0.025; rw_gn [ 48 ] = 0.025; rw_gn [ 49 ] = 0.025; rw_gn [ 50 ] = 0.025; rw_gn [ 51 ] = 0.025; rw_gn [ 52 ] = 0.025; rw_gn [ 53 ] = 0.025; rw_gn [ 54 ] = 0.025; rw_gn [ 55 ] = 0.025; rw_gn [ 56 ] = 0.025; rw_gn [ 57 ] = 0.025; rw_gn [ 58 ] = 0.025; rw_gn [ 59 ] = 0.025; rw_gn [ 60 ] = 0.025; rw_gn [ 61 ] = 0.025; rw_gn [ 62 ] = 0.025; rw_gn [ 63 ] = 0.025; rw_gn [ 64 ] = 0.025; rw_gn [ 65 ] = 0.025; rw_gn [ 66 ] = 0.025; rw_gn [ 67 ] = 0.025; rw_gn [ 68 ] = 0.025; rw_gn [ 69 ] = 0.025; rw_gn [ 70 ] = 0.025; rw_gn [ 71 ] = 0.025; rw_gn [ 72 ] = 0.025; rw_gn [ 73 ] = 0.025; rw_gn [ 74 ] = 0.025; rw_gn [ 75 ] = 0.025; rw_gn [ 76 ] = 0.025; rw_gn [ 77 ] = 0.025; rw_gn [ 78 ] = 0.025; rw_gn [ 79 ] = 0.025;

  flt W2V = 198990953438442.969 ; 

  c_flt nmbr_3_2 = 1.5, nmbr_m2 = -2, nmbr_m4 = -4, nmbr_m6 = -6 ;
  c_flt nmbr_15_2 = 7.5, nmbr_25_2 = 12.5, nmbr_33_2 = 16.5, nmbr_m24 = -24 ;
  c_flt t_cffcnt = ( flt ) 1 / ( flt ) n_spr_fst_smpl_sm_prd / ( flt ) 3 ;
  c_flt nmbr_1_ovr_3 = ( flt ) 1 / 3 ;
  c_flt b_1_3 = ( flt ) -1 / 6, b_1_2 = 0.5, b_1_1 = -0.5, b_1_0 = ( flt ) 1 / 6 ;
  c_flt b_2_3 = 0.5, b_2_2 = -1, b_2_0 = ( flt ) 2 / 3 ;
  flt b_3_3 = -0.5, b_3_2 = 0.5, b_3_1 = 0.5, b_3_0 = ( flt ) 1 / 6 ;
  flt b_4_3 = ( flt ) 1 / 6 ;
  // Variables directly related with the simulation at the integration step time
  uint i_dsn_smpl = 0 ;
  uint i_spr_fst_smpl = 0 ;
  flt v_out_shdet = 0 ;
  flt vr_dtctr_shdet [ n_vr_shdet ] ;
  vr_dtctr_shdet [ 0 ] = 0 ;
  vr_dtctr_shdet [ 1 ] = -1 ;
  vr_dtctr_shdet [ 2 ] = -1 ;
  vr_dtctr_shdet [ 3 ] = -1 ;
  vr_dtctr_shdet [ 4 ] = 1 ;
  vr_dtctr_shdet [ 5 ] = 0 ;
  vr_dtctr_shdet [ 6 ] = 0 ; 
  vr_dtctr_shdet [ 7 ] = 0 ; 
  vr_dtctr_shdet [ 8 ] = 0 ;
  vr_dtctr_shdet [ 9 ] = 0 ; 
  vr_dtctr_shdet [ 10 ] = 0 ; 
  vr_dtctr_shdet [ 11 ] = 0 ; 
  vr_dtctr_shdet [ 12 ] = 0 ; 
  vr_dtctr_shdet [ 13 ] = 0 ; 
  flt v_out_shdet_acum = 0, opt_pwr_in = 0, t_bspl_1 = 0, t_bspl_2 = 0, t_bspl_3 = 0 ;
  int i_ctp = 0 ;
  flt c_opt [ 4 ] ;
  flt bspl [ 4 ] ;
  //-- Indices for the simulation
  uint i_smp_chnk = 0 ;

  // 3.- Initial temperature of the bolometer modules
  flt mdl_tmprtr [ mx_blmtr_mdl ] ;
    for ( uint i_mdl = 0; i_mdl != mx_blmtr_mdl; i_mdl++ ) mdl_tmprtr [ i_mdl ] = prm_rnng [ 1 ] ;

  // 4.- ADC type (0/1=Off/On)
  c_uint adc_mdl = ( c_uint ) ( prm_rnng ) [ 4 ] ;

  // 5.- Do the work
  if ( prm_rnng [ 35 ] == 0 )
  { //- First element (will be discarded in npipe)
      for ( i_dsn_smpl = 0; i_dsn_smpl < n_brn_in; i_dsn_smpl++ ) { for ( i_spr_fst_smpl = 0; i_spr_fst_smpl < n_spr_fst_smpl_sm_prd; i_spr_fst_smpl++ ) { SHDet_smltn_f( prm_rnng, prm_inst, prm_rdout, prm_blmtr, i_spr_fst_smpl, rng_arry [ 0 ], mdl_tmprtr, vr_dtctr_shdet, v_out_shdet, n_blmtr_mdl ) ; } }

    //- Noise
    //-- WN level
    uint rw_ns = 0 ;
      if ( ( prm_rnng [ 36 ] ) && ( prm_rnng [ 36 ] != shdet_sntnl ) ) rw_ns = 1 ;
    unsigned long sd_ns = ( unsigned long ) prm_rnng [ 37 ] ;
    unsigned long mrsgl = 4101842887655102017LL^sd_ns ;
    flt lv_1 = 0, lv_2 = 0, lv_3 = 0, lv_4 = 0, lv_5 = 0 ;

    //- Continue with the work (NB: the last two samples are not simulated. They belong in any case to additional samples that will be discarded in npipe)
      for ( i_smp_chnk = 1; i_smp_chnk != l_rng_lp - 2; i_smp_chnk++ )
      { v_out_shdet_acum = 0 ; 
        //--- Optical power
        c_opt [ 0 ] = nmbr_25_2 * rng_arry [ i_smp_chnk - 1 ] + nmbr_m24 * rng_arry [ i_smp_chnk ] + \
                      nmbr_33_2 * rng_arry [ i_smp_chnk + 1 ] + nmbr_m4 * rng_arry [ i_smp_chnk + 2 ] ;
        c_opt [ 1 ] = nmbr_m2 * rng_arry [ i_smp_chnk - 1 ] + nmbr_15_2 * rng_arry [ i_smp_chnk ] + \
                      nmbr_m6 * rng_arry [ i_smp_chnk + 1 ] + nmbr_3_2 * rng_arry [ i_smp_chnk + 2 ] ;
        c_opt [ 2 ] = nmbr_3_2 * rng_arry [ i_smp_chnk - 1 ] + nmbr_m6 * rng_arry [ i_smp_chnk ] + \
                      nmbr_15_2 * rng_arry [ i_smp_chnk + 1 ] + nmbr_m2 * rng_arry [ i_smp_chnk + 2 ] ;
        c_opt [ 3 ] = nmbr_m4 * rng_arry [ i_smp_chnk - 1 ] + nmbr_33_2 * rng_arry [ i_smp_chnk ] + \
                      nmbr_m24 * rng_arry [ i_smp_chnk + 1 ] + nmbr_25_2 * rng_arry [ i_smp_chnk + 2 ] ;
          for ( i_spr_fst_smpl = 0; i_spr_fst_smpl < n_spr_fst_smpl_sm_prd; i_spr_fst_smpl++ ) 
          { t_bspl_1 = nmbr_1_ovr_3 + t_cffcnt * i_spr_fst_smpl ;
            t_bspl_2 = t_bspl_1 * t_bspl_1 ;
            t_bspl_3 = t_bspl_2 * t_bspl_1 ;
            bspl [ 0 ] = b_1_3 * t_bspl_3 + b_1_2 * t_bspl_2 + b_1_1 * t_bspl_1 + b_1_0 ;
            bspl [ 1 ] = b_2_3 * t_bspl_3 + b_2_2 * t_bspl_2 + b_2_0 ;
            bspl [ 2 ] = b_3_3 * t_bspl_3 + b_3_2 * t_bspl_2 + b_3_1 * t_bspl_1 + b_3_0 ;
            bspl [ 3 ] = b_4_3 * t_bspl_3 ;
            opt_pwr_in = ( c_opt [ 0 ] * bspl [ 0 ] ) + ( c_opt [ 1 ] * bspl [ 1 ] ) + ( c_opt [ 2 ] * bspl [ 2 ] ) + ( c_opt [ 3 ] * bspl [ 3 ] ) ;
            SHDet_smltn_f( prm_rnng, prm_inst, prm_rdout, prm_blmtr, i_spr_fst_smpl, opt_pwr_in, mdl_tmprtr, vr_dtctr_shdet, v_out_shdet, n_blmtr_mdl ) ;
            //-- Creating a DSN sample
              if ( ! ( ( i_spr_fst_smpl + 1 ) % n_sm_smpl_pr_fst_smpl ) ) 
              { //--- Adding Gaussian noise (polar Box-Mueller algorithm, http://en.literateprograms.org/Box-Muller_transform)
                  if ( rw_ns )
                  { //--- Full pseudo-random number generator 
                    //--- Algorithm from Leva (1992) from NR, C++. PS: Same as original paper, except for the do ... while structure
                    //--- PS: good for less than 10^12 calls, notice that seed changes per ring. And a ring has < 5*10^7 raw samples
                      do
                      { //--- 1) Marsaglia, but from NR C++, 7.1.3, Ranq1, instead of his original paper
                        //static unsigned long mrsgl = 4101842887655102017LL^sd_ns ;
                        mrsgl ^= ( mrsgl >> 21 ) ; mrsgl ^= ( mrsgl << 35 ) ; mrsgl^= ( mrsgl >> 4 ) ;
                        mrsgl *= 2685821657736338717LL ;
                        //--- First number
                        lv_1 = 5.42101086242752217e-20 * mrsgl ;
                        mrsgl ^= ( mrsgl >> 21 ) ; mrsgl ^= ( mrsgl << 35 ) ; mrsgl^= ( mrsgl >> 4 ) ;
                        mrsgl *= 2685821657736338717LL ;
                        //--- 2) Leva (1992): optimal boundary of the Box-Mueller acceptance region
                        lv_2 = 5.42101086242752217e-20 * mrsgl ;
                        lv_2 = 1.7156 * ( lv_2 - 0.5 ) ;
                        lv_3 = lv_1 - 0.449871 ;
                        lv_4 = lv_2 * ( ( lv_2 > 0 ) - ( lv_2 < 0 ) ) + 0.386595 ;
                        lv_5 = lv_3 * lv_3 + lv_4 * ( 0.19600 * lv_4 - 0.25472 * lv_3 ) ;
                        } while ( ( lv_5 > 0.27597 ) && ( lv_5 > 0.27846 || lv_2 * lv_2 > -4. * std::log ( lv_1 ) * lv_1 * lv_1 ) ) ;
                    v_out_shdet += ( lv_2 / lv_1 * ( prm_rnng [ 36 ] ) ) ;
                  }
                //--- ADC model
                  if ( adc_mdl ) 
                  { i_ctp = ( int ) ( ( v_out_shdet + v_adc_mx ) * adc_cnv_fct_inv )  ;
                      if ( i_ctp < 0 ) { i_ctp = 0 ; v_out_shdet = 0 ; continue ; }
                      if ( i_ctp > adc_rng ) { i_ctp = adc_rng ; v_out_shdet = adc_rng ; continue ; }
                    d_adc = adc_ctp [ i_ctp ] - v_out_shdet ;
                    sgn_d_adc = ( d_adc > 0 ) - ( d_adc < 0 ) ;
                      if ( sgn_d_adc < 0 )
                      {   while ( d_adc < 0 )
                          { ++i_ctp ;
                            d_adc = adc_ctp [ i_ctp ] - v_out_shdet ;
                          } ;
                      }
                      if ( sgn_d_adc > 0 )
                      {   while ( d_adc > 0 )
                          { --i_ctp ;
                            d_adc = adc_ctp [ i_ctp ] - v_out_shdet ;
                          } ;
                      }

                      if ( sgn_d_adc <= 0 ) { v_out_shdet = ( flt ) ( i_ctp ) ; }
                      else { v_out_shdet = ( flt ) ( i_ctp + 1 ) ; }
                  }
                v_out_shdet_acum += v_out_shdet ;
              }
          }
        //-- Storing a DSN sample
        rng_arry [ i_smp_chnk - 1 ] = v_out_shdet_acum ;
      } 
      //--- Recall the NB on the last two samples above.
      rng_arry [ l_rng_lp - 2 ] = rng_arry [ l_rng_lp - 3 ] ;
      rng_arry [ l_rng_lp - 1 ] = rng_arry [ l_rng_lp - 3 ] ;
  }
  else 
  { //-- Generic local variables and initialization
    //--- Notice how the raw sample index is used, and not 'super' fast samples, as when dealing with the full simulation.
    uint idx_rw_smpl = 0 ;
    uint sm_prd_shft = 0 ;
    flt prty_sgn = 1 ;
    flt opt_dff_pwr_in = 0 ;
    flt v_out_shdet = 0 ;
    flt v_out_shdet_acum = 0 ;
    i_smp_chnk = 0 ;
    //- Noise
    //-- WN level
    uint rw_ns = 0 ;
      if ( ( prm_rnng [ 36 ] ) && ( prm_rnng [ 36 ] != shdet_sntnl ) ) rw_ns = 1 ;
    unsigned long sd_ns = ( unsigned long ) prm_rnng [ 37 ] ;
    unsigned long mrsgl = 4101842887655102017LL^sd_ns ;
    flt lv_1 = 0, lv_2 = 0, lv_3 = 0, lv_4 = 0, lv_5 = 0 ;
    //-- ADC related
    i_ctp = 0 ;
    //-- Core work:
    rng_arry [ 0 ] = 0 ; // Discarded anyways in npipe
      for ( i_smp_chnk = 1; i_smp_chnk != l_rng_lp ; i_smp_chnk++ )
      { v_out_shdet_acum = 0 ;
        opt_dff_pwr_in = ( rng_arry [ i_smp_chnk ] - prm_opt [ 0 ] ) * W2V ;
        sm_prd_shft = n_smpl_sm_prd * ( 1 + ( int ) prty_sgn ) / 2 ;
          for ( idx_rw_smpl = sm_prd_shft; idx_rw_smpl < n_smpl_sm_prd + sm_prd_shft; idx_rw_smpl++ )
          { v_out_shdet = rw_cnst [ idx_rw_smpl ] + rw_gn [ idx_rw_smpl ] * opt_dff_pwr_in ; // W to V
            //--- Adding Gaussian noise (polar Box-Mueller algorithm, http://en.literateprograms.org/Box-Muller_transform)
              if ( rw_ns )
              { //--- Full pseudo-random number generator
                //--- Algorithm from Leva (1992) from NR, C++. PS: Same as original paper, except for the do ... while structure
                //--- PS: good for less than 10^12 calls, notice that seed changes per ring. And a ring has < 5*10^7 raw samples
                  do
                  { //--- 1) Marsaglia, but from NR C++, 7.1.3, Ranq1, instead of his original paper
                    //static unsigned long mrsgl = 4101842887655102017LL^sd_ns ;
                    mrsgl ^= ( mrsgl >> 21 ) ; mrsgl ^= ( mrsgl << 35 ) ; mrsgl^= ( mrsgl >> 4 ) ;
                    mrsgl *= 2685821657736338717LL ;
                    //--- First number
                    lv_1 = 5.42101086242752217e-20 * mrsgl ;
                    mrsgl ^= ( mrsgl >> 21 ) ; mrsgl ^= ( mrsgl << 35 ) ; mrsgl^= ( mrsgl >> 4 ) ;
                    mrsgl *= 2685821657736338717LL ;
                    //--- 2) Leva (1992): optimal boundary of the Box-Mueller acceptance region
                    lv_2 = 5.42101086242752217e-20 * mrsgl ;
                    lv_2 = 1.7156 * ( lv_2 - 0.5 ) ;
                    lv_3 = lv_1 - 0.449871 ;
                    lv_4 = lv_2 * ( ( lv_2 > 0 ) - ( lv_2 < 0 ) ) + 0.386595 ;
                    lv_5 = lv_3 * lv_3 + lv_4 * ( 0.19600 * lv_4 - 0.25472 * lv_3 ) ;
                    } while ( ( lv_5 > 0.27597 ) && ( lv_5 > 0.27846 || lv_2 * lv_2 > -4. * std::log ( lv_1 ) * lv_1 * lv_1 ) ) ;
                v_out_shdet += ( lv_2 / lv_1 * ( prm_rnng [ 36 ] ) ) ;
              }
            //--- ADC
              if ( adc_mdl )
              { i_ctp = ( int ) ( ( v_out_shdet + v_adc_mx ) * adc_cnv_fct_inv )  ;
                  if ( i_ctp < 0 ) { i_ctp = 0 ; v_out_shdet = 0 ; continue ; }
                  if ( i_ctp > adc_rng ) { i_ctp = adc_rng ; v_out_shdet = adc_rng ; continue ; }
                d_adc = adc_ctp [ i_ctp ] - v_out_shdet ;
                sgn_d_adc = ( d_adc > 0 ) - ( d_adc < 0 ) ;
                  if ( sgn_d_adc < 0 )
                  {   while ( d_adc < 0 )
                      { ++i_ctp ;
                        d_adc = adc_ctp [ i_ctp ] - v_out_shdet ;
                      } ;
                  }
                  if ( sgn_d_adc > 0 )
                  {   while ( d_adc > 0 )
                      { --i_ctp ;
                        d_adc = adc_ctp [ i_ctp ] - v_out_shdet ;
                      } ;
                  }

                  if ( sgn_d_adc <= 0 ) { v_out_shdet = ( flt ) ( i_ctp ) ; }
                  else { v_out_shdet = ( flt ) ( i_ctp + 1 ) ; }
                }
              v_out_shdet_acum += v_out_shdet ;
            }
          rng_arry [ i_smp_chnk - 1 ] = v_out_shdet_acum ;
          prty_sgn *= -1 ;
      }
  }
}

//- SHDet_smltn_f: simulation at the super fast sample level
void SHDet_smltn_f( flt ( & prm_rnng ) [ n_prm_shdet ], flt ( & prm_inst ) [ n_prm_shdet ], flt ( & prm_rdout ) [ n_prm_shdet ], flt ( & prm_blmtr ) [ n_prm_shdet ] [ mx_blmtr_mdl ], c_uint & i_smpl_arg, c_flt & opt_pwr_in, flt ( & mdl_tmprtr ) [ mx_blmtr_mdl ], flt ( & vr_dtctr_shdet ) [ n_vr_shdet ], flt & v_out_shdet, c_flt & n_blmtr_mdl )
{ 

  vr_dtctr_shdet [ 1 ] = vr_dtctr_shdet [ 2 ] ;
 
  // Local variables
  uint n_spr_fst_smpl_sm_prd = 40 * ( int ) prm_rnng [ 3 ] ;
  c_flt v_mx = 12 ;
  c_flt v_mn = -12 ;
  flt pwr_1 = 0, pwr_2 = 0, pwr_3 = 0, inv_tau_stry = 0, gmm_1 = 0, v_d_c_2_nw = 0, v_h_c_1_nw = 0, v_h_c_2_nw = 0 ; 
  uint exp_md = 0 ; // No NAN for integers
 
  flt r_blmtr = ( prm_rdout [ 0 ] ) * exp ( pow( ( prm_rdout [ 1 ] ) / mdl_tmprtr [ 0 ], prm_rdout [ 2 ] ) ) ;
  
    for ( int i_mdl = n_blmtr_mdl - 1; i_mdl >= 0; i_mdl-- )
    {   if ( i_mdl != 0 )
        { pwr_2 = ( prm_blmtr [ 1 ] [ i_mdl - 1 ] ) / \
                    ( pow( mdl_tmprtr [ i_mdl ], prm_blmtr [ 3 ] [ i_mdl - 1 ] ) ) / \
                    ( prm_blmtr [ 7 ] [ i_mdl - 1 ] ) * \
                    ( pow( mdl_tmprtr [ i_mdl - 1 ], prm_blmtr [ 7 ] [ i_mdl - 1 ] ) - pow( mdl_tmprtr [ i_mdl ], prm_blmtr [ 7 ] [ i_mdl - 1 ] ) ) ;
          //- Balance of input power and output power between modules without including the sink
          pwr_3 = pwr_2 - pwr_1 ;
          //- Saving the link power between two modules for the next step
          pwr_1 = pwr_2 ;
        }
        //- Last instance: bolometer itself: optical power + electrothermal feedback and link between modules (=0 if n_blmtr_mdl=1, by construction)
        else pwr_3 = ( vr_dtctr_shdet [ 0 ] ) * ( vr_dtctr_shdet [ 0 ] ) / r_blmtr + opt_pwr_in - pwr_1 ;
      //- Re-using pwr_2
      pwr_2 = ( ( prm_blmtr [ 0 ] [ i_mdl ] ) * prm_rnng [ 1 ] ) /
              ( prm_blmtr [ 6 ] [ i_mdl ] ) * \
              ( pow( mdl_tmprtr [ i_mdl ] / prm_rnng [ 1 ], prm_blmtr [ 6 ] [ i_mdl ] ) - 1 ) ;
      pwr_3 -= pwr_2 ;
      // Updated module temperature. The case i_mdl = 0 is the bolometer temperature
      mdl_tmprtr [ i_mdl ] += ( pwr_3 / ( ( prm_blmtr [ 4 ] [ i_mdl ] ) * pow( mdl_tmprtr [ i_mdl ], prm_blmtr [ 5 ] [ i_mdl ] ) ) ) ;
  }

  inv_tau_stry = ( ( prm_rdout [ 6 ] + prm_rdout [ 7 ] ) / r_blmtr + prm_rdout [ 6 ] / ( prm_rdout [ 4 ] ) ) / ( prm_rdout [ 8 ] ) ;

  exp_md = ( i_smpl_arg + 1 ) % n_spr_fst_smpl_sm_prd ;

    if ( prm_rnng [ 30 ] )
    { vr_dtctr_shdet [ 2 ] = 1 ;
    flt sqr_tmp_1 = ( flt ) i_smpl_arg + 1.5 ;
    flt sqr_tmp_2 = sqr_tmp_1 / ( prm_rdout [ 18 ] ) ;
    sqr_tmp_1 = ( ( ( flt ) n_spr_fst_smpl_sm_prd ) - sqr_tmp_1 ) / ( prm_rdout [ 19 ] ) ;
    sqr_tmp_1 = fmin( 1, fabs( sqr_tmp_1 ) ) ;
    vr_dtctr_shdet [ 2 ] = fmin( sqr_tmp_1, fabs( sqr_tmp_2 ) ) ;
      if ( !exp_md ) {  vr_dtctr_shdet [ 3 ] *= ( -1 ) ; }
    vr_dtctr_shdet [ 2 ] = ( vr_dtctr_shdet [ 3 ] ) * ( vr_dtctr_shdet [ 2 ] ) ;
    }
    if ( !( prm_rnng [ 30 ] ) && ( !exp_md ) ) { vr_dtctr_shdet [ 2 ] *= ( -1 )  ; }

  vr_dtctr_shdet [ 0 ] *= ( ( flt ) 1 - ( prm_rnng [ 11 ] ) * inv_tau_stry );
  vr_dtctr_shdet [ 0 ] += ( ( prm_rdout [ 9 ] ) * vr_dtctr_shdet [ 5 ] ) ;
  vr_dtctr_shdet [ 0 ] += ( ( prm_rdout [ 10 ] ) * ( vr_dtctr_shdet [ 2 ] - vr_dtctr_shdet [ 1 ] ) ) ;
  vr_dtctr_shdet [ 0 ] += ( ( prm_rdout [ 11 ] ) * ( vr_dtctr_shdet [ 1 ] ) ) ;
  vr_dtctr_shdet [ 0 ] += ( ( prm_rdout [ 12 ] ) * ( vr_dtctr_shdet [ 4 ] ) ) ;
  vr_dtctr_shdet [ 4 ] += ( vr_dtctr_shdet [ 1 ] ) ;
  gmm_1 = ( prm_rdout [ 7 ] / r_blmtr - prm_rdout [ 17 ] / prm_rdout [ 4 ] ) / prm_rdout [ 8 ] * prm_rnng [ 11 ] ;
  vr_dtctr_shdet [ 5 ] = gmm_1 * vr_dtctr_shdet [ 0 ] ;
  vr_dtctr_shdet [ 5 ] = vr_dtctr_shdet [ 5 ] + prm_rdout [ 14 ] * vr_dtctr_shdet [ 5 ] ;
  vr_dtctr_shdet [ 5 ] += ( prm_rdout [ 14 ] * vr_dtctr_shdet [ 1 ] ) ;
  vr_dtctr_shdet [ 5 ] += ( prm_rdout [ 15 ] * ( vr_dtctr_shdet [ 2 ] - vr_dtctr_shdet [ 1 ] ) ) ;
  vr_dtctr_shdet [ 5 ] += ( prm_rdout [ 16 ] * vr_dtctr_shdet [ 4 ] ) ;
  v_out_shdet = vr_dtctr_shdet [ 0 ] * prm_inst [ 0 ] ;
  v_out_shdet = ( v_out_shdet >= v_mx ) ? v_mx : v_out_shdet ;
  v_out_shdet = ( v_out_shdet <= v_mn ) ? v_mn : v_out_shdet ;
  vr_dtctr_shdet [ 6 ] *= ( prm_inst [ 2 ] ) ;
  vr_dtctr_shdet [ 6 ] += ( vr_dtctr_shdet [ 7 ] * prm_inst [ 1 ] ) ;
  vr_dtctr_shdet [ 7 ] = ( prm_inst [ 3 ] * v_out_shdet ) + ( prm_rdout [ 5 ] * vr_dtctr_shdet [ 1 ] ) ;
  v_out_shdet = vr_dtctr_shdet [ 6 ] + vr_dtctr_shdet [ 7 ] ;
  v_out_shdet = ( v_out_shdet >= v_mx ) ? v_mx : v_out_shdet ;
  v_out_shdet = ( v_out_shdet <= v_mn ) ? v_mn : v_out_shdet ;
  v_d_c_2_nw = prm_inst [ 5 ] * vr_dtctr_shdet [ 8 ] ;
  v_d_c_2_nw += ( prm_inst [ 4 ] * ( v_out_shdet - vr_dtctr_shdet [ 9 ] ) ) ;
  vr_dtctr_shdet [ 9 ] *= prm_inst [ 6 ] ;
  vr_dtctr_shdet [ 9 ] += ( prm_inst [ 7 ] * vr_dtctr_shdet [ 8 ] ) ;
  vr_dtctr_shdet [ 9 ] += ( prm_inst [ 8 ] * v_out_shdet ) ;
  v_out_shdet = ( v_d_c_2_nw - vr_dtctr_shdet [ 8 ] ) / prm_inst [ 4 ] ;
  v_out_shdet *=  prm_inst [ 9 ] ;
  vr_dtctr_shdet [ 8 ] = v_d_c_2_nw ;
  v_out_shdet = ( v_out_shdet >= v_mx ) ? v_mx : v_out_shdet ;
  v_out_shdet = ( v_out_shdet <= v_mn ) ? v_mn : v_out_shdet ;
  vr_dtctr_shdet [ 10 ] *= prm_inst [ 12 ] ;
  vr_dtctr_shdet [ 10 ] += ( prm_inst [ 10 ] * v_out_shdet ) ;
  v_out_shdet = prm_inst [ 11 ] * vr_dtctr_shdet [ 10 ] ;
  v_out_shdet = ( v_out_shdet >= v_mx ) ? v_mx : v_out_shdet ;
  v_out_shdet = ( v_out_shdet <= v_mn ) ? v_mn : v_out_shdet ;
  v_out_shdet *= prm_inst [ 13 ] ;
  v_h_c_1_nw = prm_inst [ 14 ] * v_out_shdet + prm_inst [ 15 ] * vr_dtctr_shdet [ 11 ] + prm_inst [ 16 ] * vr_dtctr_shdet [ 12 ] + prm_inst [ 17 ] * vr_dtctr_shdet [ 13 ] ;
  v_h_c_2_nw = prm_inst [ 18 ] * v_out_shdet + prm_inst [ 19 ] * vr_dtctr_shdet [ 11 ] + prm_inst [ 20 ] * vr_dtctr_shdet [ 12 ] + prm_inst [ 21 ] * vr_dtctr_shdet [ 13 ] ;
  vr_dtctr_shdet [ 13 ] *= prm_inst [ 23 ] ;
  vr_dtctr_shdet [ 13 ] += ( prm_inst [ 22 ] * vr_dtctr_shdet [ 12 ] ) ;
  v_out_shdet = prm_inst [ 24 ] * vr_dtctr_shdet [ 13 ] ;
  v_out_shdet = ( v_out_shdet >= v_mx ) ? v_mx : v_out_shdet ;
  v_out_shdet = ( v_out_shdet <= v_mn ) ? v_mn : v_out_shdet ;
  vr_dtctr_shdet [ 11 ] = v_h_c_1_nw ;
  vr_dtctr_shdet [ 12 ] = v_h_c_2_nw ;
}
