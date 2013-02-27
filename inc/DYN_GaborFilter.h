#ifndef GABOR_FILTER_H
#define GABOR_FILTER_H

#include "types.h"

class DYN_GaborFilter
{
public:
	DYN_GaborFilter();
	~DYN_GaborFilter();

	static int apply_gabor_filtering( gabor_mask_t gabor_mask, float* level_data, mod_t mod, siz_t level_size, siz_t mod_size);

	static void projection( motion_vect_t v_curr, motion_vect_t v_prev, siz_t v_curr_size, siz_t v_prev_size);

	static void interpolation( float *dec, motion_vect_t v_proj, float *niv, siz_t level_size);

	static void modulation( cmplx_t out, cmplx_t prev, cmplx_t mod, siz_t level_size, siz_t mod_size, float dpf0, float teta);

	static void demodulation( cmplx_t out, cmplx_t in, cmplx_t mod, siz_t level_size, siz_t mod_size, float dpf0, float teta);

	static int high_pass_prefilter( cmplx_t out, float *in, siz_t level_size, float sigma);

};

#endif