#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "types.h"

class DYN_Gradients
{
public:
	static int calculate_gradients( gradient_t grad, float* level_data, gabor_mask_t mask, gabor_mask_t mask_prev, mod_t mod, siz_t level_size);

	static void DYN_Gradients::calculate_gradients_x_y( cmplx_t gx, cmplx_t gy, cmplx_t in, siz_t level_size);

	static void DYN_Gradients::calculate_gradients_t( cmplx_t gt, cmplx_t curr, cmplx_t prev, siz_t level_size);
};

#endif