#ifndef MOTION_ESTIMATION_H
#define MOTION_ESTIMATION_H

#include "types.h"

class DYN_MotionEstimation
{
	public:
	DYN_MotionEstimation();

	~DYN_MotionEstimation();

	static int estimate_motion(motion_vect_t v, gradient_t grad, siz_t level_size, bool init);

	static int apply_projection( motion_vect_t v_out, motion_vect_t v, std::complex<float> mask, siz_t level_size);

	static int apply_gaussian_recursive( motion_vect_t v_out, motion_vect_t v_in, siz_t level_size, float sigma);
};

#endif