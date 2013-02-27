#ifndef GAUSSIAN_RECURSIVE_H
#define GAUSSIAN_RECURSIVE_H

#include "types.h"

class DYN_GaussianRecursive
{
protected:

	static void transpose( float* out, float* in, const unsigned int size_x, const unsigned int size_y );

	static void gaussian_recursive_causal (float* out, float *in, int w, int h, float b0, float b1, float b2, float b3);

	static void gaussian_recursive_anticausal( float* out, float *in, int w, int h, float b0, float b1, float b2, float b3);

public:
	static int gaussian_recursive (float *out, float* in, int w, int h, float sigma);

};

#endif