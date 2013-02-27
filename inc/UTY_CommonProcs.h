
#ifndef COMMON_H
#define COMMON_H

#include "types.h"
#include "fftw3.h"

class UTY_CommonProcs
{
private:
	UTY_CommonProcs();
	~UTY_CommonProcs();

public:	
	static int UTY_retina_gaussion_filter( float *kernel, float *im, siz_t kernel_size, siz_t im_size
		, float sigma, float *out);

	static int UTY_normalization_nl( float *map, siz_t map_size);

	static int UTY_normalization_itti( float *map, siz_t map_size);

	static int UTY_normalization_pc( float *map, siz_t map_size);

	static int UTY_frequency_to_spatial( fftw_complex *io, siz_t map_size);

	static int UTY_spatial_to_frequency( float *in, siz_t map_size, fftw_complex *out);

	static int UTY_retina_bipolar( float *in_photo, float *in_hor, siz_t map_size, int beta, float *out);

	static int UTY_mask_create( float *mask, siz_t map_size, int mu);

	static int UTY_mesh_create( float *in, float d1, float d2, int n);

	static int UTY_gabor_masks( float *u1, float *v1, double *teta, siz_t map_size);

	static int MotionCompensation(float *in, int w, int h, const std::vector<float> lsMC, float *out);
};

#endif