#ifndef PYRAMID_H
#define PYRAMID_H

#include "types.h"

class DYN_Pyramid
{

public:
	//pyramid_t *pyramid;
	
	DYN_Pyramid();

	~DYN_Pyramid();
	
	void init( pyramid_t *pyramid, float *im_data, siz_t im_size, bool flag);

	void init_pyramid( pyramid_t *pyramid);

	void create_pyramid( pyramid_t *pyramid);

	void create_pyramid_precedent( pyramid_t *pyramid);

	void reinit_pyramid_precedent( pyramid_t *pyramid);

	void calculate_modulation_matrix (mod_t out, float f0, siz_t im_size);

	void gabor_filter (float* im_data, std::complex<float> *b, mod_t mod, float* temp);

	void free_pyramid( pyramid_t *pyramid);

};

#endif