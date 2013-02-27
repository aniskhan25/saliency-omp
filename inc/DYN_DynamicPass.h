#ifndef DYNAMIC_PASS_H
#define DYNAMIC_PASS_H

#include "types.h"

#include "DYN_Pyramid.h"
#include "STA_Pathway.h"

class DYN_DynamicPass
{

public:	
	pyramid_t pyramid;

	DYN_Pyramid oPyramid;

	float *h_idata_com;

	DYN_DynamicPass( siz_t im_size)
		: oStaticPathway( im_size)
	{
		pyramid.im_size = im_size;	

		oPyramid.init_pyramid( &pyramid);

		h_idata_com = (float *)malloc( im_size.w*im_size.h * sizeof(float));
	}

	~DYN_DynamicPass()
	{
		oPyramid.free_pyramid(&pyramid);

		free(h_idata_com);
	}

	
	STA_Pathway oStaticPathway;

	void calculate_saliency_map( 
		float* out
		, float *in							 
		, siz_t im_size	
		, std::string path
		, bool is_first		
		, unsigned int frame_no
		);

	void calculate_modulation_matrix (mod_t mod, float dpf0, float teta, siz_t im_size);

};

#endif