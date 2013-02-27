
#ifndef STATIC_MAIN_H
#define STATIC_MAIN_H

#include "types.h"

class STA_StaticPass
{
public:
	STA_StaticPass( siz_t im_size);
	~STA_StaticPass();

	float* calculate_saliency_map( float *in);

protected:
	float frequencies[NO_OF_BANDS], sig_hor[NO_OF_BANDS];
	double teta[NO_OF_ORIENTS];
	float **maps, **mapsT;
	siz_t im_size;
};

#endif