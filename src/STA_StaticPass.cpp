
//#include <cstdlib>
//#include <cmath>

#include "types.h"

#include "STA_Pathway.h"
#include "STA_StaticPass.h"

STA_StaticPass::STA_StaticPass( siz_t _im_size)
{
	im_size = _im_size;

	int 
		i, index
		, size;

	size = im_size.h*im_size.w * sizeof( float);

	maps = ( float **)malloc( NO_OF_ORIENTS*NO_OF_BANDS * sizeof( float *));

	for( i = 0; i < NO_OF_ORIENTS*NO_OF_BANDS ; i++)
		maps[i] = ( float *)malloc( size);

	mapsT =( float **)malloc( NO_OF_ORIENTS*NO_OF_BANDS * sizeof( float *));

	for( i = 0; i < NO_OF_ORIENTS*NO_OF_BANDS ; i++)
		mapsT[i] = ( float *)malloc( size);

	for( i = 0 ; i < NO_OF_ORIENTS ; i++)
		teta[i] = i*180.0 / NO_OF_ORIENTS + 90.0;

	index = 0;
	for( i = NO_OF_BANDS ; i > 0 ; i--){

		frequencies[index]  = FREQ_MAX /( powf( SCALE, i*1.0f - 1)* 1.0f);
		sig_hor[index]		= STD_MAX  /( powf( SCALE, i*1.0f - 1)* 1.0f);

		index++;
	}
}

STA_StaticPass::~STA_StaticPass()
{
	int i;

	for( i = 0 ; i < NO_OF_ORIENTS*NO_OF_BANDS; i++){
		free( maps[i]);
		free( mapsT[i]);
	}

	free( maps);
	free( mapsT);
}

float* STA_StaticPass::calculate_saliency_map( float *in)
{
	STA_Pathway oStaticPathway( im_size);

	float *out = ( float *)malloc( im_size.w*im_size.h * sizeof(float));

	oStaticPathway.retina_filter		( in, im_size, 0, 0, 0, in);

	oStaticPathway.apply_mask			( in, MU);		

	oStaticPathway.gabor_bank			( in, teta, sig_hor, frequencies, maps);

	oStaticPathway.interactions_short	( maps, mapsT);

	oStaticPathway.normalizations		( mapsT);

	oStaticPathway.summation			( mapsT, out);

	oStaticPathway.apply_mask			( out, MU);

	return out;
}