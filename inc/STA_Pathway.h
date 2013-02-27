
#ifndef PATHWAY_H
#define PATHWAY_H

class STA_Pathway
{
public:
	STA_Pathway(){}

	STA_Pathway( siz_t _im_size);

	~STA_Pathway();

	int apply_mask( float *image, int mu);

	int retina_filter( float *in, siz_t im_size, int gang, int display, int n, float *out);
	
	int gabor_bank( float *inputImage, double *teta, float *sig_hor, float *frequencies, float **maps);
	
	int interactions_short( float **maps, float **mapsT);
	
	int normalizations( float **maps);
	
	int summation( float **maps, float *odata);

protected:
	siz_t im_size;

};

#endif