//#include <cstdlib>
//#include <cmath>

#include "DYN_GaussianRecursive.h"

void DYN_GaussianRecursive::transpose( float* out, float* in, const unsigned int size_x, const unsigned int size_y ) 
{
#pragma omp parallel
	{
#pragma omp for
		for (int xy = 0; xy < size_x*size_y; ++xy) 
		{
			int x = xy / size_y;
			int y = xy % size_y;

			out[(x * size_y) + y] = in[(y * size_x) + x];
		}
	}  
}

void DYN_GaussianRecursive::gaussian_recursive_causal (float* out, float *in, int w, int h, float b0, float b1, float b2, float b3){

	float B;
	float tampon[10];    
	int i, j;

	B = 1 + b1 + b2 + b3;

	/* Horizental causal filtering */

	for( j=0; j<h ; ++j){

		int idx = j*w;

		tampon[0] = B*in[idx];
		tampon[1] = B*in[idx+1] + (b1/b0)*tampon[0];
		tampon[2] = B*in[idx+2] + (b1*tampon[1] + b2*tampon[0]) / b0;

		for( i=3 ; i<10 ; i++)
			tampon[i] = B*in[idx+i] + ( b1*tampon[i-1] + b2*tampon[i-2] + b3*tampon[i-3]) / b0;

		out[idx++] = B*in[idx] + ( b1*tampon[9] + b2*tampon[8] + b3*tampon[7]) / b0;
		out[idx++] = B*in[idx] + ( b1*out[idx-1]  + b2*tampon[9] + b3*tampon[8]) / b0;
		out[idx++] = B*in[idx] + ( b1*out[idx-1]  + b2*out[idx-2]  + b3*tampon[9]) / b0;

		for( i=3 ; i<w ; ++i)
			out[idx++] = B*in[idx] + ( b1*out[idx-1] + b2*out[idx-2] + b3*out[idx-3]) / b0;		
	}
}

void DYN_GaussianRecursive::gaussian_recursive_anticausal( float* out, float *in, int w, int h, float b0, float b1, float b2, float b3){

	float B;    
	float tampon[10];    
	int i, j;

	B = 1 + b1 + b2 + b3;

	/* Horizental anti-causal filtering */

	for( j=0 ; j<h ; ++j){

		int idx = j*w + w-1;

		tampon[0] = B*in[idx];
		tampon[1] = B*in[idx-1] + (b1/b0)*tampon[0];
		tampon[2] = B*in[idx-2] + (b1*tampon[1] + b2*tampon[0]) / b0;

		for( i=3 ; i<10 ; i++)
			tampon[i] = B*in[idx-i] + ( b1*tampon[i-1] + b2*tampon[i-2] + b3*tampon[i-3]) / b0;

		out[idx--] = B*in[idx] + ( b1*tampon[9] + b2*tampon[8] + b3*tampon[7]) / b0;
		out[idx--] = B*in[idx] + ( b1*out[idx+1]  + b2*tampon[9] + b3*tampon[8]) / b0;
		out[idx--] = B*in[idx] + ( b1*out[idx+1]  + b2*out[idx+2]  + b3*tampon[9]) / b0;

		for( i=w-4 ; i>=0 ; --i)
			out[idx--] = B*in[idx] + ( b1*out[idx+1] + b2*out[idx+2] + b3*out[idx+3]) / b0;
	}
}


int DYN_GaussianRecursive::gaussian_recursive (float *out, float* in, int w, int h, float sigma){
	

	float q, b0, b1, b2, b3;
	float d1r, d1i, d3, b;    

	q = ( sigma/2.1234f) + 0.0467f;

	d1r = powf( 1.7387f,1/q) * cosf( 0.61861f/q);
	d1i = powf( 1.7387f,1/q) * sinf( 0.61861f/q);    
	d3  = powf( 1.8654f,1/q);

	b  =  1 / ( d3 * ( d1r*d1r + d1i*d1i));
	b1 = -b * ( d1r*d1r + d1i*d1i + 2*d3*d1r);
	b2 =  b * (2*d1r + d3);
	b3 = -b;    
	b0 = -1;

	gaussian_recursive_causal	 ( tmp_01, in, w, h, b0, b1, b2, b3);
	gaussian_recursive_anticausal( tmp_02, tmp_01, w, h, b0, b1, b2, b3);

	transpose( tmp_01, tmp_02, w, h);

	gaussian_recursive_causal	 ( tmp_02, tmp_01, h, w, b0, b1, b2, b3);
	gaussian_recursive_anticausal( tmp_01, tmp_02, h, w, b0, b1, b2, b3);

	transpose( out, tmp_01, h, w);
	
	return 1;
}
