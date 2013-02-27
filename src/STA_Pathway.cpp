#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "UTY_HelperProcs.h"
#include "UTY_CommonProcs.h"
#include "STA_Pathway.h"

#include "types.h"
#include "fftw3.h"

STA_Pathway::STA_Pathway(siz_t _im_size)
{
	im_size = _im_size;
}

STA_Pathway::~STA_Pathway(){}

//Apply mask the input image
int STA_Pathway::apply_mask( float *image, int mu)
{
	int i, j;

	float var1 =( float)floor( im_size.w / 2.0f);
	float var2 = pow( var1, mu);

	float var3 =( float)floor( im_size.h / 2.0f);
	float var4 = pow( var3, mu);

	if( !image)return 0;

	for( i = 0 ; i < im_size.h ; i++)
		for( j = 0 ; j < im_size.w ; j++)
			image[i*im_size.w + j] *= 
			( 
			( 1.0f -( pow( ( j - var1), mu)/ var2))*
			( 1.0f -( pow( ( i - var3), mu)/ var4))
			);

	return 1;
}

int STA_Pathway::retina_filter( float *in, siz_t im_size, int gang, int display, int n, float *out)
{	
	siz_t kernel_size;
	int beta = 1;
	float sigma
		, *kernel_photo, *kernel_hor
		, *photoreceptor, *horizental;

	if( !in || !out)return 0;

	kernel_size.w = kernel_size.h = 15;

	sigma = 1.0f /( 2.0f * PI * 0.5f);

	kernel_photo  = ( float*)malloc( sizeof( float)* kernel_size.w*kernel_size.h);
	photoreceptor = ( float*)malloc( sizeof( float)* im_size.h*im_size.w);

	UTY_CommonProcs::UTY_retina_gaussion_filter( kernel_photo, in, kernel_size, im_size, sigma, photoreceptor);

	kernel_size.w = kernel_size.h = 21;

	sigma = 0.96f;	

	kernel_hor = ( float *)malloc( sizeof( float)* kernel_size.w*kernel_size.h);
	horizental = ( float *)malloc( sizeof( float)* im_size.h*im_size.w);

	UTY_CommonProcs::UTY_retina_gaussion_filter( kernel_hor, photoreceptor, kernel_size, im_size, sigma, horizental);

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// Bipolar Cells

	// When ON
	UTY_CommonProcs::UTY_retina_bipolar( photoreceptor, horizental, im_size, beta, out);

	//	DisplayMatrix( out, im_size);

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// Ganglion Cells

	//	if( gang == 0)
	// Parvocellular Output
	//		out = out( n + 1 : n + Nrows, n + 1 : n + Mcolumns);

	//	else if gang == 1
	//		// Magnocellular Output in
	//		Mx  = filter2( G, out);
	//	Mx  = filter2( G, Mx);
	//	Mx  = filter2( G, Mx);
	//	out = Mx;

	//	out = out( n + 1 : n + Nrows, n + 1 : n + Mcolumns);

	//	else if gang == 2
	//		// Magnocellular output Y( used to track movement)
	//		A = Y - horizental;

	//	sigma1  = 0.62;
	//	fg2     = fspecial( 'gaussian', 21, sigma1);
	//	out     = filter2( fg2, A);

	kernel_photo = kernel_hor = photoreceptor = horizental = NULL;
	free( kernel_photo);
	free( kernel_hor);
	free( photoreceptor);
	free( horizental);

	return 1;
}

int STA_Pathway::gabor_bank( float *inputImage, double *teta, float *sig_hor, float *frequencies, float **maps){

	int i, j
		, row, col;

	float temp1, temp2
		, *h_gaborMaskU, *h_gaborMaskV;

	fftw_complex *in, *gabor;

	if( !inputImage)return 0;

	h_gaborMaskU =( float *)malloc( sizeof( float)* im_size.h*im_size.w * NO_OF_ORIENTS);
	h_gaborMaskV =( float *)malloc( sizeof( float)* im_size.h*im_size.w * NO_OF_ORIENTS);

	in		=( fftw_complex *)fftw_malloc( sizeof( fftw_complex)* im_size.h * im_size.w);
	gabor	=( fftw_complex *)fftw_malloc( sizeof( fftw_complex)* im_size.h * im_size.w);

	UTY_CommonProcs::UTY_spatial_to_frequency( inputImage, im_size, in);

	UTY_CommonProcs::UTY_gabor_masks( h_gaborMaskU, h_gaborMaskV, teta, im_size);

	for( j = 0; j < NO_OF_ORIENTS ; j++){
		for( i = 0 ; i < NO_OF_BANDS ; i++){

			temp1 = 2.0f * sig_hor[i]*sig_hor[i];

#pragma omp parallel for schedule(dynamic,1)
			for (int xy = 0; xy < im_size.w*im_size.h; ++xy) 
			{
				int x = xy / im_size.h;
				int y = xy % im_size.h;

				temp2 = exp( - (
					powf( h_gaborMaskU[j * im_size.h*im_size.w +( y*im_size.w + x)] - frequencies[i], 2.0f)
					/ temp1 
					+ powf( h_gaborMaskV[j * im_size.h*im_size.w +( y*im_size.w + x)]				, 2.0f)
					/ temp1));

				gabor[y*im_size.w + x][0] = in[y*im_size.w + x][0] * temp2;
				gabor[y*im_size.w + x][1] = in[y*im_size.w + x][1] * temp2;
			}			

			UTY_CommonProcs::UTY_frequency_to_spatial( gabor, im_size);

#pragma omp parallel for schedule(dynamic,1)
			for (int xy = 0; xy < im_size.w*im_size.h; ++xy) 
			{
				int x = xy / im_size.h;
				int y = xy % im_size.h;

				maps[j*NO_OF_BANDS + i][y*im_size.w + x] = fabs( 
					powf( ( float)( gabor[y*im_size.w + x][0]/( float)( im_size.h*im_size.w)), 2.0f)+ 
					powf( ( float)( gabor[y*im_size.w + x][1]/( float)( im_size.h*im_size.w)), 2.0f));
			}
		}
	}

	gabor = in = NULL;
	free( gabor);
	free( in);

	return 1;
}

//Apply neuronal interaction
int STA_Pathway::interactions_short( float **maps, float **mapsT)
{   
	int i, j, ii, jj;
	int jp, jm;

	if( !maps)return 0;

	for( j = 0; j < NO_OF_ORIENTS ; j++){
		for( i = 0 ; i < NO_OF_BANDS ; i++){

			jp =( j == NO_OF_ORIENTS - 1)? 0				  : j + 1;
			jm =( j == 0				)? NO_OF_ORIENTS - 1 : j - 1;

			if( i == 0){

#pragma omp parallel for schedule(dynamic,1)
				for (int xy = 0; xy < im_size.w*im_size.h; ++xy) 
				{
					int x = xy / im_size.h;
					int y = xy % im_size.h;

					mapsT[j*NO_OF_BANDS + i][y*im_size.w + x] 
					= 1.0f  * maps[ j*NO_OF_BANDS + i     ][y*im_size.w + x] 
					+ 0.5f  * maps[ j*NO_OF_BANDS + i + 1 ][y*im_size.w + x] 
					- 0.25f * maps[jp*NO_OF_BANDS + i     ][y*im_size.w + x] 
					- 0.25f * maps[jm*NO_OF_BANDS + i     ][y*im_size.w + x];
				}
			}
			else if( i == NO_OF_BANDS-1){

#pragma omp parallel for schedule(dynamic,1)
				for (int xy = 0; xy < im_size.w*im_size.h; ++xy) 
				{
					int x = xy / im_size.h;
					int y = xy % im_size.h;

					mapsT[j*NO_OF_BANDS + i][y*im_size.w + x] 
					= 0.5f  * maps[ j*NO_OF_BANDS + i - 1][y*im_size.w + x] 
					+ 1.0f  * maps[ j*NO_OF_BANDS + i    ][y*im_size.w + x]
					- 0.25f * maps[jp*NO_OF_BANDS + i     ][y*im_size.w + x] 
					- 0.25f * maps[jm*NO_OF_BANDS + i     ][y*im_size.w + x];
				}
			}
			else{

#pragma omp parallel for schedule(dynamic,1)
				for (int xy = 0; xy < im_size.w*im_size.h; ++xy) 
				{
					int x = xy / im_size.h;
					int y = xy % im_size.h;

					mapsT[j*NO_OF_BANDS + i][y*im_size.w + x] 
					= 0.5f  * maps[ j*NO_OF_BANDS + i - 1][y*im_size.w + x] 
					+ 1.0f  * maps[ j*NO_OF_BANDS + i    ][y*im_size.w + x]
					+ 0.5f  * maps[ j*NO_OF_BANDS + i + 1 ][y*im_size.w + x] 
					- 0.5f * maps[jp*NO_OF_BANDS + i     ][y*im_size.w + x] 
					- 0.5f * maps[jm*NO_OF_BANDS + i     ][y*im_size.w + x];
				}
			}
		}
	}

	return 1;
}

//Apply different normaization
int STA_Pathway::normalizations( float **maps)
{
	int i;

	if( !maps)return 0;

	for( i = 0 ; i < NO_OF_ORIENTS*NO_OF_BANDS ; i++){

		UTY_CommonProcs::UTY_normalization_nl	( maps[i], im_size);						
		UTY_CommonProcs::UTY_normalization_itti	( maps[i], im_size);
		UTY_CommonProcs::UTY_normalization_pc	( maps[i], im_size);
	}

	return 1;
}

//Fusion of partial energy maps into a saliency map
int STA_Pathway::summation( float **maps, float *odata)
{
	int i, j;

	if( !maps)return 0;

	#pragma omp parallel for schedule(dynamic,1) private(j)
	for( i = 0 ; i < im_size.h*im_size.w ; ++i){

		float sum = 0.0f;
		for( j = 0; j < NO_OF_ORIENTS*NO_OF_BANDS ; j++)				
			sum += maps[j][i];

		if( sum > 1.0f)
			odata[i] = 1.0f;
		else if( sum < 0.0f)
			odata[i] = 0.0f;
		else
			odata[i] = sum;
	}

	return 1;
}