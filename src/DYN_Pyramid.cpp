
#include "types.h"

#include "DYN_GaussianRecursive.h"
#include "DYN_GaborFilter.h"
#include "DYN_MotionEstimation.h"

#include "DYN_pyramid.h"

DYN_Pyramid::DYN_Pyramid(){}

DYN_Pyramid::~DYN_Pyramid(){}

void DYN_Pyramid::init( pyramid_t *pyramid, float *im_data, siz_t im_size, bool flag)
{
	if( flag)
	{	
		pyramid->im_data02 = im_data;		
	
		DYN_Pyramid::create_pyramid(pyramid);
	}
	else
	{
		pyramid->im_data01 = im_data;
	
		DYN_Pyramid::create_pyramid_precedent(pyramid);
	}
}


void DYN_Pyramid::init_pyramid( pyramid_t *pyramid)
{
	for( int n=0 ; n<N ; n++){
		pyramid->mod.mt_mod[n].re = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
		pyramid->mod.mt_mod[n].im = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
	}

	DYN_Pyramid::calculate_modulation_matrix (pyramid->mod, F0, pyramid->im_size);

	for( int k=0 ; k<K ; k++){

		if( k==0){				
			pyramid->pre_levels[k].level_size = pyramid->im_size;
			pyramid->levels[k].level_size	  = pyramid->im_size;
		}
		else{

			pyramid->levels[k].level_size.w = pyramid->levels[k-1].level_size.w/2;
			pyramid->levels[k].level_size.h = pyramid->levels[k-1].level_size.h/2;

			pyramid->pre_levels[k].level_size.w = pyramid->levels[k].level_size.w;
			pyramid->pre_levels[k].level_size.h = pyramid->levels[k].level_size.h;
		}

		for( int n=0 ; n<N ; n++){
			pyramid->levels[k].gabor_mask.masks[n].re = (float *)malloc( pyramid->levels[k].level_size.w *pyramid->levels[k].level_size.h * sizeof(float));
			pyramid->levels[k].gabor_mask.masks[n].im = (float *)malloc( pyramid->levels[k].level_size.w *pyramid->levels[k].level_size.h * sizeof(float));
		}

		pyramid->levels[k].dec  = (float *)malloc( pyramid->levels[k].level_size.w*pyramid->levels[k].level_size.h * sizeof(float));

		pyramid->levels[k].motion_vect.vx = (float *)malloc( pyramid->levels[k].level_size.w*pyramid->levels[k].level_size.h * sizeof(float));
		pyramid->levels[k].motion_vect.vy = (float *)malloc( pyramid->levels[k].level_size.w*pyramid->levels[k].level_size.h * sizeof(float));

		pyramid->levels[k].im_data = (float *)malloc( pyramid->levels[k].level_size.w*pyramid->levels[k].level_size.h * sizeof(float));

		// Precedent
		for( int n=0 ; n<N ; n++){
			pyramid->pre_levels[k].gabor_mask.masks[n].re = (float *)malloc( pyramid->pre_levels[k].level_size.w*pyramid->pre_levels[k].level_size.h * sizeof(float));
			pyramid->pre_levels[k].gabor_mask.masks[n].im = (float *)malloc( pyramid->pre_levels[k].level_size.w*pyramid->pre_levels[k].level_size.h * sizeof(float));
		}

		pyramid->pre_levels[k].im_data = (float *)malloc( pyramid->levels[k].level_size.w*pyramid->levels[k].level_size.h * sizeof(float));				
	}

	for( int n=0 ; n<N ; n++){
		pyramid->grad.gx[n].re = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
		pyramid->grad.gx[n].im = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));

		pyramid->grad.gy[n].re = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
		pyramid->grad.gy[n].im = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
		
		pyramid->grad.gt[n].re = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
		pyramid->grad.gt[n].im = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
	}

	pyramid->temp_01 = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
	pyramid->temp_02 = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));
	pyramid->seuil_tmp = (float *)malloc( pyramid->im_size.w*pyramid->im_size.h * sizeof(float));

}

void DYN_Pyramid::create_pyramid( pyramid_t *pyramid)
{	
#pragma omp parallel
	{
#pragma omp for
		for( int m=0 ; m<pyramid->im_size.w*pyramid->im_size.h ; m++)
			pyramid->levels[0].im_data[m] = pyramid->im_data02[m];
	}

	for( int k=1 ; k<K ; k++)
	{
		DYN_GaussianRecursive::gaussian_recursive( pyramid->levels[k-1].gabor_mask.masks[0].re, pyramid->levels[k-1].im_data, pyramid->levels[k-1].level_size.w, pyramid->levels[k-1].level_size.h, S3);

#pragma omp parallel
		{
#pragma omp for
			for (int xy = 0; xy < pyramid->levels[k].level_size.w*pyramid->levels[k].level_size.h; ++xy) 
			{
				int x = xy / pyramid->levels[k].level_size.h;
				int y = xy % pyramid->levels[k].level_size.h;

				int m = x + pyramid->levels[k].level_size.w*y;

				pyramid->levels[k].im_data[m] = pyramid->levels[k-1].gabor_mask.masks[0].re[2*x + 2*y*pyramid->levels[k-1].level_size.w];
			}
		}
	}
}

void DYN_Pyramid::create_pyramid_precedent( pyramid_t *pyramid)
{		
#pragma omp parallel
	{
#pragma omp for
		for( int m=0 ; m<pyramid->im_size.w*pyramid->im_size.h ; m++)
			pyramid->pre_levels[0].im_data[m] = pyramid->im_data01[m];
	}

	for( int k=1 ; k<K ; k++){

#pragma omp parallel
		{
#pragma omp for
			for( int m=0 ; m<pyramid->pre_levels[k-1].level_size.w*pyramid->pre_levels[k-1].level_size.h ; m++)
				pyramid->pre_levels[k-1].gabor_mask.masks[0].re[m] = pyramid->pre_levels[k-1].im_data[m];
		}

		DYN_GaussianRecursive::gaussian_recursive( pyramid->pre_levels[k-1].gabor_mask.masks[0].re, pyramid->pre_levels[k-1].gabor_mask.masks[0].re, pyramid->pre_levels[k-1].level_size.w, pyramid->pre_levels[k-1].level_size.h, S3);

#pragma omp parallel
		{
#pragma omp for
			for (int xy = 0; xy < pyramid->pre_levels[k].level_size.w*pyramid->pre_levels[k].level_size.h; ++xy) 
			{
				int x = xy / pyramid->pre_levels[k].level_size.h;
				int y = xy % pyramid->pre_levels[k].level_size.h;

				int m = x + pyramid->pre_levels[k].level_size.w*y;

				pyramid->pre_levels[k].im_data[m] = pyramid->pre_levels[k-1].gabor_mask.masks[0].re[2*x + 2*y*pyramid->pre_levels[k-1].level_size.w];		
			}
		}
	}

	DYN_GaborFilter::apply_gabor_filtering( pyramid->pre_levels[K-1].gabor_mask, pyramid->pre_levels[K-1].im_data, pyramid->mod, pyramid->pre_levels[K-1].level_size, pyramid->im_size);
}

void DYN_Pyramid::reinit_pyramid_precedent( pyramid_t *pyramid)
{
	for( int k=0 ; k<K ; k++)
#pragma omp parallel
	{
#pragma omp for
		for( int m=0 ; m<pyramid->pre_levels[k].level_size.w*pyramid->pre_levels[k].level_size.h ; m++)
			pyramid->pre_levels[k].im_data[m] = pyramid->levels[k].im_data[m];
	}

	for( int n=0 ; n<N ; n++)
#pragma omp parallel
	{
#pragma omp for
		for( int m=0 ; m<pyramid->pre_levels[K-1].level_size.w*pyramid->pre_levels[K-1].level_size.h ; m++){
			pyramid->pre_levels[K-1].gabor_mask.masks[n].re[m] = pyramid->levels[K-1].gabor_mask.masks[n].re[m];
			pyramid->pre_levels[K-1].gabor_mask.masks[n].im[m] = pyramid->levels[K-1].gabor_mask.masks[n].im[m];
		}
	}
}

void DYN_Pyramid::calculate_modulation_matrix( mod_t mod, float f, siz_t im_size)
{	
	float dpf0, teta, cos_teta, sin_teta;
	
	dpf0 = _2PI*f;

	for( int n=0 ; n<N ; n++){
		teta = ( (n+1) * _PI) / N;

		cos_teta = cosf(teta);
		sin_teta = sinf(teta);

#pragma omp parallel
		{
#pragma omp for
			for (int xy = 0; xy < im_size.w*im_size.h; ++xy) 
			{
				int x = xy / im_size.h;
				int y = xy % im_size.h;

				float tmp = dpf0 * ( y*cos_teta + x*sin_teta);
				int m = x + im_size.w*y;

				mod.mt_mod[n].re[m] = cos(tmp);
				mod.mt_mod[n].im[m] = sin(tmp);
			}
		}
	}
}

void DYN_Pyramid::gabor_filter (float* im_data, std::complex<float> *b, mod_t mod, float* temp)
{

	//high_pass_prefilter( b[N-1], im_data, level_size, S1);

	//for (int n=0;n<N;n++)
	//{
	//	modulation( b[n], b[N-1], mod[n], level_size, mod_size, 2.0, 1.2);

	//	gaussian_recursive (b[n], b[n], level_size, S0);

	//	demodulation( b[n], b[n], mod, level_size, mod_size, 2.0, 1.2);
	//}

}

void DYN_Pyramid::free_pyramid( pyramid_t *pyramid)
{
	for( int n=0 ; n<N ; n++){
		free( pyramid->mod.mt_mod[n].re );
		free( pyramid->mod.mt_mod[n].im );
	}

	for( int k=0 ; k<K ; k++){

		for( int n=0 ; n<N ; n++){
			free( pyramid->levels[k].gabor_mask.masks[n].re );
			free( pyramid->levels[k].gabor_mask.masks[n].im );
		}

		free( pyramid->levels[k].dec );

		free( pyramid->levels[k].motion_vect.vx );
		free( pyramid->levels[k].motion_vect.vy );

		free( pyramid->levels[k].im_data );

		// Precedent
		for( int n=0 ; n<N ; n++){
			free( pyramid->pre_levels[k].gabor_mask.masks[n].re );
			free( pyramid->pre_levels[k].gabor_mask.masks[n].im );			
		}

		free( pyramid->pre_levels[k].im_data );
	}

	for( int n=0 ; n<N ; n++){
		free( pyramid->grad.gx[n].re );
		free( pyramid->grad.gx[n].im );
		
		free( pyramid->grad.gy[n].re );
		free( pyramid->grad.gy[n].im );
		
		free( pyramid->grad.gt[n].re );
		free( pyramid->grad.gt[n].im );		
	}

	free(pyramid->temp_01);
	free(pyramid->temp_02);
}

