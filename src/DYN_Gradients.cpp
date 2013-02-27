
#include "DYN_Gradients.h"

int DYN_Gradients::calculate_gradients( gradient_t grad, float* level_data, gabor_mask_t mask, gabor_mask_t mask_prev, mod_t mod, siz_t level_size)
{
	for( int n=0 ; n<N ; ++n)
	{
		DYN_Gradients::calculate_gradients_x_y ( grad.gx[n], grad.gy[n], mask.masks[n], level_size);
		DYN_Gradients::calculate_gradients_t( grad.gt[n], mask.masks[n], mask_prev.masks[n], level_size);
	}

	return 0;
}

void DYN_Gradients::calculate_gradients_x_y ( cmplx_t gx, cmplx_t gy, cmplx_t in, siz_t level_size) 
{
#pragma omp parallel
	{
#pragma omp for
		for (int xy = 0; xy < level_size.w*level_size.h; ++xy) 
		{
			int x = xy / level_size.h;
			int y = xy % level_size.h;

			int idx = x + level_size.w*y;

			// default constructor sets to (0,0)
			gx.re[idx] = gx.im[idx] = gy.re[idx] = gy.im[idx] = 0.0f;			

			if ( !( x<2 || x>=level_size.w-2 || y<2 || y>=level_size.h-2))
			{
				gx.re[idx] = ( in.re[idx-2] - 8.0f*in.re[idx-1] + 8.0f*in.re[idx+1] - in.re[idx+2]) / 12.0f;
				gx.im[idx] = ( in.im[idx-2] - 8.0f*in.im[idx-1] + 8.0f*in.im[idx+1] - in.im[idx+2]) / 12.0f;

				gy.re[idx] = ( in.re[idx-2*level_size.w] - 8.0f*in.re[idx-level_size.w] + 8.0f*in.re[idx+level_size.w] - in.re[idx+2*level_size.w]) / 12.0f;
				gy.im[idx] = ( in.im[idx-2*level_size.w] - 8.0f*in.im[idx-level_size.w] + 8.0f*in.im[idx+level_size.w] - in.im[idx+2*level_size.w]) / 12.0f;
			}
		}
	}
}

void DYN_Gradients::calculate_gradients_t ( cmplx_t gt, cmplx_t curr, cmplx_t prev, siz_t level_size) 
{
#pragma omp parallel
	{
#pragma omp for
		for(int m=0 ; m<level_size.w*level_size.h ; ++m){
			gt.re[m] = curr.re[m] - prev.re[m];
			gt.im[m] = curr.im[m] - prev.im[m];
		}
	}
}