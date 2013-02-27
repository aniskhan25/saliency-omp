
#include "DYN_GaborFilter.h"
#include "DYN_Gradients.h"
#include "DYN_MotionEstimation.h"
#include "DYN_GaussianRecursive.h"

#include "DYN_DynamicPass.h"
#include "STA_Pathway.h"

#include "UTY_HelperProcs.h"
#include "UTY_CommonProcs.h"

void DYN_DynamicPass::calculate_saliency_map( 
								 float* out
							 , float *in							 
							 , siz_t im_size	
							 , std::string path
							 , bool is_first	
							 , unsigned int frame_no
							 )
{			
	std::vector<float> compensation_vars;

		if( is_first)
	{		
		oPyramid.init( &pyramid, in,  im_size, false);

		#pragma omp parallel
	{
#pragma omp for nowait
		for( int k=K-1 ; k>=0 ; k--)
		{
			oStaticPathway.retina_filter( pyramid.pre_levels[k].im_data, pyramid.levels[k].level_size, 0, 0, 0, pyramid.pre_levels[k].im_data);		
		}
	}

		return;
	}

			if( !path.empty())
	{
		//compensation_vars = oMotionCompensation.FetchCompensationVariables( path, frame_no);
		compensation_vars = UTY_HelperProcs::iof_get_compensation( path, frame_no-1);

		if( !compensation_vars.empty())
		{
			UTY_CommonProcs::MotionCompensation( in, im_size.w, im_size.h, compensation_vars, h_idata_com);			
		}
	}

	oPyramid.init( &pyramid, h_idata_com,  im_size, true);

	#pragma omp parallel
	{
#pragma omp for nowait
	for ( int id = 0;id<pyramid.im_size.w*pyramid.im_size.h;++id)
		pyramid.seuil_tmp[id] = 100 * ( ( ( fabs( pyramid.im_data01[id]-pyramid.im_data02[id] ) ) < 0.2f ) ? 1.0f : 0.0f );
	}

	for( int k=K-1 ; k>=0 ; k--){
		
		oStaticPathway.retina_filter( pyramid.levels[k].im_data, pyramid.levels[k].level_size, 0, 0, 0, pyramid.levels[k].im_data);

		if( k<K-1){
			DYN_GaborFilter::projection( pyramid.levels[k].motion_vect, pyramid.levels[k+1].motion_vect, pyramid.levels[k].level_size, pyramid.levels[k+1].level_size);			

			DYN_GaborFilter::interpolation( pyramid.levels[k].dec, pyramid.levels[k].motion_vect, pyramid.levels[k].im_data, pyramid.levels[k].level_size);

			DYN_GaborFilter::apply_gabor_filtering( pyramid.pre_levels[k].gabor_mask, pyramid.pre_levels[k].im_data, pyramid.mod, pyramid.pre_levels[k].level_size, pyramid.im_size);
			DYN_GaborFilter::apply_gabor_filtering( pyramid.levels[k].gabor_mask,	  pyramid.levels[k].im_data,	 pyramid.mod, pyramid.levels[k].level_size, pyramid.im_size);
		}
		else 
			DYN_GaborFilter::apply_gabor_filtering( pyramid.levels[k].gabor_mask, pyramid.levels[k].im_data, pyramid.mod, pyramid.levels[k].level_size, pyramid.im_size);

		DYN_Gradients::calculate_gradients( pyramid.grad, pyramid.levels[k].im_data, pyramid.levels[k].gabor_mask, pyramid.pre_levels[k].gabor_mask, pyramid.mod, pyramid.levels[k].level_size);			

		if (k==K-1) 
			DYN_MotionEstimation::estimate_motion( pyramid.levels[k].motion_vect, pyramid.grad, pyramid.levels[k].level_size, true);
		else
			DYN_MotionEstimation::estimate_motion( pyramid.levels[k].motion_vect, pyramid.grad, pyramid.levels[k].level_size, false);

		DYN_GaussianRecursive::gaussian_recursive( pyramid.levels[k].motion_vect.vx, pyramid.levels[k].motion_vect.vx, pyramid.levels[k].level_size.w, pyramid.levels[k].level_size.h, S2);
		DYN_GaussianRecursive::gaussian_recursive( pyramid.levels[k].motion_vect.vy, pyramid.levels[k].motion_vect.vy, pyramid.levels[k].level_size.w, pyramid.levels[k].level_size.h, S2);
	}

#pragma omp parallel
	{
#pragma omp for nowait
		for ( int id = 0;id<pyramid.im_size.w*pyramid.im_size.h;++id){

			if (pyramid.seuil_tmp[id]<=VP)
				out[id] = sqrt( 
				powf( pyramid.levels[0].motion_vect.vx[id], 2) + 
				powf( pyramid.levels[0].motion_vect.vy[id], 2)
				);
			else
				out[id] = 0.0f;		
		}
	}
}
