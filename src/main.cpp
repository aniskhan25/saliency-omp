/*
* Saliency map for grayscale clippets
*
* Author	anis rahman
*
* 05/10/10	19:43
*/
/*
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector_types.h>

#include "cutil_inline.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "types.h"

#include "UTY_CommonProcs.h"
#include "UTY_HelperProcs.h"
#include "STA_StaticPass.h"

#include "DYN_DynamicPass.h"

using namespace std;

void temporal_filtering( float *out, float** in, int w, int h );

#define QUEUE_LEN 13

inline void exit(){	cout << endl << "Press enter to continue..." << endl; cin.get();}

int main( int argc, char** argv) 
{
	float mvt_lst[16];

	cv::Mat	im001, im002;

	float *h_idata, *h_idata_01, *h_idata_02, *h_idata_com
		, *h_odata_sta, *h_odata_dyn, *h_odata;

	float **maps = NULL;

	std::vector<int> params;

	params.push_back( CV_IMWRITE_PXM_BINARY );
	params.push_back( 100 );

	int width, height;

	im001 = cv::imread( "C:\\Sophie_these\\Program_Marat_21-01-10\\saillance\\data\\original\\pgm\\ClipRMX_3_0000i.pgm", 0 );

	siz_t im_size;
	im_size.w = im001.cols;
	im_size.h = im001.rows;

	h_idata	   = (float *)calloc( im_size.w*im_size.h, sizeof(float) );	
	h_idata_01 = (float *)calloc( im_size.w*im_size.h, sizeof(float) );	
	h_idata_02 = (float *)calloc( im_size.w*im_size.h, sizeof(float) );	
	h_idata_com = (float *)calloc( im_size.w*im_size.h, sizeof(float) );
	h_odata	   = (float *)calloc( im_size.w*im_size.h, sizeof(float) );	

	maps = (float**) malloc ( QUEUE_LEN * sizeof(float*) );
	for( unsigned int i=0;i<QUEUE_LEN ;++i )
		maps[i] = (float*) malloc ( im_size.w*im_size.h * sizeof(float) );

	char filename[100];
	char command[100];

	unsigned int queue_idx, queue_full;

	std::cout << "Starting calculating saliency maps ..." << std::endl;

	clock_t start = clock();

	std::cout << "Opening video " << std::endl;

	sprintf( filename, "%s%d%s", "C:\\Sophie_these\\Program_Marat_21-01-10\\saillance\\data\\original\\pgm\\ClipRMX_", k, "_0000i.pgm" );
	im001 = cv::imread( filename, 0 );

	queue_full = 0;
	queue_idx = 0;

	unsigned int i=0;

	while(1){

		std::cout << "Image " << i << "\r" ;

		im
		sprintf( filename, "%s%d%s%s%d%s", "C:\\Sophie_these\\Program_Marat_21-01-10\\saillance\\data\\original\\pgm\\ClipRMX_", k, "_", (i<10)?"000":(i<100)?"00":(i<1000)?"0":"", i, "i.pgm" );
		im002 = cv::imread( filename, 0 );		

		for (unsigned int id = 0;id<im_size.w*im_size.h;++id)
		{
			h_idata[id]	   = im002.data[id];

			h_idata_01[id] = im001.data[id];
			h_idata_02[id] = im002.data[id];
		}

		sprintf( filename, "%s%d%s", "C:\\Sophie_these\\Program_Marat_21-01-10\\saillance\\data\\ClipRMX_", k, ".txt" );
		UTY_HelperProcs::iof_get_compensation( filename, mvt_lst, i);
		UTY_CommonProcs::MotionCompensation( h_idata_01, im_size.w, im_size.h, mvt_lst, h_idata_com);

		STA_StaticPass oStaticPass( im_size);
		h_odata_sta = oStaticPass.calculate_saliency_map( h_idata);

		DYN_DynamicPass oDynamicPass( h_idata_01, h_idata_02, im_size);
		h_odata_dyn = oDynamicPass.calculate_saliency_map();

		float max_s, max_d;
		max_s = max_d = 0.0f;

		for (unsigned int id = 0;id<im_size.w*im_size.h;++id){

			if ( h_odata_sta[id] > max_s)
				max_s = h_odata_sta[id];
			if ( h_odata_dyn[id] > max_d)
				max_d = h_odata_dyn[id];
		}

		for (unsigned int id = 0;id<im_size.w*im_size.h;++id){

			h_odata_sta[id] = h_odata_sta[id]/max_s * 255.0f;
			h_odata_dyn[id] = h_odata_dyn[id]/max_d * 255.0f;
		}

		UTY_HelperProcs::UTY_fusion( h_odata_sta, h_odata_dyn, im_size.w, im_size.h, h_odata );

		float max_sd = 0.0f;

		for (unsigned int id = 0;id<im_size.w*im_size.h;++id)
			if ( h_odata[id] > max_sd)
				max_sd = h_odata[id];

		if( !queue_full ){

			for (unsigned int id = 0;id<im_size.w*im_size.h;++id)
				maps[queue_idx][id] = h_odata[id]/max_sd * 255.0f;						

			for (unsigned int id = 0;id<im_size.w*im_size.h;++id){

				im001.data[id] = im002.data[id];
				im002.data[id] = (char)(unsigned char)( maps[queue_idx][id] );
			}

			++queue_idx;

			if( queue_idx == QUEUE_LEN ){
				queue_full = 1;
				queue_idx = 0;
			}
		}
		else{

			if( queue_idx == QUEUE_LEN ){
				queue_idx = 0;
			}

			for (unsigned int id = 0;id<im_size.w*im_size.h;++id)			
				maps[queue_idx][id] = h_odata[id]/max_sd * 255.0f;						

			temporal_filtering( h_odata, maps, im_size.w, im_size.h );

			for (unsigned int id = 0;id<im_size.w*im_size.h;++id){

				im001.data[id] = im002.data[id];
				im002.data[id] = (char)(unsigned char)( h_odata[id] );
			}

			++queue_idx;
		}

		sprintf( filename, "%s%d%s%s%d%s", "D:\\tmp\\dynamic\\ClipRMX_", k, "_", (i<10)?"000":(i<100)?"00":(i<1000)?"0":"", i, "_fus.pgm" );
		cv::imwrite( filename, im002, params );
	}

	std::cout << "Calculation of saliency maps completed." << std::endl;

	cout << "Time elapsed: " << ( (float)clock() - start) / CLOCKS_PER_SEC << endl;

	for( unsigned int i=0;i<QUEUE_LEN;++i )
		free(maps[i]);
	free(maps);

	h_idata = h_odata = h_odata_sta = h_odata_dyn = h_idata_01 = h_idata_02 = NULL;

	free( h_idata);
	free( h_odata);
	free( h_odata_sta);		
	free( h_odata_dyn);		
	free( h_idata_01);
	free( h_idata_02);

	exit();
}

void temporal_filtering( float *out, float** in, int w, int h )
{
	float arr[QUEUE_LEN];

	float tmp;

	unsigned int min_idx;

	int middle = QUEUE_LEN/2;

	float average;

	for( unsigned int idx=0;idx<w*h;++idx )
	{
		for( unsigned int i=0;i<QUEUE_LEN;++i )
			arr[i] = in[i][idx];

		for( unsigned int i=0;i<QUEUE_LEN-1;++i ){

			min_idx = i;
			for( unsigned int j=i+1;j<QUEUE_LEN;++j ){

				if( arr[min_idx]>arr[j] )
					min_idx = j;
			}

			tmp = arr[i];
			arr[i] = arr[min_idx];
			arr[min_idx] = tmp;
		}

		if( QUEUE_LEN%2==0 ) average = (float)(arr[middle-1]+arr[middle])/2;
		else average = arr[middle];

		out[idx] = average;
	}
}
*/