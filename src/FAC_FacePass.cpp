
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/gpu/gpu.hpp"

#include "FAC_FacePass.h"

#include <iostream>
#include <iomanip>

template<class T>
void convertAndResize(const T& src, T& gray, T& resized, double scale)
{
	if( src.channels() == 3)
	{
		cvtColor( src, gray, CV_BGR2GRAY );
	}
	else
	{
		gray = src;
	}

	cv::Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

	if( scale != 1)
	{
		cv::resize(gray, resized, sz);
	}
	else
	{
		resized = gray;
	}
}

void Face::Pathway::GetFaceMap( 
							   float *out
							   , const std::vector<cv::Rect> faces
							   , const std::vector<int> weights
							   )
{
	unsigned int m;

	int X0, Y0;

	float 
		X, Y
		, a, b, c
		, sigma = 5
		, tmp
		, thr;

	for( int i = 0; i < im_size_.w*im_size_.h; ++i)
		out[i] = 0.0f;

	for( int i = 0; i < faces.size(); ++i)
	{
		X0 = ( faces[i].width  - 1) / 2;
		Y0 = ( faces[i].height - 1) / 2;

		a = 1.0f / ( 2*powf( faces[i].width, 2.0f));
		b = 0.0f;
		c = 1.0f / ( 2*powf( faces[i].height, 2.0f));

		thr = weights[i] * exp( - (
			a * powf( - faces[i].width/2, 2.0f)));

		for( int y = 0; y < faces[i].height; ++y) {
			for( int x = 0; x < faces[i].width; ++x) {

				m = ( faces[i].y + y)*im_size_.w + ( faces[i].x + x);

				X = ( x - X0);
				Y = ( y - Y0);

				tmp = weights[i] * exp( - (
					a * powf( X, 2.0f)
					+ b * X*Y
					+ c * powf( Y, 2.0f)
					));

				if( tmp>thr) out[m] += tmp;

				//out[m] = weights[i] * 
				//	( 1 / (2*_PI*faces[i].width*faces[i].height )) * 
				//	exp( - 
				//	( powf( X, 2.0f) / 2*powf( faces[i].width, 2.0f) + powf( Y, 2.0f) / 2*powf( faces[i].height, 2.0f))
				//	);
			}
		}
	}
}

void Face::Pathway::Init()
{

	if( !cascade_cpu_frontal.load(cascadeNameFrontal_))
	{
		std::cout << "ERROR: Could not load cascade classifier \"" << cascadeNameFrontal_ << "\"" << std::endl;
	}

	if( is_two_pass)
	{
		if( !cascade_cpu_profile.load(cascadeNameProfile_))
		{
			std::cout << "ERROR: Could not load cascade classifier \"" << cascadeNameProfile_ << "\"" << std::endl;
		}
	}

	h_idata = (float*) malloc( size_ * sizeof(float));
	h_odata = (float*) malloc( size_ * sizeof(float));
}

void Face::Pathway::Apply( 
						  std::vector<cv::Rect> & faces
						  , std::vector<int> & weights
						  , std::vector<int> & types
						  , const cv::Mat & im
						  )
{
	for( unsigned int i = 0; i < im.rows; i++ )
	{
		for( unsigned int j = 0; j < im.cols; j++ )
			h_idata[j + i*im.cols] = im.data[ im.step[0]*i + im.step[1]*j];
	}

	oStaticPathway.retina_filter( h_idata, im_size_, 0, 0, 0, h_odata);

	for( unsigned int i = 0; i < im.rows; i++ )
	{
		for( unsigned int j = 0; j < im.cols; j++ )
			h_odata[j + i*im.cols] = h_odata[j + i*im.cols]/255 + im.data[ im.step[0]*i + im.step[1]*j];
	}

	float mn = h_odata[0];	
	for( unsigned int i=0;i<size_;++i ){		
		if( mn>h_odata[i])
			mn = h_odata[i];
	}

	for( unsigned int i=0;i<size_;++i ){
		h_odata[i] -=  mn;
	}

	float mx = h_odata[0];
	for( unsigned int i=0;i<size_;++i ){
		if( mx<h_odata[i])
			mx = h_odata[i];		
	}

	for( unsigned int i=0;i<size_;++i ){
		h_odata[i] =  h_odata[i]/mx * 255;

		if( h_odata[i] < 0)
			h_odata[i] = 0;
		else if( h_odata[i] > 255)
			h_odata[i] = 255;
	}

	for( unsigned int i = 0; i < im.rows; i++ )
	{
		for( unsigned int j = 0; j < im.cols; j++ )
			im.data[ im.step[0]*i + im.step[1]*j] = 
			( char)( unsigned char)(h_odata[j + i*im.cols]);
	}

	///////////////////////////////////////////////////////////////////////////////////////////

	bool findLargestObject = false;
	bool filterRects = false;

	convertAndResize(im, gray_cpu, resized_cpu, scale_);

	///////////////////////////////////////////////////////////////////////////////////////////

	std::vector<cv::Rect> l_faces;
	std::vector<int> l_weights;

	cascade_cpu_frontal.detectMultiScale(resized_cpu, l_faces, 1.1, 0, 0, cv::Size(20,20));

	const double GROUP_EPS = 0.2;

	groupRectangles(l_faces, 5, GROUP_EPS, &l_weights, 0);

	if (!l_faces.empty())
	{
		for( unsigned int i = 0 ; i<l_faces.size(); ++i)
		{
			faces.push_back( l_faces[i]);
			weights.push_back( l_weights[i]);
			types.push_back( 2);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////

	if( is_two_pass)
	{		
		cascade_cpu_profile.detectMultiScale(resized_cpu, l_faces, 1.1, 0, 0, cv::Size(22,20));

		const double GROUP_EPS = 0.2;

		groupRectangles(l_faces, 2, GROUP_EPS, &l_weights, 0);

		if (!l_faces.empty())
		{
			for( unsigned int i = 0 ; i<l_faces.size(); ++i)
			{
				faces.push_back( l_faces[i]);
				weights.push_back( l_weights[i]);
				types.push_back( 1);
			}
		}	
	}
}

void Face::Pathway::Clean() 
{	
	resized_cpu.release();	
	gray_cpu.release();

	h_idata = NULL;	free(h_idata);
	h_odata = NULL; free(h_odata);
}

