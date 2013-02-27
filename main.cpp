
// Utilities and system includes
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

#include <omp.h>

#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"

//#include "opencv2/gpu/gpu.hpp"

#include "types.h"

#include "UTY_CommonProcs.h"
#include "UTY_HelperProcs.h"

#include "STA_StaticPass.h"
#include "DYN_DynamicPass.h"
#include "FAC_FacePass.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;
namespace po = boost::program_options;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
	return os;
}

#define SCALE_FACTOR 1.0f
#define SCALE_FACTOR_DYNAMIC 1.0f

#define QUEUE_LEN 13

boost::mutex g_mutex;	// mutex to synchronize access to buffer

int frame_idx, iter, niter, queue_len;

string 
input_path, compensation_path
, static_path, dynamic_path, face_path, master_path
, input_file, config_file
, cascadename
, input_ext
, output_ext;

bool is_completed, is_input_ready, is_static_ready, is_dynamic_ready, is_face_ready;

void GetFrameThread();
void ComputeStaticThread();
void ComputeDynamicThread();
void ComputeFaceThread();
void ComputeFusionTwoPathThread();
void ComputeFusionThreePathThread();

float 
*h_idata
, *h_odata_s, *h_odata_d, *h_odata_f, *h_odata_f_cropped;

unsigned int 
in_nbpixels, out_nbpixels
,in_nbpixels_cropped, out_nbpixels_cropped;

siz_t 
in_size, out_size
, in_size_cropped, out_size_cropped;

unsigned int 
in_offset_w, in_offset_h
, out_offset_w, out_offset_h;

bool 
one_path_static, one_path_dynamic, one_path_face, two_path, three_path
, save_videos, save_images, save_text
, display_input, display_static, display_dynamic, display_face, display_master
, compensate;

vector<cv::Rect> faces;
vector<int> types;
vector<int> weights;	

Mat frame, gray_curr, gray_prev;
VideoCapture capture; // open the default camera

inline void SaveVideo( cv::Mat mat, cv::VideoWriter outputVideo)
{ outputVideo << mat; }

inline void SaveText( float *data, std::stringstream & ss_path, siz_t im_size)
{ UTY_HelperProcs::WriteFile( (char*)ss_path.str().c_str(), data, im_size); }

inline void SaveImage( cv::Mat mat, std::stringstream & ss_path)
{ imwrite( ss_path.str(), mat ); }

inline void GetImageMat( cv::Mat mat, float *data, siz_t out_size, siz_t in_size, siz_t offset)
{			
	float mx = std::numeric_limits<int>::min();	
	for( unsigned int j=0;j<in_size.w*in_size.h;++j) { if( data[j]>mx) mx = data[j]; }

	for( unsigned int j=0;j<in_size.h;++j) { for( unsigned int i=0;i<in_size.w;++i) {
		mat.data[(i+offset.w) + (j+offset.h)*out_size.w] =  ( char)( unsigned char)( data[i + j*in_size.w]/mx*255.0f);
	}
	}
}

inline void GetImageMatPrime( cv::Mat mat, float *data, siz_t out_size, siz_t in_size, siz_t offset)
{	
	float mn = std::numeric_limits<int>::max();	
	float mx = std::numeric_limits<int>::min();	

	for( unsigned int j=0;j<in_size.w*in_size.h;++j) {
		if( data[j]<mn) mn = data[j]; 
		if( data[j]>mx) mx = data[j]; 
	}

	for( unsigned int j=0;j<in_size.h;++j) { 
		for( unsigned int i=0;i<in_size.w;++i) {
			mat.data[(i+offset.w) + (j+offset.h)*out_size.w] =  ( char)( unsigned char)( (data[i + j*in_size.w]-mn)/(mx-mn)*255.0f);
		}
	}
}

/**
* Resizes the image to specified @a width and @a height via reduced resolution.
* The above sentence will be treated as brief.  Here are some more details. These
* sentences are treated as in one paragraph.
*
* And more details requiring a new paragraph.
* @param width no need to comment about a parameter if its name is descriptive
*     enough. Indent 4 characters if a param line wraps.
* @param height parameters should be documented all or none. If you document one
*     parameter, you must document all the others (leave the description blank
*     if nothing to say), to let doxygen check the consistency of docs.
* @param[out] an output parameter. @c 0 if ...
* @return @c true if succeeds.
*/
int main(int ac, char* av[])
{
	try {

		omp_set_num_threads(8);

		cout << "\n";
		cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
		cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";

		//////////////////////////////////////////////////////////////////////////

		// Declare a group of options that will be 
		// allowed only on command line
		po::options_description generic("Generic options");
		generic.add_options()
			("verbose,v", "print status messages (TODO)")
			("version", "print version string")
			("help,h", "produce help message")
			("one-path-static", "compute static saliency")
			("one-path-dynamic", "compute dynamic saliency")
			("one-path-face", "compute face saliency")
			("two-path", "compute two pathway visual saliency")
			("three-path", "compute three pathway visual saliency")
			("save-videos", "save saliency maps as video stream")
			("save-images", "save saliency maps as image sequence")
			("save-text-files", "save saliency maps as text files")
			("display-input", "visualize input video frames")
			("display-static", "visualize static saliency maps")
			("display-dynamic", "visualize dynamic saliency maps")
			("display-face", "visualize face saliency maps")
			("display-master", "visualize master saliency maps")			
			("config,c", po::value<string>(&config_file)->default_value("saliency.cfg"),
			"name of a file of a configuration.")
			;

		// Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		po::options_description config("Configuration");
		config.add_options()
			("temporal-len,t", po::value<int>(&queue_len)->default_value(5), 
			"queue length of temporal filter. Maximum length of 13 allowed.")
			("iter,I", po::value<int>(&iter)->default_value(0), 
			"start of iterations.")
			("niter,N", po::value<int>(&niter)->default_value(10), 
			"number of iterations.")
			("input-path,i", po::value<string>(&input_path)->composing(), 
			"input path of videos.")			
			("static-path,s", po::value<string>(&static_path)->composing(), 
			"image path of static saliency map.")
			("dynamic-path,d", po::value<string>(&dynamic_path)->composing(), 
			"image path of dynamic saliency map.")
			("face-path,f", po::value<string>(&face_path)->composing(), 
			"image path of face saliency map.")
			("master-path,m", po::value<string>(&master_path)->composing(), 
			"image path of fused master saliency map.")
			("compensation-path,C", po::value<string>(&compensation_path)->composing(), 
			"text file path of camera compensation.")
			("iext", po::value<string>(&input_ext)->default_value("mpg"), 
			"input video extension.")
			("oext", po::value<string>(&output_ext)->default_value("png"), 
			"output video, or images extension.")
			("compensate", po::value<bool>(&compensate)->default_value(false), 
			"output video, or images extension.")
			("cascade-name", po::value<string>(&cascadename)->composing(), 
			"haar cascade file path.")
			;

		// Hidden options, will be allowed both on command line and
		// in config file, but will not be shown to the user.
		po::options_description hidden("Hidden options");
		hidden.add_options()
			("input-file", po::value<string>(&input_file), "input video file")
			;

		po::options_description cmdline_options;
		cmdline_options.add(generic).add(config).add(hidden);

		po::options_description config_file_options;
		config_file_options.add(config).add(hidden);

		po::options_description visible("Allowed options");
		visible.add(generic).add(config);

		po::positional_options_description p;
		p.add("input-file", -1);

		po::variables_map vm;
		store(po::command_line_parser(ac, av).
			options(cmdline_options).positional(p).run(), vm);
		notify(vm);

		ifstream ifs(config_file.c_str());
		if (!ifs)
		{
			cout << "can not open config file: " << config_file << "\n";
			return 0;
		}
		else
		{
			store(parse_config_file(ifs, config_file_options), vm);
			notify(vm);
		}

		one_path_static  = (vm.count("one-path-static")) ? true : false;
		one_path_dynamic = (vm.count("one-path-dynamic"))? true : false;
		one_path_face	 = (vm.count("one-path-face"))   ? true : false;

		two_path   = (vm.count("two-path"))   ? true : false;
		three_path = (vm.count("three-path")) ? true : false;

		save_videos = (vm.count("save-videos"))     ?  true : false;
		save_images = (vm.count("save-images"))     ?  true : false;
		save_text   = (vm.count("save-text-files")) ?  true : false;

		display_input   = (vm.count("display-input")) 	?  true : false;
		display_static  = (vm.count("display-static")) 	?  true : false;
		display_dynamic = (vm.count("display-dynamic")) ?  true : false;
		display_face    = (vm.count("display-face")) 	?  true : false;
		display_master  = (vm.count("display-master")) 	?  true : false;

		if( display_input)  namedWindow( "Input",  CV_WINDOW_AUTOSIZE);
		if( display_static) namedWindow( "Static", CV_WINDOW_AUTOSIZE);
		if( display_dynamic)namedWindow( "Dynamic",CV_WINDOW_AUTOSIZE);
		if( display_face)   namedWindow( "Face",   CV_WINDOW_AUTOSIZE);
		if( display_master) namedWindow( "Master", CV_WINDOW_AUTOSIZE);

		if( vm.count("version")) {
			cout << "STVS, version 1.0\n";
			return 0;
		}

		if( vm.count("help") || 
			input_path.empty() || 
			input_file.empty() || 
			( cascadename.empty() && three_path) || 
			queue_len>QUEUE_LEN) 
		{
			cout << visible << "\n";
			return 0;
		}

		if( save_images && !input_path.empty())
			cout << "Input video path is: " << input_path << "\n";

		if( save_videos && !input_path.empty())
			cout << "Input video name is: " << input_file << "\n";

		if( !static_path.empty())
			cout << "Static saliency image path is: " << static_path << "\n";
		if( !dynamic_path.empty())
			cout << "Dynamic saliency image path is: " << dynamic_path << "\n";
		if( !face_path.empty())
			cout << "Face saliency image path is: "	<< face_path << "\n";
		if( !master_path.empty())
			cout << "Master saliency image path is: " << master_path << "\n";
		if( !compensation_path.empty())
			cout << "Camera compensation text file path is: " << compensation_path << "\n";
		if( !cascadename.empty())
			cout << "Haar cascade file path: " << cascadename << "\n";

		cout << "Queue length of temporal filtering is " << queue_len << "\n";

		cout << "Start of iterations is "  <<  iter << "\n";
		cout << "Number of iterations is " << niter << "\n";

		cout << "Compute two pathway visual saliency model is "   << ((two_path)  ?"true":"false") << "\n";
		cout << "Compute three pathway visual saliency model is " << ((three_path)?"true":"false") << "\n";

		cout << "Save saliency maps as image sequence is " << ((save_images)?"true":"false") << "\n";
		cout << "Save saliency maps as video stream is "   << ((save_videos)?"true":"false") << "\n";
		cout << "Save saliency maps as text files is "     << ((save_text)  ?"true":"false") << "\n";

		cout << "Visualize input frames is " 			<< ((display_input)  ?"true":"false") << "\n";
		cout << "Visualize static saliency maps is " 	<< ((display_static) ?"true":"false") << "\n";
		cout << "Visualize dynamic saliency maps is " 	<< ((display_dynamic)?"true":"false") << "\n";
		cout << "Visualize face saliency maps is " 		<< ((display_face)   ?"true":"false") << "\n";
		cout << "Visualize master saliency maps is " 	<< ((display_master) ?"true":"false") << "\n";

		//////////////////////////////////////////////////////////////////////////

		stringstream ss_vidname;
		ss_vidname << input_path << "/" << input_file << "." << input_ext;

		capture.open( ss_vidname.str());

		if(!capture.isOpened()) { // check if we succeeded
			cout << "Unable to read input video." << endl;
			return -1;
		}
		cout << "Input video : " << ss_vidname.str() << endl;

		frame_idx = -1;
		while( iter!=frame_idx)
		{
			capture >> frame; // get a new frame from camera

			cvtColor(frame,gray_curr,CV_RGB2GRAY);

			++frame_idx;
		}

		in_size.h = frame.rows;
		in_size.w = frame.cols;

		// reset all flags
		is_completed 	 = false;
		is_input_ready 	 = false;
		is_static_ready  = false;
		is_dynamic_ready = false;
		is_face_ready 	 = false;

		////////////////////////////////////////////////////////////////////////////

		/// <summary>
		/// Input images are rescaled, if the size dimensions are not a multiple of 16.
		/// The reason is multiple of 16 being the ideal size of data access and processing on GPUs.
		/// Also, cufft library requires the data to be a multiple of 16.
		/// </summary>
		out_size.w = in_size.w * SCALE_FACTOR;
		out_size.h = in_size.h * SCALE_FACTOR;

		in_size_cropped.w = (int) ( floor(in_size.w/16.0f) * 16.0f);
		in_size_cropped.h = (int) ( floor(in_size.h/16.0f) * 16.0f);

		in_offset_w = (in_size.w - in_size_cropped.w) * 0.5f;
		in_offset_h = (in_size.h - in_size_cropped.h) * 0.5f;

		out_size_cropped.w = ( int)( in_size_cropped.w * SCALE_FACTOR);
		out_size_cropped.h = ( int)( in_size_cropped.h * SCALE_FACTOR);

		out_offset_w = (int)( ( out_size.w - out_size_cropped.w) * 0.5f);
		out_offset_h = (int)( ( out_size.h - out_size_cropped.h) * 0.5f);

		in_nbpixels  =  in_size.w *  in_size.h;
		out_nbpixels = out_size.w * out_size.h;

		in_nbpixels_cropped  =  in_size_cropped.w *  in_size_cropped.h;
		out_nbpixels_cropped = out_size_cropped.w * out_size_cropped.h;

		////////////////////////////////////////////////////////////////////////////

		h_idata	  = ( float *)malloc(  in_nbpixels_cropped * sizeof( float));
		h_odata_s = ( float *)malloc( out_nbpixels_cropped * sizeof( float));
		h_odata_d = ( float *)malloc( out_nbpixels_cropped * sizeof( float));

		h_odata_f 			= ( float *)malloc( out_nbpixels	   * sizeof( float));
		h_odata_f_cropped	= ( float *)malloc( out_nbpixels_cropped * sizeof( float));

		if( one_path_static)
		{
			two_path = false;

			is_dynamic_ready = true;

			boost::thread get_frame_thread( 				&GetFrameThread); 		// start input frame thread
			boost::thread compute_static_thread( 			&ComputeStaticThread); 		// start static pass thread
			boost::thread compute_fusion_two_path_thread( 	&ComputeFusionTwoPathThread); 	// start two pathway fusion thread

			get_frame_thread.join(); // wait for timer_thread to finish
			compute_static_thread.join();
			compute_fusion_two_path_thread.join();

		}	
		else if( one_path_dynamic)
		{
			two_path = false;

			is_static_ready = true;

			boost::thread get_frame_thread(                                 &GetFrameThread);               // start input frame thread
			boost::thread compute_dynamic_thread(                   &ComputeDynamicThread);         // start dynamic pass thread
			boost::thread compute_fusion_two_path_thread(   &ComputeFusionTwoPathThread);   // start two pathway fusion thread

			get_frame_thread.join(); // wait for timer_thread to finish
			compute_dynamic_thread.join();
			compute_fusion_two_path_thread.join();

			//two_path = false;

			//is_static_ready = true;

			//boost::thread_group processors;

			//boost::thread get_frame_thread( 				&GetFrameThread); 		// start input frame thread
			//boost::thread compute_dynamic_thread( 			&ComputeDynamicThread); 	// start dynamic pass thread
			//boost::thread compute_fusion_two_path_thread( 	&ComputeFusionTwoPathThread); 	// start two pathway fusion thread

			//processors.add_thread(&get_frame_thread);			
			//processors.add_thread(&compute_dynamic_thread);			
			//processors.add_thread(&compute_fusion_two_path_thread);

			//processors.join_all();
		}
		else if( one_path_face)
		{
			three_path = false;

			is_static_ready  = true;
			is_dynamic_ready = true;

			boost::thread get_frame_thread(					&GetFrameThread);               // start input frame thread
			boost::thread compute_face_thread(				&ComputeFaceThread);         // start dynamic pass thread
			boost::thread compute_fusion_three_path_thread(	&ComputeFusionThreePathThread);   // start three pathway fusion thread

			get_frame_thread.join(); // wait for timer_thread to finish
			compute_face_thread.join();
			compute_fusion_three_path_thread.join();
		}
		else if( three_path)
		{
			boost::thread get_frame_thread( 				&GetFrameThread); 		// start input frame thread
			boost::thread compute_static_thread( 			&ComputeStaticThread); 		// start static pass thread
			boost::thread compute_dynamic_thread( 			&ComputeDynamicThread); 	// start dynamic pass thread
			boost::thread compute_face_thread(				&ComputeFaceThread);		// start face pass thread
			boost::thread compute_fusion_three_path_thread(	&ComputeFusionThreePathThread); // start three pathway fusion thread

			get_frame_thread.join(); // wait for timer_thread to finish
			compute_static_thread.join();
			compute_dynamic_thread.join();
			compute_face_thread.join();
			compute_fusion_three_path_thread.join();
		}
		else // default two_path
		{
			two_path = true;

			boost::thread get_frame_thread( 				&GetFrameThread); 		// start input frame thread
			boost::thread compute_static_thread( 			&ComputeStaticThread); 		// start static pass thread
			boost::thread compute_dynamic_thread( 			&ComputeDynamicThread); 	// start dynamic pass thread
			boost::thread compute_fusion_two_path_thread( 	&ComputeFusionTwoPathThread); 	// start two pathway fusion thread

			get_frame_thread.join(); // wait for timer_thread to finish
			compute_static_thread.join();
			compute_dynamic_thread.join();			
			compute_fusion_two_path_thread.join();
		}

		free( h_idata);
		free( h_odata_s);
		free( h_odata_d);
		free( h_odata_f);
		free( h_odata_f_cropped);
	}
	catch(exception& e)
	{
		cout << "CATCH" << endl;
		cout << e.what() << endl;
		return 1;
	}
}

void GetFrameThread() // get input frame
{	
	int framecount = (int)capture.get( CV_CAP_PROP_FRAME_COUNT);

	while(1) {

		while( is_input_ready && (!is_completed));

		if( is_completed) break;

		cout << "Frame : " << frame_idx << " / " << framecount << "\r";

		if( frame_idx > iter ) // first frame already in buffer on video capture
		{
			capture >> frame;

			gray_curr.copyTo(gray_prev);
		}

		cvtColor(frame,gray_curr,CV_RGB2GRAY);

		//		if( !gray_curr.empty()) {
		if( !gray_curr.empty() && ( frame_idx < ( iter + niter) || niter==0)) {

			for( int j=0;j<in_size_cropped.h;++j) {
				for( int i=0;i<in_size_cropped.w;++i) {
					h_idata[i + j*in_size_cropped.w] =  gray_curr.data[(i+in_offset_w) + (j+in_offset_h)*in_size.w];					
				}
			}

			if( display_input)
			{
				imshow("Input", gray_curr);
				char c = waitKey(30);

				if( c == 27 ) 
				{
					// get exclusive ownership of mutex (wait for light to tuen green)
					boost::mutex::scoped_lock lock_it( g_mutex ) ;
					// ok, now we have exclusive access to the light

					is_completed = true;
					// destructor for lock_it will release the mutex
				}
			}

			{
				// get exclusive ownership of mutex (wait for light to tuen green)
				boost::mutex::scoped_lock lock_it( g_mutex ) ;
				// ok, now we have exclusive access to the light

				is_input_ready = true;

				// destructor for lock_it will release the mutex
			}
		}			
		else {
			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_completed = true;
			// destructor for lock_it will release the mutex
		}
	}

	cout << "Thread input : Ended" << endl;
}


void ComputeStaticThread() // get input frame
{
	//cudaSetDevice(1);	

	double wtime;

	float mx;

	Mat im_s = Mat( out_size.h, out_size.w, CV_8UC1, Scalar(0) );	
	vector<int> params;

	params.push_back( CV_IMWRITE_PXM_BINARY);
	params.push_back( 100);
	/*
	for( int i=0;i<out_nbpixels;++i) {		
	im_s.data[i] = ( char)( unsigned char)( 0);
	}
	*/
	float *temp = (float*) calloc(sizeof(float), out_nbpixels);

	//Static::Pathway oStaticPathway( in_size_cropped, 1.0f);

	STA_StaticPass oStaticPass( in_size_cropped);

	////////////////////////////////////////////////////////////////////////////////////////////////////

	const bool askOutputType = false;

	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

	Size S = Size((int) out_size.w, (int)out_size.h);    //Acquire input size

	VideoWriter outputVideo;	// Open the output

	stringstream ss_vidname;

	if( !static_path.empty() && save_videos)
	{
		ss_vidname << static_path << "/" << input_file << "_sta" << "." << output_ext;

		if (askOutputType)
			outputVideo.open(ss_vidname.str(), ex=-1, capture.get(CV_CAP_PROP_FPS), S, true);
		else   
			outputVideo.open(ss_vidname.str(), ex,    capture.get(CV_CAP_PROP_FPS), S, true);

		if (!outputVideo.isOpened())
		{
			cout  << "Could not open the static output video for write: " << endl;

			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_completed = true;
			// destructor for lock_it will release the mutex
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////

	while(1){

		while( !is_input_ready && !is_completed);

		if( is_completed) break;

		wtime = omp_get_wtime ( );

		//oStaticPathway.Apply( h_odata_s, h_idata, out_size_cropped, in_size_cropped);
		h_odata_s = oStaticPass.calculate_saliency_map( h_idata);

		wtime = omp_get_wtime ( ) - wtime;
		cout << "\n";
		cout << "  Elapsed cpu time for main computation:\n";
		cout << "  " << wtime << " seconds.\n";		

		GetImageMat( im_s, h_odata_s, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));

		if( !static_path.empty())
		{
			stringstream ss_path;
			ss_path << static_path << "/" << input_file << "_"<< frame_idx << "_sta" << "." << output_ext;

			if( save_videos) SaveVideo( im_s, outputVideo);
			if( save_images) SaveImage( im_s, ss_path);
			if( save_text) SaveText( h_odata_s, ss_path, out_size_cropped);
		}

		if( display_static)
		{
			imshow("Static", im_s);
			char c = waitKey(30);

			if( c == 27 ) 
			{
				// get exclusive ownership of mutex (wait for light to tuen green)
				boost::mutex::scoped_lock lock_it( g_mutex ) ;
				// ok, now we have exclusive access to the light

				is_completed = true;
				// destructor for lock_it will release the mutex
			}
		}

		{
			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_static_ready = true;	

			// destructor for lock_it will release the mutex	
		}

		while( is_static_ready && !is_completed);
	}

	im_s.release();
	free(temp);

	cout << "Thread static : Ended" << endl;
}


void ComputeDynamicThread() // get input frame
{
	//cudaSetDevice(0);

	float mx;

	double wtime;

	Mat im_d = Mat( out_size.h, out_size.w, CV_8UC1 );
	vector<int> params;

	params.push_back( CV_IMWRITE_PXM_BINARY);
	params.push_back( 100);

	vector<float> compensation_vars;

	for( int i=0;i<out_nbpixels;++i) {
		im_d.data[i] = ( char)( unsigned char)( 0);
	}

	stringstream ss_comname;
	if( !compensation_path.empty())
		ss_comname << compensation_path << "/" << input_file << ".txt";

	//Dynamic::Pathway oDynamicPathway( in_size_cropped, SCALE_FACTOR_DYNAMIC, queue_len);

	DYN_DynamicPass oDynamicPass( in_size_cropped);

	////////////////////////////////////////////////////////////////////////////////////////////////////

	const bool askOutputType = false;

	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

	Size S = Size((int) out_size.w, (int)out_size.h);    //Acquire input size

	VideoWriter outputVideo;	// Open the output

	stringstream ss_vidname;

	if( !dynamic_path.empty() && save_videos)
	{
		ss_vidname << dynamic_path << "/" << input_file << "_dyn" << "." << output_ext;

		if (askOutputType)
			outputVideo.open(ss_vidname.str(), ex=-1, capture.get(CV_CAP_PROP_FPS), S, true);
		else   
			outputVideo.open(ss_vidname.str(), ex,    capture.get(CV_CAP_PROP_FPS), S, true);

		if (!outputVideo.isOpened())
		{
			cout  << "Could not open the dynamic output video for write: " << endl;

			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_completed = true;
			// destructor for lock_it will release the mutex
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////

	bool is_first = true;

	while(1) {

		while( !is_input_ready && !is_completed);

		if( is_completed) break;

		wtime = omp_get_wtime ( );

		//oDynamicPathway.Apply( h_odata_d, h_idata, out_size_cropped, in_size_cropped, ss_comname.str(), is_first, frame_idx);
		oDynamicPass.calculate_saliency_map(h_odata_d, h_idata, in_size_cropped, ss_comname.str(), is_first, frame_idx);

		wtime = omp_get_wtime ( ) - wtime;
		cout << "\n";
		cout << "  Elapsed cpu time for main computation:\n";
		cout << "  " << wtime << " seconds.\n";		

		if( !is_first) 
		{
			GetImageMat( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));

			if( !dynamic_path.empty())
			{
				stringstream ss_path;
				ss_path << dynamic_path << "/" << input_file << "_"<< frame_idx << "_dyn" << "." << output_ext;

				if( save_videos) SaveVideo( im_d, outputVideo);
				if( save_images) SaveImage( im_d, ss_path);
				if( save_text) SaveText( h_odata_d, ss_path, out_size_cropped);
			}

			if( display_dynamic)
			{
				imshow("Dynamic", im_d);
				char c = waitKey(30);

				if( c == 27 ) 
				{
					// get exclusive ownership of mutex (wait for light to tuen green)
					boost::mutex::scoped_lock lock_it( g_mutex ) ;
					// ok, now we have exclusive access to the light

					is_completed = true;
					// destructor for lock_it will release the mutex
				}
			}
		}

		is_first = false;

		{
			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_dynamic_ready = true;

			// destructor for lock_it will release the mutex
		}

		while( is_dynamic_ready && !is_completed);
	}
	/*
	#if defined(DEBUG)

	for( unsigned int i=0;i<K;i++)
	{
	bool type = 1;

	stringstream path;		
	path << "data/images/" << ((type) ? "curr" : "prev") << i << ".png";

	oDynamicPathway.GetPyramid( h_odata_d, i, type);

	GetImageMatPrime( im_d, h_odata_d, siz_t(out_size.w,out_size.h), siz_t(out_size_cropped.w/powf(2,i),out_size_cropped.h/powf(2,i)), siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<K;i++)
	{
	bool type = 0;

	stringstream path;		
	path << "data/images/" << ((type) ? "curr" : "prev") << i << ".png";

	oDynamicPathway.GetPyramid( h_odata_d, i, type);

	GetImageMatPrime( im_d, h_odata_d, siz_t(out_size.w,out_size.h), siz_t(out_size_cropped.w/powf(2,i),out_size_cropped.h/powf(2,i)), siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	{
	stringstream path;
	path << "data/images/shift.png";

	oDynamicPathway.GetShift(h_odata_d);		

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<2*N;i++)
	{
	stringstream path;
	path << "data/images/gx" << i << ".png";

	oDynamicPathway.GetGradientX(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<2*N;i++)
	{
	stringstream path;
	path << "data/images/gy" << i << ".png";

	oDynamicPathway.GetGradientY(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<2*N;i++)
	{
	stringstream path;
	path << "data/images/gt" << i << ".png";

	oDynamicPathway.GetGradientT(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<2*N;i++)
	{
	stringstream path;
	path << "data/images/mod" << i << ".png";

	oDynamicPathway.GetModulationMasks(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<K;i++)
	{
	stringstream path;
	path << "data/images/vx" << i << ".png";

	oDynamicPathway.GetVelocityX(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, siz_t(out_size.w,out_size.h), siz_t(out_size_cropped.w/powf(2,i),out_size_cropped.h/powf(2,i)), siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<K;i++)
	{
	stringstream path;
	path << "data/images/vy" << i << ".png";

	oDynamicPathway.GetVelocityY(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, siz_t(out_size.w,out_size.h), siz_t(out_size_cropped.w/powf(2,i),out_size_cropped.h/powf(2,i)), siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	{
	stringstream path;
	path << "data/images/v.png";

	oDynamicPathway.GetVelocity(h_odata_d);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	{
	stringstream path;
	path << "data/images/t.png";

	oDynamicPathway.GetThreshold(h_odata_d);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<2*N;i++)
	{
	stringstream path;
	path << "data/images/filt" << i << ".png";

	oDynamicPathway.GetCurrentGaborMasks(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	for( unsigned int i=0;i<2*N;i++)
	{
	stringstream path;
	path << "data/images/prefilt" << i << ".png";

	oDynamicPathway.GetPreviousGaborMasks(h_odata_d, i);

	GetImageMatPrime( im_d, h_odata_d, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));
	SaveImage( im_d, path);
	}

	#endif
	*/
	cout << "Thread dynamic : Ended" << endl;
}

void ComputeFaceThread() // get input frame
{
	//cudaSetDevice(0);

	double wtime;

	float mx;

	cv::Mat im_f = cv::Mat::zeros(in_size.h, in_size.w, CV_8UC1 );

	std::vector<int> params;

	params.push_back( CV_IMWRITE_PXM_BINARY);
	params.push_back( 100);

	for( int i=0;i<out_nbpixels;++i) {		
		im_f.data[i] = ( char)( unsigned char)( 0);
	}

	float *temp = (float*) calloc(sizeof(float), out_nbpixels);

	////////////////////////////////////////////////////////////////////////////////////////////////////

	const bool askOutputType = false;

	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

	Size S = Size((int) out_size.w, (int)out_size.h);    //Acquire input size

	VideoWriter outputVideo;	// Open the output

	stringstream ss_vidname;

	if( !face_path.empty() && save_videos)
	{
		ss_vidname << face_path << "/" << input_file << "_fac" << "." << output_ext;

		if (askOutputType)
			outputVideo.open(ss_vidname.str(), ex=-1, capture.get(CV_CAP_PROP_FPS), S, true);
		else   
			outputVideo.open(ss_vidname.str(), ex,    capture.get(CV_CAP_PROP_FPS), S, true);

		if (!outputVideo.isOpened())
		{
			cout  << "Could not open the face output video for write: " << endl;

			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_completed = true;
			// destructor for lock_it will release the mutex
		}
	}

	Face::Pathway oFacePathway( cascadename, in_size, 1.0f);	
	//Face::Pathway oFacePathway( cascadeNameFrontal, cascadeNameProfile, in_size, 1.0f);	

	////////////////////////////////////////////////////////////////////////////////////////////////////

	while(1){

		while( !is_input_ready && !is_completed);

		if( is_completed) break;

		faces.clear();
		weights.clear();
		types.clear();

		wtime = omp_get_wtime ( );
		
		oFacePathway.Apply( faces, weights, types, gray_curr);

		wtime = omp_get_wtime ( ) - wtime;
		cout << "\n";
		cout << "  Elapsed cpu time for main computation:\n";
		cout << "  " << wtime << " seconds.\n";

		oFacePathway.GetFaceMap( h_odata_f, faces, weights);

		for( int j=0;j<out_size_cropped.h;++j) {
			for( int i=0;i<out_size_cropped.w;++i) {

				h_odata_f_cropped[i + j*out_size_cropped.w] = h_odata_f[(i+out_offset_w) + (j+out_offset_h)*out_size.w];
			}
		}

		GetImageMat( im_f, h_odata_f, out_size, out_size, siz_t(0,0)); // not cropped

		if( !face_path.empty())
		{
			stringstream ss_path;
			ss_path << face_path << "/" << input_file << "_"<< frame_idx << "_fac" << "." << output_ext;

			if( save_videos) SaveVideo( im_f, outputVideo);
			if( save_images) SaveImage( im_f, ss_path);
			if( save_text) SaveText( h_odata_f, ss_path, out_size_cropped);
		}

		if( display_face)
		{
			imshow("Face", im_f);
			char c = waitKey(30);

			if( c == 27 ) 
			{
				// get exclusive ownership of mutex (wait for light to tuen green)
				boost::mutex::scoped_lock lock_it( g_mutex ) ;
				// ok, now we have exclusive access to the light

				is_completed = true;
				// destructor for lock_it will release the mutex
			}
		}

		{
			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_face_ready = true;	

			// destructor for lock_it will release the mutex	
		}

		while( is_face_ready && !is_completed);
	}

	im_f.release();
	free(temp);

	faces.clear();
	weights.clear();
	types.clear();

	cout << "Thread face : Ended" << endl;
}

void ComputeFusionTwoPathThread() // get input frame
{
	float mx;

	float *h_odata_sd = (float*) malloc( out_size_cropped.w*out_size_cropped.h * sizeof(float));

	Mat im_sd = Mat( out_size.h, out_size.w, CV_8UC1 );
	vector<int> params;

	params.push_back( CV_IMWRITE_PXM_BINARY);
	params.push_back( 100);

	for( int i=0;i<out_nbpixels;++i) {

		im_sd.data[i] = ( char)( unsigned char)( 0);		
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////

	const bool askOutputType = false;

	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

	Size S = Size((int) out_size.w, (int)out_size.h);    //Acquire input size

	VideoWriter outputVideo;	// Open the output

	stringstream ss_vidname;

	if( !master_path.empty() && save_videos)
	{
		ss_vidname << master_path << "/" << input_file << "_fus" << "." << output_ext;

		if (askOutputType)
			outputVideo.open(ss_vidname.str(), ex=-1, capture.get(CV_CAP_PROP_FPS), S, true);
		else   
			outputVideo.open(ss_vidname.str(), ex,    capture.get(CV_CAP_PROP_FPS), S, true);

		if (!outputVideo.isOpened())
		{
			cout  << "Could not open the master output video for write: " << endl;

			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_completed = true;
			// destructor for lock_it will release the mutex
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////

	while(1) {

		while( ( !is_static_ready || !is_dynamic_ready) && !is_completed);

		if( is_completed) break;

		if( frame_idx>iter && two_path)
		{
			UTY_HelperProcs::Fusion( h_odata_sd, h_odata_s, h_odata_d, out_size_cropped);			

			GetImageMat( im_sd, h_odata_sd, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));

			if( !master_path.empty())
			{
				stringstream ss_path;
				ss_path << master_path << "/" << input_file << "_"<< frame_idx << "_fus" << "." << output_ext;

				if( save_videos) SaveVideo( im_sd, outputVideo);
				if( save_images) SaveImage( im_sd, ss_path);
				if( save_text) SaveText( h_odata_sd, ss_path, out_size_cropped);
			}

			if( display_master)
			{
				imshow("Master", im_sd);
				char c = waitKey(30);

				if( c == 27 ) 
				{
					// get exclusive ownership of mutex (wait for light to tuen green)
					boost::mutex::scoped_lock lock_it( g_mutex ) ;
					// ok, now we have exclusive access to the light

					is_completed = true;
					// destructor for lock_it will release the mutex
				}
			}
		}

		{
			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			++frame_idx;

			is_input_ready   = false;

			is_static_ready  = ( two_path) ? false : ( one_path_static ) ? false : true;
			is_dynamic_ready = ( two_path) ? false : ( one_path_dynamic) ? false : true;

			// destructor for lock_it will release the mutex	
		}
	}

	free( h_odata_sd);

	cout << "Thread fusion : Ended" << endl;
}

void ComputeFusionThreePathThread() // get input frame
{
	float mx;

	float *h_odata_sdf = (float*) malloc( out_size_cropped.w*out_size_cropped.h * sizeof(float));	

	Mat im_sdf = Mat( out_size.h, out_size.w, CV_8UC1 );
	vector<int> params;

	params.push_back( CV_IMWRITE_PXM_BINARY);
	params.push_back( 100);

	for( int i=0;i<out_nbpixels;++i) {

		im_sdf.data[i] = ( char)( unsigned char)( 0);		
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////

	const bool askOutputType = false;

	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

	Size S = Size((int) out_size.w, (int)out_size.h);    //Acquire input size

	VideoWriter outputVideo;	// Open the output

	stringstream ss_vidname;

	if( !master_path.empty() && save_videos)
	{
		ss_vidname << master_path << "/" << input_file << "_fus" << "." << output_ext;

		if (askOutputType)
			outputVideo.open(ss_vidname.str(), ex=-1, capture.get(CV_CAP_PROP_FPS), S, true);
		else   
			outputVideo.open(ss_vidname.str(), ex,    capture.get(CV_CAP_PROP_FPS), S, true);

		if (!outputVideo.isOpened())
		{
			cout  << "Could not open the master output video for write: " << endl;

			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			is_completed = true;
			// destructor for lock_it will release the mutex
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////

	while(1) {

		while( (!is_static_ready || !is_dynamic_ready || !is_face_ready) && !is_completed);

		if( is_completed) break;

		while( !is_static_ready || !is_dynamic_ready || !is_face_ready);

		if( frame_idx>iter && three_path) // frame counter already updated to next frame, so -1
		{
			UTY_HelperProcs::Fusion( h_odata_sdf, h_odata_s, h_odata_d, h_odata_f_cropped, weights, out_size_cropped);

			GetImageMat( im_sdf, h_odata_sdf, out_size, out_size_cropped, siz_t(out_offset_w,out_offset_h));

			if( !master_path.empty())
			{
				stringstream ss_path;
				ss_path << master_path << "/" << input_file << "_"<< frame_idx << "_fus" << "." << output_ext;

				if( save_videos) SaveVideo( im_sdf, outputVideo);
				if( save_images) SaveImage( im_sdf, ss_path);
				if( save_text) SaveText( h_odata_sdf, ss_path, out_size_cropped);
			}

			if( display_master)
			{
				imshow("Master", im_sdf);
				char c = waitKey(30);

				if( c == 27 ) 
				{
					// get exclusive ownership of mutex (wait for light to tuen green)
					boost::mutex::scoped_lock lock_it( g_mutex ) ;
					// ok, now we have exclusive access to the light

					is_completed = true;
					// destructor for lock_it will release the mutex
				}
			}
		}

		{
			// get exclusive ownership of mutex (wait for light to tuen green)
			boost::mutex::scoped_lock lock_it( g_mutex ) ;
			// ok, now we have exclusive access to the light

			++frame_idx;

			is_input_ready   = false;

			is_static_ready  = ( three_path) ? false : ( one_path_static ) ? false : true;
			is_dynamic_ready = ( three_path) ? false : ( one_path_dynamic) ? false : true;
			is_face_ready  	 = ( three_path) ? false : ( one_path_face   ) ? false : true;

			// destructor for lock_it will release the mutex	
		}
	}

	free( h_odata_sdf);

	cout << "Thread fusion : Ended" << endl;	
}