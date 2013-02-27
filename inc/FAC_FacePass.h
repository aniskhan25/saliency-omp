
#ifndef _FAC_PATHWAY_H
#define _FAC_PATHWAY_H

#include "types.h"

#include "STA_Pathway.h"

/**
* This namespace wraps the face pathway's functionality.
*/
namespace Face {

	/**
	* A class with functions to compute the face visual saliency map for the STVS model.
	*/
	class Pathway {

	private:
		/**
		* Source image size.
		*/
		siz_t im_size_;

		/**
		* Scaled image size.
		*/
		siz_t im_size_scaled_;

		/**
		* No. of pixels in source image.
		*/
		unsigned int size_;

		/**
		* No. of pixels in scaled image.
		*/
		unsigned int size_scaled_;

		/**
		* Haar cascade name used by the detector.
		*/
		std::string cascadeNameFrontal_;

		/**
		* Haar cascade name used by the detector.
		*/
		std::string cascadeNameProfile_;

		/**
		* Two pass detector.
		*/
		boolean is_two_pass;

		/**
		* Scale factor for detector's input image size.
		*/
		float scale_;

		/**
		* Host matrix for resized input image.
		*/
		cv::Mat resized_cpu;

		cv::Mat gray_cpu;

		/**
		* Haar cascade classifier object.
		*/
		cv::CascadeClassifier cascade_cpu_frontal;

		/**
		* Haar cascade classifier object.
		*/
		cv::CascadeClassifier cascade_cpu_profile;

		/**
		* 
		*/
		float *h_idata;

		/**
		* 
		*/
		float *h_odata;

		/**
		* Initializes the face pathway of STVS model.
		*/
		void Init();

	public:

		/**
		* Cleans up the face pathway of STVS model.
		*/
		void Clean();

		/**
		* Default contructor for Pathway class.		
		* @param cascadename a constant string for cascade file path		
		* @param im_size a source image size
		* @param scale a constant scale factor for image for as detector's input		
		*/
		inline Pathway( 
			std::string const & cascadeName
			, siz_t const & im_size
			, float const & scale = 1.0f
			)
			: oStaticPathway( im_size)
		{
			im_size_.w = im_size.w;
			im_size_.h = im_size.h;

			size_ = im_size_.w*im_size_.h;

			im_size_scaled_.w = (int)( im_size.w*scale);
			im_size_scaled_.h = (int)( im_size.h*scale);

			size_scaled_ = im_size_scaled_.w*im_size_scaled_.h;

			scale_ = scale;

			cascadeNameFrontal_ = cascadeName;

			is_two_pass = false;

			Init();
		}

		inline Pathway( 
			std::string const & cascadeNameFrontal
			, std::string const & cascadeNameProfile
			, siz_t im_size
			, float const & scale = 1.0f
			)
			: oStaticPathway( im_size)
		{
			im_size_.w = im_size.w;
			im_size_.h = im_size.h;

			size_ = im_size_.w*im_size_.h;

			im_size_scaled_.w = (int)( im_size.w*scale);
			im_size_scaled_.h = (int)( im_size.h*scale);

			size_scaled_ = im_size_scaled_.w*im_size_scaled_.h;

			scale_ = scale;

			cascadeNameFrontal_ = cascadeNameFrontal;
			cascadeNameProfile_ = cascadeNameProfile;

			is_two_pass = true;

			Init();
		}

		/**
		* Destructor for Pathway class.
		*/
		inline ~Pathway(){ Clean();}
		//inline ~Pathway(){}

		/**
		Object of Reduce class used for retinal filtering operations.
		*/
		//Dynamic::Retina oRetina;
		STA_Pathway oStaticPathway;

		/**
		* Detects the faces along with their confidence scores for an input image.
		* The function computes face saliency map for STVS model. 
		* It takes an input video frames or image, and computes the face salience map as:
		* Step 1: Resize the image to get robust detections.
		* Step 2: Runs the face detector.
		* Step 3: Returns the resulting face detections.
		* @param faces a detected faces pointer.
		* @param weights a face confidence scores pointer.
		* @param im a source image pointer.
		* @return detected faces with confidence scores.
		*/
		void Apply( 
			std::vector<cv::Rect> & faces
			, std::vector<int> & weights
			, std::vector<int> & types
			, const cv::Mat & im
			);

		/**
		* Computes face saliency map from detected faces and their confidence scores.
		* @param faces a detected faces pointer.
		* @param weights a face confidence scores pointer.
		* @param im a destination image pointer.
		* @param im_size a destination image size.
		* @return face saliency map.
		*/
		void GetFaceMap( 
			float *out
			, const std::vector<cv::Rect> faces
			, const std::vector<int> weights
			);

	}; // class Pathway

} // namespace Face

#endif // _FAC_PATHWAY_H
