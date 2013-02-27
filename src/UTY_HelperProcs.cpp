
//#include <cmath>
#include <iostream>

#include "cutil_inline.h"

#include "UTY_HelperProcs.h"

using namespace std;

UTY_HelperProcs::UTY_HelperProcs(){};
UTY_HelperProcs::~UTY_HelperProcs(){};

siz_t UTY_HelperProcs::im_size;

float UTY_HelperProcs::UTY_rint( float x)
{
	//middle value point test
	if( ceil( x+0.5)== floor( x+0.5))
	{
		int a =( int)ceil( x);
		if( a%2 == 0)
		{return ceil( x);}
		else
		{return floor( x);}
	}

	else return floor( x+0.5f);
}

int UTY_HelperProcs::UTY_rotate_90( float *in, siz_t map_size, float *out)
{
	if( !in || !out) return 0;

#pragma omp parallel
	{
#pragma omp for
		for (int xy = 0; xy < map_size.w*map_size.h; ++xy) 
		{
			int x = xy / map_size.h;
			int y = xy % map_size.h;

			int m = y*map_size.w + x;
			int l = x*map_size.h + y;

			out[m] = in[l];
		}
	}

	return 1;
}

int UTY_HelperProcs::UTY_r2c( float *in, siz_t map_size, fftw_complex *out)
{
	int i, j;

	if( !in)return 0;

#pragma omp parallel
	{
#pragma omp for
		for( i = 0; i < map_size.h*map_size.w ; i++){
			out[i][0] = in[i];
			out[i][1] = 0.0f;
		}
	}

	return 1;
}

int UTY_HelperProcs::UTY_c2r( fftw_complex *in, siz_t map_size, float *out)
{
	int i, j;

	if( !in)return 0;

#pragma omp parallel
	{
#pragma omp for
		for( i = 0; i < map_size.h*map_size.w ; i++)
			out[i] =( float)in[i][0] /( float)( map_size.h * map_size.w);
	}

	return 1;
}

int UTY_HelperProcs::UTY_shift(fftw_complex *in ,siz_t map_size)
{		
	unsigned int centerW, centerH;

	centerW = map_size.w/2;
	centerH	= map_size.h/2;

#pragma omp parallel
	{
#pragma omp for
		for (int xy = 0; xy < map_size.w*centerH; ++xy) 
		{
			int x = xy / centerH;
			int y = xy % centerH;

			int m = x + y*map_size.w;

			fftw_complex tmp;

			if (x < centerW)
			{
				int l = ( y+centerH)*map_size.w + x + centerW;

				tmp[0] = in[m][0];
				tmp[1] = in[m][1];

				in[m][0] = in[l][0];
				in[m][1] = in[l][1];

				in[l][0] = tmp[0];
				in[l][1] = tmp[1];
			}
			else
			{
				int l = ( y+centerH)*map_size.w + x - centerW;

				tmp[0] = in[m][0];
				tmp[1] = in[m][1];

				in[m][0] = in[l][0];
				in[m][1] = in[l][1];

				in[l][0] = tmp[0];
				in[l][1] = tmp[1];
			}
		}
	}

	return 1;
}

int UTY_HelperProcs::UTY_ishift(fftw_complex *in ,siz_t map_size)
{
	int i, j;
	int widthEO, heightEO ,centerW ,centerH ,xx ,yy;

	fftw_complex *temp;

	if( !in)return 0;

	temp = ( fftw_complex *)fftw_malloc ( sizeof ( fftw_complex ) * map_size.h * map_size.w );

	centerW		=	map_size.w/2;
	centerH		=	map_size.h/2;
	widthEO		= ( map_size.w%2 == 0) ? 0 : 1;
	heightEO	= ( map_size.h%2 == 0) ? 0 : 1;

#pragma omp parallel
	{
#pragma omp for
		for (int xy = 0; xy < map_size.w*map_size.h; ++xy) 
		{
			int x = xy / map_size.h;
			int y = xy % map_size.h;

			int yy = (y < centerH) ? ( y+centerH + heightEO ) : ( y-centerH );
			int xx = (x < centerW) ? ( x+centerW + widthEO  ) : ( x-centerW );

			int m = x + y*map_size.w;
			int l = xx + yy*map_size.w;

			temp[m][0] = in[l][0];
			temp[m][1] = in[l][1];       
		}
	}

	memcpy(in,temp, sizeof ( fftw_complex ) * map_size.h * map_size.w);

	temp = NULL;
	free(temp);

	return 1;
}

int UTY_HelperProcs::UTY_fft_2d( fftw_complex *in, fftw_complex *out, int rows, int cols)
{     
	fftw_plan plan_forward;

	if( !in)return 0;

	plan_forward = fftw_plan_dft_2d( rows, cols, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute( plan_forward);

	fftw_destroy_plan( plan_forward);

	return 1;
}

int UTY_HelperProcs::UTY_ifft_2d( fftw_complex *in, fftw_complex *out, int rows, int cols)
{  	
	fftw_plan plan_backward;

	if( !in)return 0;

	plan_backward = fftw_plan_dft_2d( rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute( plan_backward);

	fftw_destroy_plan( plan_backward);

	fftw_cleanup_threads();

	return 1;
}

int UTY_HelperProcs::UTY_convolve_2d( float* in, float* out, int dataSizeX, int dataSizeY, 
									 float* kernel, int kernelSizeX, int kernelSizeY)
{
	int i, j, m, n;
	float *inPtr, *inPtr2, *outPtr, *kPtr;
	int kCenterX, kCenterY;
	int rowMin, rowMax;                             // to check boundary of input array
	int colMin, colMax;                             //

	// check validity of params
	if( !in || !out || !kernel)return 0;
	if( dataSizeX <= 0 || kernelSizeX <= 0)return 0;

	// find center position of kernel( half of kernel size)
	kCenterX = kernelSizeX >> 1;
	kCenterY = kernelSizeY >> 1;

	// init working  pointers
	inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted( kCenterX, kCenterY), 
	outPtr = out;
	kPtr = kernel;

	// start convolution
	for( i= 0; i < dataSizeY; ++i)                  // number of rows
	{
		// compute the range of convolution, the current row of kernel should be between these
		rowMax = i + kCenterY;
		rowMin = i - dataSizeY + kCenterY;

		for( j = 0; j < dataSizeX; ++j)             // number of columns
		{
			// compute the range of convolution, the current column of kernel should be between these
			colMax = j + kCenterX;
			colMin = j - dataSizeX + kCenterX;

			*outPtr = 0;                            // set to 0 before accumulate

			// flip the kernel and traverse all the kernel values
			// multiply each kernel value with underlying input data
			for( m = 0; m < kernelSizeY; ++m)       // kernel rows
			{
				// check if the index is out of bound of input array
				if( m <= rowMax && m > rowMin)
				{
					for( n = 0; n < kernelSizeX; ++n)
					{
						// check the boundary of array
						if( n <= colMax && n > colMin)
							*outPtr += *( inPtr - n)* *kPtr;
						++kPtr;                     // next kernel
					}
				}
				else
					kPtr += kernelSizeX;            // out of bound, move to next row of kernel

				inPtr -= dataSizeX;                 // move input data 1 raw up
			}

			kPtr = kernel;                          // reset kernel to( 0, 0)
			inPtr = ++inPtr2;                       // next input
			++outPtr;                               // next output
		}
	}

	return 1;
}

int UTY_HelperProcs::UTY_gaussian_kernel( float *h, siz_t kernel_size, float sig_p)
{
	//case 'gaussian' % Gaussian filter
	//  siz   =( p2-1)/2;
	int siz =( kernel_size.w - 1)/ 2;
	float std   = sig_p;

	//  [x, y] = meshgrid( -siz( 2):siz( 2), -siz( 1):siz( 1));
	float *X, *Y;

	float hmax;
	float temp;
	int i, j, index;	

	float sumh;

	if( !h)return 0;

	X =( float *)calloc( sizeof( float),( size_t)kernel_size.w*kernel_size.h);
	Y =( float *)calloc( sizeof( float),( size_t)kernel_size.w*kernel_size.h);

#pragma omp parallel
	{
#pragma omp for
		for (int xy = 0; xy < kernel_size.w*kernel_size.h; ++xy) 
		{
			int x = xy / kernel_size.h;
			int y = xy % kernel_size.h;

			int m = y*kernel_size.w + x;

			X[m] =( float)( x - siz);
			Y[m] =( float)( y - siz);					
		}
	}

	//  arg   = -( x.*x + y.*y)/( 2*std*std);
	//  h     = exp( arg);	
	hmax = exp( -( ( ( X[0] * X[0])+( Y[0] * Y[0]))/( 2 * std * std)));

#pragma omp parallel
	{
#pragma omp for
		for( int m = 0 ; m < kernel_size.w*kernel_size.h ; m++){			

			h[m] = exp( -( ( ( X[m] * X[m])+( Y[m] * Y[m]))/( 2 * std * std)));					

			if( hmax < h[m])hmax = h[m];
		}
	}

	//  h( h<eps*max( h( :)))= 0;
	temp =( float)EPS * hmax;

	for( i = 0 ; i < kernel_size.h*kernel_size.w ; i++)
		if( h[i] < temp) h[i] = 0;

	//  sumh = sum( h( :));
	sumh = 0.0;

#pragma omp parallel shared(h) private(i) 
	{
#pragma omp for reduction(+:sumh)
		for( i = 0 ; i < kernel_size.h*kernel_size.w ; i++)
			sumh += h[i];
	}

	//  if sumh ~= 0, 
	//    h  = h/sumh;
	//  end;
	if( sumh != 0){
#pragma omp parallel
		{
#pragma omp for
			for( i = 0 ; i < kernel_size.h*kernel_size.w ; i++)
				h[i] /= sumh;
		}
	}

	return 1;
}

float UTY_HelperProcs::Mean(
							float *im
							, siz_t im_size
							)
{
	int i;
	float sum = 0.0f;

#pragma omp parallel shared(im) private(i) 
	{
#pragma omp for reduction(+:sum)
		for( i = 0; i < im_size.w*im_size.h; i++)
			sum += im[i];
	}

	return( sum  / (float)( im_size.w*im_size.h));
}

float UTY_HelperProcs::Mean(
							std::vector<int> values
							)
{
	if( values.empty()) return 0;
	if( values.size()==1) return values[0];

	float sum = 0.0f;

	for( unsigned int i = 1; i < values.size(); i++)
		sum += values[i];

	return( sum / (float) values.size());
}

float UTY_HelperProcs::StandardDeviation(
	float *im
	, siz_t im_size
	)
{
	unsigned int i;
	float avg, sum01 = 0.0f, sum02 = 0.0f;

	for( i=1;i<im_size.w*im_size.h;i++)
	{
		sum01 += im[i];
		sum02 +=( im[i]*im[i]);
	}

	return( sqrt( sum02/( im_size.w*im_size.h) - powf( sum01/( im_size.w*im_size.h), 2)));
}

float UTY_HelperProcs::Skewness(
								float *im
								, siz_t im_size
								, float mu
								, float std
								)
{
	unsigned int i;
	float sum = 0.0f;

	for( i=1;i<im_size.w*im_size.h;i++)
		sum +=( im[i] - mu)*( im[i] - mu)*( im[i] - mu);

	return( ( float)( sum/( im_size.w*im_size.h)) / powf( std, 3));
}

float UTY_HelperProcs::Min(
						   float *im
						   , siz_t im_size
						   )
{
	unsigned int i;
	float min = im[0];

	for( i=1;i<im_size.w*im_size.h;i++)
		if( im[i]< min)
			min = im[i];

	return min;
}

float UTY_HelperProcs::Max(
						   float *im
						   , siz_t im_size
						   )
{
	unsigned int i;
	float max = im[0];

	for( i=1;i<im_size.w*im_size.h;i++)
		if( im[i]> max)
			max = im[i];

	return max;
}

void UTY_HelperProcs::Fusion(
							 float *out
							 , float *im_static
							 , float *im_dynamic
							 , siz_t im_size				
							 )
{	
	float max_sta, max_dyn, min_sta, min_dyn, skw_dyn;

	min_dyn = UTY_HelperProcs::Min( im_dynamic, im_size);
	max_dyn = UTY_HelperProcs::Max( im_dynamic, im_size);

	min_sta = UTY_HelperProcs::Min( im_static, im_size);
	max_sta = UTY_HelperProcs::Max( im_static, im_size);

	skw_dyn = UTY_HelperProcs::Skewness( im_dynamic, im_size, UTY_HelperProcs::Mean( im_dynamic, im_size), UTY_HelperProcs::StandardDeviation( im_dynamic, im_size));

	skw_dyn = ( skw_dyn < 0.0f) ? 0.0f : skw_dyn;

	for( unsigned int i = 0; i < im_size.w*im_size.h; i++)
		out[i] = max_sta*im_static[i] + skw_dyn*im_dynamic[i] + max_sta*skw_dyn*im_static[i]*im_dynamic[i];
}

void UTY_HelperProcs::Fusion(
							 float *out
							 , float *im_static
							 , float *im_dynamic
							 , float *im_face
							 , std::vector<int> weights
							 , siz_t im_size			
							 )
{	
	float max_sta, skw_dyn, mu_fac;

	max_sta = UTY_HelperProcs::Max( im_static, im_size);

	skw_dyn = UTY_HelperProcs::Skewness( im_dynamic, im_size, UTY_HelperProcs::Mean( im_dynamic, im_size), UTY_HelperProcs::StandardDeviation( im_dynamic, im_size));
	skw_dyn = ( skw_dyn < 0.0f) ? 0.0f : skw_dyn;

	mu_fac  = UTY_HelperProcs::Mean( weights);

	for( unsigned int i = 0; i < im_size.w*im_size.h; i++)
		out[i] = max_sta*im_static[i] 
	+ skw_dyn*im_dynamic[i]
	+  mu_fac*im_face[i]
	+ max_sta*skw_dyn * im_static[i] *im_dynamic[i]
	+ max_sta*mu_fac  * im_static[i] *im_face[i]
	+ skw_dyn*mu_fac  * im_dynamic[i]*im_face[i];
}

/**************************************************************************
Compensation
**************************************************************************/

#define DATA_LINE_LEN 16

std::vector<float> UTY_HelperProcs::iof_get_compensation(	
	const std::string filename
	, unsigned int frame_no
	)
{
	int n;
	float tmp;

	FILE *file = fopen ( (char*)filename.c_str(), "r");

	std::vector<float> compensation_vars;

	if ( file != NULL){
		char line[1000];

		while( fgets( line, sizeof line, file) != NULL)
		{
			n = -1;
			sscanf( &line[0], "%d", &n);

			if ( n==frame_no) 
			{
				char *p;

				p = strtok ( line, " ");

				if ( p != NULL) {

					sscanf( p, "%d", &n);

					for ( unsigned int i=0;i<DATA_LINE_LEN;i++) {

						p = strtok ( NULL, " ");						
						sscanf( p, "%f ", &tmp);

						compensation_vars.push_back( tmp);
					}

					break;
				}
			}
		}
		fclose( file);
	}
	else 
	{
		perror( (char*)filename.c_str());
	}

	return compensation_vars;
}
/*
int UTY_HelperProcs::iof_get_compensation (const char *filename, float list[], unsigned int frameNumber)
{
int n;

FILE *file = fopen ( filename, "r" );
if ( file != NULL ){
char line[LINE_MAX] ;

while( fgets( line, sizeof line, file ) != NULL )
{
if (atoi(&line[0]) == frameNumber)
{	
char *p;

p = strtok (line," ");

if (p != NULL)
{
sscanf(p, "%d", &n);
//printf("Frame No. = %d\n", n);

for (unsigned int i=0 ; i<DATA_LINE_LEN ; i++)
{
p = strtok (NULL, " ");
sscanf(p, "%f ", &list[i]);

//printf("list[%d] = %f\n", i, list[i]);
}
}
}
}

fclose ( file );
}
else {
perror ( filename );
}

return 0;
}
*/

int UTY_HelperProcs::BilinearInterpolation(float *in, float *vx, float *vy, int w, int h, float *out)
{	
#pragma omp parallel
	{
#pragma omp for
		for (int xy = 0; xy < w*h; ++xy) 
		{
			int x = xy / h;
			int y = xy % h;

			int m = y*w+x;

			if (vx[m]<0.0f || vx[m]>w-1 || vy[m]<0.0f || vy[m]>h-1) { out[m] = 0.0f; continue; }

			int floor_x = (int)floor(vx[m]);
			int floor_y = (int)floor(vy[m]);

			if (floor_x < 0) floor_x = 0;
			if (floor_y < 0) floor_y = 0;

			int ceil_x = floor_x + 1;
			if (ceil_x >= w) ceil_x = floor_x;

			int ceil_y = floor_y + 1;
			if (ceil_y >= h) ceil_y = floor_y;

			float fraction_x = vx[m] - (float)floor_x;
			float fraction_y = vy[m] - (float)floor_y;

			float one_minus_x = 1.0 - fraction_x;
			float one_minus_y = 1.0 - fraction_y;

			float pix[4];

			pix[0] = in[floor_y*w + floor_x];
			pix[1] = in[floor_y*w + ceil_x];
			pix[2] = in[ceil_y*w + floor_x];
			pix[3] = in[ceil_y*w + ceil_x];

			out[m] = one_minus_y * 
				(one_minus_x * pix[0] + fraction_x * pix[1]) + fraction_y * (one_minus_x * pix[2] + fraction_x * pix[3]);
		}
	}

	return 0;
}

int UTY_HelperProcs::WriteFile(
							   char* filename
							   , float *im
							   , siz_t im_size
							   )
{
	FILE *file;
	file = fopen(filename,"w+");
	for(unsigned int i=0;i<im_size.w*im_size.h;++i)
		fprintf(file,"%.2f ", im[i]);
	fclose(file); /*done!*/
	return 0;
}