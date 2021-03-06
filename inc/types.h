
#ifndef TYPES_H
#define TYPES_H

#ifndef FALSE
#  define FALSE			0
#endif
#ifndef TRUE
#  define TRUE			1
#endif

#define  PI				3.1416f
#define  EPS			2.2204e-016

#define  sizeTAI		0   
#define  SIG			12

#define  NO_OF_BANDS			4
#define  NO_OF_ORIENTS			6

#define  STD_MAX		0.081617728f		// 0.25 / 0.125 * 1 /( 2 * 3.1416 * 3.9)
#define  FREQ_MAX		0.25f

#define  SCALE			2.0f

#define  PC_LEVEL		0.2f
#define  LOWER_BOUND	0.0f
#define  UPPER_BOUND	1.0f

#define  MU				8

#define  VP				15

#define  DEBUG				1

/// <summary>Image size.</summary>
typedef struct siz_t{	

	siz_t(){}
	siz_t(int w_, int h_){ w = w_; h = h_;}

	/// <summary>Width.</summary>
	int w;
	/// <summary>Height.</summary>
	int h;
} siz_t;

////////////////////////////////////////////

#include <complex>

#define N 6 /* N = nb de filtres du banc */
#define K 3 /* K = nb de niveaux de la pyramide */
#define L N*(2*N-1) /* L = nb de solutions du syst�me surdimensionn� */

#define pas_max 3 /* nombre maximum d'incr�mentation de la boucle mcpi */
#define dif_min 0.01 /* diff�rence en dessous de laquelle on stoppe la boucle mcpi*/
#define C 0.5 /* param�tre du M-estimateur biweight de Tukey */
#define est_max 2 /* valeur au dessus de laquelle une valeur estim�e est consid�r�e comme aberrante
		    (entrez un nombre n�gatif ou nul pour qu'aucune valeur ne soit exclue ainsi,
		     dans ce cas pensez � adapter la valeur du paramètre C) */

#define _PI  3.141592654
#define _2PI 6.283185307

#define F0 0.125f /* f0 = freq de modulation des filtres de Gabor */
#define S0 3.500f /* s0 = cart type des filtres de Gabor */
#define S1 2.200f /* s1 = cart type du filtre passe haut */
#define S2 3.500f /* s2 = cart type du filtre lisseur de l'estimation */
#define S3 1.000f /* s3 = cart type du filtre pour sous-échantillonner la pyramide */

typedef struct{
	float *re;
	float *im;
}cmplx_t;

typedef struct{
	cmplx_t mt_mod[N];
}mod_t;

typedef struct{
	cmplx_t masks[N];	
}gabor_mask_t;

typedef struct{
	float *vx;
	float *vy;
} motion_vect_t;

typedef struct{
	cmplx_t gx[N];
	cmplx_t gy[N];
	cmplx_t gt[N];
}gradient_t;

typedef struct{
	float	*im_data;
	siz_t	level_size;
	gabor_mask_t gabor_mask;
} pre_level_info_t;

typedef struct{	
	float	*im_data;
	siz_t	level_size;
	gabor_mask_t	gabor_mask;
	float	*dec;
	motion_vect_t motion_vect;
} level_info_t;

typedef struct{	
	level_info_t		levels[K];
	pre_level_info_t	pre_levels[K];
	gradient_t			grad;
	mod_t				mod;
	siz_t				im_size;
	float				*im_data01;
	float				*im_data02;
	float				*temp_01;
	float				*temp_02;
	float				*seuil_tmp;
} pyramid_t;

//std::complex<float> operator +( std::complex<float> &op1, std::complex<float> &op2){
//	return std::complex<float>( std::real(op1)+std::real(op2), std::imag(op1)+std::imag(op2));
//}
//
//std::complex<float> operator -( std::complex<float> &op1, std::complex<float> &op2){
//	return std::complex<float>( std::real(op1)-std::real(op2), std::imag(op1)-std::imag(op2));
//}
//
//std::complex<float> operator *( std::complex<float> &op1, std::complex<float> &op2){
//	return std::complex<float>( std::real(op1)*std::real(op2), std::imag(op1)*std::imag(op2));
//}
//
//std::complex<float> operator *( float &op1, std::complex<float> &op2){
//	return std::complex<float>( op1*std::real(op2), op1*std::imag(op2));
//}
//
//std::complex<float> operator /( std::complex<float> &op1, float &op2){
//	return std::complex<float>( std::real(op1)/op2, std::imag(op1)/op2);
//}

static void iof_save_txt_simple (const char *name, float *mat, int w, int h) {

    int x,y,n;
    FILE* fp;
    
	fp=fopen(name,"wt");

	if (fp==NULL) {
		fprintf(stderr,"Error: '%s' can't be edited\n",name);
		exit(EXIT_FAILURE);
	}

	for(y=0,n=0;y<h;y++) {
		for(x=0;x<w;x++,n++) {			
			fprintf(fp,"%lf ",mat[n]);
			
		}
		
		}

	fclose(fp);
}

static float tmp_01[640*480];
static float tmp_02[640*480];

#endif /* TYPES_H */
