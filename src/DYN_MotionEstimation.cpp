
#include "DYN_MotionEstimation.h"

DYN_MotionEstimation::DYN_MotionEstimation(){}
DYN_MotionEstimation::~DYN_MotionEstimation(){}

int DYN_MotionEstimation::estimate_motion( motion_vect_t v, gradient_t grad, siz_t level_size, bool init)
{	
	int o, n, m, l, pas
		, size = level_size.w*level_size.h
		, neq1[L], neq2[L];

	float 
		*eqx[2*N], *eqy[2*N], *eqt[2*N]
	, dx[L], dy[L], r[L], w[L]
	, Mx, My, W, dif, mxp, myp, dist
	, C2=C*C;

	#pragma omp parallel for schedule(dynamic,1)
	for( n=0 ; n<N ; n++) {
		eqx[2*n]   = grad.gx[n].re;
		eqy[2*n]   = grad.gy[n].re;
		eqt[2*n]   = grad.gt[n].re;
		eqx[2*n+1] = grad.gx[n].im;
		eqy[2*n+1] = grad.gy[n].im;
		eqt[2*n+1] = grad.gt[n].im;
	}

	for( m=0 ; m<size ; m++){
		for( l=0, n=0 ; n<2*N ; n++)
			for( o=0 ; o<n ; o++, l++){
				dif = eqx[o][m]*eqy[n][m] - eqy[o][m]*eqx[n][m];

				if (dif==0)
					dx[l] = dy[l] = 0.0f;
				else{
					w[l]  = 1.0f;
					dx[l] = ( eqy[o][m]*eqt[n][m] - eqt[o][m]*eqy[n][m]) / dif;
					dy[l] = ( eqt[o][m]*eqx[n][m] - eqx[o][m]*eqt[n][m]) / dif;
				}
			}

		Mx = My = W = 0.0f;

		for( l=0 ; l<L ; l++){
			if( est_max>0 && ( fabs(dx[l])>est_max || fabs(dy[l])>est_max)) 
				w[l] = 0.0f;
			else 
				w[l] = 1.0f;

			Mx += w[l]*dx[l];
			My += w[l]*dy[l];
			W  += w[l];
		}
		if (W!=0){ Mx/=W; My/=W;}

		for( pas=0 ; pas<pas_max ; pas++){
			for( l=0 ; l<L ; l++){
				r[l] = sqrt( pow(dx[l]-Mx,2) + pow(dy[l]-My,2));

				if( fabs(r[l])>C || w[l]==0) 
					w[l]=0; 
				else{
					w[l]  = (r[l]*r[l] - C2) / C2; 
					w[l] *= w[l];
				}
			}

			mxp = Mx; myp = My;
			Mx=0; My=0; W=0;
			for(l=0;l<L;l++) {
				Mx+=w[l]*dx[l];
				My+=w[l]*dy[l];
				W+=w[l];
			}
			if (W!=0) {Mx/=W; My/=W;}

			dif=sqrt(pow(Mx-mxp,2)+pow(My-myp,2))/sqrt(pow(mxp,2)+pow(myp,2));
			if (dif<=dif_min) break;
		}

		if (init) {v.vx[m]=Mx; v.vy[m]=My;} else {v.vx[m]=(v.vx[m])+Mx; v.vy[m]=(v.vy[m])+My;}
	}

	return 0;
}

int DYN_MotionEstimation::apply_projection( motion_vect_t v_out, motion_vect_t v, std::complex<float> mask, siz_t level_size)
{

	return 0;
}

int DYN_MotionEstimation::apply_gaussian_recursive( motion_vect_t v_out, motion_vect_t v_in, siz_t level_size, float sigma)
{

	return 0;
}