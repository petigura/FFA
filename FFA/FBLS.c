#include "FBLS.h"
#include <math.h>
#include <stdio.h>
/*
Find the Maximize DeltaT and t0.

Inputs
------

XsumP : Length P array. Contains the sum of points at every epoch
        between 0 and P-1
XcntP : Length P array. Contains the number of non-zero values that
        were summed into XcntP
P : length of XsumP and Xcnt

DelTarr : Length nDelT array. Contains the lengths of the trial
          transit durations that we will evaluate and maximize.
noiseG : Length nDelT array. Per-transit noise corresponding to each
DelTarr value.
nDelT : length of DelTarra and noiseG

Outputs
s2nMa : Maximum s2n over all epochs and widths
iMa   : Index of the DelTarr corresponding to s2nMa. [0, nDelT-1]
kMa   : Epoch corresponding to s2nMa. [0, P-1]
*/


void maxDelTt0(double* XsumP, double* XXsumP, double* XcntP, int P, 
	       int* DelTarr, int nDelT,
	       double* s2nMa, int* iMa, int* kMa)
{
  int i,k,DelT;
  double noise, s2n, mBefore,mAfter,mDuring,mDepth,sigma;

  double XsumPDelT[P];
  double XXsumPDelT[P];
  double XcntPDelT[P];
  

  int kBefore, kAfter;
  for (i=0; i<nDelT; i++)
    {

      DelT = DelTarr[i];

      boxsum(XsumP,P,DelT,XsumPDelT);
      boxsum(XXsumP,P,DelT,XXsumPDelT);
      boxsum(XcntP,P,DelT,XcntPDelT);

      // now find the max
      k=0;
      kBefore = P - DelT;
      kAfter  = k + DelT;
      for ( ; k<P ; k++, kBefore++, kAfter++ )
	{
	  if (kBefore==P){kBefore=0;}
	  if (kAfter ==P){kAfter=0;}

	  mBefore  = XsumPDelT[kBefore] / XcntPDelT[kBefore];
	  mAfter   = XsumPDelT[kAfter]  / XcntPDelT[kAfter] ;
	  mDuring  = XsumPDelT[k]       / XcntPDelT[k]      ;

	  mDepth   = 0.5* (mBefore + mAfter) - mDuring ;
	  
	  sigma = sqrt( XXsumPDelT[k] / XcntPDelT[k] - mDuring*mDuring );
	  s2n   = mDepth / sigma * sqrt(XcntPDelT[k]) ;

	  if (s2n > *s2nMa)
	    {
	      *s2nMa = s2n;
	      *iMa   = i;
	      *kMa   = k;
	    }
	      
	}

      printf("%f %f %f %i %f %f\n",mDepth,XXsumPDelT[*kMa], XcntPDelT[*kMa], DelT,sqrt( XXsumPDelT[*kMa] / XcntPDelT[*kMa] - (XsumPDelT[*kMa] / XcntPDelT[*kMa])*(XsumPDelT[*kMa] / XcntPDelT[*kMa] ) )  ,*s2nMa);
		  


    }
}

//void calcs2n(double* XsumP, double* XcntPDelT, int DelT, double* s2n)
//{
//  int iLast,iFirst;
//  double total = 0;
//  iFirst = 0;
//  
//  for(iLast=0; iLast<DeltaT; iLast++)
//    {
//      totsum += XsumP[iLast];
//      totcnt += XcntP[iLast];
//    }
//    
//
//  while (iFirst < N)
//    {
//      XsumPbox[iFirst] = totcnt;
//
//
//
//      //  increase the limits by 1
//      iFirst += 1;
//      iLast  = (iFirst + DeltaT) % N ;
//        
//      total -= X[ iFirst-1 ] ; // Kick out the oldest point
//      total += X[ iLast ]  ;   // Drop in the first point
//    }
//
//
//
//
//}

void boxmean(double *XsumP, double* XcntP, int N, int DeltaT, double* meanVal);
void boxmean(double *XsumP, double* XcntP, int N, int DeltaT, double* meanVal)
/*
  Box Sum
  
  For element in `X` sum points between X[i] and X[i+DeltaT],
  wrapping as needed. We first sum up X[0:DeltaT]. The we move along
  the arary, adding one new point to the front and subtracting one
  old point from the back.
*/
{
  int iLast,iFirst;
  double totsum,totcnt;
  totsum=0;
  totcnt=0;
  iFirst = 0;

  // initalize running totals
  for(iLast=0; iLast<DeltaT; iLast++)
    {
      totsum += XsumP[iLast];
      totcnt += XcntP[iLast];
    }
    
  while (iFirst < N)
    {
      meanVal[iFirst] = totsum / totcnt;

      //  increase the limits by 1
      iFirst += 1;
      iLast  = (iFirst + DeltaT) % N ;
        
      totsum -= XsumP[ iFirst-1 ] ; // Kick out the oldest point
      totcnt -= XcntP[ iFirst-1 ] ; // Kick out the oldest point

      totsum += XsumP[ iLast ] ; // Kick out the oldest point
      totcnt += XcntP[ iLast ] ; // Kick out the oldest point
    }
}



void boxsum(double *X, int N, int DeltaT, double* Xout)
/*
  Box Sum
  
  For element in `X` sum points between X[i] and X[i+DeltaT],
  wrapping as needed. We first sum up X[0:DeltaT]. The we move along
  the arary, adding one new point to the front and subtracting one
  old point from the back.
*/
{
  int iLast,iFirst;
  double total = 0;
  iFirst = 0;
  
  for(iLast=0; iLast<DeltaT; iLast++)
    {
      total += X[iLast];
    }
    
  while (iFirst < N)
    {
      Xout[iFirst] = total;

      //  increase the limits by 1
      iFirst += 1;
      iLast  = (iFirst + DeltaT) % N ;
        
      total -= X[ iFirst-1 ] ; // Kick out the oldest point
      total += X[ iLast ]  ;   // Drop in the first point
    }
}


void hat(double* X,int N, int DeltaT, double* Xout)
{
/* 
   For every index i in X We calculate the difference between the
   X[i] and the average of X[i-DeltaT] and X[i+Delta] wrapping if
   necessary
*/
  int i, jBefore, jAfter;
  for (i=0; i<N; i++)
    {
      jBefore = (i+N - DeltaT) % N ; // The i+N correctly wraps negative 
      jAfter  = (i+N + DeltaT) % N ; // indecies
      Xout[i] = 0.5* (X[jBefore] + X[jAfter]) - X[i] ;
    }
}

int main(int argc, char *argv[])
{
  return 0;
}
