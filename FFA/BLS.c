#include "BLS.h"
#include <math.h>
void BLS(double* t, double* f, long N,
	 double* Parr, long nP,
	 int nb, double qmi, double qma,
	 double* SNR)
{
  int delT,delT1,delT2,j;
  double s,ss,s2,snr,snrma,Nin,ph,freq,P,rN;

  long iP, i; // counter for period and flux values
  int ib,ib1,ib2; // counters for the bins

  rN    = (double)N;
  delT1 = (int)((float)nb * qmi);
  delT2 = (int)((float)nb * qma);
  
  double ibi[nb],y[nb],yy[nb]; // temp arrays
  double u[N];
  int min_pts_trans = 40; // minimum number of (unbinned points per transit, this prevents the SNR from blowing up when there are just a few points);

  for (i=0;i<N;i++)
    {
      u[i] = t[i] - t[0];
    }
  
  for (iP=0; iP<nP; iP++)
    {
      P    = Parr[iP];
      freq = 1 / P;
      
      // Zeroing out the binned arrays
      for (ib=0; ib<nb; ib++)
	{
	  y[ib]   = 0;
	  yy[ib]  = 0;
	  ibi[ib] = 0;
	}

      //  Filling in the binned arrays
      for (i=0; i<N; i++)
	{
	  ph = u[i] * freq;  // phase of ith observation
	  ph = ph - (int)ph; // save only the remainder
	  j  = (int)(nb*ph); // the bin that the ith measurement is in

	  // Increment up each of the binned values
	  ibi[j] += 1  ;
	  y[j]   += f[i];
	  yy[j]  += f[i]*f[i];
        
	}
      snr   = 0.; 
      snrma = 0.; 

      // Loop over epochs
      for (ib1=0; ib1<nb; ib1++)
	{
	  delT  = 1 ; // counter that keeps track of the width of the boxes 
	  ib2   = ib1 ; 
	  s     = 0 ;  // sum of all the measurements in a box
	  ss    = 0 ;  // sum of squares of all measurements in a box
	  Nin   = 0 ;  // sum of the number of measurements in a box  

	  // Loop over durations
	  while (delT <= delT2)
	    {
	      // If index runs past the end of the array, wrap arounda
	      if (ib2 == nb) ib2 = 0;
	      
	      s   += y[ib2];
	      ss  += yy[ib2];
	      Nin += ibi[ib2];

	      s2  = s*s; // squart of the sum of points
	      if (delT >= delT1) //  Start computing SR
		{
		  snr = s2 / (ss - s2/Nin);
		  if ( (snr > snrma) & (s < 0) & (Nin >= min_pts_trans)   )
		    {
		      snrma  = snr ; 
		    }
		}
	      delT += 1;
	      ib2  += 1;
	    }
	}
      SNR[iP] = sqrt(snrma);
    }
}
