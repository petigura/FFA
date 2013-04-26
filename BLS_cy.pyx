import numpy as np

cimport cython
cimport numpy as np
DTYPE  = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double sqrt(double)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def BLS_SNR(np.ndarray[DTYPE_t] t, np.ndarray[DTYPE_t] f, 
            np.ndarray[DTYPE_t] Parr, int nb, float qmi, float qma):
    """
    BLS based on SNR
    """

    cdef long nP  = Parr.size
    cdef long N   = t.size

    cdef int delT,delT1,delT2
    cdef float s,ss,snr,snrma,Nin,ph,freq,P,rN

    rN = float(N) # for computation of SNR I need a float 

    delT1 = int(nb * qmi)
    delT2 = int(nb * qma)
                                                # For every bin count the 
    cdef np.ndarray[DTYPE_t] ibi = np.empty(nb) # number of measurements
    cdef np.ndarray[DTYPE_t] y   = np.empty(nb) # sum of measurements
    cdef np.ndarray[DTYPE_t] yy  = np.empty(nb) # sum of measurements squared
    cdef np.ndarray[DTYPE_t] u = t - t[0]

    cdef int iP,ib,i,i1,i2,j # counters

    # Stores the maximum value of SNR at each trial period.
    cdef np.ndarray[DTYPE_t] SNR = np.empty(nP) 
    for iP in range(nP):
        P = Parr[iP]
        freq = 1./P

        # Zeroing out the binned arrays
        for ib in range(nb): 
            y[ib]   = 0.
            yy[ib]  = 0.
            ibi[ib] = 0.
            
        # Filling in the binned arrays
        for i in range(N):
            ph      = u[i] * freq  # phase of ith observation
            ph      = ph - int(ph) # save only the remainder
            j       = int(nb*ph)   # the bin that the ith measurement is in

            # Increment up each of the binned values
            ibi[j] += 1  
            y[j]   += f[i]
            yy[j]  += f[i]*f[i]
        
        snr   = 0.
        snrma = 0.
        
        # Loop over epochs
        for i1 in range(nb): # i is index of starting transit
            delT  = 1  # counter that keeps track of the width of the boxes 
            i2    = i1

            s     = 0. # sum of all the measurements in a box
            ss    = 0. # sum of squares of all measurements in a box
            Nin   = 0. # sum of the number of measurements in a box  

            # Loop over durations
            while delT <= delT2:
                # If index runs past the end of the array, wrap arounda
                if i2 == nb: 
                    i2 = 0

                s   += y[i2]
                ss  += yy[i2]
                Nin += ibi[i2]
    
                if delT >= delT1: # Start computing SR
                    snr = - s*s / (ss - s*s/Nin)
                    if (snr > snrma):
                        snrma  = snr

                delT +=1
                i2   += 1 

        SNR[iP] = sqrt(snrma)

    return SNR
