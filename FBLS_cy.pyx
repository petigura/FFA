"""
Wrap all the C-extensions individually
"""
import numpy as np
from numpy import ma
import FFA_cy as FFA


cimport cython
cimport numpy as np

DTYPE  = np.float
ctypedef np.float_t DTYPE_t

I_DTYPE = np.int
ctypedef np.int_t I_DTYPE_t

#cython: cdivision=True
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def FBLS(np.ndarray[DTYPE_t] XsumP, np.ndarray[DTYPE_t] XcntP, 
         int delT1, int delT2, long nt0):
    """
    Fast BLS.

    Copy of BLS from Kovacs. Compute signal residue `SR` using the
    output of the FFA.

    Parameters 
    ----------

    XsumP - sums of data values (slice of FFA)
    XcntP - sums of weights (slice of FFA)
    delT1 - Shortest width to search over
    delT2 - Longest width to search over.
    nt0 - number of epochs searched over.

    Returns
    -------
    SRma   = Maximum Signal Residue
    delTma = Transit width corresponding to SRma
    t0ma   = Index of the begining of the transit corresponding to SRma
    """
    cdef int i1,i2,delT
    cdef float s,Nin,pow,powma
    cdef float N = np.sum(XcntP) # Sum the weights

    powma = 0
    for i1 in range(nt0): # i is index of starting transit
        delT  = 1  # counter that keeps track of the width of the boxes 
        i2    = i1
        s     = 0 # running total of data values 
        Nin   = 0 # running total of weights
        
        while delT <= delT2:
            if i2 == nt0: # If index runs past the end of the array, wrap around
                i2 = 0

            s    += XsumP[i2]
            Nin  += XcntP[i2]

            if delT >= delT1: # Start computing SR
                pow = s*s/(Nin*(N - Nin))
                if pow > powma:
                    powma  = pow
                    t0ma   = i1
                    delTma = delT
            delT +=1
            i2   += 1 
    SRma = sqrt(powma)
    return SRma,delTma,t0ma

cdef extern from "math.h":
    double sqrt(double)

#cython: cdivision=True
@cython.wraparound(False)
@cython.boundscheck(True)
@cython.cdivision(True)
def FBLS_SNR(np.ndarray[DTYPE_t] XsumP, np.ndarray[DTYPE_t] XcntP,
             np.ndarray[DTYPE_t] XXsumP, 
             int delT1, int delT2, long nt0):
    """
    Fast BLS.

    Copy of BLS from Kovacs. Compute signal residue `SR` using the
    output of the FFA.

    Parameters 
    ----------

    XsumP - sums of data values (slice of FFA)
    XcntP - sums of weights (slice of FFA)
    delT1 - Shortest width to search over
    delT2 - Longest width to search over.
    nt0 - number of epochs searched over.

    Returns
    -------
    SRma   = Maximum Signal Residue
    delTma = Transit width corresponding to SRma
    t0ma   = Index of the begining of the transit corresponding to SRma
    """

    cdef int i1,i2,delT
    cdef float s,ss,Nin,SNR,SNRma
    cdef float N = np.sum(XcntP) # Sum the weights

    SNRma = 0
    for i1 in range(nt0): # i is index of starting transit
        delT  = 1  # counter that keeps track of the width of the boxes 
        i2    = i1
        s     = 0 # running total of data values 
        ss    = 0
        Nin   = 0 # running total of weights        
        while delT <= delT2:
            if i2 == nt0: # If index runs past the end of the array, wrap around
                i2 = 0
                
            s    += XsumP[i2]
            ss   += XXsumP[i2]
            Nin  += XcntP[i2]

            if delT >= delT1: # Start computing SR
                SNR = - s * N / (N - Nin) / sqrt(ss - s*s/Nin)
                if SNR > SNRma:
                    t0ma   = i1
                    delTma = delT
                    SNRma  = SNR
            
            delT +=1
            i2   += 1 
    return SNRma,delTma,t0ma

#cython: cdivision=True
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def FBLS_SRpos(np.ndarray[DTYPE_t] XsumP, np.ndarray[DTYPE_t] XcntP,
               int delT1, int delT2, long nt0):
    """
    Maximizes based on SR, only keeping case where there are dimmings
    """
    cdef int i1,i2,delT
    cdef float s,ss,Nin
    cdef float N = np.sum(XcntP) # Sum the weights
    cdef float pow,powma
    for i1 in range(nt0): # i is index of starting transit
        delT  = 1  # counter that keeps track of the width of the boxes 
        i2    = i1
        s     = 0 # running total of data values 
        ss    = 0
        Nin   = 0 # running total of weights
        
        while delT <= delT2:
            if i2 == nt0: # If index runs past the end of the array, wrap around
                i2 = 0
                
            s    += XsumP[i2]
            Nin  += XcntP[i2]

            if delT >= delT1: # Start computing SR
                pow = s*s/(Nin*(N - Nin))

                if (pow > powma) & (s < 0):
                    powma  = pow
                    t0ma   = i1
                    delTma = delT

            delT +=1
            i2   += 1 
    SRma = sqrt(powma)
    return SRma,delTma,t0ma


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def FBLS_SRCC(np.ndarray[DTYPE_t] XsumP, np.ndarray[DTYPE_t] XcntP,
              float qmi, float qma, long nt0):
    """
    Maximizes based on SR, but is insensitive to dimming / brightening
    """

    cdef int i1,i2,delT
    cdef float s,Nin
    cdef float N = np.sum(XcntP) # Sum the weights
    cdef float pow,powma

    cdef int delT1 = int(nt0 * qmi)
    cdef int delT2 = int(nt0 * qma)

    powma = 0.
    for i1 in range(nt0): # i is index of starting transit
        delT  = 1  # counter that keeps track of the width of the boxes 
        i2    = i1
        s     = 0 # running total of data values 
        Nin   = 0 # running total of weights        
        while delT <= delT2:
            if i2 == nt0: # If index runs past the end of the array, wrap around
                i2 = 0

            s    += XsumP[i2]
            Nin  += XcntP[i2]

            if delT >= delT1: # Start computing SR
                pow = s*s/(Nin*(N - Nin))
                if (pow > powma):
                    powma  = pow
                    t0ma   = i1
                    delTma = delT

            delT +=1
            i2   += 1 
    SRma = sqrt(powma)
    return SRma,delTma,t0ma

