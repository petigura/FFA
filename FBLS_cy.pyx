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
def FBLS(np.ndarray[DTYPE_t, ndim=2, mode='c'] XsumP, np.ndarray[DTYPE_t, ndim=2, mode='c'] XcntP, np.ndarray[DTYPE_t, ndim=1, mode='c'] DeltaTarr, np.ndarray[DTYPE_t, ndim=1, mode='c'] noiseG, long nP, long nt0, long nDeltaT):
    """
    Fast BLS

    Given an evenly-spaced time series:

    P = P0 + i / M - 1

    Where i ranges from 0 to M-1.

    DeltaT : various widths of the transit

    Xdata elements are summed (set to 0 to ignore). Xmask keeps track
    of how many points went into the sum.
    
    Xmask : 1 if point is to be counted, 0 if not 


    Returns
    -------
    s2n : length M array with s2n (marginalized over deltaT and ep)
    kMa : length M array with k index of maximum s2n
    iMa : length M array with i index of maximum s2n

              i - specifies delta T
              j - specifies P
              k - specifies epoch 
    """
    cdef int i,j,iMa2,kMa2,kMa1
    cdef float s2nMa1, s2nMa2
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] Xsum, Xcnt

    cdef np.ndarray[I_DTYPE_t] iMa = np.zeros(nP,dtype=I_DTYPE)
    cdef np.ndarray[I_DTYPE_t] kMa = np.zeros(nP,dtype=I_DTYPE)
    cdef np.ndarray[DTYPE_t] s2nMa = np.zeros(nP,dtype=DTYPE)

    for j in range(nP):
        Xsum  = XsumP[j]
        Xcnt  = XcntP[j]

        iMa2   = 0
        kMa2   = 0 
        s2nMa2 = 0
        for i in range(nDeltaT):
            DeltaT = DeltaTarr[i]
            XsumPDeltaT  = boxsum(Xsum,nt0,DeltaT)
            XcntPDeltaT  = boxsum(Xcnt,nt0,DeltaT)
            
            # Average of in transit points
            meanVal = XsumPDeltaT / XcntPDeltaT 
            mDepth  = hat(meanVal,nt0,DeltaT)
            
            s2n     = mDepth / noiseG[i] * np.sqrt( XcntPDeltaT / DeltaT )
            kMa1    = np.argmax(s2n)
            s2nMa1  = s2n[kMa1]
            if s2nMa1 > s2nMa2:
                s2nMa2 = s2nMa1
                iMa2   = i
                kMa2   = kMa1

        iMa[j]   = iMa2
        kMa[j]   = kMa2
        s2nMa[j] = s2nMa2
    return s2nMa,iMa,kMa


@cython.profile(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def boxsum(np.ndarray[DTYPE_t, ndim=1] X, int N, int DeltaT):
    """
    Box Sum

    For element in `X` sum points between X[i] and X[i+DeltaT],
    wrapping as needed. We first sum up X[0:DeltaT]. The we move along
    the arary, adding one new point to the front and subtracting one
    old point from the back.
    """
    cdef int iFirst,iLast,ind
    cdef DTYPE_t total=0 # keeps a running total.
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(N) 

    iFirst = 0
    for iLast in range(DeltaT):
        total += X[iLast]
    
    while iFirst < N:
        res[iFirst] = total

        # increase the limits by 1
        iFirst += 1
        iLast  = (iFirst + DeltaT) % N
        
        total -= X[ iFirst-1 ] # Kick out the oldest point
        total += X[ iLast ]    # Drop in the first point

    return res

@cython.profile(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def hat(np.ndarray[DTYPE_t, ndim=1,mode='c'] X,
        int N, int DeltaT):
    """
    For every index i in X We calculate the difference between the
    X[i] and the average of X[i-DeltaT] and X[i+Delta] wrapping if
    necessary
    """
    cdef int i, jBefore, jAfter
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(N)
    for i in range(N):
        jBefore = (i+N - DeltaT) % N
        jAfter  = (i+N + DeltaT) % N        
        res[i] = 0.5* (X[jBefore] + X[jAfter]) - X[i]
        if i < 10:
            print jBefore,jAfter

    return res


#def FFAGroupShiftAdd(cnp.ndarray[cnp.float32_t, ndim=2,mode='c'] group0,
#                     cnp.ndarray[cnp.float32_t, ndim=2,mode='c'] group,
#                     int nRowGroup,
#                     int nColGroup):
#
#    """
#    FFA Shift and Add
#
#    Add the rows of `group` to each other.
#    
#    Parameters
#    ----------
#
#    group0 : Initial group before shuffling and adding. 
#             shape(group0) = (M,P0) where M is a power of 2.
#
#    """
#    cdef int iRow,iCol,iA,iB,Bs,i,j,jB
#    cdef int nRowGroupOn2 = nRowGroup / 2 # Half the group size
#
#    # Grow group by the maximum shift value
#    # Loop over rows in group
#    for i in range(nRowGroup):
#        iA = i/2                 # Row in the group that A is draw from
#        iB = iA + nRowGroupOn2   # Row in group that B is drawn from
#        Bs = (i + 1) / 2
#        # Loop over the columns in the group
#        for j in range(nColGroup):
#            jB = (j + Bs + nColGroup) % nColGroup
#            group[i,j] = group0[iA,j] + group0[iB,jB]
