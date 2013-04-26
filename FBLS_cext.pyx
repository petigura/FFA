"""
Wrap all the C-extensions individually
"""
import numpy as np
cimport cython
cimport numpy as cnp

cdef extern from "FBLS.h": 
    void maxDelTt0(double* XsumP, double* XXsumP, double* XcntP, int P, 
                   int* DelTarr, int nDelT,
                   double* s2nMa, int* iMa, int* kMa)
    void boxsum(double *X, int N, int DeltaT, double* Xout)
    void hat(double* X, int N, int DeltaT, double* Xout)

@cython.boundscheck(False)
@cython.wraparound(False)
def cmaxDelTt0( cnp.ndarray[double, ndim=2,mode='c'] XsumP,
                cnp.ndarray[double, ndim=2,mode='c'] XXsumP,
                cnp.ndarray[double, ndim=2,mode='c'] XcntP,
                int P,
                cnp.ndarray[int, ndim=1,mode='c'] DelTarr,
                int nDelT):
                

    cdef int M = XsumP.shape[0]
    cdef int j

    
    cdef cnp.ndarray[double, ndim=1,mode='c'] s2nMa = np.zeros(M) - 1
    cdef cnp.ndarray[int, ndim=1,mode='c']    iMa   = np.zeros(M,np.int32) -1 
    cdef cnp.ndarray[int, ndim=1,mode='c']    kMa   = np.zeros(M,np.int32) -1

    for j in range(M):
        if j==7:
            maxDelTt0( &XsumP[j,0], &XXsumP[j,0], &XcntP[j,0], P,
                        &DelTarr[0], nDelT,
                        &s2nMa[j], &iMa[j], &kMa[j])

    return s2nMa,iMa,kMa
