import numpy as np
cimport cython
cimport numpy as cnp

cdef extern from "BLS.h":
     void BLS(double* t, double* f, long N,
              double* Parr, long nP,
              int nb, double qmi, double qma,
              double* SNR)

def cBLS( cnp.ndarray[double, mode='c'] t,
         cnp.ndarray[cnp.float64_t, mode='c'] f,
         cnp.ndarray[cnp.float64_t, mode='c'] Parr, 
         int nb, double qmi, double qma ):

    cdef long N = t.size
    cdef long nP = Parr.size
    cdef cnp.ndarray[double, mode='c'] SNR = np.empty(nP)

    BLS( &t[0], &f[0], N, 
         &Parr[0], nP,nb, qmi, qma, &SNR[0] )
    return SNR
