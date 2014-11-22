#cython: boundscheck=False, wraparound=False

"""
Cython functions for folding
"""
import cython
cimport numpy as np
from libcpp cimport bool

import numpy as np
from numpy import ma



cpdef np.ndarray[double] fold_sum(np.ndarray[np.int64_t, ndim=1] col,np.ndarray[np.float64_t, ndim=1] x):
    cdef float total
    cdef int icad, icol
    cdef int ncol = np.max(col)
    cdef int ncad = x.shape[0]
    cdef np.ndarray[double] totcol = np.empty(ncol)
    
    for icol in range(ncol):
        total = 0.
        for icad in range(ncad):
            if col[icad]==icol:
                total+=x[icad]
        totcol[icol] = total

    return totcol

cpdef np.ndarray[double] fold_sum2(np.ndarray[np.int64_t, ndim=1] col,np.ndarray[np.float64_t, ndim=1] x, double Pcad):
    cdef float total
    cdef int icad, icol
    cdef int ncol = np.max(col)
    cdef int ncad = x.shape[0]
    cdef np.ndarray[double] totcol = np.empty(ncol)
    
    for icol in range(ncol):
        total = 0.
        for icad in range(ncad):
            if col[icad]==wrap_col(icad, Pcad):
                total+=x[icad]

        totcol[icol] = total

    return totcol

cpdef fold_ma(np.ndarray[np.float64_t, ndim=1] data,
              np.ndarray[np.int64_t, ndim=1] mask, 
              np.ndarray[np.int64_t, ndim=1] col):
    """
    Fold the data along the columns
    """

    cdef int ncad = data.shape[0]
    cdef int ncol = np.max(col)
    cdef np.ndarray[np.int64_t] ccol = np.empty(ncol,dtype=int)
    cdef np.ndarray[np.float64_t] scol = np.empty(ncol)
    cdef np.ndarray[np.float64_t] sscol = np.empty(ncol)

    cdef float s
    cdef int icad, icol, c_sum
    
    for icol in range(ncol):
        ss_sum = 0.
        s_sum = 0.
        c_sum = 0
        for icad in range(ncad):
            if (col[icad]==icol) & (mask[icad]==0):
                s = data[icad]
                c_sum+=1
                s_sum+=s
                ss_sum+=s**2

        ccol[icol] = c_sum
        scol[icol] = s_sum
        sscol[icol] = ss_sum
    return ccol,scol,sscol

def wrap_icad(icad,Pcad):
    """
    rows and column identfication to each one of the
    measurements in df

    Parameters
    ----------
    icad : Measurement number starting with 0
    Pcad : Period to fold on
    """

    row = np.floor( icad / Pcad ).astype(int)
    col = np.floor(np.mod(icad,Pcad)).astype(int)
    return row,col

cdef int wrap_col(int icad, double Pcad):
    """
    Test equivalence between wrap_icad and wrap_col

    col1 = array(map(lambda x : fold.wrap_col(x,3.3),arange(10)))
    col2 = fold.wrap_icad(arange(10),3.3)
    """
    cdef int col
    col = int ( cython.cmod(icad , Pcad) )
    return col
    


