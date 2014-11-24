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

cpdef fold_col(np.ndarray[np.float64_t, ndim=1] data,
              np.ndarray[np.int64_t, ndim=1] mask, 
              np.ndarray[np.int64_t, ndim=1] col):
    """
    Fold Columns

    Assign each point of time series to a column (bin) compute the
    following aggregating statristics for each column.

    Parameters
    ----------
    data : (float) data array
    mask : (int) mask for data array. 1 = masked out.
    col : (int) column corresponding to phase bin of measurement.

    Return
    ------
    ccol : total number of non-masked elements
    scol : sum of elements
    sscol : sum of squares of elements
    """

    cdef int icad, icol,ncad, ncol
    ncad = data.shape[0]
    ncol = np.max(col)+1
    
    # Define column arrays
    cdef np.ndarray[np.int64_t] ccol = np.zeros(ncol,dtype=int)
    cdef np.ndarray[np.float64_t] scol = np.zeros(ncol)
    cdef np.ndarray[np.float64_t] sscol = np.zeros(ncol)

    # Loop over cadences
    for icad in range(ncad):
        if (mask[icad]==0):
            icol = col[icad]

            # Increment counters
            ccol[icol]+=1 
            scol[icol]+=data[icad] 
            sscol[icol]+=data[icad]**2

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

    


