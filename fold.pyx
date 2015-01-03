#cython: boundscheck=False, wraparound=False

"""
Cython functions for folding
"""
import cython
cimport numpy as np

import numpy as np
from numpy import ma

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
