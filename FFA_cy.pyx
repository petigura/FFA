import numpy as np
from numpy import ma
cimport cython
cimport numpy as cnp
# cython: cdivision   = True

@cython.profile(True)
def FFA(XW):
    """
    Fast Folding Algorithm

    Consider an evenly-spaced timeseries of length N.  We can fold it
    on P0, by creating a new array XW, shape = (P0,M), M = N/P0.
    There are M ways to fold XW, yielding the following periods

    P = P0 + i / M - 1

    Where i ranges from 0 to M-1.  Summing all these values requires P
    * M**2 = N**2 / P0 sums.  If M is a power of 2, the FFA eliminates
    some of the redundant summing and requires only N log2 (N/P0)
    sums.

    Algorithm
    ---------
    The columns of XW are shifted and summed in a pairwise fashion.

    - `FFAButterfly` : for a group of size nGroup, `FFAButterfly`
      computes the amount pairwise combinations of rows and the amount
      the second row is shifted.
    - `FFAShiftAdd` : Adds the rows in the manner specified by
      `FFAButterfly`
        
    Parameters
    ----------
    XW : Wrapped array folded on P0.  shape(XW) = (P0,M) and M must be
         a power of 2
    
    Returns
    -------
    XWFS : XW Folded and Summed. 

    References
    ----------
    [1] Staelin (1969)
    [2] Kondratiev (2009)
    """

    # Make sure array is the right shape.
    nRow,P0  = XW.shape
    nStage   = np.log2(nRow)
    assert np.allclose(nStage,np.round(nStage)),"nRow must be power of 2"    
    nStage = int(nStage)

    XWFS = XW.copy()
    for stage in range(1,nStage+1):
        XWFS = FFAShiftAdd(XWFS.astype(np.float32),stage) 
    return XWFS

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(True)
@cython.cdivision(True)
def FFAGroupShiftAdd(cnp.ndarray[cnp.float32_t, ndim=2,mode='c'] group0,
                     cnp.ndarray[cnp.float32_t, ndim=2,mode='c'] group,
                     int nRowGroup,
                     int nColGroup):

    """
    FFA Shift and Add

    Add the rows of `group` to each other.
    
    Parameters
    ----------

    group0 : Initial group before shuffling and adding. 
             shape(group0) = (M,P0) where M is a power of 2.

    """
    cdef int iRow,iCol,iA,iB,Bs,i,j,jB
    cdef int nRowGroupOn2 = nRowGroup / 2 # Half the group size

    # Grow group by the maximum shift value
    # Loop over rows in group
    for i in range(nRowGroup):
        iA = i/2                 # Row in the group that A is draw from
        iB = iA + nRowGroupOn2   # Row in group that B is drawn from
        Bs = (i + 1) / 2
        # Loop over the columns in the group
        for j in range(nColGroup):
            jB = (j + Bs + nColGroup) % nColGroup
            group[i,j] = group0[iA,j] + group0[iB,jB]
    

@cython.profile(True)
def FFAShiftAdd(cnp.ndarray[cnp.float32_t, ndim=2] XW0,
                int stage):
    """
    FFA Shift and Add

    Shuffle pairwise add the rows of the FFA data array corresponding
    to stage
    
    Parameters
    ----------
    XW0   : array
    stage : The stage in the FFA.  An integer ranging from 1 to K
            where 2**K = M
            
    Returns
    -------
    XW    : Shifted and added array


    Test Cases
    ----------

    >>> tfind.FFAShiftAdd(eye(4),1)
    >>> array([[ 1.,  1.,  0.,  0.],
               [ 2.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  1.],
               [ 0.,  0.,  2.,  0.]])
    """
    cdef int nRow,nCol,nGroup,nRowGroup,iGroup,start,stop
    cdef cnp.ndarray[cnp.float32_t, ndim=2] XW

    nRow = XW0.shape[0]
    nCol = XW0.shape[1]
    nRowGroup = 2**stage
    nGroup    = nRow/nRowGroup
    XW =  np.zeros((nRow,nCol),dtype=np.float32)
    for iGroup in range(nGroup):
        start = iGroup*nRowGroup
        stop  = (iGroup+1)*nRowGroup
        FFAGroupShiftAdd(XW0[start:stop],XW[start:stop],nRowGroup,nCol)

    return XW

def XWrap(x,Pcad,fill_value=0):
    """
    Extend and wrap array.
    
    Fold array every y indecies.  There will typically be a hanging
    part of the array.  This is padded out.

    Parameters
    ----------
    x     : input
    Pcad  : Period to fold on.  Can be non-integer, only accurate to 1./Pcad0

    Return
    ------

    xwrap : Wrapped array.
    """

    ncad = x.size # Number of cadences
    # for some reason np.ceil(ncad/Pcad0) doesn't work!
    Pcad0 = np.floor(Pcad)
    nrow = int( np.floor(ncad/Pcad0) +1 )
    rem  = int(np.round(Pcad0 * (Pcad-Pcad0)  ))
    nExtend = nrow * Pcad0 - ncad # Pad out remainder of array with 0s.

    if type(x) is np.ma.core.MaskedArray:
        pad = ma.empty(nExtend)
        pad.mask = True
        x = ma.hstack( (x ,pad) )
    else:    
        pad = np.empty(nExtend) 
        pad[:] = fill_value
        x = np.hstack( (x ,pad) )

    xwrap = x.reshape( nrow,-1 )
    idShf = remShuffle(xwrap.shape,rem)
    xwrap = xwrap[idShf]
    return xwrap

def XWrap2(x,P0,fill_value=0,pow2=False):
    """
    Extend and wrap array.
    
    Fold array every y indecies.  There will typically be a hanging
    part of the array.  This is padded out.

    Parameters
    ----------

    x     : input
    P0    : Base period, units of elements
    pow2  : If true, pad out nRows so that it's the next power of 2.

    Return
    ------

    xwrap : Wrapped array.

    """

    ncad = x.size # Number of cadences
    # for some reason np.ceil(ncad/P0) doesn't work!
    nrow = int( np.floor(ncad/P0) +1 )
    nExtend = nrow * P0 - ncad # Pad out remainder of array with 0s.

    if type(x) is np.ma.core.MaskedArray:
        pad = ma.empty(nExtend)
        pad.mask = True
        x = ma.hstack( (x ,pad) )
    else:    
        pad = np.empty(nExtend) 
        pad[:] = fill_value
        x = np.hstack( (x ,pad) )

    xwrap = x.reshape( nrow,-1 )

    if pow2:
        k = np.ceil(np.log2(nrow)).astype(int)
        nrow2 = 2**k
        fill    = ma.empty( (nrow2-nrow,P0) )
        fill[:] = fill_value
        fill.mask=True
        xwrap = ma.vstack([xwrap,fill])

    return xwrap

def remShuffle(shape,rem):
    """
    Remainder shuffle

    For a 2-D array with shape (Pcad0,nrow), this rearanges the
    indecies such that the last row is shifted by rem.  rem can be any
    integer between 0 and Pcad0-1
    
    Parameters
    ----------
    shape : Shape of array to be shuffled.
    rem   : Shift the last row by rem.

    Returns
    -------
    id    : Shuffled indecies.

    """
    nrow,ncol = shape

    assert (rem >= 0) & (rem<=ncol), 'rem must be >= 0 and <= ncol '

    irow,icol = np.mgrid[0:nrow,0:ncol]
    colshift  = np.linspace(0,rem,nrow)
    colshift  = np.round(colshift).astype(int)
    for i in range(nrow):
        icol[i] = np.roll(icol[i],-colshift[i])

    return irow,icol
