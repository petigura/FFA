#include "FFAGroupShiftAdd.h"
void FFAGroupShiftAdd(float* group0, float* group, int nRowGroup,int nColGroup)
{
  int iA,iB,Bs,jB,i,j,step,stepA,stepB;
  int nRowGroupOn2 = nRowGroup / 2; // Half the group size

  // Grow group by the maximum shift value
  // Loop over rows in group
  for(i=0; i<nRowGroup; i++)
    {
      iA = i/2;                 //# Row in the group that A is draw from
      iB = iA + nRowGroupOn2; //  # Row in group that B is drawn from
      Bs = (i + 1) / 2;
      //# Loop over the columns in the group

      step = i*nColGroup;
      stepA = iA*nColGroup;
      stepB = iB*nColGroup;
      for (j=0; j<nColGroup; j++)
	  {
            jB = (j + Bs + nColGroup) % nColGroup;
	    group[step+j] = group0[stepA+j] + group0[stepB+jB];
	  }
    }
}

void FFAShiftAdd(float* XW0, float* XW, int stage, int nRow, int nCol)
{
  int nGroup,nRowGroup,iGroup,startRow,offset;
  float* group0, *group;
  nRowGroup = 1 << stage; // Equivalent to 2**stage
  nGroup    = nRow/nRowGroup;
  for (iGroup=0; iGroup<nGroup; iGroup++)
    {
      startRow = iGroup*nRowGroup;
      offset   = startRow*nCol;
      group0   = XW0 + offset;
      group    = XW  + offset;
      FFAGroupShiftAdd(group0,group,nRowGroup,nCol); 
     }
}

void FFA_ext(float* XW0, float* XW, int nRow, int nCol, int nStage)
{
  int stage;
  float *old, *new;
  for (stage=1; stage<nStage+1;stage++)
    {
      if (stage % 2 == 1)
	{
	  old = XW0;
	  new = XW;
	}
      else
	{
	  old = XW;
	  new = XW0;
	}

      FFAShiftAdd(old,new,stage,nRow,nCol) ;
    }

}
