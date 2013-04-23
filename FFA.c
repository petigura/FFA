#include "FFA.h"
#include <stdio.h>

void FFAGroupShiftAdd(float* group0, float* group, int nRowGroup,int nColGroup)
{
  int iA,iB,Bs,jB,i,j,step,stepA,stepB;
  int nRowGroupOn2 = nRowGroup / 2; // Half the group size
  /*
  Grow group by the maximum shift value
  Loop over rows in group
  nColGroup number of columns in group

  jB is the column number from which to pull the second value. For some reason, 


  //  jB = (j + Bs + nColGroup) % nColGroup;
  //  if (jB==nColGroup) {jB=0;} //protect jb from rolling off the edge
  */

  for(i=0; i<nRowGroup; i++)
    {
      iA = i/2;                 //# Row in the group that A is draw from
      iB = iA + nRowGroupOn2; //  # Row in group that B is drawn from
      Bs = (i + 1) / 2;
      //# Loop over the columns in the group

      step = i*nColGroup;
      stepA = iA*nColGroup;
      stepB = iB*nColGroup;
      j = 0;
      jB = Bs;
      while (j < nColGroup)
	  {
	    jB = (j + Bs + nColGroup) % nColGroup;
	    // if (jB==nColGroup) {jB=0;} //protect jb from rolling off the edge
	    group[step+j] = group0[stepA+j] + group0[stepB+jB];
	    j++;
	    jB++;
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

int main(int argc, char *argv[])
{

  int N  = atoi(argv[2]);
  int P0 = atoi(argv[3]);
  float XW0[N];
  float XW[N];

  FILE *infile;

  infile = fopen(argv[1], "r");

  int i=0;
  while(!feof(infile))
    {
      fscanf(infile,"%f",&XW0[i]);
      i++;
    }

  fclose(infile);

  int nStage = 1;
  int nRow   = N/P0;
  while (nRow >> nStage != 1)
    {
      nStage++;
    }
  printf("%d %d %d \n",N,P0,nStage);
  for (i=0;i<1;i++)
    {
      FFA_ext(XW0,XW,nRow,P0,nStage);
    }
  return 0;
}
