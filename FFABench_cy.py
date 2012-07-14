import numpy as np
from numpy import ma
import FFA_cy as FFA

def FFABench():
   def seg(X,P0):
       XW = FFA.XWrap2(X,P0,pow2=True)
       M   = XW.shape[0]  # number of rows
       
       idCol = np.arange(P0,dtype=int)   # id of each column
       idRow = np.arange(M,dtype=int)   # id of each row
       P  = P0 + idRow.astype(float) / (M - 1)
   
       XW.fill_value=0
       data = XW.filled()
       mask = (~XW.mask).astype(int)
   
       sumF   = FFA.FFA(data) # Sum of array elements folded on P0, P0 + i/(1-M)
       countF = FFA.FFA(mask) # Number of valid data points
       meanF  = sumF/countF
   
       names = ['mean','count','s2n','P']
       dtype = zip(names,[float]*len(names) )
       rep   = np.empty(M,dtype=dtype)
       
       # Take maximum epoch
       idColMa      = meanF.argmax(axis=1)
       rep['mean']  = meanF[idRow,idColMa]
       rep['count'] = countF[idRow,idColMa]
       rep['s2n']   = rep['mean']*np.sqrt(rep['count'])
       rep['P']     = P
   
       return rep
   
   X = np.load('pulse_train_data.npy')
   Xmask = np.load('pulse_train_mask.npy')
   X = ma.masked_array(X,Xmask,fill_value=0)
   
   X = X[:10000] # Modify this to change execution time.
   
   Pmin,Pmax = 250,2500
   PGrid = np.arange(Pmin,Pmax)
   
   func = lambda P0: seg(X,P0)
   rep = map(func,PGrid)
   rep = np.hstack(rep)
   return rep

