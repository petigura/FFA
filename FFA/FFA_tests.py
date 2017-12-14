import numpy as np
#import FFA_cy as FFA
import FFA_cext as FFA

datadir = 'sample_data/'
class TestClass:
   def test_FFA(self):
       for input in ['eye-4x6.npy','eye-32x2500.npy','P_2500+1.npy']:
          output = 'res_'+input
          inputArr = np.load(datadir+input) 
          outputArr = np.load(datadir+output)

          print inputArr
          print outputArr
          assert( np.allclose(FFA.FFA(inputArr),outputArr))

   def test_time(self):
       input = datadir+'eye-32x2500.npy'
       inputArr = np.load(input) 
       for i in range(100):
           FFA.FFA(inputArr)
