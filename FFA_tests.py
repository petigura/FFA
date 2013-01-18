import numpy as np
import FFA_cy as FFA

class TestClass:
   def test_FFA(self):
       for input in ['eye-4x6.npy','eye-32x2500.npy','P_2500+1.npy']:
           inputArr = np.load(input) 
           outputArr = np.load('res_'+input) 
           print inputArr
           print outputArr
           assert( np.allclose(FFA.FFA(inputArr),outputArr))
   def test_time(self):
       input = 'eye-32x2500.npy'
       inputArr = np.load(input) 
       for i in range(100):
           FFA.FFA(inputArr)
