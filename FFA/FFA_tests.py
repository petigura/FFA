from __future__ import print_function
import unittest
import numpy as np
# import FFA_cy as FFA
import FFA_cext as FFA
import os
import datetime

datadir = os.path.dirname(__file__) + '/sample_data/'


class TestClass(unittest.TestCase):
    def test_FFA(self):
        for input in ['eye-4x6.npy', 'eye-32x2500.npy', 'P_2500+1.npy']:
            output = 'res_' + input
            inputArr = np.load(datadir + input)
            outputArr = np.load(datadir + output)

            print(inputArr)
            print(outputArr)
            self.assertTrue(np.allclose(FFA.FFA(inputArr), outputArr))

    def test_time(self):
        ntries = 100
        t0 = datetime.datetime.now()
        input = datadir + 'eye-32x2500.npy'
        inputArr = np.load(input)
        for i in range(ntries):
            FFA.FFA(inputArr)
        dt = (datetime.datetime.now() - t0).total_seconds()
        print("Folding {}x{} array requires {} s (average of {} runs)".format(*inputArr.shape, dt/ntries, ntries))

if __name__ == '__main__':
    unittest.main()
