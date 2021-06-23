#! /usr/bin/env python3

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

class FilterPerformance:

    def __init__(self):
        self.n1 = 4096 # FFT size.
        self.n2 = 2049 # FFT result size.

        self.evalcount = 0

    def __call__(self, x):
        self.evalcount += 1
        print("@@@@", self.evalcount)
        fx = np.fft.rfft(x, self.n1)
        assert len(fx) == self.n2

        magnitude = np.abs(fx)

        stopband1 = (magnitude[:200] - 0.0)
        passband = 100 * (magnitude[250:350] - 1.0)
        stopband2 = (magnitude[400:] - 0.0)

        err = np.hstack((stopband1, passband, stopband2))

        return err


ntaps = 1000

func = FilterPerformance()

x0 = np.random.randn(ntaps)

(x, b_arr, c_dict, d_str, e_int) = scipy.optimize.leastsq(func, x0, full_output=True, maxfev=2**31-1)

print(x.shape, x.dtype)
#print(b_arr.shape, b_arr.dtype)
#print(c_dict.keys())
print(d_str)
print(e_int)

plt.subplot(311)
plt.plot(x)

fx = np.fft.rfft(x, func.n1)
magnitude = np.abs(fx)
arg = np.angle(fx)

plt.subplot(312)
plt.plot(10 * np.log10(magnitude))

plt.subplot(313)
plt.plot(arg)

plt.show()

#ftaps = np.fft.fft(taps)

#mag   = np.abs(ftaps)
#phase = np.angle(ftaps)

#plt.plot(mag)
#plt.show()
