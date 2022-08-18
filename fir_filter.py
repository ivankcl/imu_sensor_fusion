from data import IMUData
from util import Util
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import kaiserord, lfilter, firwin, freqz

def fir_filter(x, nyq = 40, width = 2.0, ripple = 60.0, cutoff = 4.0):
    nyq_rate = nyq / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = width/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = ripple

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = cutoff

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, x)

    return filtered_x

def main(data):
    x1ddot = np.array((data.f1 - np.mean(data.f1))).reshape(len(data.f1))
    x2ddot = np.array((data.f2 - np.mean(data.f2))).reshape(len(data.f2))     
    x3ddot = np.array((data.f3 - np.mean(data.f3))).reshape(len(data.f3))

    x1ddot_f = fir_filter(x1ddot)
    x2ddot_f = fir_filter(x2ddot)
    x3ddot_f = fir_filter(x3ddot)

    x1dot = integrate.cumtrapz(x1ddot, data.t, initial = 0)
    x2dot = integrate.cumtrapz(x2ddot, data.t, initial = 0)
    x3dot = integrate.cumtrapz(x3ddot, data.t, initial = 0)

    x1dot_f = fir_filter(x1dot)
    x2dot_f = fir_filter(x2dot)
    x3dot_f = fir_filter(x3dot)

    x1 = integrate.cumtrapz(x1dot, data.t, initial=0)
    x2 = integrate.cumtrapz(x2dot, data.t, initial=0)
    x3 = integrate.cumtrapz(x3dot, data.t, initial=0)

    x1_f = fir_filter(x1)
    x2_f = fir_filter(x2)
    x3_f = fir_filter(x3)

    xddot = (x1ddot, x2ddot, x3ddot)
    xddot_f = (x1ddot_f, x2ddot_f, x3ddot_f)
    xdot = (x1dot, x2dot, x3dot)
    xdot_f = (x1dot_f, x2dot_f, x3dot_f)
    x = (x1, x2, x3)
    x_f = (x1_f, x2_f, x3_f)

    Util().draw_filter_diff_3x2(data.t, xddot, xddot_f, xdot, xdot_f, x, x_f)
    Util().draw3D(x1_f, x2_f, x3_f, 'Trajectory')
    plt.show()


if __name__ == '__main__':
    d = IMUData('Data/data_freeAcc.csv.nosync.csv')
    main(d)