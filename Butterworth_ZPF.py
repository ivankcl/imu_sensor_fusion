from data import IMUData
from util import Util
import numpy as np
from scipy import integrate, signal
import matplotlib.pyplot as plt


def zpf(x, samplePeriod, order = 6, cutoff = 0.2, highpass = True, gust = False):
    wn = (2*cutoff)/(1/samplePeriod)
    if(highpass):
        b, a = signal.butter(order, wn, btype='highpass', analog=False)
    else:
        b, a = signal.butter(order, cutoff, btype='lowpass', analog=False)

    if(gust):
        return signal.filtfilt(b, a, x, method='gust')
    else:
        return signal.filtfilt(b, a, x)
        

def main(data, a2v, v2d):

    f1, f2, f3 = data.f1, data.f2, data.f3
    x1ddot = np.array((f1 - np.mean(f1))).reshape(len(f1))
    x2ddot = np.array((f2 - np.mean(f2))).reshape(len(f2))     
    x3ddot = np.array((f3 - np.mean(f3))).reshape(len(f3))

    td = [j-i for i, j in zip(data.t[:-1], data.t[1:])]
    period = np.mean(td)

    x1dot = x1dot_f= integrate.cumtrapz(x1ddot, data.t, initial = 0)
    x2dot = x2dot_f = integrate.cumtrapz(x2ddot, data.t, initial = 0)
    x3dot = x2dot_f = integrate.cumtrapz(x3ddot, data.t, initial = 0)

    if(a2v[0]):
        x1dot_f = zpf(x1dot, period, a2v[1], a2v[2])
        x2dot_f = zpf(x2dot, period, a2v[3], a2v[4])
        x3dot_f = zpf(x3dot, period, a2v[5], a2v[6])
        
    x1 = x1_f = integrate.cumtrapz(x1dot_f, data.t, initial=0)
    x2 = x2_f = integrate.cumtrapz(x2dot_f, data.t, initial=0)
    x3 = x3_f = integrate.cumtrapz(x3dot_f, data.t, initial=0)

    if(v2d[0]):
        x1_f = zpf(x1, period, v2d[1], v2d[2])
        x2_f = zpf(x2, period, v2d[3], v2d[4])
        x3_f = zpf(x3, period, v2d[5], v2d[6])

    xddot = (x1ddot, x2ddot, x3ddot)
    xddot_f = (x1ddot, x2ddot, x3ddot)
    xdot = (x1dot, x2dot, x3dot)
    xdot_f = (x1dot_f, x2dot_f, x3dot_f)
    x = (x1, x2, x3)
    x_f = (x1_f, x2_f, x3_f)

    Util().draw_filter_diff_3x2(data.t, xddot, xddot_f, xdot, xdot_f, x, x_f)
    Util().draw3D(x1_f, x2_f, x3_f, 'Trajectory')

    plt.show()

if __name__ == '__main__':
    d = IMUData('Data/data_freeAcc.csv.nosync.csv')
    # filter parameter:
    a2v = [True, 4, 0.00056, 4, 0.00056, 4, 0.00056]
    v2d = [True, 4, 0.00056, 4, 0.00056, 4, 0.00056]

    main(d, a2v, v2d)