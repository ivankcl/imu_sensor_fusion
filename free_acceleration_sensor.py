from data import IMUData
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from util import Util

def main(data):
    
    
    f = np.vstack([data.fE, data.fN, data.fU])
    T = np.array([
        [0.999981, -0.0041867, -0.00428401],
        [0.00462431, 0.999989, 0.000388165],
        [0.00470699, -0.000219299, 0.999989]
    ])

    newf = T.dot(f)
    F1 = newf[0,:]
    F2 = newf[1,:]
    F3 = newf[2,:]

    x1ddot = np.array((F1 - np.mean(F1))).reshape(len(F1))        
    x2ddot = np.array((F2 - np.mean(F2))).reshape(len(F2))      
    x3ddot = np.array((F3 - np.mean(F3))).reshape(len(F3))

    x1dot = integrate.cumulative_trapezoid(x1ddot, data.t, initial = 0)
    x2dot = integrate.cumulative_trapezoid(x2ddot, data.t, initial = 0)
    x3dot = integrate.cumulative_trapezoid(x3ddot, data.t, initial = 0)

    x1 = integrate.cumulative_trapezoid(x1dot, data.t, initial=0)
    x2 = integrate.cumulative_trapezoid(x2dot, data.t, initial=0)
    x3 = integrate.cumulative_trapezoid(x3dot, data.t, initial=0)

    Util().draw_motion(data.t, [x1ddot, x2ddot, x3ddot], [x1dot, x2dot, x3dot], [x1, x2, x3], 'Motion')
    Util().draw3D(x1, x2, x3, 'trajectory')
    plt.show()


if __name__ == '__main__':
    d = IMUData('Data/calibration1.csv.nosync.csv')
    main(d)