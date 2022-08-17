import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from data import IMUData
from util import Util

def main(data):

    A = np.array(
        [
            [1, data.dt, data.dt**2/2],
            [0, 1, data.dt],
            [0, 0, 1]
        ]
    )
    B = 0
    H = np.identity(3)

    u = 0
    # acceleration matrix
    k1 = np.array([data.f1, data.f2, data.f3])
    Q = np.zeros((3, 1))
    Q_est = Q
    P = np.identity(3)
    Ex = np.identity(3)
    R = np.identity(3)

    Z_p = np.zeros((3, 1))
    Z_v = np.zeros((3, 1))
    Z_a = np.zeros((3, 1))

    x_estimate_az = np.zeros((3, 1))
    y_estimate_az = np.zeros((3, 1)) 
    z_estimate_az = np.zeros((3, 1))

    ll = np.mean(data.f3)
    u3_bias = data.f3 - ll
    u3_perfect = data.f3 - u3_bias

    for t in range(data.length):
        Q_est_curr = A.dot(Q_est)+ B * u
        Z_p = np.vstack([Z_p, Q_est_curr[0]])
        Z_v = np.vstack([Z_v, Q_est_curr[1]])
        Z_a = np.vstack([Z_a, Q_est_curr[2]])

        P = A.dot(P).dot(A.T) + Ex
        K = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T)+ R))
        y = np.vstack([Q_est_curr[0], Q_est_curr[1], data.f1[t]])
        Q_est = Q_est_curr + K.dot(y - H.dot(Q_est_curr))

        P = (np.identity(3) - K.dot(H)).dot(P)

        x_estimate_az = np.vstack([x_estimate_az, Q_est[0]])
        y_estimate_az = np.vstack([y_estimate_az, Q_est[1]])
        z_estimate_az = np.vstack([z_estimate_az, Q_est[2]])

    # print(np.shape(z_estimate_az), np.shape(tdatac))

    x_estimate_az = x_estimate_az[3:]   
    y_estimate_az = y_estimate_az[3:]
    z_estimate_az = z_estimate_az[3:]


    x1ddot = x_estimate_az.flatten()
    x2ddot = y_estimate_az.flatten()
    x3ddot = z_estimate_az.flatten()


    x1dot = integrate.cumulative_trapezoid(x1ddot, data.t, initial = 0)
    x2dot = integrate.cumulative_trapezoid(x2ddot, data.t, initial = 0)
    x3dot = integrate.cumulative_trapezoid(x3ddot, data.t, initial = 0)

    x1 = integrate.cumulative_trapezoid(x1dot, data.t, initial=0)
    x2 = integrate.cumulative_trapezoid(x2dot, data.t, initial=0)
    x3 = integrate.cumulative_trapezoid(x3dot, data.t, initial=0)


    Util().draw3D(x1, x2, x3, 'Trajectory')
    plt.show()

    

if __name__ == '__main__':

    d = IMUData('Data/calibration1.csv.nosync.csv')

    main(d)