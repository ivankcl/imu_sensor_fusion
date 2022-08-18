from data import IMUData
from util import Util
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def main(data):

    # define the initial condition:
    # define the initial condition:
    psi0 = 0; theta0 = 0; phi0 = 0
    y0 = np.array([psi0 , theta0, phi0])
    t = data.t
    x10 = 0            #  m
    x20 = 0            #  m
    x30 = 0            #  m

    x1dot0 = 0         #  m/s
    x2dot0 = 0        	#  m/s
    x3dot0 = 0       	#  m/s

    # declare the array for storing the results of integration
    y = np.zeros((data.length, len(y0))) 
    y[0, :] = y0

    # data = Data()
    arg = [data.t, data.omega1, data.omega2, data.omega3]

    # integration:
    r = integrate.ode(Util().vdp1).set_integrator("dopri5")
    r.set_initial_value(y0, t[0]).set_f_params(arg)
    for i in range(1, t.size):
        y[i, :] = r.integrate(t[i]) 
        # print(i)
        if not r.successful():
            raise RuntimeError("Could not integrate")

    psi = y[:,0]
    theta = y[:,1]
    phi = y[:,2]

    Util().draw_euler(data.t, psi, theta, phi, 'Euler Angles')

    # declare the array for storing linear acceleration
    F1 = []
    F2 = []
    F3 = []

    # coordinate reorientation: 
    # transforming corotational specific force measurements to space-fixed components
    for k in range(len(psi)):
        R1=(np.array(
            [
                [Util().cos(psi[k]), -Util().sin(psi[k]), 0],
                [Util().sin(psi[k]), Util().cos(psi[k]), 0],
                [0, 0, 1]
            ]))
        R2=(np.array(
            [
                [Util().cos(theta[k]), 0, Util().sin(theta[k])],
                [0, 1, 0],
                [-Util().sin(theta[k]), 0, Util().cos(theta[k])]
            ]))
        R3=(np.array(
            [
                [1, 0, 0],
                [0, Util().cos(phi[k]), -Util().sin(phi[k])],
                [0, Util().sin(phi[k]), Util().cos(phi[k])]
            ]
        ))
        f = np.array([data.f1[k], data.f2[k], data.f3[k]]).reshape((3,1))
        
        F123 = R1.dot(R2).dot(R3).dot(f)
        F1.append(F123[0])
        F2.append(F123[1])
        F3.append(F123[2])

    # remove bias and acceleration due to gravity
    x1ddot = np.array((F1 - np.mean(F1))).reshape(len(F1))        
    x2ddot = np.array((F2 - np.mean(F2))).reshape(len(F2))      
    x3ddot = np.array((F3 - np.mean(F3))).reshape(len(F3))    

    # integrate acceleration twice using the trapezoidal rule
    x1dot = integrate.cumulative_trapezoid(x1ddot, data.t, initial = x1dot0)
    x2dot = integrate.cumulative_trapezoid(x2ddot, data.t, initial = x2dot0)
    x3dot = integrate.cumulative_trapezoid(x3ddot, data.t, initial = x3dot0)

    x1 = integrate.cumulative_trapezoid(x1dot, data.t, initial=x10)
    x2 = integrate.cumulative_trapezoid(x2dot, data.t, initial=x20)
    x3 = integrate.cumulative_trapezoid(x3dot, data.t, initial=x30)

    Util().draw_motion(data.t, [x1ddot, x2ddot, x3ddot], [x1dot, x2dot, x3dot], [x1, x2, x3], 'Motion')
    Util().draw3D(x1, x2, x3, 'Trajectory')

    plt.show()

if __name__ == '__main__':

    d = IMUData('Data/data_freeAcc.csv.nosync.csv')
    main(d)
