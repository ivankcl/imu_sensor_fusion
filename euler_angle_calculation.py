from data import IMUData
from util import Util
import numpy as np
from scipy import integrate, ndimage
import matplotlib.pyplot as plt

def integration(data, sample_size = 0, batch = False):
    psi0 = 0
    theta0 = 0
    phi0 = 0

    t = data.t

    y0 = np.array([ psi0 , theta0, phi0])
    y = np.zeros((len(data.t), len(y0)))

    y[0, :] = y0
    arg = [data.t, data.omega1, data.omega2, data.omega3]

     # initial values

    if(batch):
        for i in range(0, t.size//sample_size):
            r = integrate.ode(Util().vdp1).set_integrator("dopri5").set_f_params(arg)  
            r.set_initial_value(y[-1,:], t[i*sample_size])
            # print('runned: ', i*sample_size)
            for j in range(1, sample_size+1):       
                if(i*sample_size+j<t.size):
                    y[i*sample_size+j, :] = r.integrate(t[i*sample_size+j], ) # get one more value, add it to the array
                    if not r.successful():
                        raise RuntimeError("Could not integrate")
    else:
        r = integrate.ode(Util().vdp1).set_integrator("dopri5")  # choice of method
        r.set_initial_value(y0, t[0]).set_f_params(arg)  
        for i in range(1, t.size):
            y[i, :] = r.integrate(t[i]) # get one more value, add it to the array
            if not r.successful():
                raise RuntimeError("Could not integrate")

    psi = y[:,0]
    theta = y[:,1]
    phi = y[:,2]

    return (psi, theta, phi)

def Cenu(roll, pitch, yaw):
    return np.array([
        [Util().cos(yaw)*Util().cos(pitch), -Util().cos(roll)*Util().sin(yaw)+Util().sin(roll)*Util().cos(yaw)*Util().sin(pitch), Util().sin(roll)*Util().sin(yaw)+Util().cos(roll)*Util().cos(yaw)*Util().sin(pitch)],
        [Util().sin(yaw)*Util().cos(pitch), Util().cos(roll)*Util().cos(yaw)+Util().sin(roll)*Util().sin(yaw)*Util().sin(pitch), -Util().sin(roll)*Util().cos(yaw)+Util().cos(roll)*Util().sin(yaw)*Util().sin(pitch)],
        [-Util().sin(pitch), Util().sin(roll)*Util().cos(pitch), Util().cos(roll)*Util().cos(pitch)]
    ])

def cal(data, alpha = 1, sigma = 200):


    f1 = (data.f1 - np.mean(data.f1))*alpha
    f2 = (data.f2 - np.mean(data.f2))*alpha
    f3 = (data.f3 - np.mean(data.f3))*alpha

    roll = np.arctan2(f2, f3) * 180/np.pi
    pitch = np.arctan2(-f1, np.sqrt(f2**2 + f3**2)) * 180/np.pi

    magx = data.mag1.dot(Util().cos(pitch)) + np.array(data.mag2.dot(Util().sin(roll))).dot(Util().sin(pitch)) + np.array(data.mag3.dot(Util().sin(pitch))).dot(Util().cos(roll))
    magy = -data.mag2.dot(Util().cos(roll)) + data.mag3.dot(Util().sin(roll))

    yaw = np.arctan2(-magy, magx) * 180/np.pi

    roll_p = []
    pitch_p = []
    yaw_p = []

    for r, p, y in zip(roll, pitch, yaw):
        Axyz = Cenu(r, p, y).transpose().dot(np.array([[0],[0],[1]]))
        roll_p.append(np.arctan2(Axyz[1], Axyz[2]))
        pitch_p.append(np.arctan2((-Axyz[0]), (np.sqrt(Axyz[1]**2 + Axyz[2]**2))))
        yaw_p.append(0)

    roll_p = np.array(roll_p).flatten()
    pitch_p = np.array(pitch_p).flatten()
    # print(np.shape(roll_p), np.shape(pitch_p), np.shape(yaw_p))

    yaw_p = []

    # print(np.shape(roll), np.shape(data.mag1))

    for i, (r, p, y, m1, m2, m3) in enumerate(zip(roll, pitch, yaw, data.mag1, data.mag2, data.mag3)):
        Mxyz = Cenu(r, p, y).transpose().dot(np.array([[m1],[m2],[m3]]))   
        # print(np.shape(Mxyz), np.shape(np.array([Cxyz_p[i]]).T))
        Cxyz_p = Cenu(r, p, y).transpose()
        mxyz_p = Cxyz_p.dot(Mxyz)
        yaw_p.append(np.arctan2(mxyz_p[1],mxyz_p[2]))
  

    roll_p = ndimage.gaussian_filter1d(roll_p, sigma=sigma)
    pitch_p = ndimage.gaussian_filter1d(pitch_p, sigma=sigma)
    yaw_p = np.array(ndimage.gaussian_filter(yaw_p, sigma=sigma)).flatten()

    # print(np.shape(roll_p), np.shape(pitch_p), np.shape(yaw_p))
    
    return (roll_p, pitch_p, yaw_p)

def main(data):

    intE = integration(data)
    calE = cal(data, alpha = 0.9, sigma= 250)

    Util().draw_euler_3x3(data.t, (data.roll, data.pitch, data.yaw), 'Sensor Algo', intE, 'Integration', calE, 'Calculation')

    plt.show()


if __name__ == '__main__':
    d = IMUData('Data/calibration1.csv.nosync.csv')
    main(d)