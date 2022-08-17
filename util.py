from errno import E2BIG
import math
from re import T
import numpy as np
import matplotlib.pyplot as plt
# from data import IMUData as Data

class Util:

    def deg2rad(self, x):
        return x*np.pi/180

    def rad2deg(self, x):
        return x*180/np.pi

    def sin(self, x, radian = True):
        if(not radian):
            x = self.deg2rad(x)
        return np.sin(x)

    def cos(self, x, radian = True):
        if(not radian):
            x = self.deg2rad(x)
        return np.cos(x)

    def tan(self, x, radian = True):
        if(not radian):
            x = self.deg2rad(x)
        return np.tan(x)

    # state equation to integrate:
    # rate of change of Eular angles = mat * corotational angular velocity
    # arg should contains [t_a, omega1_a, omega2_a, omega3_a]
    def vdp1(self, t, y, arg):
        
        # interpolate the angular velocity data to obtain the value needed to
        # evaluate the state equations at current integration step

        omega1 = np.interp(t, arg[0], arg[1])
        omega2 = np.interp(t, arg[0], arg[2])
        omega3 = np.interp(t, arg[0], arg[3])

        psi = y[0]
        theta = y[1]
        phi = y[2]

        mat = np.array([
            [0, self.sin(phi)/self.cos(theta), self.cos(phi)/self.cos(theta)],
            [0, self.cos(phi), -self.sin(phi)],
            [1, self.sin(phi)*self.tan(theta), self.cos(phi)*self.tan(theta)]])

        
        return  mat.dot(np.array([
                    [omega1],
                    [omega2],
                    [omega3]
                    ]))

    def draw3D(self, x1, x2, x3, title):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title(title)
        ax.plot(x1, x2, x3, label='parametric curve')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.draw()

    def draw_euler(self, t, roll, pitch, yaw, title, roll_title = 'Roll', pitch_title = 'Pitch', yaw_title = 'Yaw'):
        fig, ax = plt.subplots(3)
        fig.set_size_inches(12.5, 6.5)
        plt.title(title)
        ax[0].plot(t, roll)
        ax[0].set_title(roll_title)
        ax[1].plot(t, pitch)
        ax[1].set_title(pitch_title)
        ax[2].plot(t, yaw)
        ax[2].set_title(yaw_title)
        plt.draw()

    def draw_motion(self, t, xddot, xdot, x, title):
        fig, ax = plt.subplots(3)
        fig.set_size_inches(12.5, 6.5)

        ax[0].plot(t, xddot[0])
        ax[0].plot(t, xddot[1])
        ax[0].plot(t, xddot[2])
        ax[0].legend(['ax', 'ay', 'az'])
        ax[0].set_ylabel('Acceleration', rotation=0, size='small')

        ax[1].plot(t, xdot[0])
        ax[1].plot(t, xdot[1])
        ax[1].plot(t, xdot[2])
        ax[1].legend(['vx', 'vy', 'vz'])
        ax[1].set_ylabel('Velocity', rotation=0, size='small')

        ax[2].plot(t, x[0])
        ax[2].plot(t, x[1])
        ax[2].plot(t, x[2])
        ax[2].legend(['x', 'y', 'z'])
        ax[2].set_ylabel('Displacement', rotation=0, size='small')

        plt.draw()

    def draw_euler_3x3(self, t, e1, e1_title, e2, e2_title, e3, e3_title):
        fig, ax = plt.subplots(3, 3)

        roll, pitch, yaw = e1

        e1_title = e1_title + '\nroll'
        ax[0, 0].plot(t, roll)
        ax[0, 0].set_title(e1_title)
        ax[1, 0].plot(t, pitch)
        ax[1, 0].set_title('\npitch')
        ax[2, 0].plot(t, yaw)
        ax[2, 0].set_title('\nyaw')

        roll, pitch, yaw = e2
        e2_title = e2_title + '\nroll'
        ax[0, 1].plot(t, roll)
        ax[0, 1].set_title(e2_title)
        ax[1, 1].plot(t, pitch)
        ax[1, 1].set_title('\npitch')
        ax[2, 1].plot(t, yaw)
        ax[2, 1].set_title('\nyaw')

        roll, pitch, yaw = e3
        e3_title = e3_title + '\nroll'
        ax[0, 2].plot(t, roll)
        ax[0, 2].set_title(e3_title)
        ax[1, 2].plot(t, pitch)
        ax[1, 2].set_title('\npitch')
        ax[2, 2].plot(t, yaw)
        ax[2, 2].set_title('\nyaw')

        plt.draw()

    def draw_filter_diff_3x2(self, t, a_0, a, v_0, v, x_0, x):
        fig, ax = plt.subplots(3, 2)
        fig.set_size_inches(12.5, 6.5)

        a0, a1, a2 = a_0
        ax[0, 0].plot(t, a0)
        ax[0, 0].plot(t, a1)
        ax[0, 0].plot(t, a2)
        ax[0, 0].legend(['ax', 'ay', 'az'])
        ax[0, 0].set_title('Before filter')
        ax[0, 0].set_ylabel('Acceleration', rotation=0, size='small')

        a0, a1, a2 = a
        ax[0, 1].plot(t, a0)
        ax[0, 1].plot(t, a1)
        ax[0, 1].plot(t, a2)
        ax[0, 1].legend(['ax', 'ay', 'az'])
        ax[0, 1].set_title('Before filter')

        v0, v1, v2 = v_0
        ax[1, 0].plot(t, v0)
        ax[1, 0].plot(t, v1)
        ax[1, 0].plot(t, v2)
        ax[1, 0].legend(['vx', 'vy', 'vz'])
        ax[1, 0].set_ylabel('Velocity', rotation=0, size='small')        

        v0, v1, v2 = v
        ax[1, 1].plot(t, v0)
        ax[1, 1].plot(t, v1)
        ax[1, 1].plot(t, v2)
        ax[1, 1].legend(['vx', 'vy', 'vz'])

        x0, x1, x2 = x_0
        ax[2, 0].plot(t, x0)
        ax[2, 0].plot(t, x1)
        ax[2, 0].plot(t, x2)
        ax[2, 0].legend(['x', 'y', 'z'])
        ax[2, 0].set_ylabel('Displacement', rotation=0, size='small') 

        x0, x1, x2 = x
        ax[2, 1].plot(t, x0)
        ax[2, 1].plot(t, x1)
        ax[2, 1].plot(t, x2)
        ax[2, 1].legend(['x', 'y', 'z'])

        plt.draw()