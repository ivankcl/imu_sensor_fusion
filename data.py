import pandas as pd
import numpy as np
from util import Util

class IMUData:

    def __init__(self, path):
    
        # read file and remove rows with no data
        df = pd.read_csv(path)
        data = df[df['Acc_X'].notna()]

        # define the data range for extraction
        start = 0
        end = len(data)

        self.length = end

        # define the global variable for function uses
        omega1 = np.array(data['Gyr_X'][start:end], dtype=float) 
        omega2 = np.array(data['Gyr_Y'][start:end], dtype=float)
        omega3 = np.array(data['Gyr_Z'][start:end], dtype=float)
        self.omega1 = Util().deg2rad(omega1.reshape(len(omega1)))
        self.omega2 = Util().deg2rad(omega2.reshape(len(omega2)))
        self.omega3 = Util().deg2rad(omega3.reshape(len(omega3)))
        # self.omega1 = omega1.reshape(len(omega1))
        # self.omega2 = omega2.reshape(len(omega2))
        # self.omega3 = omega3.reshape(len(omega3))

        tdata = np.array(data['SampleTimeFine'][start:end], dtype=float)
        tdata = tdata - tdata[0]
        tdata = tdata/1e3
        self.dt = tdata[1] - tdata[0]
        self.t = tdata.reshape(len(tdata))

        self.f1 = np.array(data['Acc_X'][start:end], dtype=float)
        self.f2 = np.array(data['Acc_Y'][start:end], dtype=float)
        self.f3 = np.array(data['Acc_Z'][start:end], dtype=float)

        try:

            self.fE = np.array(data['FreeAcc_E'][start:end], dtype=float)
            self.fN = np.array(data['FreeAcc_N'][start:end], dtype=float)
            self.fU = np.array(data['FreeAcc_U'][start:end], dtype=float)
        except:
            self.fE = np.zeros((self.length,))
            self.fN = np.zeros((self.length,))
            self.fU = np.zeros((self.length,))

        self.roll = np.array(data['Roll'][start:end], dtype=float)
        self.pitch = np.array(data['Pitch'][start:end], dtype=float)
        self.yaw = np.array(data['Yaw'][start:end], dtype=float)

        mag1 = np.array(data['Mag_X'][start:end], dtype=float)
        mag2 = np.array(data['Mag_Y'][start:end], dtype=float)
        mag3 = np.array(data['Mag_Z'][start:end], dtype=float)

        mask = np.isnan(mag1)
        mag1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mag1[~mask])
        mask = np.isnan(mag2)
        mag2[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mag2[~mask])
        mask = np.isnan(mag3)
        mag3[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mag3[~mask])

        self.mag1 = mag1
        self.mag2 = mag2
        self.mag3 = mag3
