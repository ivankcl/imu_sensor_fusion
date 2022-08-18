# IMU Sensor Fusion
## Purpose

The purpose of the project is pathing the trajectory and finding the orientation of an IMU sensor with accelerometer, gyroscope and magnetometer data. Mti-680G is used.
## Filter and Algorithm Used

### 0. Helper Classes

`Util()` in `util.py` contains helper functions for plotting, trigonometric functions and commonly used functions.

`IMUData()` in `data.py` decodes the imu data in the data file.

### 1. Simple Cumulative Trapezoidal Method

It can be found in `simple_cumtrap_integration.py`. This is a simple integration method using the trapezoidal rule without any filters or algorithms. Before the integration, we would subtract $a_i$ from its mean to remove gravitational acceleration and bias.

The integration is based on the formula:

$x_i(t) = x_i(0) + \int_0^t[v_i(0) + \int_0^\tau(a_i(u) du)]d\tau$

We can integrate acceleration twice to obtain the displacement, and to estimate the integral, trapezoidal rule can be used:

$v_i(t) \approx \frac{\Delta t}{2}(a_i(0) + a_i(t) + \sum_{i=1}^{t-1}2a_i(t))$

$x_i(t) \approx \frac{\Delta t}{2}(v_i(0) + v_i(t) + \sum_{i=1}^{t-1}2v_i(t))$

Note that since the sensor is fixed, we can treat its lcoal acceleration as the space-fixed acceleration.

In python, `integrate.cumulative_trapezoid()` from `scipy` is used. And this method is the integration function that would be used in other files as well.

### 2. Butterworth Filter w/ Zero-phase Filter

Butterworth filter smooths out the frequency and produces a maximally flat response in the passband and rolls off towards zero in the stopband (No ripples)
The filter takes 3 parameters:
order of filter, cut-off frequency and the functionality

The filter could reduce noise by reducing the amplitude of wave above or below a certain frequency of choice.
However, using Butterworth filter would cause a phase shift. To eliminate the shift, the filter is applied forward and backward once.

We used the filters twice, after the integration of accleration and velocity. The files can be found in `Butterworth_ZPF.py`.

```python
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
```

### 3. lfilter w/ FIR filter

lfilter can filter data along one-dimention with a FIR filter. And the FIR filter can be used to implement its frequency response.

The program can be found in `fir_filter.py`.

```python
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

```

### 4. Combinatoin of #2 and #3

The results produced above are not optimal. Therefore, we combine the two filters and try to construct a better result. A FIR filter is used to process the acceleration data before integration. It could reduce outliers and again smooth out the data.

The result can be found in `fir_Butterworth_ZPF.py`.

### 5. Free Acceleration

Although the sensor is being fixed on the robot arm and should not have any orientation, vibrations and movement could still cause a small amount of rotation. Therefore, we want to take that into account in our calculation too.

We want to convert the local acceleration of the sensor to a space-fixed acceleration.

Based on the formula below, we can first related the change of Euler angles and angular velocity. 

Then, by integrating the formula, we could obtain the roll, pitch and yaw of the sensor at any given time. The integration module used this time is 
`integrate.ode(Util().vdp).set_integrator("dopri5")`
as it could solve differential equations.

```python
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
```

Based on the Euler angles we obtained, we can use the following formula to calculate the space-fixed acceleration.

However, the Euler angle result is inaccurate. Instead of treating the whole acceleration data as a continuous set of data, an interval of 20 is taken for example. Hence, every 20 data points in the function are integrate using the ode method.

```python    
y = np.zeros((data.length, len(y0))) 
y[0, :] = y0

arg = [data.t, data.omega1, data.omega2, data.omega3]

# integration:
for i in range(0, t.size//sample_size):
    r = integrate.ode(Util().vdp1).set_integrator("dopri5").set_f_params(arg)
    r.set_initial_value(y[-1,:], t[i*sample_size])
    # print('runned: ', i*sample_size)
    for j in range(1, sample_size+1):       
        if(i*sample_size+j<t.size):
            y[i*sample_size+j, :] = r.integrate(t[i*sample_size+j], ) # get one more value, add it to the array
            if not r.successful():
                raise RuntimeError("Could not integrate")

psi = y[:,0]
theta = y[:,1]
phi = y[:,2]

```

A more accurate result is produced. The two files could be found in `free_acceleration_calculation.py` and `free_acceleration_segmented.py`.

Lastly, the sensor also has an inbuilt algorithm to generate space-fixed acceleration. They could be used directly for integration as shown in `free_acceleration_sensor.py`.

### 6. Euler Angle Calculation

There is another method to calculate Euler angles using given data besides integration. Based on the following formula, they can be calculated by the accelerations.

The comparison of this method, integraion method and sensor generated one is shown in `euler_angle_calculation.py`.

### 7. MATLAB Functions

Several MATLAB functions could fulfil our purpose.

1. `ahrsfilter` object in `ahrsfilter.mlx` fuses accelerometer, magnetometer, and gyroscope sensor data to estimate device orientation.
2. `insfilterMARG` and `insfilterasync` in `insfilter.mlx` and `insfilterasync.mlx` estimate pose based on accelerometer, gyroscope, GPS, and magnetometer using an extended Kalman filter.