## IMU Sensor Fusion
### Purpose

The purpose of the project is pathing the trajectory and finding the orientation of an IMU sensor with accelerometer, gyroscope and magnetometer data. Mti-680G is used.
### Filter and Algorithm Used

#### 1. Simple Cumulative Trapezoidal Method

It can be found in `simple_cumtrap_integration.py`. This is a simple integration method using the trapezoidal rule without any filters or algorithms. Before the integration, we would subtract $a_i$ from its mean to remove gravitational acceleration and bias.

The integration is based on the formula:

$x_i(t) = x_i(0) + \int_0^t[v_i(0) + \int_0^\tau(a_i(u) du)]d\tau$

We can integrate acceleration twice to obtain the displacement, and to estimate the integral, trapezoidal rule can be used:

$v_i(t) \approx \frac{\Delta t}{2}(a_i(0) + a_i(t) + \sum_{i=1}^{t-1}2a_i(t))$

$x_i(t) \approx \frac{\Delta t}{2}(v_i(0) + v_i(t) + \sum_{i=1}^{t-1}2v_i(t))$

Note that since the sensor is fixed, we can treat its lcoal acceleration as the space-fixed acceleration.