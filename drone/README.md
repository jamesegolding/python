# Drone Simulation and Control

Python drone model using 9dof sensor

Overview:
- Orientation sensor fusion using Kalman filter
- Orientation control using quaternion target (http://folk.ntnu.no/skoge/prost/proceedings/ecc-2013/data/papers/0927.pdf)
- Translational estimation based on simple accelerometer integration
- Vertical control using LQR

More details:
- Quaternion state and angular velocity derivative
- Linear capped motor forces with first order dynamics
- Gaussian sensor noise and disturbance accelerations
- Quaternion numerical integration using exponential function
- Numba just in time compilation
