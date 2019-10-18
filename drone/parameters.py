
# physical constants
G = 9.81                # [m/s/s]
rho = 1.225             # [kg/m/m/m]

# mass and inertia
m = 1.2                 # [kg]
J_xx = 7.0e-3           # [kg*m*m]
J_yy = 7.0e-3           # [kg*m*m]
J_zz = 1.3e-2           # [kg*m*m]

# geometry
l_arm = 0.2             # [m] arm length of drone (cg to motor)
k_z_arm = 0.005         # [m] influence of single motor on yaw

# aerodynamics
cd_xy = 0.6             # [-] Fx / 0.5 / rho / v_x^2 (copy Fy vs v_y)
cd_z = 0.6              # [-] Fz / 0.5 / rho / v_z^2
cd_axy = 0.1            # [-] Mx / 0.5 / rho / n_x^2 (copy My vs n_y)
cd_az = 0.1             # [-] Mz / 0.5 / rho / n_z^2

# disturbances
std_g_dist = 0.        # [m/s/s]
std_dn_dist = 0.       # [m/s/s]

# sensor noise
std_g_xyz_noise = 0.   # [m/s/s]
std_n_axyz_noise = 0.  # [rad/s/s]
std_e_mag_noise = 0.  # [-] orientation vector

# motor_max
f_motor_max = 6.5727    # motor max force
tau_motor = 0.05        # motor response time constant

# grounding
k_ground = 1e2
d_ground = 5e0
