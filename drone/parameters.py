
# physical constants
G = 9.81                # [m/s/s]
rho = 1.225             # [kg/m/m/m]

# mass and inertia
m = 1.2                 # [kg]
J_xx = 0.01             # [kg*m*m]
J_yy = 0.01             # [kg*m*m]
J_zz = 0.02             # [kg*m*m]

# geometry
l_xy_arm = 0.2          # [m] half width/length of drone (cg to motor)
k_z_arm = 0.2           # [m] influence of single motor on yaw

# aerodynamics
cd_xy = 0.01            # [-] Fx / 0.5 / rho / v_x^2 (copy Fy vs v_y)
cd_z = 0.04             # [-] Fz / 0.5 / rho / v_z^2
cd_axy = 0.02           # [-] Mx / 0.5 / rho / n_x^2 (copy My vs n_y)
cd_az = 0.04            # [-] Mz / 0.5 / rho / n_z^2

# disturbances
std_g_xy_dist = 0.002    # [m/s/s] body frame
std_g_z_dist = 0.002     # [m/s/s] body frame
std_dn_axy_dist = 0.002  # [rad/s/s] body frame
std_dn_az_dist = 0.002   # [rad/s/s] body frame

# sensor noise
std_g_xyz_noise = 0.1   # [m/s/s]
std_dn_axyz_noise = 0.1 # [rad/s/s]
std_a_mag_noise = 0.03  # [-] orientation vector

# motor_max
f_motor_max = 6.5727

# grounding
k_ground = 1e2
d_ground = 5e0
