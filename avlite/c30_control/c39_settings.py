import numpy as np

class ControlSettings:
    exclude = ["exclude"]
    filepath: str = "configs/c30_control.yaml"

    # c33 PID controller
    c33_pid_alpha = 0.01
    c33_pid_beta = 0.001
    c33_pid_gamma = 0.6
    c33_pid_valpha = 0.8
    c33_pid_vbeta = 0.01
    c33_pid_vgamma = 0.3
    c33_pid_lookahead = 2

    # c34 Stanley controller
    c34_stanley_k = 5
    c34_stanley_k_soft = 0.01
    c34_stanley_lookahead = 5
    c34_stanley_valpha = 0.8
    c34_stanley_vbeta = 0.01
    c34_stanley_vgamma = 0.3
    c34_stanley_slow_down_cte = 0.5
    c34_stanley_slow_down_heading_cte = np.pi / 6
    c34_stanley_slow_down_vel_threshold = 3

    # c30 shared emergency braking (c33 + c34)
    c30_emergency_velocity_threshold: float = 0.5
    c30_emergency_min_moving_velocity: float = 1.0
    c30_emergency_braking_factor: float = 0.9

    # c32 ego vehicle kinematic constraints
    c32_ego_distance_front_axle: float = 2.5
    c32_ego_max_velocity: float = 30.0
    c32_ego_max_acceleration: float = 10.0
    c32_ego_min_acceleration: float = -20.0
    c32_ego_max_steering: float = 0.7
    c32_ego_min_steering: float = -0.7
