import numpy as np

class ControlSettings:
    exclude = []
    filepath: str="configs/c30_control.yaml"

    # PID controller settings
    pid_alpha=0.01
    pid_beta=0.001
    pid_gamma=0.6
    pid_valpha=0.8
    pid_vbeta=0.01
    pid_vgamma=0.3
    pid_lookahead=2


    # Stanley controller controller setting
    stanley_k=5
    stanley_k_soft = 0.01
    stanley_lookahead=5
    stanley_valpha=0.8
    stanley_vbeta=0.01
    stanley_vgamma=0.3
    stanley_slow_down_cte = 0.5  # threshold for slowing down based on steering CTE
    stanley_slow_down_heading_cte = np.pi / 6  # threshold for slowing down based on heading CTE
    stanley_slow_down_vel_threshold = 3 # threshold for slowing down based on steering CTE
