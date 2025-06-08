
class ControlSettings:
    exclude = []
    filepath: str="configs/c39_control.yaml"

    # PID controller settings
    alpha=0.01
    beta=0.001
    gamma=0.6
    pid_valpha=0.8
    pid_vbeta=0.01
    pid_vgamma=0.3
    pid_lookahead=2


    # Stanley controller controller setting
    k=5
    k_soft = 0.01
    lookahead=4
    valpha=0.8
    vbeta=0.01
    vgamma=0.3
    slow_down_cte = 0.5  # threshold for slowing down based on steering CTE
    slow_down_vel_threshold = 2 # threshold for slowing down based on steering CTE
