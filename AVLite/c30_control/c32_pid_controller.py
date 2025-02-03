from c10_perceive.c12_state import EgoState
from c20_plan.c24_trajectory import Trajectory
from c30_control.c31_base_controller import BaseController, ControlComand
import numpy as np
import copy

import logging

log = logging.getLogger(__name__)


class PIDController(BaseController):
    def __init__(self, tj:Trajectory=None, alpha=0.1, beta=0.001, gamma=0.6, valpha=0.8, vbeta=0.01, vgamma=0.3):
        super().__init__(tj)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        self.valpha, self.vbeta, self.vgamma = valpha, vbeta, vgamma

        self.cte_s_sum = 0
        self.cte_v_sum = 0


    def control(self, ego: EgoState, tj: Trajectory = None) -> ControlComand:
        if tj is not None:
            self.tj = tj
        elif tj is None and self.tj is None:
            log.error("Trajectory is not provided")
            return

        s, cte = self.tj.convert_xy_to_sd(ego.x, ego.y)
        # self.past_cte.append(cte)

        self.cte_s_sum += self.cte_steer
        # Compute P, I, D components for steering
        P = -self.alpha * cte
        I = -self.beta * self.cte_s_sum
        D = -self.gamma * (cte - self.cte_steer)

        self.cte_steer = cte

        # Compute the steering angle
        steer = P + I + D
        steer = np.clip(steer, ego.min_steering, ego.max_steering)
        # Logging with formatted string for clarity
        log.debug(
            f"Steer: {steer:+6.2f} [P={P:+.3f}, I={I:+.3f}, D={D:+.3f}] based on CTE: {cte:+.3f}"
        )
        self.last_steer = steer


        # Compute P, I, D components for velocity
        idx = self.tj.current_wp
        target_velocity = self.tj.velocity[idx]
        v_error = ego.velocity - target_velocity
        self.cte_v_sum += self.cte_velocity

        vP = -self.valpha * v_error
        vI = -self.vbeta * self.cte_v_sum
        vD = -self.vgamma * (v_error - self.cte_velocity)

        self.cte_velocity = v_error

        # Compute the acceleration
        acc = vP + vI + vD
        acc = np.clip(acc, ego.min_acceleration, ego.max_acceleration)

        # Logging with formatted string for clarity
        log.debug(
            f"Acc  : {acc:+6.2f} [P={vP:+.3f}, I={vI:+.3f}, D={vD:+.3f}] based on CTE: {v_error:+.2f} ({ego.velocity:.2f} vs target: {target_velocity:.2f})"
        )

        cmd = ControlComand(steer=steer, acc=acc)
        self.cmd = cmd
        return cmd

    def reset(self):
        self.cte_s_sum = 0
        self.cte_steer = 0
        self.cte_v_sum = 0
        self.cte_velocity = 0
    
    def get_copy(self):
        return copy.deepcopy(self)
    

