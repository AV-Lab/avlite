from c10_perceive.c12_state import EgoState
from c20_plan.c24_trajectory import Trajectory
from c30_control.c31_base_controller import BaseController, ControlComand
import numpy as np

import logging

log = logging.getLogger(__name__)


class PIDController(BaseController):
    def __init__(self, alpha=0.05, beta=0.001, gamma=0.7, valpha=0.4, vbeta=0.3, vgamma=0.5):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        self.valpha, self.vbeta, self.vgamma = valpha, vbeta, vgamma

        self.cte_sum = 0
        self.cte_prev = 0

        self.v_error_sum = 0
        self.v_error_prev = 0

    def control(self, ego: EgoState, tj: Trajectory = None) -> ControlComand:
        if tj is not None:
            self.tj = tj
        elif tj is None and self.tj is None:
            raise ValueError("Trajectory is not provided")

        s, cte = self.tj.convert_xy_to_sd(ego.x, ego.y)
        # self.past_cte.append(cte)

        self.cte_sum += self.cte_prev
        # Compute P, I, D components for steering
        P = -self.alpha * cte
        I = -self.beta * self.cte_sum
        D = -self.gamma * (cte - self.cte_prev)

        self.cte_prev = cte

        # Compute the steering angle
        steer = P + I + D
        steer = np.clip(steer, -ego.max_steering, ego.max_steering)
        # Logging with formatted string for clarity
        log.debug(
            f"Steer: {steer:+6.2f} [P={P:+.3f}, I={I:+.3f}, D={D:+.3f}] based on CTE: {cte:+.3f}"
        )
        self.last_steer = steer


        # Compute P, I, D components for velocity
        idx = self.tj.current_wp
        target_velocity = self.tj.velocity[idx]
        v_error = ego.velocity - target_velocity
        self.v_error_sum += self.v_error_prev

        vP = -self.valpha * v_error
        vI = -self.vbeta * self.v_error_sum
        vD = -self.vgamma * (v_error - self.v_error_prev)

        self.v_error_prev = v_error

        # Compute the acceleration
        acc = vP + vI + vD
        acc = np.clip(acc, -ego.max_deceleration, ego.max_acceleration)

        # Logging with formatted string for clarity
        log.debug(
            f"Acc  : {acc:+6.2f} [P={vP:+.3f}, I={vI:+.3f}, D={vD:+.3f}] based on CTE: {v_error:+.2f} ({ego.velocity:.2f} vs target: {target_velocity:.2f})"
        )

        cmd = ControlComand(steer=steer, acc=acc)
        return cmd

    def reset(self):
        self.cte_sum = 0
        self.cte_prev = 0
        self.v_error_sum = 0
        self.v_error_prev = 0
