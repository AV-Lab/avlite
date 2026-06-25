import numpy as np
import copy
import logging
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c39_settings import ControlSettings

log = logging.getLogger(__name__)

class PIDController(ControlStrategy):
    def __init__(self, tj:Optional[TrajectoryTracker]=None, alpha=ControlSettings.c33_pid_alpha, beta=ControlSettings.c33_pid_beta, gamma=ControlSettings.c33_pid_gamma,
                 valpha=ControlSettings.c33_pid_valpha, vbeta=ControlSettings.c33_pid_vbeta, vgamma=ControlSettings.c33_pid_vgamma, pid_lookahead=ControlSettings.c33_pid_lookahead):
        super().__init__(tj)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        self.valpha, self.vbeta, self.vgamma = valpha, vbeta, vgamma
        self.lookahead = pid_lookahead
        
        self.cte_steer = 0
        self.cte_velocity = 0  # Track velocity error

        self.cte_s_sum = 0
        self.cte_v_sum = 0


    def control(self, ego: EgoState, plan: Optional[LocalPlan]=None, control_dt=None) -> ControlCommand:
        if plan is not None:
            self.tj = plan.as_trajectory()
        elif self.tj is None:
            log.warning("Trajectory is not provided. Steering and acceleration set to zero. Please provide a trajectory.")
            return ControlCommand(steer=0, acceleration=0)

        # to deal with fast replanning, need to have a lookahead to the next trajectory
        if self.tj.parent_trajectory is not None:  
            parent = self.tj.parent_trajectory
            sp, dp =  parent.convert_xy_to_sd(ego.x, ego.y)
            sp = sp + self.lookahead
            x, y =  parent.convert_sd_to_xy(sp, dp)
            s, cte = self.tj.convert_xy_to_sd(x, y)
            s_, cte_ = self.tj.convert_xy_to_sd(ego.x, ego.y)
            log.debug(f"CTE with Lookahead: {self.lookahead}, cte: {cte:.2f}, W.O LA cte: {cte_:.2f}")
            # Also update current_wp for local trajectory to get correct target velocity
            self.tj.update_waypoint_by_xy(ego.x, ego.y)
        else:   
            self.tj.update_waypoint_by_xy(ego.x, ego.y)
            s, cte = self.tj.convert_xy_to_sd(ego.x, ego.y)
            # self.past_cte.append(cte)
        

        ##################################
        # Compute the steer control PID
        ##################################

        self.cte_s_sum += self.cte_steer
        # Compute P, I, D components for steering
        P = -self.alpha * cte
        I = -self.beta * self.cte_s_sum
        D = -self.gamma * (cte - self.cte_steer)

        self.cte_steer = cte

        # Compute the steering angle
        steer = P + I + D
        steer = np.clip(steer, self.ego_min_steering, self.ego_max_steering)
        # Logging with formatted string for clarity
        log.debug( f"Steer: {steer:+6.2f} [P={P:+.3f}, I={I:+.3f}, D={D:+.3f}] based on CTE: {cte:+.3f}")
        self.last_steer = steer


        ##################################
        # Compute the velocity control PID
        ##################################
        idx = self.tj.current_wp
        target_velocity = self.tj.velocity[idx]

        prev_cte_v = self.cte_velocity
        self.cte_velocity = ego.velocity - target_velocity
        self.cte_v_sum += self.cte_velocity

        vP = -self.valpha * self.cte_velocity
        vI = -self.vbeta * self.cte_v_sum
        vD = -self.vgamma * (self.cte_velocity - prev_cte_v)  # D-term: rate of change of error

        # Compute the acceleration
        acc = vP + vI + vD
        
        # Emergency braking: if target velocity is 0 (or very low) and we're still moving,
        # apply maximum braking force regardless of PID output
        if target_velocity < ControlSettings.c30_emergency_velocity_threshold and ego.velocity > ControlSettings.c30_emergency_min_moving_velocity:
            # Emergency stop requested - apply max deceleration
            emergency_acc = self.ego_min_acceleration * ControlSettings.c30_emergency_braking_factor  # 90% of max braking
            if acc > emergency_acc:
                log.warning(f"Emergency braking: overriding PID acc {acc:.2f} with {emergency_acc:.2f}")
                acc = emergency_acc
        
        acc = np.clip(acc, self.ego_min_acceleration, self.ego_max_acceleration)

        # Anti-windup: clear integral when stopped so accumulated braking error
        # does not keep pushing the car backwards past zero velocity.
        if ego.velocity <= 0 and self.cte_v_sum > 0:
            self.cte_v_sum = 0.0
        # Velocity floor: never command further deceleration when already at rest.
        if ego.velocity <= 0 and acc < 0:
            acc = 0.0

        log.debug(f"Acc  : {acc:+6.2f} [P={vP:+.3f}, I={vI:+.3f}, D={vD:+.3f}] based on CTE: {self.cte_velocity:+.2f} ({ego.velocity:.2f} vs target: {target_velocity:.2f})")
        cmd = ControlCommand(steer=steer, acceleration=acc)
        self.cmd = cmd
        return cmd

    def reset(self):
        self.cte_s_sum = 0
        self.cte_steer = 0
        self.cte_v_sum = 0
        self.cte_velocity = 0
    
    def get_copy(self):
        return copy.deepcopy(self)
    

