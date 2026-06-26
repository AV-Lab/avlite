import logging
from typing import Optional
import numpy as np

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c39_settings import ControlSettings, ControlSettingsSchema

log = logging.getLogger(__name__)

class StanleyController(ControlStrategy):
    def __init__(self, tj:Optional[TrajectoryTracker]=None, setting: ControlSettingsSchema = ControlSettings):
        """
        Stanley Controller for trajectory following. The controller also slows down the vehicle if steer CTE is > 0.5

        :param tj: Trajectory to follow.
        :param setting: Control settings the gains/thresholds are read from (defaults to
            the live ``ControlSettings`` singleton; inject a stub for tests).

        Reads (from ``setting``): ``c34_stanley_k`` (steering gain), ``c34_stanley_k_soft``
        (low-speed softening), ``c34_stanley_lookahead``, velocity gains
        ``c34_stanley_valpha``/``vbeta``/``vgamma``, and the slow-down thresholds
        ``c34_stanley_slow_down_cte``/``slow_down_heading_cte``/``slow_down_vel_threshold``.
        """
        super().__init__(tj)
        self.lookahead = setting.c34_stanley_lookahead
        self.k = setting.c34_stanley_k
        self.k_soft = setting.c34_stanley_k_soft
        self.cte_steer = 0
        self.slow_down_cte = setting.c34_stanley_slow_down_cte  # threshold for slowing down based on steering CTE
        self.slow_down_heading_cte = setting.c34_stanley_slow_down_heading_cte
        self.slow_down_vel_threshold = setting.c34_stanley_slow_down_vel_threshold # threshold for slowing down based on steering CTE

        self.valpha, self.vbeta, self.vgamma = setting.c34_stanley_valpha, setting.c34_stanley_vbeta, setting.c34_stanley_vgamma
        self.cte_v_sum = 0
        self.cte_velocity = 0
        self.previous_cte_velocity = 0  # For D-term calculation
        self.previous_heading = None


    def control(self, ego: EgoState, plan: Optional[LocalPlan]=None, control_dt = None) -> ControlCommand:
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

        self.cte_steer = cte

        ##################################
        # Compute the steering: Stanley
        ##################################
            
        heading_error = normalize_angle(self.tj.get_current_heading() - ego.theta)
        log.debug(f"heading error: {heading_error:+6.2f} [tj: {self.tj.get_current_heading():+6.2f}, ego: {ego.theta:+6.2f}]")
        steer1 = heading_error + np.arctan2(self.k * -cte, ego.velocity + self.k_soft)
        log.debug( f"Steer: {steer1:+6.2f} ")
        steer = np.clip(steer1, -self.ego_max_steering, self.ego_max_steering)
        # if steer1 !=  steer:
        #     log.warning(f"Steering angle {steer1:+6.2f} clipped to {steer:+6.2f} due to limits [{ego.min_steering:+6.2f}, {ego.max_steering:+6.2f}]. Heading error: {heading_error:+6.2f} ")


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

        # lower the speed if abs(steer) > 0.5
        if (np.abs(self.cte_steer) > self.slow_down_cte or np.abs(heading_error) > self.slow_down_heading_cte) \
            and ego.velocity > self.slow_down_vel_threshold:
            acc2 = acc - 3 * np.e**np.abs(self.cte_steer)  # reduce acceleration based on steering error
            acc2 = np.clip(acc2, self.ego_min_acceleration, self.ego_max_acceleration)
            log.debug(f"Steering error {self.cte_steer:+6.2f} is large, reducing acceleration from {acc:.2f} to {acc2:.2f}")
            acc = acc2

        log.debug(f"Acc  : {acc:+6.2f} [P={vP:+.3f}, I={vI:+.3f}, D={vD:+.3f}] based on CTE: {self.cte_velocity:+.2f} ({ego.velocity:.2f} vs target: {target_velocity:.2f})")

        cmd = ControlCommand(steer=steer, acceleration=acc)
        self.cmd = cmd
        return cmd

    def reset(self):
        self.cte_v_sum = 0
        self.cte_velocity = 0
        self.previous_cte_velocity = 0


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi
