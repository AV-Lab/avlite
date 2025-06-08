import logging
from typing import Optional
import numpy as np

from c10_perception.c11_perception_model import EgoState
from c20_planning.c28_trajectory import Trajectory
from c30_control.c31_control_model import ControlComand
from c30_control.c32_control_strategy import ControlStrategy
from c30_control.c39_settings import ControlSettings

log = logging.getLogger(__name__)

class StanleyController(ControlStrategy):
    def __init__(self, tj:Optional[Trajectory]=None, k=ControlSettings.k, k_soft = ControlSettings.k_soft,
                 lookahead=ControlSettings.lookahead, valpha=ControlSettings.valpha, vbeta=ControlSettings.vbeta,
                 vgamma=ControlSettings.vgamma, slow_down_cte=ControlSettings.slow_down_cte, 
                 slow_down_vel_threshold=ControlSettings.slow_down_vel_threshold):
        """
        Stanley Controller for trajectory following. The controller also slows down the vehicle if steer CTE is > 0.5
        :param tj: Trajectory to follow.
        :param k: Gain for steering control.
        :param k_soft: Softening factor for steering control (at low speed).
        :param lookahead: Lookahead distance for trajectory following.
        :param valpha, vbeta, vgamma: Parameters for velocity control (not used in this implementation).
        :param slow_down_cte: Threshold for slowing down based on steering CTE.
        :param slow_down_vel_threshold: Threshold for slowing down based on steering CTE.
        """
        super().__init__(tj)
        self.lookahead = lookahead
        self.k = k
        self.k_soft = k_soft
        self.cte_steer = 0
        self.slow_down_cte = slow_down_cte  # threshold for slowing down based on steering CTE
        self.slow_down_vel_threshold = slow_down_vel_threshold # threshold for slowing down based on steering CTE
        
        self.valpha, self.vbeta, self.vgamma = valpha, vbeta, vgamma
        self.cte_v_sum = 0
        self.cte_velocity = 0
        self.previous_heading = None


    def control(self, ego: EgoState, tj: Optional[Trajectory]=None, control_dt = None) -> ControlComand:
        if tj is not None:
            self.tj = tj
        elif tj is None and self.tj is None:
            log.warning("Trajectory is not provided. Steering and acceleration set to zero. Please provide a trajectory.")
            return ControlComand(steer=0, acceleration=0)

        # to deal with fast replanning, need to have a lookahead to the next trajectory
        if self.tj.parent_trajectory is not None:  
            parent = self.tj.parent_trajectory
            sp, dp =  parent.convert_xy_to_sd(ego.x, ego.y)
            sp = sp + self.lookahead
            x, y =  parent.convert_sd_to_xy(sp, dp)
            s, cte = self.tj.convert_xy_to_sd(x, y)
            s_, cte_ = self.tj.convert_xy_to_sd(ego.x, ego.y)
            log.debug(f"CTE with Lookahead: {self.lookahead}, cte: {cte:.2f}, W.O LA cte: {cte_:.2f}")
            # cte = cte_
        else:   
            s, cte = self.tj.convert_xy_to_sd(ego.x, ego.y)

        self.cte_steer = cte

        ##################################
        # Compute the steering: Stanley
        ##################################
            
        heading_error = normalize_angle(self.tj.get_current_heading() - ego.theta)
        log.debug(f"heading error: {heading_error:+6.2f} [tj: {self.tj.get_current_heading():+6.2f}, ego: {ego.theta:+6.2f}]")
        steer1 = heading_error + np.arctan2(self.k * -cte, ego.velocity + self.k_soft)
        log.debug( f"Steer: {steer1:+6.2f} ")
        steer = np.clip(steer1, -ego.max_steering, ego.max_steering)
        # if steer1 !=  steer:
        #     log.warning(f"Steering angle {steer1:+6.2f} clipped to {steer:+6.2f} due to limits [{ego.min_steering:+6.2f}, {ego.max_steering:+6.2f}]. Heading error: {heading_error:+6.2f} ")


        ##################################
        # Compute the velocity control PID
        ##################################
        idx = self.tj.current_wp
        target_velocity = self.tj.velocity[idx]

        self.cte_velocity = ego.velocity - target_velocity
        self.cte_v_sum += self.cte_velocity

        vP = -self.valpha * self.cte_velocity
        vI = -self.vbeta * self.cte_v_sum
        vD = -self.vgamma * self.cte_velocity

        # Compute the acceleration
        acc = vP + vI + vD
        acc = np.clip(acc, ego.min_acceleration, ego.max_acceleration)

        # lower the speed if abs(steer) > 0.5
        if np.abs(self.cte_steer) > self.slow_down_cte and ego.velocity > self.slow_down_vel_threshold:
            acc2 = acc - 3 * np.e**np.abs(self.cte_steer)  # reduce acceleration based on steering error
            acc2 = np.clip(acc2, ego.min_acceleration, ego.max_acceleration)
            log.info(f"Steering error {self.cte_steer:+6.2f} is large, reducing acceleration from {acc:.2f} to {acc2:.2f}")
            acc = acc2

        log.debug(f"Acc  : {acc:+6.2f} [P={vP:+.3f}, I={vI:+.3f}, D={vD:+.3f}] based on CTE: {self.cte_velocity:+.2f} ({ego.velocity:.2f} vs target: {target_velocity:.2f})")

        cmd = ControlComand(steer=steer, acceleration=acc)
        self.cmd = cmd
        return cmd

    def reset(self):
        self.cte_v_sum = 0
        self.cte_velocity = 0


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi
