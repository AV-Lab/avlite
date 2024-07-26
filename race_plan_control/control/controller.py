import logging 
from race_plan_control.plan.trajectory import Trajectory
from abc import ABC, abstractmethod
import logging
log = logging.getLogger(__name__)

class Controller(ABC):
    def __init__(self):
        self.last_steer = None
        self.last_acc = None

    @abstractmethod
    def control(self, cte:float, tj:Trajectory=None):
        pass

    @abstractmethod
    def reset():
        pass

class PIDController(Controller):
    def __init__(self, alpha=0.05, beta=0.001, gamma=0.7): 
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.past_cte = []

        self.cte_sum = 0
        self.cte_prev = 0


    def control(self, cte):
        self.past_cte.append(cte)

        # Compute P, I, D components separately
        P = -self.alpha * cte
        if len(self.past_cte) < 2:
            I = 0
            D = 0
        else:
            self.cte_sum += self.cte_prev
            I = -self.beta * self.cte_sum #sum(self.past_cte[-100:])
            D = -self.gamma * (cte - self.cte_prev) # self.past_cte[-2])
        
        self.cte_prev = cte

        # Compute the steering angle
        steer = P + I + D

        # Logging with formatted string for clarity
        log.info(f"Steering Angle: {steer:+.2f} [P={P:+.3f}, I={I:+.3f}, D={D:+.3f}] based on CTE: {cte:+.3f}")
        self.last_steer = steer
        return steer

    def reset(self):
        self.past_cte = []
        self.cte_sum = 0
        self.cte_prev = 0


        
import race_plan_control.main as main
if __name__== "__main__":
    main.run()