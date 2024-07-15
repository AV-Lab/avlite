import logging 
from abc import ABC, abstractmethod
import logging


class Controller:
    class Controller(ABC):

        @abstractmethod
        def control(self, cte):
            pass

class PIDController(Controller):

    def __init__(self, alpha=0.05, beta=0.001, gamma=0.7): 
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.past_cte = []


    def control(self, cte):
        self.past_cte.append(cte)

        # Compute P, I, D components separately
        P = -self.alpha * cte
        if len(self.past_cte) < 2:
            I = 0
            D = 0
        else:
            I = -self.beta * sum(self.past_cte[-100:])
            D = -self.gamma * (cte - self.past_cte[-2])

        # Compute the steering angle
        steer = P + I + D

        # Logging with formatted string for clarity
        logging.info(f"Steering Angle: {steer:+.2f} [P={P:+.3f}, I={I:+.3f}, D={D:+.3f}] based on CTE: {cte:+.3f}")

        return steer

        
