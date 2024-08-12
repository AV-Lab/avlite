import numpy as np


class VehicleState:
    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=-np.pi / 4,
        speed=0,
        max_speed=30,
        max_acceleration=10,
        max_deceleration=10,
        l_f=2.5,
        width=2.0,
        length=4.5,
        max_steering=0.5,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed

        self.init_x = x
        self.init_y = y
        self.init_theta = theta
        self.init_speed = speed

        # TODO this should be read from a config file
        # car parameters
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_steering = max_steering
        self.L_f = l_f  # Distance from center of mass to front axle
        self.width = width
        self.length = length

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.theta = self.init_theta
        self.speed = self.init_speed

    def __repr__(self):
        return f"VehicleState(x={self.x}, y={self.y}, speed={self.speed}, theta={self.theta})"
