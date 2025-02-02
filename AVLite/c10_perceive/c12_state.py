import numpy as np
from shapely.geometry import Polygon
import copy

import logging

log = logging.getLogger(__name__)


class State:
    x: float
    y: float
    theta: float
    width: float
    length: float

    def __init__(self, x=0.0, y=0.0, theta=-np.pi / 4, width=2.0, length=4.5):
        self.x: float = x
        self.y: float = y
        self.theta: float = theta
        self.width: float = width
        self.length: float = length

        # initial x,y position, useful for reset
        self.__init_x = x
        self.__init_y = y
        self.__init_theta = theta

    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, width={self.width}, length={self.length}"

    def reset(self):
        self.x = self.__init_x
        self.y = self.__init_y
        self.theta = self.__init_theta

    def get_copy(self):
        return copy.deepcopy(self)

    def get_bb_corners(self) -> np.ndarray:
        # Calculate the four corners of the rectangle
        corners_x = np.array(
            [
                -self.length / 2,
                +self.length / 2,
                +self.length / 2,
                -self.length / 2,
            ]
        )
        corners_y = np.array(
            [
                -self.width / 2,
                -self.width / 2,
                +self.width / 2,
                +self.width / 2,
            ]
        )
        # Rotate the corners around the center of the car
        rotation_matrix = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )
        rotated_corners = np.dot(rotation_matrix, np.array([corners_x, corners_y]))

        rotated_corners_x = rotated_corners[0, :] + self.x
        rotated_corners_y = rotated_corners[1, :] + self.y

        return np.c_[rotated_corners_x, rotated_corners_y]

    def get_bb_polygon(self):
        return Polygon(self.get_bb_corners())

    def get_transformed_bb_corners(self, func):
        corners = self.get_bb_corners()
        return np.apply_along_axis(func, 1, corners)


class AgentState(State):
    velocity: float

    def __init__(self, x=0.0, y=0.0, theta=-np.pi / 4, speed=0.0, width=2.0, length=4.5):
        self.velocity: float = speed

        self.__init_speed = 0.0
        super().__init__(x, y, theta, width, length)

    # constant velocity model
    def predict(self, dt):
        self.x += self.velocity * np.cos(self.theta) * dt
        self.y += self.velocity * np.sin(self.theta) * dt

    def reset(self):
        super().reset()
        self.velocity = self.__init_speed

    def __repr__(self):
        return f"VehicleState(x={self.x}, y={self.y}, speed={self.velocity}, theta={self.theta})"


class EgoState(AgentState):
    max_valocity: float
    max_acceleration: float
    min_acceleration: float
    max_steering: float
    min_steering: float
    L_f: float

    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=-np.pi / 4,
        speed=0.0,
        width=2.0,
        length=4.5,
        max_velocity=30,
        max_acceleration=10,
        min_acceleration=-20,
        max_steering=0.7, # in radians
        min_steering=-0.7,
        l_f=2.5,
    ):
        super().__init__(x, y, theta, speed, width, length)

        # car parameters
        self.max_valocity = max_velocity
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.max_steering = max_steering
        self.min_steering = min_steering
        self.L_f = l_f  # Distance from center of mass to front axle

    def __reduce__(self):
            state = {
                'x': self.x,
                'y': self.y,
                'theta': self.theta,
                'width': self.width,
                'length': self.length,
                'velocity': self.velocity,
            }
            return (self.__class__, (), state)

    def __setstate__(self, state):
        self.__dict__.update(state)
        
