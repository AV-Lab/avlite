import numpy as np


class State:
    def __init__(self, x=0.0, y=0.0, width=2.0, length=4.5):
        self.x = x
        self.y = y
        self.width = width
        self.length = length

        # initial x,y position, useful for reset
        self.init_x = x
        self.init_y = y

    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, width={self.width}, length={self.length}"

    def get_corners(self) -> np.ndarray:

        # Calculate the four corners of the rectangle
        corners_x = np.array(
            [
                self.x - self.length / 2,
                self.x + self.length / 2,
                self.x + self.length / 2,
                self.x - self.length / 2,
            ]
        )
        corners_y = np.array(
            [
                self.y - self.width / 2,
                self.y - self.width / 2,
                self.y + self.width / 2,
                self.y + self.width / 2,
            ]
        )
        return np.c_[corners_x, corners_y]


class VehicleState(State):
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
        super().__init__(x, y, width, length)
        self.theta = theta
        self.speed = speed

        self.init_theta = theta
        self.init_speed = speed

        # car parameters
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_steering = max_steering
        self.L_f = l_f  # Distance from center of mass to front axle

    def get_corners(self) -> np.ndarray:

        # Calculate the four corners of the rectangle
        corners_x = np.array(
            [
                 - self.length / 2,
                 + self.length / 2,
                 + self.length / 2,
                 - self.length / 2,
            ]
        )
        corners_y = np.array(
            [
                 - self.width / 2,
                 - self.width / 2,
                 + self.width / 2,
                 + self.width / 2,
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

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.theta = self.init_theta
        self.speed = self.init_speed

    def __repr__(self):
        return f"VehicleState(x={self.x}, y={self.y}, speed={self.speed}, theta={self.theta})"
