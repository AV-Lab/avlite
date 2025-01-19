import numpy as np


class State:
    def __init__(self, x=0.0, y=0.0, theta=-np.pi / 4, width=2.0, length=4.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width
        self.length = length
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


class AgentState(State):

    def __init__(self, x=0.0, y=0.0, theta=-np.pi / 4, speed=0.0, width=2.0, length=4.5):
        self.speed: float = speed
        self.__init_speed = 0.0
        super().__init__(x, y, theta, width, length)

    # constant velocity model
    def predict(self, dt):
        self.x += self.speed * np.cos(self.theta) * dt
        self.y += self.speed * np.sin(self.theta) * dt

    def reset(self):
        super().reset()
        self.speed = self.__init_speed
    
    def __repr__(self):
        return f"VehicleState(x={self.x}, y={self.y}, speed={self.speed}, theta={self.theta})"


class EgoState(AgentState):
    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=-np.pi / 4,
        speed=0.0,
        width=2.0,
        length=4.5,
        max_speed=30,
        max_acceleration=10,
        max_deceleration=10,
        l_f=2.5,
        max_steering=0.5,
    ):
        super().__init__(x, y, theta, speed, width, length)

        # car parameters
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_steering = max_steering
        self.L_f = l_f  # Distance from center of mass to front axle

