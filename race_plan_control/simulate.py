from math import cos, sin

class Car:
    def __init__(self, x=0,y=0, theta=0, speed = 0, max_speed=30, max_acceleration=10, max_deceleration=10, l_f=2.5, width=2.0, length=4.5):
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration

        # car parameters
        self.L_f = l_f # Distance from center of mass to front axle
        self.width = width
        self.length = length

        # Vehicle state
        self.x, self.y = x,y
        self.speed = speed
        self.theta = theta # Heading angle

    def set_state(self, x = None, y = None, theta=None, speed=None):
        self.x = x if x is not None else self.x 
        self.y = y if y is not None else self.y
        self.theta = theta if theta is not None else self.theta
        self.speed = speed if speed is not None else self.speed

    def step(self, dt=0.01, acceleration=0, steering_angle=0):
        self.x += self.speed * cos(self.theta) * dt
        self.y += self.speed * sin(self.theta) * dt
        self.speed += acceleration * dt
        self.theta += self.speed/self.L_f * steering_angle * dt

    def print_state(self):
        print(f"x: {self.x}, y: {self.y}, theta: {self.theta}, speed: {self.speed}")

