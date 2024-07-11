import matplotlib.pyplot as plt
import numpy as np
import pytest
import sys

sys.path.append("../race_plan_control/")
from race_plan_control.util.simulate import Car


@pytest.fixture
def car():
    return Car(x=0, y=0, theta=0, speed=0)

def test_car_step(car):
    dt = 0.05
    acceleration = 5
    steering_angle = np.pi/4

    x_values = [car.x]
    y_values = [car.y]

    for _ in range(70):
        car.step(dt, acceleration, steering_angle)
        x_values.append(car.x)
        y_values.append(car.y)

    plt.plot(x_values, y_values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Car Path')
    plt.show()



if __name__ == '__main__':
    pytest.main()