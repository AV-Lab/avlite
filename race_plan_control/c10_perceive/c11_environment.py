from shapely.geometry import Polygon

from c10_perceive.c12_state import VehicleState, State

class Environment:
    def __init__(self):
        self.static_obstacles:list[State] = []
        self.dynamic_obstacles:list[VehicleState] = []
        self.controlled_vehicles:list[VehicleState] = []

    def is_collision_free(self):
        for v in self.controlled_vehicles:
            for o in self.static_obstacles:
                if v.polygon.intersects(o.polygon):
                    return False
            for o in self.dynamic_obstacles:
                if v.polygon.intersects(o.polygon):
                    return False




