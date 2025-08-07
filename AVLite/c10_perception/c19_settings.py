from dataclasses import dataclass


@dataclass
class PerceptionSettings:
    exclude = []
    filepath: str="configs/c10_perception.yaml"
    # Ego
    max_valocity: float = 30
    max_acceleration: float = 10    
    min_acceleration: float = -20
    max_steering: float = 0.7  # in radians
    min_steering: float = -0.7
    L_f: float = 2.5  # Distance from center of mass to front axle


    # Perception Model
    max_agent_vehicles: int = 12
    grid_size: int = 100  # Size of the occupancy grid -> 100x100



