from dataclasses import dataclass


@dataclass
class PerceptionSettings:
    exclude = []
    filepath: str="configs/c19_perception.yaml"
