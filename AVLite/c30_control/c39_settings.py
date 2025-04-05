from dataclasses import dataclass


@dataclass
class ControlSettings:
    exclude = []
    filepath: str="configs/c39_control.yaml"

