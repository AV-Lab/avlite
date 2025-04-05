from dataclasses import dataclass


@dataclass
class ExecutionSettings:
    exclude = []
    filepath: str="configs/c49_settings.yaml"

