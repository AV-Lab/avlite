from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class ExtensionSettingsSchema(SettingsSchema):
    device: str = Field(default="cuda:0", description="Torch device for ML models (e.g. cuda:0, cpu).")
    max_agent_distance: float = Field(default=50.0, description="Max distance (m) of agents included in prediction.")
    detector: str = Field(default="ground_truth", description="Detector backend name.")
    tracker: str | None = Field(default=None, description="Tracker backend name; None disables tracking.")
    predictor: str = Field(default="AttentionGMM", description="Predictor model name.")
    prediction_mode: str = Field(default="grid", description="Prediction output mode: single, multi, GMM, or grid.")
    pred_horizon: int = Field(default=3, description="Prediction horizon in seconds.")


class ExtensionSettings:
    schema = ExtensionSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/ext_multi_object_predictor.yaml"

    device: str = "cuda:0"
    max_agent_distance: float = 50.0
    detector: str = "ground_truth"
    tracker: str | None = None
    predictor: str = "AttentionGMM"
    prediction_mode: str = "grid"
    pred_horizon: int = 3
