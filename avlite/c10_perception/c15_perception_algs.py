import numpy as np

from avlite.c10_perception.c11_perception_model import PerceptionModel, PredictionMode
from avlite.c10_perception.c12_perception_strategy import PredictionStrategy
from avlite.c10_perception.c19_settings import PerceptionSettings


class ConstantVelocityPrediction(PredictionStrategy):
    """Predict each agent's future positions assuming constant velocity.

    Writes results into ``pm.trajectories`` (shape ``[n_agents, n_steps, 2]``)
    and sets ``pm.prediction_mode = PredictionMode.TRAJECTORY``.
    """

    @property
    def requirements(self):
        return set()

    def predict(self, perception_model: PerceptionModel) -> PerceptionModel:
        agents = perception_model.agent_vehicles
        if not agents:
            perception_model.prediction_mode = PredictionMode.TRAJECTORY
            perception_model.trajectories = np.empty((0, 0, 2))
            return perception_model

        dt = perception_model.predict_delta_t
        horizon = PerceptionSettings.prediction_horizon
        n_steps = max(1, int(round(horizon / dt)))

        trajectories = np.empty((len(agents), n_steps, 2))
        for i, agent in enumerate(agents):
            for t in range(n_steps):
                time = (t + 1) * dt
                trajectories[i, t, 0] = agent.x + agent.velocity * np.cos(agent.theta) * time
                trajectories[i, t, 1] = agent.y + agent.velocity * np.sin(agent.theta) * time

        perception_model.trajectories = trajectories
        perception_model.prediction_mode = PredictionMode.TRAJECTORY
        return perception_model
