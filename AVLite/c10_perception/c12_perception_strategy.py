from c10_perception.c11_perception_model import State, AgentState, EgoState, PerceptionModel
from c20_planning.c28_trajectory import Trajectory
import numpy as np
import logging
from dataclasses import dataclass, field
import importlib
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

class PerceptionStrategy(ABC):
    """
    Abstract base class for perception strategies.
    This class defines the interface for perception strategies, including methods for detection, tracking, prediction, and perception.
    """

    def __init__(self, perception_config):
        self.perception_model: PerceptionModel
        self.detector = self._get_config_value(perception_config, 'detector')
        self.tracker = self._get_config_value(perception_config, 'tracker')
        self.predictor = self._get_config_value(perception_config, 'predictor')
        self.device = self._get_config_value(perception_config, 'device') or 'cpu'
        
        

    def _get_config_value(self, config, key):
        """Get config value, treating string 'None' as actual None."""
        if not config or key not in config:
            return None
        value = config[key]
        return None if isinstance(value, str) and value.lower() == 'none' else value
    
    def import_models(self, module: str) -> type:
        """
        Dynamically import a model class from the c70_extension package.
        
        Args:
            module: Name of the module/class to import
            
        Returns:
            The imported model class
            
        Raises:
            ImportError: If the module or class cannot be imported
        """
        
        try:
            module_name = f"c70_extension.{module}"
            imported_module = importlib.import_module(module_name)
            ModelClass = getattr(imported_module, module)
            return ModelClass
        except (ImportError, AttributeError) as e:
            log.error(f"Failed to import {module} from {module_name}: {e}")
            raise ImportError(f"Could not import {module} from {module_name}: {e}")

    def detect(self, rgb_img=None, depth_img=None, lidar_data=None):
        """
        Detect objects in the environment using the specified detection method.
        """
        raise NotImplementedError("Detection method not implemented.")

    def track(self):
        """
        Track detected objects over time.
        """
        raise NotImplementedError("Tracking method not implemented.")

    def predict(self):
        """
        Predict future states of tracked objects.
        """
        raise NotImplementedError("Prediction method not implemented.")

    @abstractmethod
    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None, perception_model=None):
        """
        Main perception method that combines detection, tracking, and prediction.
        """
        raise NotImplementedError("Perception method not implemented.")
    


