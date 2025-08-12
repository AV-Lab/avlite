import logging
import importlib

from c10_perception.c11_perception_model import PerceptionModel
from c10_perception.c19_settings import PerceptionSettings
from extensions.multi_object_prediction.setting import ExtensionSettings
from c10_perception.c12_perception_strategy import PerceptionStrategy

log = logging.getLogger(__name__)


class MultiObjectPredictor(PerceptionStrategy):

    def __init__(self, perception_model: PerceptionModel,):
        super().__init__(perception_model)
        self.supports_prediction = True

        self.detector = ExtensionSettings.detector
        self.tracker = ExtensionSettings.tracker
        self.predictor = ExtensionSettings.predictor
        self.device = ExtensionSettings.device
        self.output_mode = ExtensionSettings.prediction_mode 
        self.max_agent_distance = ExtensionSettings.max_agent_distance 
        self.pm = perception_model
        self.detector_model = None
        self.tracker_model = None
        self.predictor_model = None
        self.grid = None
        self.bounds = None
            
        self.prediction_output = []
        
        log.info(f"Initializing PerceptionStrategy with detector: {self.detector}, tracker: {self.tracker}, predictor: {self.predictor}, device: {self.device}")
        self.initialize_models()

    # def _get_config_value(self, config, key):
    #     """Get config value, treating string 'None' as actual None."""
    #     if not config or key not in config:
    #         return None
    #     value = config[key]
    #     return None if isinstance(value, str) and value.lower() == 'none' else value
    
    def import_models(self, module: str) -> type:
        """
        Dynamically import a model class from the extension package.
        
        Args:
            module: Name of the module/class to import
            
        Returns:
            The imported model class
            
        Raises:
            ImportError: If the module or class cannot be imported
        """
        
        try:
            module_name = f"extensions.multi_object_prediction.e10_perception.{module}"
            imported_module = importlib.import_module(module_name)
            ModelClass = getattr(imported_module, module)
            return ModelClass
        except (ImportError, AttributeError) as e:
            log.error(f"Failed to import {module} from {module_name}: {e}")
            raise ImportError(f"Could not import {module} from {module_name}: {e}")
    
    

    def initialize_models(self):
        """
        Initialize the perception models based on the detector, tracker, and predictor specified in the profile configuration.
        """
       
        if self.detector is not None and self.detector != 'ground_truth':
            log.info(f"Initializing detector: {self.detector}")
            checkpoint = f"data/checkpoints/detector_{self.detector}.pth"
            try:
                DetectorClass = self.import_models(self.detector)
                self.detector_model = DetectorClass(
                    checkpoint_path=checkpoint,
                    device=self.device
                )
                log.info(f"Successfully initialized detector: {self.detector}")
            except Exception as e:
                log.error(f"Failed to initialize detector {self.detector}: {e}")
                self.detector_model = None

        if self.tracker is not None:
            log.info(f"Initializing tracker: {self.tracker}")
            checkpoint = f"data/checkpoints/tracker_{self.tracker}.pth"
            try:
                TrackerClass = self.import_models(self.tracker)
                self.tracker_model = TrackerClass(
                    checkpoint_path=checkpoint,
                    device=self.device
                )
                log.info(f"Successfully initialized tracker: {self.tracker}")
            except Exception as e:
                log.error(f"Failed to initialize tracker {self.tracker}: {e}")
                self.tracker_model = None

        if self.predictor is not None:
            log.info(f"Initializing predictor: {self.predictor}")
            checkpoint = f"data/checkpoints/predictor_{self.predictor}.pth"
            try:
                PredictorClass = self.import_models(self.predictor)
                self.predictor_model = PredictorClass(
                    saving_checkpoint_path=checkpoint,
                    device=self.device, 
                    mode='predict'
                )
                log.info(f"Successfully initialized predictor: {self.predictor}")
            except Exception as e:
                log.error(f"Failed to initialize predictor {self.predictor}: {e}")
                self.predictor_model = None
    

    def detect(self, rgb_img = None, depth_img = None, lidar_data = None):
        pass
    
    def track(self):
        pass

    def predict(self):
        
        if self.pm is None or self.pm.ego_vehicle is None:
            log.debug("Prediction cancelled: Perception model or ego vehicle not initialized")
            return

        self.pm.filter_agent_vehicles(self.max_agent_distance)
        filtered_count = len(self.pm.agent_vehicles)
        log.debug(f"Filtered agent vehicles (≤{self.max_agent_distance}m): {filtered_count}")


        # Handle no available agents
        if not self.pm.agent_vehicles:
            log.debug("No agents available for prediction after filtering")
            self.prediction_output = []
            self.grid = None
            self.bounds = None
            return
        if self.output_mode == 'grid':
            try:
                ego_location = [self.pm.ego_vehicle.x, self.pm.ego_vehicle.y]
                objects_sizes = self.pm.agents_sizes_as_np()
                log.debug(f"Predicting for {len(self.pm.agent_vehicles)} agents, ego_location at : {ego_location} grid size: {self.pm.grid_size}x{self.pm.grid_size}")
                self.pm.occupancy_flow ,self.pm.occupancy_flow_per_object,self.pm.grid_bounds = self.predictor_model.predict(self.pm,output_mode=self.output_mode,sizes=objects_sizes,ego_location=ego_location,grid_steps=self.pm.grid_size)
                log.debug(f"Occupancy grid shape: {self.pm.occupancy_flow.shape}\nOccupancy grid per object shape: {self.pm.occupancy_flow_per_object.shape}\nGrid bounds: {self.pm.grid_bounds}")
                self.prediction_output = self.pm.occupancy_flow
            except Exception as e:
                log.error(f"Error during grid prediction: {e}")
                self.prediction_output = []
        else:
            log.debug(f"Predicting for {len(self.pm.agent_vehicles)} agents, ego_location at : {ego_location}")
            self.pm.trajectories = self.predictor_model.predict(self.pm,output_mode=self.output_mode)
            log.debug(f"Predicted Trajectories:{len(self.pm.trajectories)}")

    # def agents_sizes(self) -> np.ndarray:
    #     """Get the sizes of all agent vehicles."""
    #     return np.array([[agent.length,agent.width] for agent in self.agent_vehicles])

    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None, perception_model=None) :
        """
        Main perception method that combines detection, tracking, and prediction.
        """

        if perception_model is not None and self.detector == 'ground_truth':
            self.pm = perception_model
            # log.debug(f"Using ground truth perception model number of agents: {len(self.pm.agent_vehicles)}")
        elif self.detector is not None:
            self.pm = self.detect(rgb_img, depth_img, lidar_data)

        if self.tracker is not None:
            self.track()

        if self.predictor is not None:
            self.predict()

        return  self.prediction_output 

    def reset(self):
        pass
