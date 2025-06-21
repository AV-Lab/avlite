from c10_perception.c11_perception_model import PerceptionModel
from c10_perception.c12_perception_strategy import PerceptionStrategy
import logging
import time

log = logging.getLogger(__name__)



class Perception(PerceptionStrategy):

    def __init__(self, perception_config=None):
        super().__init__(perception_config=perception_config)
        self.pm = PerceptionModel
        self.detector_model = None
        self.tracker_model = None
        self.predictor_model = None
        self.output_mode = perception_config['prediction_mode']
        self.grid = None
        self.bounds = None
        
        self.initialize_models()


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
        if self.output_mode == 'grid':
            grid_steps=100 # grid steps on x and y directions
            ego_location = [self.pm.ego_vehicle.x, self.pm.ego_vehicle.y]
            # self.perception_model.occupancy_grid,  self.perception_model.grid_bounds
            #self.perception_model.occupancy_grid, self.perception_model.grid_bounds 
            self.grid, self.bounds = self.predictor_model.predict(self.pm,output_mode=self.output_mode,ego_location=ego_location,grid_steps=grid_steps)
        else:
            self.prediction_output = self.predictor_model.predict(self.pm,output_mode=self.output_mode)
    
            log.debug(f"prediction_output agents: {len(self.prediction_output)}")


    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None, perception_model=None) :
        """
        Main perception method that combines detection, tracking, and prediction.
        """

        if perception_model is not None and self.detector == 'ground_truth':
            self.pm = perception_model
            log.debug(f"Using ground truth perception model number of agents: {len(self.pm.agent_vehicles)}")
        elif self.detector is not None:
            self.pm = self.detect(rgb_img, depth_img, lidar_data)

        if self.tracker is not None:
            self.track()

        if self.predictor is not None:
            self.predict()

        return  self.prediction_output 

    def reset(self):
        pass
