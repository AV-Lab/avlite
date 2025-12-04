class ExtensionSettings:
    exclude = ["exclude", "filepath"] # attributes to exclude from saving/loading
    filepath: str="configs/ext_multi_object_predictor.yaml"
    
    device: str = "cuda:0" #cpu
    max_agent_distance: float =  50.0 # max distance (memters) of agents to be considered in the prediction 
    detector: str =  "ground_truth"
    tracker:str =  "None"
    predictor: str = "AttentionGMM"
    prediction_mode: str =  "grid"  # single multi GMM or grid
    pred_horizon: int = 3 # this is property of predictor -> how many secs in the future it predicts
