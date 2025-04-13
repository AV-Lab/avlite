from dataclasses import dataclass


@dataclass
class PlanningSettings:
    exclude = []
    filepath: str="configs/c29_planning.yaml"
    
    global_trajectory:str = "data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    hd_map:str = "data/Town10HD_Opt.xodr"
    

