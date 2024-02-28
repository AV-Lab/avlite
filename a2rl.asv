scenario = drivingScenario;

roadCenters = [yasmarinaconverted.x_m, yasmarinaconverted.y_m, zeros([size(yasmarinaconverted, 1) 1])];

road1 = road(scenario,roadCenters, 25);

% add barriers

right_bound = [trackData(:,3),trackData(:,4), zeros(1500,1)];

left_bound = [trackData(:,5),trackData(:,6), zeros(1500,1)];
% barc1 = [1.2030   -7.3260; 11.5399   -5.7949];
barrier(scenario,right_bound(1:10:end,:), 'ClassID',5,'Width',0.1)
barrier(scenario,left_bound(1:10:end,:),'ClassID',5,'Width',0.1)
% add vehicle options
% scenario.barrier
egoVehicle = vehicle(scenario,'ClassID',1,'Position',[0 0 0]);
waypoints = trajMCP;
speed = 30;
smoothTrajectory(egoVehicle,waypoints,speed)

radar = drivingRadarDataGenerator('MountingLocation',[0 0 0]);
camera = visionDetectionGenerator('SensorLocation',[0 0],'Yaw',-180);
%clc

drivingScenarioDesigner(scenario)



%% add barriers
% scenario= load('A2RL_SCENE.mat');
right_bound = [trackData(:,3),trackData(:,4)];
left_bound = [trackData(:,5),trackData(:,6)];


% loop for barrier
barrier(scenario,right_bound,'SegmentGap',0.1)
barrier(scenario,left_bound,'SegmentGap',0.1)
%%

