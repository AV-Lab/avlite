scenario = drivingScenario;
roadCenters = [0 0 0; 50 0 0];
road(scenario,roadCenters);

egoVehicle = vehicle(scenario,'ClassID',1,'Position',[5 0 0]);
waypoints = [5 0 0; 45 0 0];
speed = 30;
smoothTrajectory(egoVehicle,waypoints,speed)


