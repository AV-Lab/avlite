# This repository helps to achieve  the following four tasks:

## 1- Trajectory optimization and velocity profile generation

- Go to the trajectory optimization folder
- you will find everything documented based on the original repository 
- you can reuse the  code easily, just start by  working the raceLineOptimizer.mlx file
- this is the entry file for the code, from there, it calls other script files (1- minimum curvature path generation 2- shortest path generation 3- velocity profile generation)
 
-  the reference for this part can be found here: MW208 Raceline Optimization
- https://www.youtube.com/watch?v=Q-djflXTJGE
## 2- Visualizing the racetrack and the vehicle using the optimized trajectory
- open MATLAB APP scenario design and load the scenario file yasMarinaCircuit.mat
- you can visualize the race track and the optimal trajectory designed in step 1
- in order to change the wayponts and generate the scenario .mat file again, use the file a2rl.m


## 3- Controller Design 

- use the folder vehicle stanely controller, all comments are there in the readme
- you can also check the original reference:  https://www.youtube.com/watch?v=FHQFya0-JBs
- for model predictive control, you can also check the reference: https://youtu.be/SzEg_C-XJ14?si=pVpcZJzYnKw_oKkE

## 4- ROS Node
#### after you optimize the trajectory and designed a controller with proper parameters of the vehicle, now you need to have a ROS node
  
- reference 1: https://www.youtube.com/watch?v=BPj1bsnlDcQ&t=681s
- reference 2: https://www.mathworks.com/help/ros/ug/generate-a-standalone-ros-node-from-simulink.html
- reference 3: https://www.youtube.com/watch?v=EzYYy6ZSvZI

