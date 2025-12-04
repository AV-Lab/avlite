
import queue
import subprocess
import threading
import time
import os
from typing import Any, Dict
import logging

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import Log

from avlite.c40_execution.c41_execution_model import Executer

log = logging.getLogger(__name__)


PLANNER_NODE_FILE = 'e43_planner_node.py'
CONTROL_NODE_FILE = 'e44_control_node.py'

class Collector(Node):
    """ROS2 node that collects data from publishers and puts it in a queue"""
    
    def __init__(self, q: queue.Queue, collection_dt: float = 0.1):
        super().__init__('collector')
        self.q = q

        hz = int(1.0 / collection_dt)
        
        self.create_subscription(String, 'local_plan', self.listener_callback, hz)
        self.create_subscription(String, 'control', self.listener_callback, hz)
        self.create_subscription(Log, '/rosout', self.rosout_callback, hz)
        
    def listener_callback(self, msg):
        """Callback function for receiving messages"""
        self.q.put(msg.data)
    
    def rosout_callback(self, msg):
        # Forward ROS log to Python logging
        level = msg.level
        text = msg.msg
        if level >= 40:  # ERROR
            log.error(text)
        elif level >= 30:  # WARN
            log.warning(text)
        elif level >= 20:  # INFO
            log.info(text)
        else:
            log.debug(text)


class ROSExecuter(Executer):
    """Executer that interfaces with ROS system"""
    
    def __init__(self, collection_dt = 0.1,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collection_dt = collection_dt
        
        self.q = queue.Queue() # used to collect data from ROS collector node

        # Initialize ROS components
        self.collector_node = None
        self.ros_exec = None
        self.control_process = None
        self.plan_process = None

        self.spin_thread = None
        self.ros_started = False
        
        log.info("ROSExecuter initialized")

    def step(self, perception_dt=0.01, control_dt=0.01, replan_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True) -> None:
        """Steps the executer for one time step. This method should be implemented by the specific executer class."""
        
        # Start ROS infrastructure if not already started
        if not self.ros_started:
            self.replan_dt = replan_dt; self.control_dt = control_dt; self.perception_dt = perception_dt
            self._start_ros()
        
        try:
            data = self.q.get_nowait()
            log.info(f"Received data: {data}")
        except queue.Empty:
            pass
        

    def _start_ros(self):
        """Initialize and start ROS components"""
        if self.ros_started:
            return
            
        # Initialize ROS
        if not rclpy.ok():
            rclpy.init()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ###############
        ### Planner ###
        ###############
        
        global_planner_name = self.global_planner.__class__.__name__
        local_planner_name = self.local_planner.__class__.__name__
        # Start the planner and control node 
        planner_path = os.path.join(current_dir,PLANNER_NODE_FILE)
        self.planner_process = subprocess.Popen(
            ['python3', planner_path, '--ros-args','-p', f"global_planner_name:={global_planner_name}", '-p', f"local_planner_name:={local_planner_name}", '-p', f"replan_dt:={self.replan_dt}"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            preexec_fn=os.setsid
        )
        log.debug(f"Started planner node process: {self.planner_process.pid}")
        ###############
        ###############
        ###############

        control_path = os.path.join(current_dir, CONTROL_NODE_FILE)
        controller_name = self.controller.__class__.__name__
        self.control_process = subprocess.Popen( ['python3', control_path, '--ros-args', '-p', f"controller_name:={controller_name}", '-p', f"control_dt:={self.control_dt}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
        log.info(f"Started control node process: {self.control_process.pid}")
        
        # Create ROS node and executor
        # we give dts to ensure correct frequency
        self.collector_node = Collector(self.q, collection_dt = self.collection_dt)
        self.ros_exec = SingleThreadedExecutor()
        self.ros_exec.add_node(self.collector_node)
        
        # Start spinning in separate thread
        self.spin_thread = threading.Thread(target=self.ros_exec.spin, daemon=True)
        self.spin_thread.start()
        
        self.ros_started = True
        log.info("ROS infrastructure started")

    def stop(self):
        """Clean shutdown of ROS components"""
        if self.ros_started:
            # Stop planner process and all its children
            if self.planner_process:
                try:
                    # Kill entire process group to ensure all child processes are terminated
                    os.killpg(os.getpgid(self.planner_process.pid), 15)  # SIGTERM
                    try:
                        self.planner_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown fails
                        os.killpg(os.getpgid(self.planner_process.pid), 9)  # SIGKILL
                        self.planner_process.wait()
                except (ProcessLookupError, OSError):
                    # Process already dead
                    pass
                log.info("Planner node process stopped")
          
            if self.control_process:
                try:
                    # Kill entire process group to ensure all child processes are terminated
                    os.killpg(os.getpgid(self.control_process.pid), 15)  # SIGTERM
                    try:
                        self.control_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown fails
                        os.killpg(os.getpgid(self.control_process.pid), 9)  # SIGKILL
                        self.control_process.wait()
                except (ProcessLookupError, OSError):
                    # Process already dead
                    pass

                log.info("Controller node process stopped")
            
            # Stop executor
            if self.ros_exec:
                self.ros_exec.shutdown()
            
            # Destroy collector node
            if self.collector_node:
                self.collector_node.destroy_node()
            
            # Shutdown ROS
            if rclpy.ok():
                rclpy.shutdown()
            
            # Wait for thread to finish
            if self.spin_thread and self.spin_thread.is_alive():
                self.spin_thread.join(timeout=1.0)
            
            self.ros_started = False
            log.info("ROS 2 infrastructure stopped")
