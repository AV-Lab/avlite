
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

from avlite.c40_execution.c41_execution_model import Executer

log = logging.getLogger(__name__)


def launch_desc():
    """Get the path for the planner node script"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    planner_node_path = os.path.join(current_dir, 'e43_planner_node.py')
    control_node_path = os.path.join(current_dir, 'e44_control_node.py')
    return planner_node_path, control_node_path


class Collector(Node):
    """ROS2 node that collects data from publishers and puts it in a queue"""
    
    def __init__(self, q: queue.Queue):
        super().__init__('collector')
        self.q = q
        
        plan_subscription = self.create_subscription( String, 'topic1', self.listener_callback, 10)
        control_subscription = self.create_subscription( String, 'topic2', self.listener_callback, 10)
        
    def listener_callback(self, msg):
        """Callback function for receiving messages"""

        self.q.put(msg.data)
        self.get_logger().info(f'Received: "{msg.data}"')


class ROSExecuter(Executer):
    """Executer that interfaces with ROS system"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize ROS components
        self.q = queue.Queue()
        self.collector_node = None
        self.exec = None
        self.control_process = None
        self.plan_process = None
        self.spin_thread = None
        self.ros_started = False
        
        log.info("ROSExecuter initialized")

    def step(self, control_dt=0.01, replan_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True) -> None:
        """Steps the executer for one time step. This method should be implemented by the specific executer class."""
        
        # Start ROS infrastructure if not already started
        if not self.ros_started:
            self._start_ros()
        
        # Read from publisher queue
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
        
        # Start the planner and control node 
        planner_path, control_path  = launch_desc()
        self.planner_process = subprocess.Popen( ['python3', planner_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
        log.info(f"Started planner node process: {self.planner_process.pid}")
        self.control_process = subprocess.Popen( ['python3', control_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
        log.info(f"Started control node process: {self.control_process.pid}")
        
        # Create ROS node and executor
        self.collector_node = Collector(self.q)
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.collector_node)
        
        # Start spinning in separate thread
        self.spin_thread = threading.Thread(target=self.exec.spin, daemon=True)
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
            if self.exec:
                self.exec.shutdown()
            
            # Destroy node
            if self.collector_node:
                self.collector_node.destroy_node()
            
            # Shutdown ROS
            if rclpy.ok():
                rclpy.shutdown()
            
            # Wait for thread to finish
            if self.spin_thread and self.spin_thread.is_alive():
                self.spin_thread.join(timeout=1.0)
            
            self.ros_started = False
            log.info("ROS infrastructure stopped")
