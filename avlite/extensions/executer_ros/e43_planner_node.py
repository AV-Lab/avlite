#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy

import logging
log = logging.getLogger(__name__)

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        try:
            self.declare_parameter('planner_name', 'default_value')
            self.declare_parameter('replan_dt', 0.1)
            planner_name = self.get_parameter('planner_name').get_parameter_value().string_value
            replan_dt = self.get_parameter('replan_dt').get_parameter_value().double_value
        except Exception as e:
            self.get_logger().error(f"Error retrieving parameters: {e}")
            return

        self.get_logger().info(f"Using planner: {planner_name} with replan_dt: {replan_dt}")


        self.pub = self.create_publisher(String, 'local_plan', int(1/replan_dt)) 
        self.i = 0
        self.create_timer(replan_dt, self.tick)

    def tick(self):
        msg = String()
        msg.data = f'planner:{self.i}'
        self.pub.publish(msg)
        self.i += 1

def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
