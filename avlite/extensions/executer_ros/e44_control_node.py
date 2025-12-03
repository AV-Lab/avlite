#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        try:
            self.declare_parameter('controller_name', 'default_value')
            self.declare_parameter('control_dt', 0.1)
            controller_name = self.get_parameter('controller_name').get_parameter_value().string_value
            control_dt = self.get_parameter('control_dt').get_parameter_value().double_value
        except Exception as e:
            self.get_logger().error(f"Error retrieving parameters: {e}")
            return
        self.get_logger().info(f"Using controller: {controller_name} with control_dt: {control_dt}")
        self.pub = self.create_publisher(String, 'control', int(1/control_dt))
        self.i = 0
        self.create_timer(control_dt, self.tick)

    def tick(self):
        msg = String()
        msg.data = f'control:{self.i}'
        self.pub.publish(msg)
        self.i += 1

def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
