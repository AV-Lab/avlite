#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Pub2(Node):
    def __init__(self):
        super().__init__('topic2')
        self.pub = self.create_publisher(String, 'topic2', 10)
        self.i = 0
        self.create_timer(0.7, self.tick)

    def tick(self):
        msg = String()
        msg.data = f'topic2:{self.i}'
        self.pub.publish(msg)
        self.i += 1

def main():
    rclpy.init()
    node = Pub2()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
