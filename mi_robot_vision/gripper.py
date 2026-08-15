#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from xarm.wrapper import XArmAPI

class Lite6GripperNode(Node):
    def __init__(self):
        super().__init__('lite6_gripper_node')
        self.arm = XArmAPI('192.168.1.167')
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        
        self.subscription = self.create_subscription(
            String,
            '/lite6_gripper/command',
            self.gripper_callback,
            10)
        self.get_logger().info('Lite6 Gripper Node listo. Comandos: "open", "close", "stop"')

    def gripper_callback(self, msg):
        cmd = msg.data.lower().strip()
        self.get_logger().info(f'Recibido: {cmd}')
        if cmd == 'open':
            ret = self.arm.open_lite6_gripper()
            self.get_logger().info(f'Open ret: {ret}')
        elif cmd == 'close':
            ret = self.arm.close_lite6_gripper()
            self.get_logger().info(f'Close ret: {ret}')
        elif cmd == 'stop':
            ret = self.arm.stop_lite6_gripper()
            self.get_logger().info(f'Stop ret: {ret}')

def main():
    rclpy.init()
    node = Lite6GripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.arm.disconnect()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
