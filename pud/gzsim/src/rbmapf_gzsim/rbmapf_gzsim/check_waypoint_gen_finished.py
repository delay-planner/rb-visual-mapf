import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class WaypointGeneratorNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator_node')
        self.waypoint_generator_subscriber = self.create_subscription(
            Empty,
            '/waypoints_generated',
            self.waypoints_generated_callback,
            10
        )

    def waypoints_generated_callback(self, msg):
        self.get_logger().info('Waypoints generation finished, proceeding with mission setup.')
        sys.exit(0)


def main():
    rclpy.init()
    node = WaypointGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
