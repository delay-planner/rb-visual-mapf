import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty
from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control import argument_parser, generate_wps
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


class WaypointGeneratorNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator_node')
        args = argument_parser()
        agents_waypoints = generate_wps(args, debug=True)

        self.waypoint_publishers = []
        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        for agent_idx in range(args.num_agents):
            instance_idx = agent_idx + 1
            waypoint_publisher = self.create_publisher(Float32MultiArray, f'/px4_{instance_idx}/waypoints', qos_profile)
            agent_wps = np.array(agents_waypoints[agent_idx])
            message = Float32MultiArray(data=agent_wps.flatten().tolist())
            waypoint_publisher.publish(message)
            self.waypoint_publishers.append(waypoint_publisher)

        self.start_pub = self.create_publisher(Empty, '/start_mission', qos_profile)
        self.start_timer = self.create_timer(1.0, self.publish_start)

    def publish_start(self):
        self.start_pub.publish(Empty())
        self.get_logger().info('Mission start published')
        self.start_timer.cancel()


def main():
    rclpy.init()
    node = WaypointGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
