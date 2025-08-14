import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty
from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import argument_parser
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


class WaypointGeneratorNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator_node')
        self.declare_parameter('interface', 'px4')
        args = argument_parser()
        habitat = args.visual == 'True'

        if habitat:
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_habitatenv import generate_wps
        else:
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import generate_wps

        agents_waypoints = generate_wps(args, debug=False)

        self.waypoint_publishers = []
        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        interface = self.get_parameter('interface').get_parameter_value().string_value
        for agent_idx in range(args.num_agents):
            instance_idx = agent_idx + 1
            if interface != "cf":
                wp_topic = f"/{interface}_{instance_idx}/waypoints"
            else:
                wp_topic = f"/{interface}{instance_idx}/waypoints"
            waypoint_publisher = self.create_publisher(
                Float32MultiArray,
                wp_topic,
                qos_profile
            )
            agent_wps = np.array(agents_waypoints[agent_idx])
            message = Float32MultiArray(data=agent_wps.flatten().tolist())
            waypoint_publisher.publish(message)
            self.waypoint_publishers.append(waypoint_publisher)

        self.waypoint_gen_finished = self.create_publisher(Empty, '/waypoints_generated', qos_profile)
        self.waypoints_gen_finished_timer = self.create_timer(1.0, self.publish_waypoints_gen_finished)

    def publish_waypoints_gen_finished(self):
        self.waypoint_gen_finished.publish(Empty())
        self.get_logger().info('Waypoints generation finished published')
        self.waypoints_gen_finished_timer.cancel()


def main():
    rclpy.init()
    node = WaypointGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
