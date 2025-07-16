import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray

from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control import argument_parser, extract_walls


class WallRVizNode(Node):
    def __init__(self, height=5.0, resolution=1.0):
        super().__init__('wall_rviz_node')
        self.declare_parameter('map_topic', 'wall_markers_3d/walls_3d')

        args = argument_parser()
        self.walls, _ = extract_walls(args)

        self.publish_markers(height, resolution)

    def publish_markers(self, height=5.0, resolution=1.0):
        marker_array = MarkerArray()
        marker_array.markers = []

        # now = self.get_clock().now().to_msg()
        cols, rows = self.walls.shape
        x0, y0 = -(cols * resolution) / 2.0, -(rows * resolution) / 2.0

        idx = 0
        for i, j in zip(*np.where(self.walls == 1)):
            x = x0 + (j + 0.5) * resolution
            y = y0 + (i + 0.5) * resolution
            z = height / 2.0

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'walls_3d'
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0

            marker.scale.x = resolution
            marker.scale.y = resolution
            marker.scale.z = height

            marker.lifetime = Duration(sec=0, nanosec=0)
            marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)

            marker_array.markers.append(marker)
            idx += 1

        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.get_logger().info(f"Map topic: {map_topic}")
        self.create_publisher(MarkerArray, map_topic, 10).publish(marker_array)


def main():
    rclpy.init()
    node = WallRVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
