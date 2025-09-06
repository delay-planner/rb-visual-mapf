import os
from pathlib import Path
from dotmap import DotMap
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import argument_parser


class WallRVizNode(Node):
    def __init__(self, height=5.0, resolution=1.0):
        super().__init__('wall_rviz_node')
        self.declare_parameter('map_topic', 'wall_markers_3d/walls_3d')
        self.declare_parameter('use_hardware', False)

        args = argument_parser()
        habitat = args.visual == 'True'

        if habitat:
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_habitatenv import extract_walls
        else:
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import extract_walls

        result = extract_walls(args)
        self.walls, _ = result[0], result[1]
        self.qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        use_hardware = self.get_parameter('use_hardware').get_parameter_value().bool_value
        self.scale_factor = np.array([self.walls.shape[1] / 6., self.walls.shape[0] / 8.]) if use_hardware else np.array([1.0, 1.0])

        marker_array = self.publish_markers(height, resolution) if not habitat else self.publish_mesh_markers(args)

        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.get_logger().info(f"Map topic: {map_topic}")
        self.create_publisher(MarkerArray, map_topic, self.qos_profile).publish(marker_array)

    def publish_markers(self, height=5.0, resolution=1.0):
        marker_array = MarkerArray()
        marker_array.markers = []

        rows, cols = self.walls.shape
        x0, y0 = -(cols * resolution) / 2.0, -(rows * resolution) / 2.0

        idx = 0
        for i, j in zip(*np.where(self.walls == 1)):
            # x = x0 + (j + 0.5) * resolution
            # y = y0 + (i + 0.5) * resolution
            x = (x0 + (j + 0.5)*resolution) / self.scale_factor[0]
            y = (y0 + (i + 0.5)*resolution) / self.scale_factor[1]
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

            marker.scale.x = float(resolution / self.scale_factor[0])
            marker.scale.y = float(resolution / self.scale_factor[1])
            marker.scale.z = height

            marker.lifetime = Duration(sec=0, nanosec=0)
            marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)

            marker_array.markers.append(marker)
            idx += 1

        return marker_array

    def publish_mesh_markers(self, args):

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "stage"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE

        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
        config = DotMap(config)
        scene_name = config.env.simulator_settings.scene
        scene_path = Path(args.sdf_path).parent / f"{scene_name}/meshes/{scene_name}.dae"
        scene_path = os.path.abspath(scene_path)
        marker.mesh_resource = f"file://{scene_path}"

        marker.mesh_use_embedded_materials = True

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = marker.scale.y = marker.scale.z = 1.0

        marker_array = MarkerArray()
        marker_array.markers = []
        marker_array.markers.append(marker)

        return marker_array


def main():
    rclpy.init()
    node = WallRVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
