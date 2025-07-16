import uuid
import rclpy
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

from rclpy.node import Node

from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control import argument_parser, extract_walls


class WallSpawnerNode(Node):
    def __init__(self, height=5.0, resolution=1.0):
        super().__init__('wall_spawner_node')

        args = argument_parser()
        walls, starts = extract_walls(args)

        suffix = Path(args.sdf_path).suffix.lower()
        with open(args.sdf_path, 'r') as f:
            template_sdf = f.read()

        rows, cols = walls.shape
        x0, y0 = -(cols * resolution) / 2.0, -(rows * resolution) / 2.0
        obstacles = []
        for i, j in zip(*np.where(walls == 1)):
            x = x0 + (j + 0.5)*resolution
            y = y0 + (i + 0.5)*resolution

            name = f"wall_{i}_{j}_{uuid.uuid4().hex[:6]}"

            sdf = f"""
            <model name='{name}'>
                <pose>{x:.3f} {y:.3f} {height/2:.3f} 0 0 0</pose>
                <static>true</static>
                <link name='link_{name}'>
                <collision name='col_{name}'>
                    <geometry>
                        <box>
                            <size>{resolution:.3f} {resolution:.3f} {height:.3f}</size>
                        </box>
                    </geometry>
                </collision>
                <visual name='vis_{name}'>
                    <geometry>
                        <box>
                            <size>{resolution:.3f} {resolution:.3f} {height:.3f}</size>
                        </box>
                    </geometry>
                    <material>
                        <ambient>0.3 0.3 0.3 1</ambient>
                        <diffuse>0.7 0.7 0.7 1</diffuse>
                        <specular>1 1 1 1</specular>
                    </material>
                </visual>
                </link>
                <self_collide>false</self_collide>
            </model>"""
            obstacles.append(sdf)

        world_with_walls = template_sdf.replace("</world>", "\n" + "\n".join(obstacles) + "\n</world>")
        output_sdf_path = args.sdf_path.replace(suffix, '_walls' + suffix)
        Path(output_sdf_path).write_text(world_with_walls)

        tree = ET.parse(output_sdf_path)
        root = tree.getroot()
        world = root.find('world')
        if world is None:
            self.get_logger().error("No world element found in the SDF file.")
            return
        current_name = world.get('name')
        if current_name is None:
            self.get_logger().error("No name attribute found in the world element.")
            return
        world.set('name', current_name + '_walls')
        tree.write(output_sdf_path)

        adjusted_starts = np.array(starts) + np.array([x0, y0, 0.0])
        starts_file_path = args.sdf_path.replace(suffix, '_starts.txt')
        np.savetxt(starts_file_path, adjusted_starts, fmt='%.5f', delimiter=',')
        np.save(args.sdf_path.replace(suffix, '_walls_matrix.npy'), walls)


def main():
    rclpy.init()
    node = WallSpawnerNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
