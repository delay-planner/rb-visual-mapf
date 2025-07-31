import os
import yaml
import uuid
import numpy as np
from pathlib import Path
from dotmap import DotMap
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node

from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import argument_parser


class WallSpawnerNode(Node):
    def __init__(self, height=5.0, resolution=1.0):
        super().__init__('wall_spawner_node')

        args = argument_parser()
        habitat = args.visual == 'True'
        if habitat:
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_habitatenv import extract_walls
        else:
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import extract_walls

        result = extract_walls(args)
        walls, starts = result[0], result[1]
        if habitat:
            lower_bounds, _ = result[2]

        suffix = Path(args.sdf_path).suffix.lower()
        with open(args.sdf_path, 'r') as f:
            template_sdf = f.read()
        rows, cols = walls.shape
        x0, y0 = -(cols * resolution) / 2.0, -(rows * resolution) / 2.0

        if not habitat:
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
        else:
            with open(args.config_file, "r") as f:
                config = yaml.safe_load(f)
            config = DotMap(config)
            scene_name = config.env.simulator_settings.scene
            scene_path = Path(args.sdf_path).parent / f"{scene_name}.dae"
            scene_path = os.path.abspath(scene_path)
            world_with_walls = template_sdf.replace(
                "</world>",
                (
                    f"\n  <include>\n"
                    f"   <uri>model://{scene_name}</uri>\n"
                    f"  </include>\n"
                    f"\n</world>"
                )
            )

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

        adjusted_starts = np.array(starts)
        # if habitat:
        #     assert bounds is not None, "Bounds must be provided for Habitat environments."
        #     adjusted_starts[0] = adjusted_starts[0] * 0.4 + lower_bounds[2]
        #     adjusted_starts[1] = adjusted_starts[1] * 0.4 + lower_bounds[0]
        #     print(f"Adjusted starts for Habitat: {adjusted_starts}")
        # else:
        #     adjusted_starts += np.array([x0, y0, 0.0])
        starts_file_path = args.sdf_path.replace(suffix, '_starts.txt')
        np.savetxt(starts_file_path, adjusted_starts, fmt='%.5f', delimiter=',')
        np.save(args.sdf_path.replace(suffix, '_walls_matrix.npy'), walls)

        if habitat:
            bounds_file_path = args.sdf_path.replace(suffix, '_bounds.txt')
            np.savetxt(bounds_file_path, lower_bounds, fmt='%.5f', delimiter=',')
            self.get_logger().info(f"Bounds saved to {bounds_file_path}")


def main():
    rclpy.init()
    node = WallSpawnerNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
