import os
from pathlib import Path

import numpy as np
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, RegisterEventHandler, TimerAction


def spawn_processes(context, *args, **kwargs):
    px4_bin = LaunchConfiguration('px4_bin_path').perform(context)
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    world_file = LaunchConfiguration('world_file').perform(context)
    config_file = LaunchConfiguration('config_file').perform(context)
    illustration_pb_file = LaunchConfiguration('illustration_pb_file').perform(context)
    constrained_ckpt_file = LaunchConfiguration('constrained_ckpt_file').perform(context)
    unconstrained_ckpt_file = LaunchConfiguration('unconstrained_ckpt_file').perform(context)

    starts_file = world_file.replace('.sdf', '_starts.txt')
    starts = np.loadtxt(starts_file, delimiter=',')
    if starts.ndim == 1:
        starts = starts.reshape(1, -1)
    if starts.shape[0] != num_drones:
        raise RuntimeError("mismatch in start count")

    walls_file = world_file.replace('.sdf', '_walls_matrix.npy')

    z_start = 1.0
    prev_action = None
    actions = []

    env = os.environ.copy()
    env['PX4_SIM_MODEL'] = 'gz_x500'
    env['PX4_GZ_WORLD'] = Path(world_file).stem + '_walls'

    for idx in range(num_drones):
        x, y = float(starts[idx, 0]), float(starts[idx, 1])
        env['PX4_GZ_MODEL_POSE'] = f"{x},{y},{z_start}"

        px4 = ExecuteProcess(
            cmd=[px4_bin, '-i', str(idx+1)],
            env=env,  # type: ignore
            name=f'px4_{idx+1}_sitl',
            output='screen'
        )

        if idx == 0:
            actions.append(px4)
            prev_action = px4
        else:
            timed_action = TimerAction(
                period=3.0,
                actions=[px4],
            )
            actions.append(
                timed_action
            )
            prev_action = px4

    if num_drones > 0:
        microxrce_agent_cmd = ExecuteProcess(
            cmd=[
                'MicroXRCEAgent', 'udp4', '--port', '8888'
            ],
            name='microxrce_agent',
            output='screen'
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=prev_action, on_start=[microxrce_agent_cmd]),
        ))

        waypoint_generator = Node(
            package="rbmapf_gzsim",
            executable="waypoint_generator",
            name="waypoint_generator",
            output="screen",
            arguments=[
                '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                '--config_file', config_file,
                '--illustration_pb_file', illustration_pb_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=microxrce_agent_cmd, on_start=[waypoint_generator]),
        ))

        drone_ids = ','.join(str(i + 2) for i in range(num_drones))
        drone_namespaces = ','.join(f'/px4_{i + 1}' for i in range(num_drones))
        multi_drone_control = Node(
            package="rbmapf_gzsim",
            executable="multi_drone_control",
            name="multi_drone_control",
            output="screen",
            arguments=[
                '--drone_ids', drone_ids,
                '--drone_namespaces', drone_namespaces,
                '--config_file', config_file,
                '--ckpt_file', constrained_ckpt_file,
                '--walls_file', walls_file,
            ]
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=microxrce_agent_cmd, on_start=[multi_drone_control]),
        ))

        rviz_config = os.path.join(
            get_package_share_directory('rbmapf_gzsim'),
            'launch',
            'multi_drone.rviz'
        )
        print(f"RViz config file: {rviz_config}")
        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=multi_drone_control, on_start=[rviz]),
        ))

    return actions


def launch_setup(context, *args, **kwargs):
    world_file = LaunchConfiguration('world_file').perform(context)
    config_file = LaunchConfiguration('config_file').perform(context)
    illustration_pb_file = LaunchConfiguration('illustration_pb_file').perform(context)
    constrained_ckpt_file = LaunchConfiguration('constrained_ckpt_file').perform(context)
    unconstrained_ckpt_file = LaunchConfiguration('unconstrained_ckpt_file').perform(context)

    actions = []

    wall_spawner_node = Node(
            package="rbmapf_gzsim",
            executable="wall_spawner",
            name="wall_spawner",
            output="screen",
            arguments=[
                '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                '--sdf_path', world_file,
                '--config_file', config_file,
                '--illustration_pb_file', illustration_pb_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
        )
    actions.append(wall_spawner_node)

    actions.append(
      RegisterEventHandler(
        OnProcessExit(
          target_action=wall_spawner_node,
          on_exit=[OpaqueFunction(function=spawn_processes)]
        )
      )
    )

    return actions


def generate_launch_description():
    declare_num_drones = DeclareLaunchArgument(
        'num_drones', default_value='3', description='Number of x500 drones to spawn'
    )
    declare_px4_bin_path = DeclareLaunchArgument(
        'px4_bin_path',
        description='Path to the PX4 binary'
    )
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        description='Path to control config YAML'
    )
    declare_illustration_pb_file = DeclareLaunchArgument(
        'illustration_pb_file',
        description='Path to illustration PB file'
    )
    declare_constrained_ckpt_file = DeclareLaunchArgument(
        'constrained_ckpt_file',
        description='Path to constrained checkpoint file'
    )
    declare_unconstrained_ckpt_file = DeclareLaunchArgument(
        'unconstrained_ckpt_file',
        description='Path to unconstrained checkpoint file'
    )
    declare_world_file = DeclareLaunchArgument(
        'world_file',
        description='Path to the world file'
    )

    return LaunchDescription([
        declare_num_drones,
        declare_world_file,
        declare_config_file,
        declare_px4_bin_path,
        declare_illustration_pb_file,
        declare_constrained_ckpt_file,
        declare_unconstrained_ckpt_file,
        OpaqueFunction(function=launch_setup)
    ])
