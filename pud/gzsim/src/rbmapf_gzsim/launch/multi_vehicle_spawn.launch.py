import os
from pathlib import Path

import numpy as np
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, RegisterEventHandler, TimerAction

# Source the ros opt
# Source the px4_msgs/ ros2_ws
# Colcon build this code
# Source this code
# Source the gazebo-classic inside PX4 --> source ~/Developer/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash ~/Developer/PX4-Autopilot/ ~/Developer/PX4-Autopilot/build/px4_sitl_default/
# Run the ros2 launch script

# Denormalizing: [0.50259066 0.26878574]
# [wall_spawner-1] Denormalized: [7.53886  4.031786]
# [wall_spawner-1] Before adjusting:  [array([7.53886 , 4.031786, 2.      ], dtype=float32)]
# [wall_spawner-1] After adjusting:  [[ 0.03885984 -3.46821404  2.        ]]


def spawn_drones(context, agent_idx, agent_x, agent_y, agent_z=1.0):
    actions = []
    
    px4_src = LaunchConfiguration('px4_src_path').perform(context)
    px4_target = LaunchConfiguration('px4_target').perform(context)
    
    px4_build_path = f"{px4_src}/build/{px4_target}"
    
    px4_bin = f"{px4_build_path}/bin/px4"
    px4_work_dir = f"{px4_build_path}/rootfs/{agent_idx}"

    px4 = ExecuteProcess(
        cmd=[px4_bin, '-i', str(agent_idx), '-d',  f'{px4_build_path}/etc'],
        cwd=px4_work_dir,
        name=f'px4_{agent_idx}_sitl',
        output='screen'
    )
    actions.append(px4)

    jinja_cmd = ExecuteProcess(
        cmd=[
            'python3', 
            px4_src + '/Tools/simulation/gazebo-classic/sitl_gazebo-classic/scripts/jinja_gen.py',
            px4_src + '/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf.jinja',
            px4_src + '/Tools/simulation/gazebo-classic/sitl_gazebo-classic',
            '--mavlink_tcp_port', str(4560 + agent_idx),
            '--mavlink_udp_port', str(14560 + agent_idx),
            '--mavlink_id', str(1 + agent_idx),
            '--gst_udp_port', str(5600 + agent_idx),
            '--video_uri', str(5600 + agent_idx),
            '--mavlink_cam_udp_port', str(14530 + agent_idx),
            '--output-file', f'/tmp/iris_{agent_idx}.sdf',
        ],
        name=f'jinja_{agent_idx}_cmd',
        output='screen',
    )
    actions.append(jinja_cmd)

    gz_cmd = ExecuteProcess(
        cmd=['gz', 'model', '--spawn-file', f'/tmp/iris_{agent_idx}.sdf', '--model-name', f'iris_{agent_idx}', '-x', str(agent_x), '-y', str(agent_y), '-z', str(agent_z)],
        name='gz_spawn',
        output='screen',
    )
    actions.append(gz_cmd)
    return actions

def spawn_processes(context, *args, **kwargs):
    gz_version = LaunchConfiguration('gz_version').perform(context)
    world_file = LaunchConfiguration('world_file').perform(context)
    config_file = LaunchConfiguration('config_file').perform(context)
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    illustration_pb_file = LaunchConfiguration('illustration_pb_file').perform(context)
    constrained_ckpt_file = LaunchConfiguration('constrained_ckpt_file').perform(context)
    unconstrained_ckpt_file = LaunchConfiguration('unconstrained_ckpt_file').perform(context)

    suffix = Path(world_file).suffix.lower()
    starts_file = world_file.replace(suffix, '_starts.txt')
    starts = np.loadtxt(starts_file, delimiter=',')
    if starts.ndim == 1:
        starts = starts.reshape(1, -1)
    if starts.shape[0] != num_drones:
        raise RuntimeError("Mismatch in start count")
    walls_file = world_file.replace(suffix, '_walls_matrix.npy')

    actions = []
    z_start = 1.0
    prev_action = None

    if gz_version == 'classic':
        updated_world_file = world_file.replace(suffix, '_walls' + suffix)
        gzserver_cmd = ExecuteProcess(
            cmd=[
                'gzserver', f'{updated_world_file}', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so' 
            ],
            name='gzserver_launch',
            output='screen'
        )
        actions.append(gzserver_cmd)

        for idx in range(1, num_drones + 1):
            x, y = float(starts[idx-1, 0]), float(starts[idx-1, 1])
            spawn_drone_cmd = OpaqueFunction(function=spawn_drones, args=[idx, x, y, z_start])
            timed_action = TimerAction(period=5.0, actions=[spawn_drone_cmd])
            actions.append(timed_action)

        gzclient_cmd = ExecuteProcess(
            cmd=['gzclient'],
            name='gzclient_cmd',
            output='screen',
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gzserver_cmd, on_start=[gzclient_cmd]),
        ))
        prev_action = gzclient_cmd
           
    elif gz_version == 'harmonic':
        env = os.environ.copy()
        env['PX4_SIM_MODEL'] = 'gz_x500'
        env['PX4_GZ_WORLD'] = Path(world_file).stem + '_walls'
        px4_bin = LaunchConfiguration('px4_bin_path').perform(context)

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
    else:
        raise RuntimeError('Incorrect gz_version provided. Options include classic or harmonic')

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
                '--num_agents', str(num_drones),
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
                '--gz_version', gz_version,
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
    declare_gz = DeclareLaunchArgument(
        'gz_version', default_value='harmonic', description='Gazebo version to run. Options include classic or harmonic'
    )
    declare_num_drones = DeclareLaunchArgument(
        'num_drones', default_value='3', description='Number of x500 drones to spawn'
    )
    declare_px4_target = DeclareLaunchArgument(
        'px4_target',
        description='Path target'
    )
    declare_px4_src_path = DeclareLaunchArgument(
        'px4_src_path',
        description='Path to the PX4 source'
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
        declare_gz,
        declare_num_drones,
        declare_world_file,
        declare_config_file,
        declare_px4_target,
        declare_px4_src_path,
        declare_illustration_pb_file,
        declare_constrained_ckpt_file,
        declare_unconstrained_ckpt_file,
        OpaqueFunction(function=launch_setup)
    ])

