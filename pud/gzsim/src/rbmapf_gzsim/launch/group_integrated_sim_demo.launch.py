import os
import yaml
import numpy as np
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python import get_package_share_directory
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    TimerAction,
    OpaqueFunction,
    ExecuteProcess,
    RegisterEventHandler,
    DeclareLaunchArgument,
    IncludeLaunchDescription
)

# Source the ros opt
# Source the px4_msgs/ ros2_ws / crazyswarm_ws
# Colcon build this code
# Source this code
# Source the gazebo-classic inside PX4 -->
#   source ~/Developer/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash
#   ~/Developer/PX4-Autopilot/ ~/Developer/PX4-Autopilot/build/px4_sitl_default/
# Run the ros2 launch script


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
        cmd=['gz', 'model',
             '--spawn-file', f'/tmp/iris_{agent_idx}.sdf',
             '--model-name', f'iris_{agent_idx}',
             '-x', str(agent_x),
             '-y', str(agent_y),
             '-z', str(agent_z)
             ],
        name='gz_spawn',
        output='screen',
    )
    actions.append(gz_cmd)
    return actions


def render_crazyflie_models(context, *args, **kwargs):
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    template_path = Path(LaunchConfiguration('model_jinja').perform(context))

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
        autoescape=False,
    )
    template = env.get_template(template_path.name)

    out_paths = []
    for drone_id in range(1, num_drones + 1):
        ns = f'crazyflie_{drone_id}'
        rendered = template.render(namespace=ns)
        out_file = Path(template_path.parent) / f'{ns}.sdf'
        out_file.write_text(rendered)
        out_paths.append(str(out_file))

    context.launch_configurations['rendered_sdfs'] = ';'.join(out_paths)
    return []


def spawn_crazyflies(context, starts):
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    sdfs = context.launch_configurations['rendered_sdfs'].split(';')
    z_start = 0.1
    actions = []
    for idx in range(1, num_drones + 1):
        x, y = float(starts[idx - 1, 0]), float(starts[idx - 1, 1])
        ns = Path(sdfs[idx - 1]).stem
        spawn = Node(
            package='ros_gz_sim',
            executable='create',
            name=f'spawn_{ns}',
            arguments=[
                '--world', 'default_walls',
                '--file',   f'file://{sdfs[idx - 1]}',
                '--name',   ns,
                '--allow-renaming', 'true',
                '--x', str(x), '--y', str(y), '--z', str(z_start),
            ],
            output='screen'
        )
        actions.append(spawn)
    return actions


def spawn_processes(context, *args, **kwargs):
    gz_version = LaunchConfiguration('gz_version').perform(context)
    world_file = LaunchConfiguration('world_file').perform(context)
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    team_size = int(LaunchConfiguration('team_size').perform(context))
    num_missions = int(LaunchConfiguration('num_missions').perform(context))
    config_file = LaunchConfiguration('config_file').perform(context)
    habitat = LaunchConfiguration('habitat').perform(context) == 'True'
    podman_image = LaunchConfiguration('podman_image').perform(context)
    use_crazyflies = LaunchConfiguration('use_crazyflies').perform(context)
    problem_set_file = LaunchConfiguration('problem_set_file').perform(context)
    kirk_server_path = Path(LaunchConfiguration('kirk_server_path').perform(context))
    execute_kirk_file_path = Path(LaunchConfiguration('execute_kirk_file_path').perform(context))
    rmpl_file_path = Path(LaunchConfiguration('rmpl_file_path').perform(context))
    generate_rmpl_file_path = Path(LaunchConfiguration('generate_rmpl_file_path').perform(context))
    hardware_demo = LaunchConfiguration('use_hardware').perform(context) == 'True'
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

    if habitat:
        bounds_file = world_file.replace(suffix, '_bounds.txt')

    z_start = 1.0
    prev_action = None
    use_sim_time = not hardware_demo
    actions = []

    if use_crazyflies == 'False':
        if gz_version == 'classic':
            updated_world_file = world_file.replace(suffix, '_walls' + suffix)
            gzserver_cmd = ExecuteProcess(
                cmd=[
                    'gzserver',
                    f'{updated_world_file}',
                    '--verbose',
                    '-s',
                    'libgazebo_ros_init.so',
                    '-s',
                    'libgazebo_ros_factory.so'
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

        elif gz_version == 'harmonic' and use_crazyflies == 'False':
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
                    '--visual', str(habitat),
                    '--config_file', config_file,
                    '--num_agents', str(num_drones),
                    '--team_size', str(team_size),
                    '--problem_set_file', problem_set_file,
                    '--constrained_ckpt_file', constrained_ckpt_file,
                    '--unconstrained_ckpt_file', unconstrained_ckpt_file,
                ],
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=microxrce_agent_cmd, on_start=[waypoint_generator]),
            ))

            drone_ids = ','.join(str(i + 2) for i in range(num_drones))
            drone_namespaces = ','.join(f'/px4_{i + 1}' for i in range(num_drones))
            px4_drone_control = Node(
                package="rbmapf_gzsim",
                executable="px4_drone_control",
                name="px4_drone_control",
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
                OnProcessStart(target_action=microxrce_agent_cmd, on_start=[px4_drone_control]),
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
                OnProcessStart(target_action=px4_drone_control, on_start=[rviz]),
            ))
    elif use_crazyflies == 'True' and use_sim_time:
        actions.append(
            OpaqueFunction(function=render_crazyflie_models)
        )
        updated_world_file = world_file.replace(suffix, '_walls' + suffix)

        gz_sim = ExecuteProcess(
            cmd=[
                'ros2', 'launch',
                'ros_gz_sim',
                'gz_sim.launch.py',
                'gz_args:=' + updated_world_file + ' -r',
            ],
            name='gz_sim',
            output='screen'
        )
        actions.append(gz_sim)

        spawn_d = TimerAction(period=2.0, actions=[OpaqueFunction(function=spawn_crazyflies, args=[starts])])
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[spawn_d]),
        ))

        waypoint_generator = Node(
            package="rbmapf_gzsim",
            executable="waypoint_generator",
            name="waypoint_generator",
            output="screen",
            arguments=[
                '--visual', str(habitat),
                '--config_file', config_file,
                '--num_agents', str(num_drones),
                '--team_size', str(team_size),
                '--problem_set_file', problem_set_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            parameters=[{'interface': 'crazyflie', 'use_sim_time': True}],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[waypoint_generator]),
        ))

        drone_ids = ','.join(str(i + 1) for i in range(num_drones))
        drone_namespaces = ','.join(f'/crazyflie_{i + 1}' for i in range(num_drones))

        if habitat:
            habitat_sensor_node = Node(
                package="rbmapf_gzsim",
                executable="habitat_sensor_node",
                name="habitat_sensor_node",
                output="screen",
                parameters=[{
                    'use_sim_time': True,
                    'config_file': config_file,
                    'drone_namespaces': drone_namespaces,
                }],
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=waypoint_generator, on_start=[habitat_sensor_node]),
            ))

        pkg_project_bringup = get_package_share_directory('rbmapf_gzsim')
        test_bridge_yaml = os.path.join(pkg_project_bringup, 'config', 'gzsim_bridge.yaml')
        with open(test_bridge_yaml, 'r') as f:
            bridge_config_template_str = f.read()

        bridge = None
        for idx in range(1, num_drones + 1):
            bridge_config_str = bridge_config_template_str.replace('{model_name}', f'crazyflie_{idx}')
            with open(os.path.join(pkg_project_bringup, 'config', f'bridge_config_crazyflie_{idx}.yaml'), 'w') as f:
                f.write(bridge_config_str)
            bridge = Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name=f'ros_gz_bridge_crazyflie_{idx}',
                parameters=[{
                    'config_file': os.path.join(pkg_project_bringup, 'config', f'bridge_config_crazyflie_{idx}.yaml'),
                    'use_sim_time': True,
                }],
                output='screen'
            )
            files = [config_file, constrained_ckpt_file, walls_file]
            if habitat:
                files.append(bounds_file)
            cf_drone_control = Node(
                package="rbmapf_gzsim",
                executable="crazyflie_drone_control",
                name=f"crazyflie_drone_control_{idx}",
                output="screen",
                parameters=[{
                    'drone_id': idx,
                    'visual': str(habitat),
                    'drone_ns': f'/crazyflie_{idx}',
                    'files': files,
                    'use_sim_time': True
                }],
            )

            timed_actions = TimerAction(period=2.0, actions=[cf_drone_control, bridge])
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=waypoint_generator, on_start=[timed_actions]),
            ))

        multi_tf = Node(
            package="rbmapf_gzsim",
            executable="multi_tf_broadcaster",
            name="multi_tf_broadcaster",
            output="screen",
            parameters=[{'namespaces': drone_namespaces, 'use_sim_time': True}],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[multi_tf]),
        ))

        rviz_config = os.path.join(
            get_package_share_directory('rbmapf_gzsim'),
            'launch',
            'multi_drone.rviz'
        )
        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{
                'use_sim_time': False,
            }],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[rviz]),
        ))
        wall_rviz_node = Node(
                package="rbmapf_gzsim",
                executable="wall_rviz_viz",
                name="wall_rviz_viz",
                output="screen",
                arguments=[
                    '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                    '--visual', str(habitat),
                    '--sdf_path', world_file,
                    '--config_file', config_file,
                    '--problem_set_file', problem_set_file,
                    '--constrained_ckpt_file', constrained_ckpt_file,
                    '--unconstrained_ckpt_file', unconstrained_ckpt_file,
                ],
                parameters=[{'use_sim_time': True}],
            )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=rviz, on_start=[wall_rviz_node]),
        ))

        waypoint_gen_checker = Node(
            package="rbmapf_gzsim",
            executable="check_waypoint_gen_finished",
            name="check_waypoint_gen_finished",
            output="screen",
            parameters=[{
                'use_sim_time': True,
            }],
        )
        actions.append(waypoint_gen_checker)

        # create_rmpl = ExecuteProcess(
        #     cmd=['python', generate_rmpl_file_path.as_posix(),
        #          '--output-rmpl-path', rmpl_file_path.as_posix(),
        #          '--num-drones', str(num_drones), '--num-missions', str(num_missions)],
        #     name='create_rmpl',
        #     output='screen',
        # )

        # upload_rmpl = ExecuteProcess(
        #     cmd=['podman', 'cp', rmpl_file_path.as_posix(), f'{podman_image}:/common-lisp/enterprise/mission.rmpl'],
        #     name='upload_rmpl',
        #     output='screen',
        # )

        # actions.append(RegisterEventHandler(
        #     OnProcessStart(target_action=gz_sim, on_start=[create_rmpl]),
        # ))

        # actions.append(RegisterEventHandler(
        #     OnProcessExit(target_action=create_rmpl, on_exit=[upload_rmpl]),
        # ))

        # Copy the execute.sh script to podman
        copy_exec_script = ExecuteProcess(
            cmd=['podman', 'cp', execute_kirk_file_path.as_posix(),
                 f'{podman_image}:/common-lisp/enterprise/execute_kirk.sh'],
            name='copy_exec_script',
            output='screen',
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[copy_exec_script]),
        ))

        for idx in range(1, num_drones + 1):
            podman_exec = ExecuteProcess(
                cmd=['podman', 'exec', '-it', podman_image, '/bin/bash', 'execute_kirk.sh', str(idx)],
                name=f'podman_exec_{idx}',
                output='screen',
            )
            actions.append(RegisterEventHandler(
                OnProcessExit(target_action=waypoint_gen_checker, on_exit=[podman_exec]),
            ))

        kirk_server = ExecuteProcess(
            cmd=['python', kirk_server_path.as_posix(), '--num-drones', str(num_drones), '--logging-level', 'CRITICAL'],
            name='kirk_server',
            output='screen',
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[kirk_server]),
        ))
    else:
        # Using crazyflie hardware

        crazyflie_yaml_path = os.path.join(
            get_package_share_directory('rbmapf_gzsim'),
            'config',
            'crazyflie_mapf.yaml'
        )
        generate_yaml = ExecuteProcess(
            cmd=[
                'ros2', 'run', 'rbmapf_gzsim', 'generate_crazyflie_yaml',
                '--output_path', crazyflie_yaml_path,
                '--num_drones', str(num_drones),
                '--starts_file', starts_file,
            ],
            output='screen'
        )
        actions.append(generate_yaml)

        rviz_config = os.path.join(
            get_package_share_directory('rbmapf_gzsim'),
            'launch',
            'multi_drone.rviz'
        )

        crazyflie_server = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(
                    get_package_share_directory('crazyflie'),
                    'launch',
                    'launch.py'
                )]
            ),
            launch_arguments={
                'crazyflies_yaml_file': crazyflie_yaml_path,
                'rviz_config_file': rviz_config,
                'debug': 'False',
                'rviz': 'False',
                'gui': 'False',
                'teleop': 'False',
                'backend': 'cflib',
                }.items()
        )
        delayed_spawn = TimerAction(period=5.0, actions=[crazyflie_server])
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=generate_yaml, on_start=[delayed_spawn]),
        ))

        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{
                'use_sim_time': False,
            }],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=generate_yaml, on_start=[rviz]),
        ))

        wall_rviz_node = Node(
            package="rbmapf_gzsim",
            executable="wall_rviz_viz",
            name="wall_rviz_viz",
            output="screen",
            arguments=[
                '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                '--visual', str(habitat),
                '--sdf_path', world_file,
                '--config_file', config_file,
                '--problem_set_file', problem_set_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            parameters=[{'use_hardware': True}],
        )
        delayed_rviz = TimerAction(period=2.0, actions=[wall_rviz_node])
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=rviz, on_start=[delayed_rviz]),
        ))

        # swarm = Crazyswarm()
        for idx in range(1, num_drones + 1):
            cf_drone_control = Node(
                package="rbmapf_gzsim",
                executable="crazyswarm_drone_control",
                name=f"crazyswarm_drone_control_{idx}",
                output="screen",
                parameters=[{
                    'drone_id': idx,
                    'drone_ns': f'/cf{idx}',
                    'files': [config_file, constrained_ckpt_file, walls_file],
                }],
                # arguments=[swarm],
            )

            timed_actions = TimerAction(period=2.0, actions=[cf_drone_control])
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=wall_rviz_node, on_start=[timed_actions]),
            ))

        waypoint_generator = Node(
            package="rbmapf_gzsim",
            executable="waypoint_generator",
            name="waypoint_generator",
            output="screen",
            arguments=[
                '--visual', str(habitat),
                '--config_file', config_file,
                '--num_agents', str(num_drones),
                '--team_size', str(team_size),
                '--use_hardware', str(hardware_demo),
                '--problem_set_file', problem_set_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            parameters=[{'interface': 'cf'}],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=wall_rviz_node, on_start=[waypoint_generator]),
        ))

        waypoint_gen_checker = Node(
            package="rbmapf_gzsim",
            executable="check_waypoint_gen_finished",
            name="check_waypoint_gen_finished",
            output="screen",
        )
        actions.append(waypoint_gen_checker)

        # create_rmpl = ExecuteProcess(
        #     cmd=['python', generate_rmpl_file_path.as_posix(),
        #          '--output-rmpl-path', rmpl_file_path.as_posix(),
        #          '--num-drones', str(num_drones), '--num-missions', str(num_missions)],
        #     name='create_rmpl',
        #     output='screen',
        # )

        # upload_rmpl = ExecuteProcess(
        #     cmd=['podman', 'cp', rmpl_file_path.as_posix(), f'{podman_image}:/common-lisp/enterprise/mission.rmpl'],
        #     name='upload_rmpl',
        #     output='screen',
        # )

        # actions.append(RegisterEventHandler(
        #     OnProcessStart(target_action=waypoint_generator, on_start=[create_rmpl]),
        # ))

        # actions.append(RegisterEventHandler(
        #     OnProcessExit(target_action=create_rmpl, on_exit=[upload_rmpl]),
        # ))

        # Copy the execute.sh script to podman
        copy_exec_script = ExecuteProcess(
            cmd=['podman', 'cp', execute_kirk_file_path.as_posix(),
                 f'{podman_image}:/common-lisp/enterprise/execute_kirk.sh'],
            name='copy_exec_script',
            output='screen',
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=waypoint_generator, on_start=[copy_exec_script]),
        ))

        for idx in range(1, num_drones + 1):
            podman_exec = ExecuteProcess(
                cmd=['podman', 'exec', '-it', podman_image, '/bin/bash', 'execute_kirk.sh', str(idx)],
                name=f'podman_exec_{idx}',
                output='screen',
            )
            actions.append(RegisterEventHandler(
                OnProcessExit(target_action=waypoint_gen_checker, on_exit=[podman_exec]),
            ))

        kirk_server = ExecuteProcess(
            cmd=['python', kirk_server_path.as_posix(), '--num-drones', str(num_drones), '--logging-level', 'CRITICAL'],
            name='kirk_server',
            output='screen',
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=waypoint_generator, on_start=[kirk_server]),
        ))

    if use_sim_time and gz_version not in ['classic', 'harmonic']:
        raise RuntimeError('Incorrect gz_version provided. Options include classic or harmonic')

    return actions


def launch_setup(context, *args, **kwargs):
    world_file = LaunchConfiguration('world_file').perform(context)
    config_file = LaunchConfiguration('config_file').perform(context)
    habitat = LaunchConfiguration('habitat').perform(context) == 'True'
    problem_set_file = LaunchConfiguration('problem_set_file').perform(context)
    hardware_demo = LaunchConfiguration('use_hardware').perform(context) == 'True'
    constrained_ckpt_file = LaunchConfiguration('constrained_ckpt_file').perform(context)
    unconstrained_ckpt_file = LaunchConfiguration('unconstrained_ckpt_file').perform(context)

    actions = []

    use_sim_time = not hardware_demo

    wall_spawner_node = Node(
            package="rbmapf_gzsim",
            executable="wall_spawner",
            name="wall_spawner",
            output="screen",
            arguments=[
                '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                '--visual', str(habitat),
                '--sdf_path', world_file,
                '--config_file', config_file,
                '--use_hardware', str(hardware_demo),
                '--problem_set_file', problem_set_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            parameters=[{'use_sim_time': use_sim_time}],
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
    ld = LaunchDescription()

    params_file = get_package_share_directory('rbmapf_gzsim') + '/config/parameters.yaml'
    specs = yaml.safe_load(Path(params_file).read_text())

    for name, meta in specs.items():
        default = meta.get("default")
        description = meta.get("description", "")
        if default is None:
            ld.add_action(DeclareLaunchArgument(name, description=description))
        else:
            ld.add_action(DeclareLaunchArgument(
                name,
                default_value=str(default),
                description=description
            ))

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
