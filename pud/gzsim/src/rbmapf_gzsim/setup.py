from setuptools import find_packages, setup

package_name = 'rbmapf_gzsim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/parameters.yaml']),
        ('share/' + package_name + '/config', ['config/gzsim_bridge.yaml']),
        ('share/' + package_name + '/launch', ['launch/multi_drone.rviz']),
        ('share/' + package_name + '/launch', ['launch/multi_vehicle_spawn.launch.py']),
        ('share/' + package_name + '/launch', ['launch/integrated_sim_demo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Viraj Parimi',
    maintainer_email='vparimi@mit.edu',
    description='Multi-Drone Gazebo Simulation for Risk-Bounded MAPF',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_spawner = rbmapf_gzsim.wall_spawner:main',
            'wall_rviz_viz = rbmapf_gzsim.wall_rviz_viz:main',
            'px4_drone_control = rbmapf_gzsim.px4_drone_control:main',
            'waypoint_generator = rbmapf_gzsim.waypoint_generator:main',
            'habitat_sensor_node = rbmapf_gzsim.habitat_sensor_node:main',
            'multi_tf_broadcaster = rbmapf_gzsim.multi_tf_broadcaster:main',
            'crazyflie_drone_control = rbmapf_gzsim.crazyflie_drone_control:main',
            'generate_crazyflie_yaml = rbmapf_gzsim.generate_crazyflie_yaml:main',
            'crazyswarm_drone_control = rbmapf_gzsim.crazyswarm_drone_control:main',
            'check_waypoint_gen_finished = rbmapf_gzsim.check_waypoint_gen_finished:main',
        ],
    },
)
