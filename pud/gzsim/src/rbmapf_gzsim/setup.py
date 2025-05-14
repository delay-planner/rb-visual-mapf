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
        ('share/' + package_name + '/launch', ['launch/multi_drone.rviz']),
        ('share/' + package_name + '/launch', ['launch/multi_vehicle_spawn.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Viraj Parimi',
    maintainer_email='vparimi@mit.edu',
    description='Multi-Drone Gazebo Simulation for Risk-Bonuded MAPF',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_spawner = rbmapf_gzsim.wall_spawner:main',
            'waypoint_generator = rbmapf_gzsim.waypoint_generator:main',
            'multi_drone_control = rbmapf_gzsim.multi_drone_control:main',
        ],
    },
)
