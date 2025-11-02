import json
import yaml
import time
import torch
import requests
import numpy as np
from dotmap import DotMap

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String, Empty, Float32MultiArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, qos_profile_sensor_data

from crazyflie_py.crazyflie import arrayToGeometryPoint
from crazyflie_interfaces.srv import Takeoff, Land, NotifySetpointsStop, GoTo

from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.vision.vision_agent import LagVisionUVFDDPG

#### Example start/goal pairs
#### x=2.25, y=-4.05
#### x=2.83, y=-0.25

##### goalx=2.7, goaly=-2.13
##### startx=2.82, starty=-1.36
##@### startx=2.924, staarty=3.950

class DroneController(Node):
    def __init__(self, waypoint_follow=False):
        super().__init__("drone_controller")

        self.declare_parameter('files', ['config.yaml', 'ckpt.pth', 'walls.npy'])
        self.declare_parameter('drone_id', 1)
        self.declare_parameter('num_drones', 1)
        self.declare_parameter('drone_ns', "/cf1")

        drone_ns = self.get_parameter('drone_ns').get_parameter_value().string_value
        drone_id = self.get_parameter('drone_id').get_parameter_value().integer_value
        num_drones = self.get_parameter('num_drones').get_parameter_value().integer_value
        files = self.get_parameter('files').get_parameter_value().string_array_value

        self.node_name = f"drone_controller_{drone_id}"

        self.viz_flag = False
        self.start_flag = False
        self.habitat_state = None
        self.drone_ns = drone_ns
        self.drone_id = drone_id
        self.num_drones = num_drones
        self.waypoint_follow = waypoint_follow

        self.ack_endpoint = "http://127.0.0.1:5000/done"
        self.mission_url = f"http://127.0.0.1:5000/?drone_id={self.drone_id}"
 
        config_file, ckpt_file, walls_file = files

        self.land_duration = 5.0
        self.hover_duration = 5.0
        self.pause_duration = 7.0
        self.takeoff_duration = 5.0

        self.pose_timeout = 0.5
        self.last_pose_time = None

        self.walls = np.load(walls_file)
        rows, cols = self.walls.shape
        self.normalize_factor = np.array([rows, cols])
        self.scale_factor = np.array([float(rows) / 6, float(cols) / 8]) # 6x8 is the highbay space
        self.origin_offset = np.array([-cols / 2.0, -rows / 2.0])

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        config = DotMap(config)
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_agent(config, ckpt_file)

        self._init_services()
        self._init_subscribers()
        self._init_visualizers()

        self.counter = 0
        self.timer = self.create_timer(0.1, self.timer_callback)
        callback_fn = self.waypoint_follower_callback if self.waypoint_follow else self.command_callback
        self.cmd_timer = self.create_timer(0.02, callback_fn)

        self.altitude = 3.0
        # self.altitude += (self.drone_id - 1) * 0.5
        self.current_state = "IDLE"
        self.offboard_mode = False
        self.distance_threshold = 0.01
        self.current_position = np.zeros(3, dtype=float)
        self.current_orientation = np.zeros(4, dtype=float)

        self.other_agent_homes = {}
        self.other_agent_positions = {}

    def _init_agent(self, config, ckpt_file):
        self.agent = DRLDDPGLag(
            4, 2, 1,
            CriticCls=GoalConditionedCritic,
            device=torch.device(config.device),
            **config.agent,
        )
        self.agent.load_state_dict(
            torch.load(ckpt_file, map_location=torch.device(config.device))
        )
        self.agent.to(torch.device(config.device))
        self.agent.eval()

    def _ready_to_start(self):
        try:
            resp = requests.get(self.mission_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            # self.get_logger().info(f"Response from server: {data}")
            mission_ready = data.get("mission_ready", False) is True
            mission_name = data.get("mission_name", "")
            land_ack = data.get("land", False) is True
            return mission_ready, mission_name, land_ack
        except (requests.RequestException, ValueError) as e:
            self.get_logger().error(f"Request failed or invalid JSON: {e}")
            return False, "", False

    def _send_ack(self, mission_name):
        try:
            data = {
                "drone_id": self.drone_id,
                "mission_name": self.mission_name
            }
            response = requests.post(self.ack_endpoint, json=data)

            if response.status_code != 200:
                self.get_logger().error(f"Request failed with status code: {response.status_code}")
                self.get_logger().error(f"Response: {response.text}")
            # else:
            #     self.get_logger().info(f"Response: {response.json()}")
        except (requests.RequestException, ValueError) as e:
            self.get_logger().error(f"Failed to send ack for mission {mission_name}: {e}")

    def _init_services(self):
        self.land_client = self.create_client(Land, self.drone_ns + '/land')
        self.goto_client = self.create_client(GoTo, self.drone_ns + '/go_to')
        self.takeoff_client = self.create_client(Takeoff, self.drone_ns + '/takeoff')
        self.notify_client = self.create_client(NotifySetpointsStop, self.drone_ns + '/notify_setpoints_stop')

        self.land_client.wait_for_service()
        self.goto_client.wait_for_service()
        self.notify_client.wait_for_service()
        self.takeoff_client.wait_for_service()

    def _init_subscribers(self):
        self.waypoint_subscriber = self.create_subscription(
            Float32MultiArray,
            f"{self.drone_ns}/waypoints",
            self.waypoints_callback,
            10,
        )

        start_qos = QoSProfile(depth=1)
        start_qos.reliability = QoSReliabilityPolicy.RELIABLE
        start_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        # self.create_subscription(
        #     Empty,
        #     "/waypoints_generated",
        #     self.start_callback,
        #     # 10,
        #     start_qos
        # )
        self.odom_subscriber = self.create_subscription(
            PoseStamped,
            f"{self.drone_ns}/pose",
            self.position_callback,
            qos_profile=qos_profile_sensor_data
            # QoSProfile(
            #     depth=1,
            #     history=QoSHistoryPolicy.KEEP_LAST,
            #     reliability=QoSReliabilityPolicy.BEST_EFFORT,
            #     deadline=rclpy.duration.Duration(seconds=0, nanoseconds=1e9/100.0)
            # ),
        )
        for i in range(1, self.num_drones + 1):
            drone_ns = f"/cf{i}"
            if drone_ns != self.drone_ns:
                self.create_subscription(
                    PoseStamped,
                    f"{drone_ns}/pose",
                    lambda msg, ns=drone_ns: self.other_position_callback(ns, msg),
                    # 10,
                    qos_profile=qos_profile_sensor_data
                )
                self.create_subscription(
                    Float32MultiArray,
                    f"{drone_ns}/waypoints",
                    lambda msg, ns=drone_ns: self.other_waypoints_callback(ns, msg),
                    10,
                )

    def _init_visualizers(self):
        self.trajectory = Path()
        self.trajectory.header.frame_id = "world"
        self.colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8),
        ]

        self.path_publisher = self.create_publisher(
            Path,
            f"{self.drone_ns}/trajectory",
            10
        )
        self.waypoint_marker_publisher = self.create_publisher(
            MarkerArray,
            f"{self.drone_ns}/waypoint_markers",
            10
        )
        # self.wall_publisher = self.create_publisher(MarkerArray, 'wall_markers', 10)
        # self.publish_wall_markers()

    def publish_markers(self):
        markers = MarkerArray()
        for idx, (xe, yn, _) in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'waypoints'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(xe)
            marker.pose.position.y = float(yn)
            marker.pose.position.z = self.altitude
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2
            marker.color = self.colors[self.drone_id - 1]
            marker.lifetime = Duration(sec=0)
            markers.markers.append(marker)  # type: ignore

        start_text = Marker()
        start_text.header.frame_id = 'world'
        start_text.header.stamp = self.get_clock().now().to_msg()
        start_text.ns = 'start'
        start_text.id = 0
        start_text.type = Marker.TEXT_VIEW_FACING
        start_text.action = Marker.ADD
        start_text.pose.position.x = float(self.waypoints[0][0])
        start_text.pose.position.y = float(self.waypoints[0][1])
        start_text.pose.position.z = self.altitude + 0.5
        start_text.text = f"START {self.drone_id}"
        start_text.scale.z = 0.4
        start_text.color = self.colors[self.drone_id - 1]
        start_text.lifetime = Duration(sec=0)
        markers.markers.append(start_text)  # type: ignore

        goal_text = Marker()
        goal_text.header.frame_id = 'world'
        goal_text.header.stamp = self.get_clock().now().to_msg()
        goal_text.ns = 'goal'
        goal_text.id = 0
        goal_text.type = Marker.TEXT_VIEW_FACING
        goal_text.action = Marker.ADD
        goal_text.pose.position.x = float(self.waypoints[-1][0])
        goal_text.pose.position.y = float(self.waypoints[-1][1])
        goal_text.pose.position.z = self.altitude + 0.5
        goal_text.text = f"GOAL {self.drone_id}"
        goal_text.scale.z = 0.4
        goal_text.color = self.colors[self.drone_id - 1]
        goal_text.lifetime = Duration(sec=0)
        markers.markers.append(goal_text)  # type: ignore

        self.waypoint_marker_publisher.publish(markers)

    # def publish_wall_markers(self, height=5.0, resolution=1.0):
    #     marker_array = MarkerArray()

    #     idx = 0
    #     for i, j in zip(*np.where(self.walls == 1)):
    #         x, y = self.env2highbay([(j + 0.5)*resolution, (i + 0.5)*resolution])

    #         marker = Marker()
    #         marker.header.frame_id = 'world'
    #         marker.header.stamp = self.get_clock().now().to_msg()
    #         marker.ns = 'walls'
    #         marker.id = idx
    #         marker.type = Marker.CUBE
    #         marker.action = Marker.ADD
    #         marker.pose.position.x = float(x)
    #         marker.pose.position.y = float(y)
    #         marker.pose.position.z = float(height / 2.0)
    #         marker.scale.x = float(resolution / self.scale_factor[0])
    #         marker.scale.y = float(resolution / self.scale_factor[1])
    #         marker.scale.z = height
    #         marker.color = ColorRGBA(r=0.3, g=0.3, b=0.3, a=1.0)
    #         marker.lifetime = Duration(sec=0)
    #         marker_array.markers.append(marker)  # type: ignore
    #         idx += 1

    #     self.wall_publisher.publish(marker_array)

    def env2highbay(self, position):
        return self.adjust_position(position, to_highbay_frame=True)

    def highbay2env(self, position):
        return self.adjust_position(position, to_highbay_frame=False)

    def adjust_position(self, position, to_highbay_frame=False):
        # If local is false then the position is the grid position in the global frame
        # If local is true then the position of the drone in the local frame
        return (position + self.origin_offset) / self.scale_factor if to_highbay_frame else (position * self.scale_factor) - self.origin_offset
        
    def waypoints_callback(self, msg: Float32MultiArray):
        # Waypoints are in global ENU frame with first entry as home, last entry as goal
        self.current_wp_index = 1  # 1 Use 1 is you want the drone to skip the start waypoint -- useful when using single drones and can start anywhere in the highbay
        self.waypoints = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        self.publish_markers()

        self.home = self.waypoints[0][:2].copy()
        # State needs to be in the coordinate frame that the GCRL policy was trained with
        # i.e origin is bottom left corner of the walls matrix not the origin of simulator!
        # self.state = self.home.copy() - self.origin_offset
        self.state = self.highbay2env(self.home)
        self.next_location = self.home.copy()  # Cannot be zeros as its in global ENU frame
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints")

    def other_waypoints_callback(self, drone_ns, msg):
        other_waypoints = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        other_home = other_waypoints[0][:2].copy()
        self.other_agent_homes[drone_ns] = other_home
        self.other_agent_positions[drone_ns] = other_home.copy()  # Initialize to home position

    # def start_callback(self, msg):
    #     self.start_flag = True
    #     self.get_logger().info(f"Drone {self.drone_id} received start signal.")

    def position_callback(self, msg):
        # if self.gz_version == 'harmonic':
        #     # Local position is in NED frame
        #     self.local_position[0] = msg.position[0]
        #     self.local_position[1] = msg.position[1]
        # else:
        #     # Global position coming from VICON is in ENU frame
        #     # Extract the local position from the global location
        #     self.local_position[0] = msg.pose.position.x - self.home[0]
        #     self.local_position[1] = msg.pose.position.y - self.home[1]

        #     # Convert to NED
        #     self.local_position = enu_to_ned(self.local_position)

        # TODO: Replace this with the correct version above after debugging
        # Local position is in NED frame
        self.current_position[0] = msg.pose.position.x
        self.current_position[1] = msg.pose.position.y
        self.current_position[2] = msg.pose.position.z
        self.current_orientation = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        self.last_pose_time = self.node_time()
        # self.publish_transform_callback()

    def other_position_callback(self, drone_ns, msg):
        # Other drone position is in NED frame
        other_position = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)
        if drone_ns in self.other_agent_positions:
            self.other_agent_positions[drone_ns] = other_position

    def timer_callback(self):
        self.start_flag, self.mission_name, self.land_ack = self._ready_to_start()
        if self.land_ack:
            self.current_state = "LAND"

        if self.current_state == "IDLE" and self.start_flag:
            self.current_state = "TAKEOFF"
            self.get_logger().info(f"Running the drone {self.drone_ns}")
        elif self.current_state == "TAKEOFF":
            self.takeoff()
        elif self.current_state == "OFFBOARD":
            self.offboard()
        elif self.current_state == "LAND":
            self.land()
        self.counter += 1

    def node_time(self):
        return self.get_clock().now().nanoseconds / 1e9

    def timehelper_sleep(self, duration):
        start = self.node_time()
        end = start + duration
        while self.node_time() < end:
            rclpy.spin_once(self, timeout_sec=0)

    def takeoff(self):
        self.offboard_mode = False

        if not hasattr(self, "_takeoff_started"):
            self._takeoff_started = False

        if self.current_state == "TAKEOFF":
            if not self._takeoff_started:
                # self.cf.takeoff(targetHeight=self.altitude, duration=5.0)
                # self.timeHelper.sleep(7.0)
                takeoff_request = Takeoff.Request()
                takeoff_request.height = self.altitude
                takeoff_request.duration = rclpy.duration.Duration(seconds=self.takeoff_duration).to_msg()
                self.takeoff_client.call_async(takeoff_request)
                # time.sleep(self.takeoff_duration + self.hover_duration)
                # self.timehelper_sleep(self.takeoff_duration + self.hover_duration)
                self._takeoff_started = True
                self._takeoff_deadline = self.node_time() + self.takeoff_duration + self.hover_duration
                self.get_logger().info(f"Takeoff command sent to {self.drone_ns}")
            else:
                if self.node_time() >= self._takeoff_deadline:
                    self._takeoff_started = False
                    self.current_state = "OFFBOARD"
                    self.get_logger().info(f"Takeoff completed for {self.drone_ns}")
                # self.current_state = "OFFBOARD"

    def land(self):
        self.offboard_mode = False

        if not hasattr(self, "_land_started"):
            self._land_started = False

        if self.current_state == "LAND":
            if not self._land_started:
                # self.cf.land(targetHeight=1.15, duration=5.0)
                # self.timeHelper.sleep(5.0)
                notify_request = NotifySetpointsStop.Request()
                self.notify_client.call_async(notify_request)
                land_request = Land.Request()
                land_request.height = 1.15
                land_request.duration = rclpy.duration.Duration(seconds=self.land_duration).to_msg()
                self.land_client.call_async(land_request)
                
                self._land_started = True
                self._land_deadline = self.node_time() + self.land_duration + self.pause_duration
                self.get_logger().info(f"Landing the drone {self.drone_ns}")
            else:
                if self.node_time() >= self._land_deadline:
                    # time.sleep(self.land_duration + self.pause_duration)
                    # self.timehelper_sleep(self.land_duration + self.pause_duration)
                    self._land_started = False
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Landing completed for {self.drone_ns}")

    def offboard(self):
        self.counter = 0
        self.offboard_mode = True

    def send_debug_trajectory(self, position):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "world"
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = self.altitude
        self.trajectory.poses.append(pose)  # type: ignore
        self.path_publisher.publish(self.trajectory)

    def send_trajectory(self, position):
        # self.cf.goTo(position, 0, 5.0)
        # self.timeHelper.sleep(5.0)
        goto_request = GoTo.Request()
        self.get_logger().info(f"Sending to position: {position}")
        float_position = [float(position[0]), float(position[1]), float(self.altitude)]
        goto_request.goal = arrayToGeometryPoint(float_position)
        goto_request.yaw = float(0.0)
        goto_request.duration = rclpy.duration.Duration(seconds=self.pause_duration).to_msg()
        self.goto_client.call_async(goto_request)
        # time.sleep(self.pause_duration)
        # self.timehelper_sleep(self.pause_duration)
        self._goto_deadline = self.node_time() + self.pause_duration

    def waypoint_follower_callback(self):
        if not (self.offboard_mode and self.start_flag):
            return
        
        if self.last_pose_time is None or (self.node_time() - self.last_pose_time) > self.pose_timeout:
            self.get_logger().warn(f"No recent position data for {self.drone_ns}, hovering at current location")
            if np.linalg.norm(self.current_position) > 0:
                if hasattr(self, "_goto_deadline") and self.node_time() < self._goto_deadline:
                    return
                self.send_trajectory(self.current_position[:2].copy())
            return
        
        if hasattr(self, "waypoints") and self.current_wp_index < len(self.waypoints):

            # Waypoints are in global ENU frame
            target = self.waypoints[self.current_wp_index][:2].copy()
            if hasattr(self, "_goto_deadline") and self.node_time() < self._goto_deadline:
                return
            self.send_trajectory(target)

            if np.linalg.norm(self.current_position[:2] - target) < self.distance_threshold:
                self.get_logger().info(
                    f"Waypoint {self.current_wp_index} reached for {self.drone_ns}"
                )
                self.current_wp_index += 1

    def command_callback(self):
        # Uses the low-level trained GC-RL policy
        if not (self.offboard_mode and self.start_flag):
            return
        
        if self.last_pose_time is None or (self.node_time() - self.last_pose_time) > self.pose_timeout:
            self.get_logger().warn(f"No recent position data for {self.drone_ns}, hovering at current location")
            if np.linalg.norm(self.current_position) > 0:
                if hasattr(self, "_goto_deadline") and self.node_time() < self._goto_deadline:
                    return
                self.send_trajectory(self.current_position[:2].copy())
            return
        
        if hasattr(self, "waypoints"):
            if self.current_wp_index < len(self.waypoints):

                target = self.waypoints[self.current_wp_index][:2].copy()

                if np.linalg.norm(self.current_position[:2] - self.home) < self.distance_threshold:
                    self.viz_flag = True

                if self.viz_flag:
                    self.send_debug_trajectory(self.current_position[:2].copy())

                # self.get_logger().info(
                #     f"Current position: {self.current_position[:2]}, Target: {target}, "
                #     f"Next location: {self.next_location}"
                # )

                # Updates the high-level waypoints towards the main goal
                if np.linalg.norm(self.current_position[:2] - target) < self.distance_threshold:
                    self.get_logger().info(
                        f"Waypoint {self.current_wp_index} reached for {self.drone_ns}"
                    )
                    self.current_wp_index += 1
                    if self.current_wp_index >= len(self.waypoints):
                        self.get_logger().info(f"Mission completed for {self.drone_ns}")
                        return
                    target = self.waypoints[self.current_wp_index][:2].copy()

                # self.get_logger().info(f"Current position: {self.current_position}")
                # self.get_logger().info(f"Target: {target}")
                # self.get_logger().info(f"Next position: {self.next_location}")
                # self.get_logger().info(f"Distance to target is {np.linalg.norm(self.current_position[:2] - target)}")
                # self.get_logger().info(f"Distance to next location is {np.linalg.norm(self.current_position[:2] - self.next_location)}")

                # Updates the low-level waypoints towards the next location
                if np.linalg.norm(self.current_position[:2] - self.next_location) < self.distance_threshold:
                    # Input to the agent is normalized global ENU positions
                    observation = self.state / self.normalize_factor
                    # goal = ((target.copy() - self.origin_offset) / self.normalize_factor)
                    goal = self.highbay2env(target) / self.normalize_factor
                    state = {
                        "observation": observation,
                        "goal": goal,
                    }
                    self.get_logger().info(f"State: {state}")
                    action = self.agent.select_action(state)
                    self.get_logger().info(f"Action: {action}")
                    # Output of applying the action is in global ENU frame
                    observation = self.step(action)
                    # observation += self.origin_offset
                    observation = self.env2highbay(observation)
                    self.next_location = observation

                    self.get_logger().info(
                        f"Next location updated to {self.next_location} for {self.drone_ns}"
                    )

                    # if (np.linalg.norm(self.next_location - target) < self.distance_threshold or
                    if (np.linalg.norm(action) < 0.01):
                        self.get_logger().info(
                            f"Next location reached for {self.drone_ns}, moving to next waypoint"
                        )
                        self.current_wp_index += 1
                        if self.current_wp_index >= len(self.waypoints):
                            self.get_logger().info(f"Mission completed for {self.drone_ns}")
                            return

                for drone_ns, other_position in self.other_agent_positions.items():
                    self_id = self.drone_id
                    other_id = int(''.join(ch for ch in drone_ns if ch.isdigit()))
                    distance = np.linalg.norm(self.next_location - other_position)
                    if distance < self.distance_threshold * 2 and other_id < self_id:
                        self.get_logger().info(
                            f"Drone {drone_ns} is too close to {self.drone_ns}, waiting at current location"
                        )
                        # if hasattr(self, "_goto_deadline") and self.node_time() < self._goto_deadline:
                        #     return
                        self.send_trajectory(self.current_position[:2].copy())
                        return

                if hasattr(self, "_goto_deadline") and self.node_time() < self._goto_deadline:
                    return
                self.send_trajectory(self.next_location)

            else:
                # self.start_flag = False
                # self.current_state = "LAND"
                if self.mission_name:
                    self.get_logger().info(f"Mission {self.mission_name} completed for {self.drone_ns}")
                    self._send_ack(self.mission_name)
                    self.start_flag = False
                    self.mission_name = ""

    def discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int64)
        if i == self.walls.shape[0]:
            i -= 1
        if j == self.walls.shape[1]:
            j -= 1
        return (i, j)

    def is_blocked(self, state):
        if not (np.all(state >= np.zeros(2)) and np.all(state <= self.normalize_factor)):
            return True
        (i, j) = self.discretize_state(state)
        return self.walls[i, j] == 1

    def step(self, action):
        action = np.clip(action, -1. * np.ones(2), np.ones(2))
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self.is_blocked(new_state):
                    self.state = new_state

        return self.state.copy()


def main(args=None):
    rclpy.init(args=args)
    drone_controller = DroneController()
    rclpy.spin(drone_controller)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
