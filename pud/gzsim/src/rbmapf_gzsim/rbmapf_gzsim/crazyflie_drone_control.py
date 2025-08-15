import json
import yaml
import torch
import requests
import numpy as np
# from PIL import Image
from dotmap import DotMap

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from nav_msgs.msg import Path, Odometry
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Duration
from rbmapf_interfaces.msg import HabitatObservations  # type: ignore
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String, Empty, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped

from pud.algos.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.vision.vision_agent import LagVisionUVFDDPG


class DroneController(Node):
    def __init__(self, waypoint_follow=False):
        super().__init__("drone_controller")

        self.declare_parameter('files', ['config.yaml', 'ckpt.pth', 'walls.npy'])
        self.declare_parameter('drone_id', 1)
        self.declare_parameter('num_drones', 1)
        self.declare_parameter('visual', 'False')
        self.declare_parameter('drone_ns', "/crazyflie_1")

        drone_ns = self.get_parameter('drone_ns').get_parameter_value().string_value
        drone_id = self.get_parameter('drone_id').get_parameter_value().integer_value
        num_drones = self.get_parameter('num_drones').get_parameter_value().integer_value
        files = self.get_parameter('files').get_parameter_value().string_array_value
        self.habitat = self.get_parameter('visual').get_parameter_value().string_value.lower() == 'true'

        self.node_name = f"drone_controller_{drone_id}"

        self.start_flag = False
        self.habitat_state = None
        self.drone_ns = drone_ns
        self.drone_id = drone_id
        self.num_drones = num_drones
        self.waypoint_follow = waypoint_follow

        self.ack_endpoint = "http://127.0.0.1:5000/done"
        self.mission_url = f"http://127.0.0.1:5000/?drone_id={self.drone_id}"

        if self.habitat:
            config_file, ckpt_file, walls_file, bounds_file = files
        else:
            config_file, ckpt_file, walls_file = files

        self.walls = np.load(walls_file)
        rows, cols = self.walls.shape
        self.normalize_factor = np.array([rows, cols]) if not self.habitat else np.ones(2)
        self.origin_offset = np.array([-cols / 2.0, -rows / 2.0])

        if self.habitat:
            self.bounds = np.loadtxt(bounds_file, delimiter=',')
            self.get_logger().info(f"Bounds loaded: {self.bounds}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        config = DotMap(config)
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_agent(config, ckpt_file)

        self._init_publishers()
        self._init_subscribers()
        self._init_visualizers()

        self.counter = 0
        self.timer = self.create_timer(0.1, self.timer_callback)
        callback_fn = self.waypoint_follower_callback if self.waypoint_follow else self.command_callback
        self.cmd_timer = self.create_timer(0.02, callback_fn)

        self.bridge = CvBridge()

        self.kp = 0.5
        self.altitude = 1.0
        self.max_speed = 1.0
        self.altitude += (self.drone_id - 1) * 0.5
        self.current_state = "IDLE"
        self.offboard_mode = False
        self.distance_threshold = 0.1 if not self.habitat else 0.15
        self.current_position = np.zeros(3, dtype=float)
        self.current_orientation = np.zeros(4, dtype=float)

        self.other_agent_homes = {}
        self.other_agent_positions = {}

    def _init_agent(self, config, ckpt_file):
        if not self.habitat:
            self.agent = DRLDDPGLag(
                4, 2, 1,
                CriticCls=GoalConditionedCritic,
                device=torch.device(config.device),
                **config.agent,
            )
        else:
            config.agent["action_dim"] = 2  # type: ignore
            config.agent["max_action"] = float(1.0)  # type: ignore
            self.agent = LagVisionUVFDDPG(
                width=config.env.simulator_settings.width,
                height=config.env.simulator_settings.height,
                in_channels=4,
                act_fn=torch.nn.SELU,
                encoder="VisualEncoder",
                device=torch.device(config.device),
                **config.agent.toDict(),
                cost_kwargs=config.agent_cost_kwargs.toDict(),
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

    def _init_subscribers(self):
        if self.habitat:
            self.create_subscription(
                HabitatObservations,
                f"{self.drone_ns}/camera/habitat_observations",
                lambda msg: self.habitat_observations_callback(msg),
                10,
            )
        self.waypoint_subscriber = self.create_subscription(
            Float32MultiArray,
            f"{self.drone_ns}/waypoints",
            self.waypoints_callback,
            10,
        )
        # self.create_subscription(
        #     Empty,
        #     "/start_mission",
        #     self.start_callback,
        #     QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        # )
        self.odom_subscriber = self.create_subscription(
            Odometry,
            f"{self.drone_ns}/odom",
            self.position_callback,
            10,
        )
        for i in range(1, self.num_drones + 1):
            drone_ns = f"/crazyflie_{i}"
            if drone_ns != self.drone_ns:
                self.create_subscription(
                    Odometry,
                    f"{drone_ns}/odom",
                    lambda msg, ns=drone_ns: self.other_position_callback(ns, msg),
                    10,
                )
                self.create_subscription(
                    Float32MultiArray,
                    f"{drone_ns}/waypoints",
                    lambda msg, ns=drone_ns: self.other_waypoints_callback(ns, msg),
                    10,
                )

    def _init_publishers(self):
        if self.habitat:
            self.state_publisher = self.create_publisher(
                String,
                f"{self.drone_ns}/state",
                10
            )
        self.twist_publisher = self.create_publisher(
            Twist,
            f"{self.drone_ns}/cmd_vel",
            10
        )
        self.tfbr = TransformBroadcaster(self)

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
        self.wall_publisher = self.create_publisher(MarkerArray, 'wall_markers', 10)
        self.publish_wall_markers()

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

    def publish_wall_markers(self, height=5.0, resolution=1.0):
        marker_array = MarkerArray()

        idx = 0
        for i, j in zip(*np.where(self.walls == 1)):
            x = self.origin_offset[0] + (j + 0.5)*resolution
            y = self.origin_offset[1] + (i + 0.5)*resolution

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'walls'
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(height / 2.0)
            marker.scale.x = marker.scale.y = resolution
            marker.scale.z = height
            marker.color = ColorRGBA(r=0.3, g=0.3, b=0.3, a=1.0)
            marker.lifetime = Duration(sec=0)
            marker_array.markers.append(marker)  # type: ignore
            idx += 1

        self.wall_publisher.publish(marker_array)

    def adjust_position(self, position, to_env_frame=False):
        # If local is false then the position is the grid position in the global frame
        # If local is true then the position of the drone in the local frame
        if not self.habitat:
            return position + self.origin_offset if to_env_frame else position - self.origin_offset
        else:
            if not to_env_frame:
                habitat_env_x = position[0]
                habitat_env_y = self.altitude
                habitat_env_z = -position[1]
                habitat_env_pos = np.array([habitat_env_x, habitat_env_y, habitat_env_z], dtype=np.float32)
                habitat_x = habitat_env_pos[2]
                habitat_y = habitat_env_pos[0]

                # habitat_x, habitat_y = -position[1], position[0]
                grid_x = (
                    ((habitat_x - self.bounds[2]) / 0.4)
                    .round()
                    .astype(int)
                )
                grid_y = (
                    ((habitat_y - self.bounds[0]) / 0.4)
                    .round()
                    .astype(int)
                )
                position = np.array([grid_x, grid_y], dtype=np.float32)
            else:
                habitat_x = position[0] * 0.4 + self.bounds[2]
                habitat_y = position[1] * 0.4 + self.bounds[0]
                habitat_env_x, habitat_env_y, habitat_env_z = habitat_y, self.altitude, habitat_x
                position = np.array([habitat_env_x, -habitat_env_z], dtype=np.float32)
                # position = np.array([-habitat_y, habitat_x, position[2]], dtype=np.float32)
            return position

    def waypoints_callback(self, msg: Float32MultiArray):
        # Waypoints are in global ENU frame with first entry as home, last entry as goal
        self.current_wp_index = 1
        self.waypoints = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        self.publish_markers()

        self.home = self.waypoints[0][:2].copy()
        # State needs to be in the coordinate frame that the GCRL policy was trained with
        # i.e origin is bottom left corner of the walls matrix not the origin of simulator!
        # self.state = self.home.copy() - self.origin_offset
        self.state = self.adjust_position(self.home, to_env_frame=False)
        self.next_location = self.home.copy()  # Cannot be zeros as its in global ENU frame

    def other_waypoints_callback(self, drone_ns, msg):
        other_waypoints = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        other_home = other_waypoints[0][:2].copy()
        self.other_agent_homes[drone_ns] = other_home
        self.other_agent_positions[drone_ns] = other_home.copy()  # Initialize position

    def habitat_observations_callback(self, msg: HabitatObservations):
        cat_goals = np.zeros((4, 32, 32, 4), dtype=np.uint8)
        cat_observations = np.zeros((4, 32, 32, 4), dtype=np.uint8)

        for idx, direction in enumerate(['forward', 'right', 'backward', 'left']):
            goal_msg = getattr(msg, f'goal_{direction}')
            goal_cv2 = self.bridge.imgmsg_to_cv2(goal_msg)
            cat_goals[idx] = goal_cv2

            obs_msg = getattr(msg, f'observation_{direction}')
            obs_cv2 = self.bridge.imgmsg_to_cv2(obs_msg)
            cat_observations[idx] = obs_cv2

        self.habitat_state = {
            "goal": cat_goals,
            "observation": cat_observations
        }

        # if self.habitat_state is not None:

        #     N, H, W, C = cat_goals.shape
        #     final_height = H + 2 * 1
        #     final_width = W * N + 1 * (N + 1)
        #     obs_frame = np.zeros((final_height, final_width, C), dtype=cat_observations.dtype)
        #     goal_frame = np.zeros_like(obs_frame, dtype=cat_goals.dtype)
        #     for i in range(C):
        #         obs_frame[:, :, i] = 1 if i == 0 else 0
        #         goal_frame[:, :, i] = 1 if i == 0 else 0
        #     for i in range(N):
        #         start_x = 1 + i * (W + 1)
        #         obs_frame[1 : H + 1, start_x : start_x + W, :] = Image.fromarray(cat_observations[i, :, :, :])
        #         goal_frame[1 : H + 1, start_x : start_x + W, :] = Image.fromarray(cat_goals[i, :, :, :])

        #     obs_image = Image.fromarray(obs_frame)
        #     obs_image.save("observation_image.png")
        #     goal_image = Image.fromarray(goal_frame)
        #     goal_image.save("goal_image.png")

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
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y
        self.current_position[2] = msg.pose.pose.position.z
        self.current_orientation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        # self.publish_transform_callback()

    def other_position_callback(self, drone_ns, msg):
        # Other drone position is in NED frame
        other_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float)
        if drone_ns in self.other_agent_positions:
            self.other_agent_positions[drone_ns] = other_position

    def timer_callback(self):
        # self.get_logger().info(f"Current state: {self.current_state} for {self.drone_ns}")
        self.start_flag, self.mission_name, self.land_ack = self._ready_to_start()
        # self.get_logger().info(f"Start flag for {self.drone_ns}: {self.start_flag} with mission {self.mission_name}")

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

    def takeoff(self):
        takeoff_cmd = Twist()
        self.offboard_mode = False

        if self.current_state == "TAKEOFF":
            takeoff_cmd.linear.z = 0.5
            if self.current_position[2] > self.altitude:
                takeoff_cmd.linear.z = 0.0
                self.current_state = "OFFBOARD"
        self.get_logger().info(f"Takeoff command sent to {self.drone_ns}")
        self.twist_publisher.publish(takeoff_cmd)

    def land(self):
        land_cmd = Twist()
        self.offboard_mode = False

        if self.current_state == "LAND":
            land_cmd.linear.z = -0.5
            if self.current_position[2] < self.distance_threshold:
                land_cmd.linear.z = 0.0
                self.current_state = "IDLE"
        self.get_logger().info(f"Landing the drone {self.drone_ns}")
        self.twist_publisher.publish(land_cmd)

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
        error_vector = position - self.current_position[:2]
        distance = np.linalg.norm(error_vector)

        twist_cmd = Twist()
        if distance >= self.distance_threshold:
            velocity = self.kp * error_vector
            magnitude = np.linalg.norm(velocity)
            if magnitude > self.max_speed:
                scale = 1.0 / magnitude
                velocity *= scale

            z_error = self.altitude - self.current_position[2]
            vz = self.kp * z_error

            twist_cmd.linear.x = velocity[0]
            twist_cmd.linear.y = velocity[1]
            twist_cmd.linear.z = np.clip(vz, -self.max_speed, self.max_speed)

            twist_cmd.angular.x = 0.0
            twist_cmd.angular.y = 0.0
            twist_cmd.angular.z = 0.0

        # self.get_logger().info(f"Sending twist command as {twist_cmd} for {self.drone_ns}")
        self.twist_publisher.publish(twist_cmd)

    def publish_transform_callback(self):
        t_base = TransformStamped()
        # t_base.header.stamp = self.get_clock().now().to_msg()
        t_base.header.frame_id = 'world'
        t_base.child_frame_id = self.drone_ns
        t_base.transform.translation.x = self.current_position[0]
        t_base.transform.translation.y = self.current_position[1]
        t_base.transform.translation.z = self.current_position[2]
        t_base.transform.rotation.x = self.current_orientation[0]
        t_base.transform.rotation.y = self.current_orientation[1]
        t_base.transform.rotation.z = self.current_orientation[2]
        t_base.transform.rotation.w = self.current_orientation[3]
        try:
            self.tfbr.sendTransform(t_base)
        except Exception:
            self.get_logger().info("Could not publish pose tf")

    def waypoint_follower_callback(self):
        if self.offboard_mode and self.start_flag:
            if hasattr(self, "waypoints") and self.current_wp_index < len(self.waypoints):

                # Waypoints are in global ENU frame
                target = self.waypoints[self.current_wp_index][:2].copy()
                self.send_trajectory(target)

                if np.linalg.norm(self.current_position[:2] - target) < self.distance_threshold:
                    self.get_logger().info(
                        f"Waypoint {self.current_wp_index} reached for {self.drone_ns}"
                    )
                    self.current_wp_index += 1

    def command_callback(self):
        # Uses the low-level trained GC-RL policy
        if self.offboard_mode and self.start_flag:
            if hasattr(self, "waypoints"):
                if self.current_wp_index < len(self.waypoints):

                    target = self.waypoints[self.current_wp_index][:2].copy()
                    self.send_debug_trajectory(self.current_position[:2])

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

                    # Updates the low-level waypoints towards the next location
                    if np.linalg.norm(self.current_position[:2] - self.next_location) < self.distance_threshold:
                        # Input to the agent is normalized global ENU positions
                        observation = self.state / self.normalize_factor
                        # goal = ((target.copy() - self.origin_offset) / self.normalize_factor)
                        goal = self.adjust_position(target, to_env_frame=False) / self.normalize_factor
                        if self.habitat:
                            state = {
                                "observation_grid": observation.tolist(),
                                "goal_grid": goal.tolist(),
                            }
                            state_msg = String()
                            state_msg.data = json.dumps(state)
                            self.state_publisher.publish(state_msg)
                            if self.habitat_state is not None:
                                action = self.agent.select_action(self.habitat_state)
                                self.habitat_state = None
                                self.get_logger().info(f"Habitat Action for {self.drone_ns}: {action}")
                            else:
                                return
                        else:
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
                        observation = self.adjust_position(observation, to_env_frame=True)
                        self.next_location = observation

                        self.get_logger().info(
                            f"Next location updated to {self.next_location} for {self.drone_ns}"
                        )

                        if (np.linalg.norm(self.next_location - target) < self.distance_threshold or
                                np.linalg.norm(action) < 0.01):
                            self.get_logger().info(
                                f"Next location reached for {self.drone_ns}, moving to next waypoint"
                            )
                            self.current_wp_index += 1
                            if self.current_wp_index >= len(self.waypoints):
                                self.get_logger().info(f"Mission completed for {self.drone_ns}")
                                return

                    for drone_ns, other_position in self.other_agent_positions.items():
                        self_id = int(self.drone_ns.split("_")[-1])
                        other_id = int(drone_ns.split("_")[-1])
                        distance = np.linalg.norm(self.next_location - other_position)
                        if distance < self.distance_threshold * 2 and other_id < self_id:
                            self.get_logger().info(
                                f"Drone {drone_ns} is too close to {self.drone_ns}, waiting at current location"
                            )
                            self.send_trajectory(self.current_position[:2])
                            return

                    self.send_trajectory(self.next_location)

                else:
                    # Send the ack back to the middleware server about mission completion
                    if self.mission_name:
                        self.get_logger().info(f"Mission {self.mission_name} completed for {self.drone_ns}")
                        self._send_ack(self.mission_name)
                        self.start_flag = False
                        # self.current_state = "LAND"
                        self.mission_name = ""

    def discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int64)
        if i == self.walls.shape[0]:
            i -= 1
        if j == self.walls.shape[1]:
            j -= 1
        return (i, j)

    def is_blocked(self, state):
        if not (np.all(state >= np.zeros(2)) and np.all(state <= self.walls.shape)):
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
