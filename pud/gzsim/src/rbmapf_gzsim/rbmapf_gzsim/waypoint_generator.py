import rclpy
import requests
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty
from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import argument_parser
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


class WaypointGeneratorNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator_node')
        self.declare_parameter('interface', 'px4')
        self.args = argument_parser()
        habitat = self.args.visual == 'True'

        if habitat:
            # TODO: The implementations of the generate_wps for habitat and pointenv have diverged!
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_habitatenv import generate_wps
        else:
            from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.control_pointenv import generate_wps

        self.replan_url = "http://127.0.0.1:5000/plan"
        self.ack_endpoint = "http://127.0.0.1:5000/done_sync"
        self._generate_waypoints_func = generate_wps
        interface = self.get_parameter('interface').get_parameter_value().string_value
        self.qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.waypoint_publishers = []
        self.current_agent_problems = []
        for agent_idx in range(self.args.num_agents):
            instance_idx = agent_idx + 1
            if interface != "cf":
                wp_topic = f"/{interface}_{instance_idx}/waypoints"
            else:
                wp_topic = f"/{interface}{instance_idx}/waypoints"
            waypoint_publisher = self.create_publisher(
                Float32MultiArray,
                wp_topic,
                self.qos_profile
            )
            self.waypoint_publishers.append(waypoint_publisher)

        self.problem_start = 0
        self._generate_waypoints()
        self.problem_start += 1
        self.replan_timer = self.create_timer(0.1, self.replan_callback)
        self.waypoint_gen_finished = self.create_publisher(Empty, '/waypoints_generated', self.qos_profile)
        self.waypoints_gen_finished_timer = self.create_timer(1.0, self.publish_waypoints_gen_finished)

    def _generate_waypoints(self, recovery=list()):

        agents_waypoints = self._generate_waypoints_func(self.args, problem_start=self.problem_start, recovery=recovery, debug=True)
        agent_idx = self.problem_start * self.args.team_size
        for waypoints in agents_waypoints:

            agent_start, agent_end = waypoints[0], waypoints[-1]
            self.current_agent_problems.append({'start': agent_start, 'end': agent_end})
            # agent_idx += (self.problem_start % self.args.team_size) * self.args.team_size
            agent_wps = np.array(waypoints)
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = agent_wps.flatten().tolist()
            self.waypoint_publishers[agent_idx].publish(waypoint_msg)
            self.get_logger().info(f'Published waypoints for agent {agent_idx + 1}')
            agent_idx += 1

    def publish_waypoints_gen_finished(self):
        self.waypoint_gen_finished.publish(Empty())
        self.get_logger().info('Waypoints generation finished published')
        self.waypoints_gen_finished_timer.cancel()

    def _plan_next_mission(self):
        try:
            resp = requests.get(self.replan_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            # self.get_logger().info(f"Response from server: {data}")
            sync_ready = data.get("sync_ready", False) is True
            sync_name = data.get("sync_name", "")
            recovery = data.get("recovery", dict())
            return sync_ready, sync_name, recovery
        except (requests.RequestException, ValueError) as e:
            self.get_logger().error(f"Request failed or invalid JSON: {e}")
            return False, "", dict()

    def _send_ack(self, mission_name):
        try:
            data = {
                "sync_name": self.sync_name,
            }
            response = requests.post(self.ack_endpoint, json=data)

            if response.status_code != 200:
                self.get_logger().error(f"Request failed with status code: {response.status_code}")
                self.get_logger().error(f"Response: {response.text}")
            # else:
            #     self.get_logger().info(f"Response: {response.json()}")
        except (requests.RequestException, ValueError) as e:
            self.get_logger().error(f"Failed to send ack for mission {mission_name}: {e}")

    def replan_callback(self):
        sync_ready, sync_name, recovery = self._plan_next_mission()
        if sync_ready:
            self.sync_name = sync_name
            self.get_logger().info(f"New mission {self.sync_name} received, replanning...")
            recovery_dict = list()
            if len(recovery) > 0:
                recovery_agent = recovery.get("recovery_agent", 0)
                to_recover = recovery.get("to_recover", [])
                for agent_idx in to_recover:
                    recovery_dict.append({'start': self.current_agent_problems[recovery_agent]['start'], 'goal': self.current_agent_problems[agent_idx]['goal']})
                self.get_logger().info("Recovery dictionary:" + str(recovery_dict))
            self._generate_waypoints(recovery_dict)
            self.problem_start += 1
            self._send_ack(self.sync_name)
            self.get_logger().info(f"Mission {self.sync_name} acknowledged.")
            self.sync_name = ""


def main():
    rclpy.init()
    node = WaypointGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
