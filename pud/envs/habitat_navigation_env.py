from pud.dependencies import *


import habitat_sim
from numpy.typing import NDArray
from typing import Tuple, Dict, Union


class HabitatNavigationEnv(gym.Env):

    def __init__(
        self, scene: str, action_noise: float = 1.0, simulator_settings: dict = {}
    ):

        self._scene = scene

        if not simulator_settings:
            # If simulator settings were not specified then use the default settings
            self._simulator_settings = {
                "scene": self._scene,  # Scene path
                "width": 256,  # Spatial resolution of the observations
                "height": 256,
                "default_agent": 0,  # Index of the default agent
                "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
            }
        else:
            self._simulator_settings = simulator_settings
            assert "scene" in self._simulator_settings
            assert "width" in self._simulator_settings
            assert "height" in self._simulator_settings
            assert "default_agent" in self._simulator_settings
            assert "sensor_height" in self._simulator_settings

        self._action_noise = action_noise

        # Height and Width of the camera resolution!
        self._width = self._simulator_settings["width"]
        self._height = self._simulator_settings["height"]

        self._configuration = self.make_habitat_configuration()
        self._simulator = habitat_sim.Simulator(self._configuration)

        self._agent = self._simulator.initialize_agent(
            self._simulator_settings["default_agent"]
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, self._height, self._width, 4), dtype=np.uint8
        )
        # The last 4 is there because the default camera sensor returns 4 channels for some reason. Not sure why

        self.observation_space = gym.spaces.Dict(
            {
                **self._simulator.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        # Ensure that the pathfinder utility of the simulator is loaded
        assert self._simulator.pathfinder.is_loaded

        # Simulator's top-down visualizer's parameters
        self._meters_per_pixel = 0.1
        self._vertical_slice = self._simulator.pathfinder.get_bounds()[0][1]
        self._walls = self._simulator.get_topdown_view(
            self._meters_per_pixel, self._vertical_slice
        )
        self._wall_height, self._wall_width = self._walls.shape
        self._apsp = self._compute_apsp(self._walls)

        self.reset()

    @property
    def walls(self):
        return self._walls

    def make_habitat_configuration(self):

        simulator_cfg = habitat_sim.SimulatorConfiguration()
        simulator_cfg.scene_id = self._simulator_settings["scene"]

        # Generate a default RBG camera specification and attach it to each robot
        rgb_sensor_spec_forward = habitat_sim.sensor.CameraSensorSpec()
        rgb_sensor_spec_forward.uuid = "color_sensor_forward"
        rgb_sensor_spec_forward.sensor_type = habitat_sim.sensor.SensorType.COLOR
        rgb_sensor_spec_forward.resolution = [self._height, self._width]
        rgb_sensor_spec_forward.position = [
            0.0,
            self._simulator_settings["sensor_height"],
            0.0,
        ]

        rgb_sensor_spec_right = habitat_sim.sensor.CameraSensorSpec()
        rgb_sensor_spec_right.uuid = "color_sensor_forward"
        rgb_sensor_spec_right.sensor_type = habitat_sim.sensor.SensorType.COLOR
        rgb_sensor_spec_right.resolution = [self._height, self._width]
        rgb_sensor_spec_right.position = [
            0.0,
            self._simulator_settings["sensor_height"],
            0.0,
        ]
        rgb_sensor_spec_right.orientation = [0.0, -np.pi / 2, 0.0]

        rgb_sensor_spec_backward = habitat_sim.sensor.CameraSensorSpec()
        rgb_sensor_spec_backward.uuid = "color_sensor_forward"
        rgb_sensor_spec_backward.sensor_type = habitat_sim.sensor.SensorType.COLOR
        rgb_sensor_spec_backward.resolution = [self._height, self._width]
        rgb_sensor_spec_backward.position = [
            0.0,
            self._simulator_settings["sensor_height"],
            0.0,
        ]
        rgb_sensor_spec_backward.orientation = [0.0, np.pi, 0.0]

        rgb_sensor_spec_left = habitat_sim.sensor.CameraSensorSpec()
        rgb_sensor_spec_left.uuid = "color_sensor_forward"
        rgb_sensor_spec_left.sensor_type = habitat_sim.sensor.SensorType.COLOR
        rgb_sensor_spec_left.resolution = [self._height, self._width]
        rgb_sensor_spec_left.position = [
            0.0,
            self._simulator_settings["sensor_height"],
            0.0,
        ]
        rgb_sensor_spec_left.orientation = [0.0, np.pi / 2, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [
            rgb_sensor_spec_forward,
            rgb_sensor_spec_right,
            rgb_sensor_spec_backward,
            rgb_sensor_spec_left,
        ]

        return habitat_sim.Configuration(simulator_cfg, agent_cfg)

    def reset(self):
        self.state = np.zeros((4, self._height, self._width, 4), dtype=np.uint8)
        observations = self._simulator.reset()
        for idx, (key, value) in enumerate(observations.items()):
            assert value.shape == (self._height, self._width, 4)
            self.state[idx] = value
        return self.state.copy()

    def step(self, action: NDArray):
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        num_substeps = 10
        dt = 1.0 / num_substeps
        for _ in np.linspace(0, 1, num_substeps):
            # The agent's positions are in 3D space where index 0 and 2 are x and z and can be moved as
            # y moves the agent up and down
            for axis, axis_action in [(2, action[0]), (0, action[1])]:
                agent_state = self._simulator.get_agent(
                    self._simulator_settings["default_agent"]
                ).get_state()
                new_state = agent_state.position.copy()
                new_state[axis] += dt * axis_action
                if self._simulator.pathfinder.is_navigable(new_state):
                    agent_state.position = new_state
                    agent_state.sensor_states = {}
                    self._agent.set_state(agent_state)

        done = False
        agent_state = self._simulator.get_agent(
            self._simulator_settings["default_agent"]
        ).get_state()
        agent_position = np.array([agent_state.position[2], agent_state.position[0]])
        rew = -1.0 * np.linalg.norm(agent_position)

        new_state = np.zeros((4, self._height, self._width, 4), dtype=np.uint8)
        observations = self._simulator.get_sensor_observations()
        for idx, (key, value) in enumerate(observations.items()):
            assert value.shape == (self._height, self._width, 4)
            new_state[idx] = value
        self.state = new_state

        return self.state.copy(), rew, done, {}

    def _compute_apsp(self, walls: NDArray):

        # NOTE: walls[i, j] is True if (i, j) is traversable and False otherwise

        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j]:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0:
                            continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if not walls[i, j]:
                            continue  # Don't add edges to walls
                        if not walls[i + di, j + dj]:
                            continue  # Don't add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float32("inf"))
        for (i1, j1), dist_dict in nx.shortest_path_length(g):
            for (i2, j2), d in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist


class GoalConditionedHabitatPointWrapper(gym.Wrapper):
    """Wrapper that appends goal to observation produced by habitat environment."""

    def __init__(
        self,
        env: HabitatNavigationEnv,
        prob_constraint: float = 0.8,
        min_dist: float = 0.0,
        max_dist: float = 4.0,
        threshold_distance: float = 1.0,
    ):
        """Initialize the environment.

        Args:
          env: an environment.
          prob_constraint: (float) Probability that the distance constraint is
            followed after resetting.
          min_dist: (float) When the constraint is enforced, ensure the goal is at
            least this far from the initial observation.
          max_dist: (float) When the constraint is enforced, ensure the goal is at
            most this far from the initial observation.
          threshold_distance: (float) States are considered equivalent if they are
            at most this far away from one another.
        """

        self._min_dist = min_dist
        self._max_dist = max_dist
        self._prob_constraint = prob_constraint
        self._threshold_distance = threshold_distance
        super(GoalConditionedHabitatPointWrapper, self).__init__(env)

        simulator_bounds = env._simulator.pathfinder.get_bounds()
        simulator_bounds_lb = np.array([simulator_bounds[0][2], simulator_bounds[0][0]])
        simulator_bounds_ub = np.array([simulator_bounds[1][2], simulator_bounds[1][0]])
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "goal": gym.spaces.Box(
                    low=simulator_bounds_lb,
                    high=simulator_bounds_ub,
                    dtype=np.float32,
                ),
            }
        )

    def _get_distance(self, position: NDArray, goal: NDArray) -> float:
        """Compute the shortest path distance.

        NOTE: This distance is *not* used for training."""
        (i1, j1) = self._discretize_state(
            position, (self.env._wall_height, self.env._wall_width)
        )
        (i2, j2) = self._discretize_state(
            goal, (self.env._wall_height, self.env._wall_width)
        )
        return self._apsp[i1, j1, i2, j2]

    def _discretize_state(
        self, position: NDArray, grid_resolution: Tuple[int, int]
    ) -> Tuple[int, int]:
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        lower_bound, upper_bound = self.env._simulator.pathfinder.get_bounds()

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        grid_x = int((position[0] - lower_bound[2]) / grid_size[0])
        grid_y = int((position[1] - lower_bound[0]) / grid_size[1])
        return grid_x, grid_y

    def _undiscretize_state(
        self, grid_position: Tuple[int, int], grid_resolution: Tuple[int, int]
    ) -> Tuple[float, float]:

        lower_bound, upper_bound = self.env._simulator.pathfinder.get_bounds()

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        realworld_x = lower_bound[2] + grid_position[0] * grid_size[0]
        realworld_y = lower_bound[0] + grid_position[1] * grid_size[1]
        return realworld_x, realworld_y

    def reset(self) -> Dict:
        count = 0
        goal = None
        while goal is None:
            obs = self.env.reset()
            agent_state = self.env._simulator.get_agent(
                self.env._simulator_settings["default_agent"]
            ).get_state()
            agent_position = np.array(
                [agent_state.position[2], agent_state.position[0]]
            )
            (agent_position, goal) = self._sample_goal(agent_position)
            count += 1
            if count > 1000:
                print("WARNING: Unable to find goal within constraints.")
        self._goal = goal
        return {
            "observation": obs,
            "goal": self._goal,
        }

    def step(self, action: NDArray) -> Tuple[Dict, float, bool, Dict]:
        obs, _, _, _ = self.env.step(action)
        rew = -1.0

        agent_state = self.env._simulator.get_agent(
            self.env._simulator_settings["default_agent"]
        ).get_state()
        agent_position = np.array([agent_state.position[2], agent_state.position[0]])
        done = self._is_done(agent_position, self._goal)
        return (
            {
                "observation": obs,
                "goal": self._goal,
            },
            rew,
            done,
            {},
        )

    def set_sample_goal_args(
        self,
        prob_constraint: Union[float, None] = None,
        min_dist: Union[float, None] = None,
        max_dist: Union[float, None] = None,
    ):
        assert min_dist is not None
        assert min_dist >= 0

        assert max_dist is not None
        assert max_dist >= min_dist

        assert prob_constraint is not None

        self._min_dist = min_dist
        self._max_dist = max_dist
        self._prob_constraint = prob_constraint

    def _is_done(self, agent_position: NDArray, goal: NDArray) -> bool:
        """Determines whether observation equals goal."""
        return np.linalg.norm(agent_position - goal) < self._threshold_distance

    def _sample_goal(self, agent_position: NDArray) -> Tuple[NDArray, NDArray]:
        """Sampled a goal observation."""
        if np.random.random() < self._prob_constraint:
            return self._sample_goal_constrained(
                agent_position, self._min_dist, self._max_dist
            )
        else:
            return self._sample_goal_unconstrained(agent_position)

    def _sample_goal_constrained(
        self, agent_position: NDArray, min_dist: float, max_dist: float
    ) -> Tuple[NDArray, NDArray]:
        """Samples a goal with dist min_dist <= d(observation, goal) <= max_dist.

        Args:
          agent_position: The current position of the agent (without goal).
          min_dist: (int) Minimum distance to goal.
          max_dist: (int) Maximum distance to goal.
        Returns:
          agent_position: The current position of the agent (without goal).
          goal: A goal observation that satifies the constraints.
        """
        (i, j) = self._discretize_state(
            agent_position, (self.env._wall_height, self.env._wall_width)
        )
        mask = np.logical_and(
            self.env._apsp[i, j] >= min_dist, self.env._apsp[i, j] <= max_dist
        )
        mask = np.logical_and(mask, self.env._walls == True)
        candidate_states = np.where(mask)
        num_candidate_states = len(candidate_states[0])
        if num_candidate_states == 0:
            return (agent_position, None)
        goal_index = np.random.choice(num_candidate_states)
        goal = np.array(
            [candidate_states[0][goal_index], candidate_states[1][goal_index]],
            dtype=np.float32,
        )
        goal += np.random.uniform(size=2)
        dist_to_goal = self._get_distance(agent_position, goal)
        assert min_dist <= dist_to_goal <= max_dist

        undiscretized_goal_x, undiscretized_goal_y = self._undiscretize_state(
            (int(goal[0]), int(goal[1])), (self.env._wall_height, self.env._wall_width)
        )
        undiscretized_goal = np.array(
            [undiscretized_goal_y, self.env._vertical_slice, undiscretized_goal_x]
        )
        assert self.env._simulator.pathfinder.is_navigable(undiscretized_goal)

        return (agent_position, np.array([undiscretized_goal_x, undiscretized_goal_y]))

    def _sample_goal_unconstrained(
        self, agent_position: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Samples a goal without any constraints.

        Args:
          agent_position: The current position of the agent (without goal)
        Returns:
          observation: observation (without goal).
          goal: a goal observation.
        """

        agent_state = self.env._simulator.get_agent(
            self.env._simulator_settings["default_agent"]
        ).get_state()
        current_island = self.env._simulator.pathfinder.get_island(agent_state.position)

        tries = 1
        sampled_goal = self.env._simulator.pathfinder.get_random_navigable_point()
        is_navigable = self.env._simulator.pathfinder.is_navigable(sampled_goal)
        sampled_island = self.env._simulator.pathfinder.get_island(sampled_goal)
        valid = is_navigable and sampled_island == current_island
        while not valid:
            sampled_goal = self.env._simulator.pathfinder.get_random_navigable_point()
            is_navigable = self.env._simulator.pathfinder.is_navigable(sampled_goal)
            sampled_island = self.env._simulator.pathfinder.get_island(sampled_goal)
            valid = is_navigable and sampled_island == current_island
            tries += 1
            if tries > 1000:
                print("WARNING: Unable to find goal without constraints.")
        return (
            agent_position,
            self.env._simulator.pathfinder.get_random_navigable_point(),
        )

    @property
    def max_goal_dist(self) -> float:
        apsp = self.env._apsp
        return np.max(apsp[np.isfinite(apsp)])
