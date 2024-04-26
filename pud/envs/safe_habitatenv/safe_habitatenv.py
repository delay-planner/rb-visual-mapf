import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.axes import Axes
from typing import Tuple, Dict, Union

from pud.envs.habitat_navigation_env import HabitatNavigationEnv


class SafeHabitatNavigationEnv(HabitatNavigationEnv):
    def __init__(
        self,
        scene: str,
        height: Union[float, None] = None,
        action_noise: float = 1.0,
        simulator_settings: dict = {},
        apsp_path: Union[str, None] = None,
        # Cost specific arguments
        cost_fn_args: dict = {},
        cost_limit: float = 0.5,
        verbose: bool = False,
    ):

        super().__init__(scene, height, action_noise, simulator_settings, apsp_path)

        self.scene = scene
        self.cost_function = None
        self.cost_limit = cost_limit
        self.action_noise = action_noise
        self.cost_fn_cfg = cost_fn_args

        obstacles_x, obstacles_y = np.where(self._walls == 0)
        self.obstacles = np.stack([obstacles_x, obstacles_y], axis=-1).astype(float)

        t0 = time.time()
        if self.cost_fn_cfg.get("name") == "cosine":
            import functools
            from pud.envs.safe_pointenv.cost_functions import cost_from_cosine_distance

            self.cost_function = functools.partial(
                cost_from_cosine_distance, r=self.cost_fn_cfg["radius"]
            )
            self._cost_map = self._build_cost_map()

        self._safe_empty_states = self._gather_safe_empty_states(self.cost_limit)
        self.reset()
        print("[INFO] SafeHabitatNavigationEnv Setup: {} s".format(time.time() - t0))

    @property
    def cost_map(self):
        return self._cost_map

    def _gather_safe_empty_states(self, cost_limit: float) -> NDArray:
        """
        Due to the increased cost in reset, precompile a list of initial states here
        """
        empty_states = np.where(self._walls == 1)
        safe_empty_states = [[], []]

        for cx, cy in zip(*empty_states):
            # Only sample states whose costs are lower than an upper bound
            if self._cost_map[cx, cy] < cost_limit:
                safe_empty_states[0].append(cx)
                safe_empty_states[1].append(cy)

        safe_empty_states = np.column_stack(safe_empty_states)  # Nxd
        return safe_empty_states

    def _dist_two_blocks(self, xy: NDArray) -> Tuple[float, int]:
        """
        Calculate the distance between a float state xy and a blocked states i.e obstacles that are ints
        (from array indices)

        A block covers an square area of
            block_x -- block_x + 1
            block_y -- block_y + 1

        Args:
            xy (NDarray): [x, y]

        Returns:
            float: Calculated distance
            int: Index of the nearest block

        Example:
            xy = np.array([0.5, 0.6])
            block_xys = np.array([[0,1],[2,5]])

        Reference:
            https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """

        bxs_min = self.obstacles[:, 0]
        bys_min = self.obstacles[:, 1]
        x, y = xy

        dxs = np.maximum(bxs_min - x, x - (bxs_min + 1))
        dxs = np.maximum(dxs, 0.0)

        dys = np.maximum(bys_min - y, y - (bys_min + 1))
        dys = np.maximum(dys, 0)

        d2 = dxs**2.0 + dys**2.0
        ind_min = np.argmin(d2)
        d_min = np.sqrt(d2[ind_min])

        return d_min, ind_min  # type: ignore

    def _build_cost_map(self) -> NDArray:
        height, width = self._walls.shape
        cost_map = np.ones([height, width], dtype=float) * np.inf

        assert self.cost_function is not None

        for i in range(height):
            for j in range(width):
                min_distance, _ = self._dist_two_blocks(np.array([i, j]))
                cost_map[i, j] = self.cost_function(min_distance)

        return cost_map

    def _get_state_cost(self, xy: NDArray) -> float:
        """
        Assumes that the xy argument is the grid position and not the continuous position
        """
        assert self.cost_function is not None
        min_distance, _ = self._dist_two_blocks(xy)
        return self.cost_function(min_distance)  # type: ignore

    def _sample_safe_empty_state(self, cost_limit: float):
        """
        Must take the intersection with the empty states because state cost is computed from the center of the block
        """
        num_candidate_states = len(self._safe_empty_states)

        idx = np.random.randint(0, num_candidate_states)
        new_state = self._safe_empty_states[idx].astype(np.float32)

        undiscretized_new_state_x, undiscretized_new_state_y = self._undiscretize_state(
            (int(new_state[0]), int(new_state[1]))
        )
        undiscretized_new_state = np.array(
            [undiscretized_new_state_x, undiscretized_new_state_y]
        )

        # NOTE: Don't remove the checks below
        assert not self._is_blocked(undiscretized_new_state)
        assert self._get_state_cost(new_state) < self.cost_limit
        return new_state

    def seed(self, seed: int) -> None:
        self._simulator.seed(seed)

    def reset(self):
        if (not hasattr(self, "cost_limit")) or (not hasattr(self, "_cost_map")):
            print(
                "[INFO] Skipping the reset in HabitatNavigationEnv.__init__ because setup is not ready yet"
            )
            return

        # TODO: Perhaps suffer from label inbalance?
        # Extract a safe (x, y) position for the agent based on the navigation grid and cost map
        safe_agent_position = self._sample_safe_empty_state(cost_limit=self.cost_limit)
        agent_cost = self._get_state_cost(xy=safe_agent_position)
        info = {"cost": agent_cost}

        # Reset the simulator
        self.state = np.zeros((4, self._height, self._width, 4), dtype=np.uint8)
        observations = self._simulator.reset()

        # Spawn the agent at the corresponding real-world coordinates in the habitat environment
        safe_agent_position_x, safe_agent_position_y = self._undiscretize_state(
            (int(safe_agent_position[0]), int(safe_agent_position[1]))
        )
        self._update_agent_position(
            np.array([safe_agent_position_x, safe_agent_position_y])
        )

        # Get the initial observations and return the state
        observations = self._simulator.get_sensor_observations()

        for idx, (key, value) in enumerate(observations.items()):
            assert value.shape == (self._height, self._width, 4)  # type: ignore
            self.state[idx] = value
        return self.state.copy(), info

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, Dict]:
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        # NOTE: Use the maximum cost along the action segment
        agent_position = self._get_agent_position()
        (i, j) = self._discretize_state(agent_position)
        cost = self._get_state_cost(np.array([i, j]))

        num_substeps = 10
        dt = 1.0 / num_substeps
        for _ in np.linspace(0, 1, num_substeps):
            for axis, axis_action in enumerate(action):
                new_state = self._get_agent_position()
                new_state[axis] += dt * axis_action
                if not self._is_blocked(new_state):
                    self._update_agent_position(new_state)

                    (i, j) = self._discretize_state(new_state)
                    state_cost = self._get_state_cost(np.array([i, j]))
                    if cost < state_cost:
                        cost = state_cost

        done = False
        agent_position = self._get_agent_position()
        rew = float(-1.0 * np.linalg.norm(agent_position))

        new_state = np.zeros((4, self._height, self._width, 4), dtype=np.uint8)
        observations = self._simulator.get_sensor_observations()
        for idx, (key, value) in enumerate(observations.items()):
            assert value.shape == (self._height, self._width, 4)  # type: ignore
            new_state[idx] = value
        self.state = new_state

        return self.state.copy(), rew, done, {"cost": cost}


def display_map(
    topdown_map: NDArray,
    scene_name: str,
    cost_map: NDArray,
    cost_limit: float,
    ax: Axes,
    key_points: Union[NDArray, None] = None,
) -> Axes:
    ax.set_title(scene_name)
    ax.axis("off")

    height, width = topdown_map.shape
    for i, j in zip(*np.where(topdown_map == 0)):
        x = np.array([j, j + 1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i + 1, i + 1]) / float(height)
        ax.fill_between(x, y0, y1, color="grey")

    unsafe_points = np.where(cost_map > cost_limit)
    unsafe_points = np.column_stack(unsafe_points)
    ax.scatter(
        unsafe_points[:, 1] / float(width),
        unsafe_points[:, 0] / float(height),
        s=2,
        marker="o",
        c="red",
    )

    if key_points is not None:
        ax.plot(
            key_points[:, 1] / float(width),
            key_points[:, 0] / float(height),
            "b-o",
            markersize=0.5,
            alpha=0.8,
        )
        ax.scatter(
            key_points[0][1] / float(width),
            key_points[0][0] / float(height),
            marker="+",
            color="cyan",
            s=10,
            label="start",
        )
        ax.scatter(
            key_points[-1][1] / float(width),
            key_points[-1][0] / float(height),
            marker="*",
            color="green",
            s=10,
            label="goal",
        )

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    return ax


if __name__ == "__main__":

    scene = (
        "/home/mers-pluto/Desktop/Work/habitat_workspace/habitat-lab/data/scene_datasets/"
        "habitat-test-scenes/skokloster-castle.glb"
    )
    env = SafeHabitatNavigationEnv(
        scene=scene,
        height=0,
        cost_fn_args={"name": "cosine", "radius": 2.0},
        cost_limit=1.0,
    )
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax = display_map(env.walls, "Skokloster Castle", env.cost_map, env.cost_limit, ax=ax)  # type: ignore
    plt.show()
