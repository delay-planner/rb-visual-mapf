import gym
import gym.spaces
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, Union, List

from pud.envs.safe_pointenv.safe_wrappers import SafeTimeLimit
from pud.envs.safe_habitatenv.safe_habitatenv import SafeHabitatNavigationEnv


class SafeGoalConditionedHabitatPointWrapper(gym.Wrapper):
    """
    Wrapper that appends the goal to the observation produced by the habitat environment.
    Samples the goal with safety constraints
    Option 1: Sample goals of any distance subject to the step cost == 0
        Potential long distances, but there exists viable solutions
    Option 2: Distance constraints subject to step cost == 0
        Distance constraints, but there exists viable solutions
    Option 3: Distance constraints subject to 0 < step cost < cost limit
        Perhaps only good for training
        Distance constraints, but there may not exist viable solutions, but trajectories whose step cost < cost limit
    """

    def __init__(
        self,
        env: SafeHabitatNavigationEnv,
        prob_constraint: float = 0.8,
        min_dist: float = 0,
        max_dist: float = 4,
        min_cost: float = 0,
        max_cost: float = 1000,
        threshold_distance: float = 1.0,
    ):
        self.env = env

        self._min_dist = min_dist
        self._max_dist = max_dist
        self._min_cost = min_cost
        self._max_cost = max_cost
        self._prob_constraint = prob_constraint
        self._threshold_distance = threshold_distance
        super(SafeGoalConditionedHabitatPointWrapper, self).__init__(env)

        simulator_bounds = env._simulator.pathfinder.get_bounds()
        simulator_bounds_lb = np.array(
            [simulator_bounds[0][2], simulator_bounds[0][0]], dtype=float
        )
        simulator_bounds_ub = np.array(
            [simulator_bounds[1][2], simulator_bounds[1][0]], dtype=float
        )

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

    def _is_done(self, agent_position: NDArray, goal: NDArray) -> bool:
        """Determines whether observation equals goal.
        NOTE: Both the agent position and goal arguments are the 2D coordinates ([x, y])
        """
        return bool(np.linalg.norm(agent_position - goal) < self._threshold_distance)

    def _set_sample_goal_args(
        self,
        prob_constraint: Union[float, None] = None,
        min_dist: Union[float, None] = None,
        max_dist: Union[float, None] = None,
        min_cost: Union[float, None] = None,
        max_cost: Union[float, None] = None,
    ):
        assert min_dist is not None
        assert min_dist >= 0

        assert max_dist is not None
        assert max_dist >= min_dist

        assert min_cost is not None
        assert min_cost >= 0

        assert max_cost is not None
        assert max_cost >= min_cost

        assert prob_constraint is not None

        self._min_dist = min_dist
        self._max_dist = max_dist
        self._min_cost = min_cost
        self._max_cost = max_cost
        self._prob_constraint = prob_constraint

    def _sample_goal(
        self, agent_position: NDArray
    ) -> Tuple[NDArray, Union[NDArray, None]]:
        """Sampled a goal observation."""
        if np.random.random() < self._prob_constraint:
            return self._sample_goal_constrained(
                agent_position, self._min_dist, self._max_dist
            )
        else:
            return self._sample_goal_unconstrained(agent_position)

    def _sample_goal_constrained(
        self, agent_position: NDArray, min_dist: float, max_dist: float
    ) -> Tuple[NDArray, Union[NDArray, None]]:
        """Samples a goal with dist min_dist <= d(observation, goal) <= max_dist.

        Args:
          agent_position: The current position of the agent (without goal).
          min_dist: (int) Minimum distance to goal.
          max_dist: (int) Maximum distance to goal.
        Returns:
          agent_position: The current position of the agent (without goal).
          goal: A goal observation that satifies the constraints.
        """
        (i, j) = self.env._discretize_state(agent_position)
        mask = np.logical_and(
            self.env._apsp[i, j] >= min_dist, self.env._apsp[i, j] <= max_dist
        )
        mask = np.logical_and(mask, self.env._walls)
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

        undiscretized_goal_x, undiscretized_goal_y = self.env._undiscretize_state(
            (goal[0], goal[1])
        )
        undiscretized_goal = np.array([undiscretized_goal_x, undiscretized_goal_y])
        dist_to_goal = self.env._get_distance(agent_position, undiscretized_goal)

        assert min_dist <= dist_to_goal <= max_dist
        assert not self.env._is_blocked(undiscretized_goal)

        return (agent_position, undiscretized_goal)

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

        agent_position = self.env._get_agent_position()
        agent_sim_position = self.env._convert_grid_to_sim(agent_position)
        current_island = self.env._simulator.pathfinder.get_island(agent_sim_position)

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
        sampled_goal = np.array([sampled_goal[2], sampled_goal[0]], dtype=np.float32)
        return (
            agent_position,
            sampled_goal,
        )

    def _reset_original(self) -> Tuple[Dict, Dict]:

        count = 0
        goal = None
        info = {"cost": 0.0}
        while goal is None:
            obs, info = self.env.reset()  # type: ignore
            agent_position = self.env._get_agent_position()
            (agent_position, goal) = self._sample_goal(agent_position)
            count += 1
            if count > 1000:
                print("WARNING: Unable to find goal within constraints.")
        self._goal = goal
        return {
            "observation": obs,
            "goal": self._goal,
        }, info

    def reset(self) -> Tuple[Dict, Dict]:
        return self._reset_original()

    def step(self, action: NDArray) -> Tuple[Dict, float, bool, Dict]:
        obs, _, _, info = self.env.step(action)
        rew = -1.0

        agent_position = self.env._get_agent_position()
        done = self._is_done(agent_position, self._goal)
        return (
            {
                "observation": obs,
                "goal": self._goal,
            },
            rew,
            done,
            info,
        )

    @property
    def max_goal_dist(self):
        apsp = self.env._apsp
        return np.max(apsp[np.isfinite(apsp)])


class SafeGoalConditionedHabitatPointQueueWrapper(
    SafeGoalConditionedHabitatPointWrapper
):

    def __init__(
        self,
        env: SafeHabitatNavigationEnv,
        prob_constraint: float = 0.8,
        min_dist: float = 0,
        max_dist: float = 4,
        min_cost: float = 0,
        max_cost: float = 1000,
        threshold_distance: float = 1.0,
    ):
        """
        Reset using problems (start - goal) pairs from an external queue.
        If the queue is empty, use the default reset method
        """

        super(SafeGoalConditionedHabitatPointQueueWrapper, self).__init__(
            env=env,
            prob_constraint=prob_constraint,
            min_dist=min_dist,
            max_dist=max_dist,
            min_cost=min_cost,
            max_cost=max_cost,
            threshold_distance=threshold_distance,
        )
        self._pb_Q = []

    def get_Q_size(self):
        return len(self._pb_Q)

    def append_pbs(self, pb_list: List[Tuple]):
        self._pb_Q.extend(pb_list)


def set_safe_habitat_env_difficulty(
    eval_env: SafeGoalConditionedHabitatPointWrapper,
    difficulty: float,
    min_cost: float = 0.0,
    max_cost: float = 1.0,
):

    assert 0 <= difficulty <= 1

    max_goal_dist = eval_env.max_goal_dist
    eval_env._set_sample_goal_args(
        prob_constraint=1,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05),
        min_cost=min_cost,
        max_cost=max_cost,
    )


def safe_habitat_env_load_fn(
    env_kwargs: dict,
    cost_fn_kwargs: dict,
    max_episode_steps: int = 0,
    gym_env_wrappers: Tuple[gym.Wrapper] = (
        SafeGoalConditionedHabitatPointWrapper,
    ),  # type: ignore
    wrapper_kwargs: List[dict] = [],
    terminate_on_timeout=False,
):
    """
    Loads the selected environment and wraps it with the specified wrappers.

    Args:
      environment_name: Name for the environment to load.
      max_episode_steps: If None the max_episode_steps will be set to the default
        step limit defined in the environment's spec. No limit is applied if set
        to 0 or if there is no timestep_limit set in the environment's spec.
      gym_env_wrappers: Iterable with references to wrapper classes to use
        directly on the gym environment.
      wrapper_kwargs: args for gym_env_wrappers, empty list or [wrapper_1_arg, wrapper_2_arg, ...],
        where wrapper_N_arg could be empty tuple as a place holder
      terminate_on_timeout: Whether to set done = True when the max episode
                            steps is reached.

    Returns:
      An environment instance.
    """

    env = SafeHabitatNavigationEnv(**env_kwargs, cost_fn_args=cost_fn_kwargs)

    for idx, wrapper in enumerate(gym_env_wrappers):
        if idx < len(wrapper_kwargs):
            env = wrapper(env, **wrapper_kwargs[idx])  # type: ignore
        else:
            env = wrapper(env)  # type: ignore

    if max_episode_steps is not None and max_episode_steps > 0:
        env = SafeTimeLimit(
            env, max_episode_steps, terminate_on_timeout=terminate_on_timeout
        )
    return env
