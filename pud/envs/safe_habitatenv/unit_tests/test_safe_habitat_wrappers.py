import unittest
import numpy as np
from copy import deepcopy

from pud.envs.safe_habitatenv.safe_habitatenv import (
    SafeHabitatNavigationEnv,
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    safe_habitat_env_load_fn,
    set_safe_habitat_env_difficulty,
)

"""
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitat_wrappers.py
"""


class TestSafeHabitatWrapper(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = {
            "scene": "/home/mers-pluto/Desktop/Work/habitat_workspace/habitat-lab/data/scene_datasets/"
            "habitat-test-scenes/skokloster-castle.glb",
            "height": 0.0,
            "apsp_path": "/home/mers-pluto/Desktop/Work/cc-sorb-rev/pud/envs/safe_habitatenv/apsp.pkl",
        }
        self.cost_fn_kwargs = {
            "name": "cosine",
            "radius": 2.0,
        }
        self.precompilation_kwargs = {
            "cost_limit": 1,
        }

        self.habitat_env = SafeHabitatNavigationEnv(
            **self.env_kwargs,  # type: ignore
            **self.precompilation_kwargs,
            cost_f_args=self.cost_fn_kwargs
        )
        self.wrapped_habitat_env = SafeGoalConditionedHabitatPointWrapper(
            self.habitat_env
        )

    def test_set_safe_habitat_env_difficulty(self):
        set_safe_habitat_env_difficulty(self.wrapped_habitat_env, 0.5)

    def test_safe_habitat_env_load_fn(self):
        """
        Test the environment loader with wrappers
        TimeLimit is loaded by default if max_episode_steps > 0
        """

        env_args = deepcopy(self.env_kwargs)
        env_args.update(self.precompilation_kwargs)
        gym_env_wrappers = [SafeGoalConditionedHabitatPointWrapper]
        env = safe_habitat_env_load_fn(
            env_args,
            self.cost_fn_kwargs,
            gym_env_wrappers=gym_env_wrappers,  # type: ignore
        )

        self.assertTrue(isinstance(env, SafeGoalConditionedHabitatPointWrapper))
        env._set_sample_goal_args(  # type: ignore
            prob_constraint=1,
            min_dist=1.0,
            max_dist=10.0,
            min_cost=0.0,
            max_cost=1.0,
        )
        env.reset()

    def test_reset_no_constraint(self):
        self.wrapped_habitat_env._set_sample_goal_args(
            prob_constraint=0.0,
            min_dist=1.0,
            max_dist=1.0,
            min_cost=0.0,
            max_cost=1.0,
        )

        for _ in range(100):
            state, info = self.wrapped_habitat_env.reset()

    def test_reset_with_constraint(self):
        max_cost = 0.5
        self.wrapped_habitat_env._set_sample_goal_args(
            prob_constraint=1.0,
            min_dist=1.0,
            max_dist=10.0,
            min_cost=0.0,
            max_cost=max_cost,
        )

        # NOTE: The max_cost is not used yet in the reset method

        for _ in range(100):
            state, info = self.wrapped_habitat_env.reset()
            agent_position = self.wrapped_habitat_env.env.get_xy_in_habitat()
            agent_grid_position_x, agent_grid_position_y = (
                self.wrapped_habitat_env.env.get_grid_xy_from_habitat_xy(agent_position)
            )
            agent_grid_position = np.array(
                [agent_grid_position_x, agent_grid_position_y]
            )
            self.assertTrue(
                self.wrapped_habitat_env.env._get_state_cost(agent_grid_position)
                <= self.wrapped_habitat_env.env.cost_limit
            )
            self.assertTrue(
                self.wrapped_habitat_env._min_dist
                <= self.wrapped_habitat_env.env.get_distance(
                    agent_position, state["goal"]
                )
                <= self.wrapped_habitat_env._max_dist
            )

    def test_step(self):
        state, info = self.wrapped_habitat_env.reset()
        action = self.wrapped_habitat_env.action_space.sample()
        next_state, reward, done, info = self.wrapped_habitat_env.step(action)


if __name__ == "__main__":
    unittest.main()
