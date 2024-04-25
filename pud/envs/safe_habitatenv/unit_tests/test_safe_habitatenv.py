import unittest
import numpy as np
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt

from pud.envs.safe_habitatenv.safe_habitatenv import (
    SafeHabitatNavigationEnv,
    display_map,
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    safe_habitat_env_load_fn,
)


class TestSafeHabitatEnv(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = {
            "scene": "/home/mers-pluto/Desktop/Work/habitat_workspace/habitat-lab/data/scene_datasets/"
            "habitat-test-scenes/skokloster-castle.glb",
            "height": 0.0,
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
            cost_fn_args=self.cost_fn_kwargs
        )

    def test_safe_habitat_env_load_fn(self):
        """
        Test the environment loader with wrappers
        TimeLimit is loaded by default if max_episode_steps > 0
        """

        env_args = deepcopy(self.env_kwargs)
        env_args.update
        gym_env_wrappers = [SafeGoalConditionedHabitatPointWrapper]
        env = safe_habitat_env_load_fn(
            env_args,
            self.cost_fn_kwargs,
            max_episode_steps=20,
            gym_env_wrappers=gym_env_wrappers,  # type: ignore
            terminate_on_timeout=False,
        )

        state, info = env.reset()  # type: ignore
        self.assertTrue("cost" in info)
        self.assertTrue("goal" in state)
        self.assertTrue("observation" in state)
        self.assertTrue(state["observation"].shape == (4, 256, 256, 4))  # type: ignore

        for _ in range(40):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.assertTrue("cost" in info)
            self.assertTrue("goal" in next_state)
            self.assertTrue("observation" in next_state)
            self.assertTrue(next_state["observation"].shape == (4, 256, 256, 4))

    def test_plot_safe_environment(self):
        output_dir = Path("pud/envs/safe_habitatenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        title = "Skokloster Castle"
        for cost_ub in [0, 1, 2]:
            fig, ax = plt.subplots()
            _ = display_map(
                self.habitat_env.walls,
                title,
                self.habitat_env.cost_map,
                cost_ub,  # type: ignore
                ax=ax,
            )
            fig.savefig(
                "pud/envs/safe_habitatenv/unit_tests/outputs/{}_cost={:.2f}.jpg".format(
                    title, cost_ub
                ),
                dpi=300,
            )
            plt.close(fig)

    def test_reset(self):
        for _ in range(100):
            state, info = self.habitat_env.reset()  # type: ignore
            agent_state = self.habitat_env._simulator.get_agent(
                self.habitat_env._simulator_settings["default_agent"]
            ).get_state()
            agent_position = np.array(
                [agent_state.position[2], agent_state.position[0]]
            )
            cx, cy = self.habitat_env._discretize_state(
                agent_position,
                (self.habitat_env._wall_height, self.habitat_env._wall_width),
            )
            sample_cost = self.habitat_env._get_state_cost(np.array([cx, cy]))
            self.assertTrue(
                sample_cost < self.habitat_env.cost_limit,
                msg="Sample = {}, Sample Cost = {}, Cost Map = {}, Cost Limit = {}".format(
                    (cx, cy),
                    sample_cost,
                    self.habitat_env.cost_map[int(cx), int(cy)],
                    self.habitat_env.cost_limit,
                ),
            )

            self.assertTrue(self.habitat_env._simulator.pathfinder.is_navigable(agent_state.position))  # type: ignore

    def test_step(self):
        s0, info = self.habitat_env.reset()  # type: ignore
        action = self.habitat_env.action_space.sample()
        self.habitat_env.step(action)


if __name__ == "__main__":
    unittest.main()
