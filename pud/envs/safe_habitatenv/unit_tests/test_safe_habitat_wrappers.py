from pathlib import Path
import unittest
from dotmap import DotMap
from matplotlib import pyplot as plt
import numpy as np
import yaml

from pud.envs.safe_habitatenv.safe_habitatenv import (
    SafeHabitatNavigationEnv,
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    safe_habitat_env_load_fn,
)
from pud.envs.safe_pointenv.safe_pointenv import plot_maze_grid_points, plot_safe_walls

"""
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitat_wrappers.py TestSafeHabitatWrapper.safe_habitat_env_load_fn
"""


class TestSafeHabitatWrapper(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = dict(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
        )
        self.cost_kwargs = dict(
            cost_f_args={"name": "linear", "radius": 2.0},
            cost_limit=10.0,
        )

    def safe_habitat_env_load_fn(self):
        env = safe_habitat_env_load_fn(
            env_kwargs=self.env_kwargs,
            **self.cost_kwargs,
            gym_env_wrappers=[SafeGoalConditionedHabitatPointWrapper],
            )
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
    
    def test_plot_safe_walls_w_grids(self):
        output_dir = Path("pud/envs/safe_habitatenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        config_file = "/home/mers-pluto/job_26896380_visual_cost_correct_flag/2024-08-28-03-58-20/lag/2024-08-30-06-17-29/bk/config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        config = DotMap(config)
        env = safe_habitat_env_load_fn(
            env_kwargs=config.env.toDict(),
            cost_f_args=config.cost_function.toDict(),
            cost_limit=config.agent_cost_kwargs.cost_limit,
            max_episode_steps=config.time_limit.max_episode_steps,
            gym_env_wrappers=[SafeGoalConditionedHabitatPointWrapper],
            terminate_on_timeout=True,
            )
        #for cost_ub in [0, 1, 2]:
        for cost_ub in [0]:
            fig, ax = plt.subplots()
            ax = plot_maze_grid_points(walls=env.get_map(), ax=ax)
            ax = plot_safe_walls(walls=env.get_map(), cost_map=env.get_cost_map(), cost_limit=cost_ub, ax=ax)
            fig.savefig(output_dir.joinpath("{}_cost={:.2f}_w_grids.jpg".format("SC2_Staging_08", cost_ub)), dpi=300)
            plt.close(fig)
    

if __name__ == "__main__":
    unittest.main()
