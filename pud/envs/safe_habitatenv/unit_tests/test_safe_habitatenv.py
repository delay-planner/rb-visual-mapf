import unittest

import numpy as np
import matplotlib.pyplot as plt

from pud.envs.habitat_navigation_env import plot_wall
from pud.envs.safe_habitatenv.safe_habitatenv import (
    SafeHabitatNavigationEnv, 
)

"""

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.reset

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.step

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.cost_map

"""


class TestSafeHabitatEnv(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = dict(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "cosine", "radius": 2.0},
            cost_limit=1.0,
        )

    def reset(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "linear", "radius": 3.0},
            cost_limit=10.0,
        )
        env.reset()

    def step(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "cosine", "radius": 2.0},
            cost_limit=10.0,
        )
        s0, info = env.reset()  # type: ignore
        action = env.action_space.sample()
        env.step(action)

    def cost_map(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "linear", "radius": 3.0},
            cost_limit=0.0,
        )
        fig, ax = plt.subplots()
        plot_wall(walls=env.walls, ax=ax)

        if env.cost_map is not None:
            unsafe_points = np.where(env.cost_map > env.cost_limit)
            unsafe_points = np.column_stack(unsafe_points)
            ax.scatter(unsafe_points[:,0]/float(env.wall_height), unsafe_points[:,1]/float(env.wall_width), s=2, marker='o', c="red")
        ax.set_title("TestSafeHabitatEnv test cost_map")
        fig.savefig(fname="runs/tmp_plots/test_plot_cost_map.jpg", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
