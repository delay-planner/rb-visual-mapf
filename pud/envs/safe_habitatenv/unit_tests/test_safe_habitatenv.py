import unittest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pud.envs.habitat_navigation_env import plot_wall
from pud.envs.safe_habitatenv.safe_habitatenv import (
    SafeHabitatNavigationEnv, 
)

"""

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.reset

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.step

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.cost_map

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.cost_contour
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
            cost_f_args={"name": "linear", "radius": 0.5},
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

    def cost_contour(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "linear", "radius": 0.5},
            cost_limit=1,
        )

        x = np.linspace(0, env.wall_height+1, int(2*(env.wall_height+1)), dtype=float)
        y = np.linspace(0, env.wall_width+1, int(2*(env.wall_width+1)), dtype=float)
        X, Y = np.meshgrid(x, y)
        Z = np.ones_like(X) * np.inf

        pbar = tqdm(total=X.shape[0], desc="outer loop")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                min_d, _  = env.dist_2_blocks([X[i,j],Y[i,j]])
                Z[i,j]= env.cost_function(min_d)
            pbar.update()
        pbar.close()

        fig, ax = plt.subplots()
        CS = ax.contour(X/float(env.wall_height), Y/float(env.wall_width), Z)
        ax.clabel(CS, inline=True, fontsize=10)

        ax = plot_wall(walls=env.walls, ax=ax)
        ax.set_title("Test Safe HabitatEnv Cost Contour")
        fig.savefig("runs/tmp_plots/test_cost_contour.jpg", dpi=300)
        plt.close(fig=fig)

if __name__ == "__main__":
    unittest.main()
