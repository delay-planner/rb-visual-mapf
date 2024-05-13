import unittest
from pud.envs.habitat_navigation_env import (HabitatNavigationEnv, GoalConditionedHabitatPointWrapper)
from pathlib import Path    
import matplotlib.pyplot as plt
from pud.envs.safe_pointenv.safe_pointenv import plot_safe_walls
import numpy as np


"""
python pud/envs/safe_habitatenv/unit_tests/test_habitat_env.py TestHabitatEnv.compare_occupancy
"""

def plot_walls(env:HabitatNavigationEnv, ax:plt.axes):
    walls = env.walls.copy()
    # 1 is navigatbale, 0 is obstacle
    # convert to the convention of pointenv
    walls = 1 - walls
    walls = walls.T
    (height, width) = walls.shape
    # only plot walls
    for (i, j) in zip(*np.where(walls)):
        x = np.array([j, j+1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i+1, i+1]) / float(height)
        ax.fill_between(x, y0, y1, color='grey')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    
    return ax


class TestHabitatEnv(unittest.TestCase):
    def setUp(self):
        self.scene = "scene_datasets/habitat-test-scenes/skokloster-castle.glb"
        self.device = "cpu"
        self.simulator_settings = dict(
            scene= "scene_datasets/habitat-test-scenes/skokloster-castle.glb",
            width= 64,
            height= 64,
            default_agent= 0,
            sensor_height= 1.5,
        )
        self.apsp_path = "pud/envs/safe_habitatenv/apsps/skokloster/apsp.pickle"

    def compare_occupancy(self):
        """compare the occupancy from the 2D maze matrix and habitat"""
        env = HabitatNavigationEnv(
            scene=self.scene,
            height=0,
            simulator_settings=self.simulator_settings,
            device=self.device,
            apsp_path=self.apsp_path,
            )
        diff_set = []

        h, w = env.walls.shape
        for i in range(h):
            for j in range(w):
                grid_xy = (i,j)
                habitat_xy = env.get_xy_in_habitat_from_xy_in_grid(grid_xy)
                # 1 is empty, 0 is blocked
                if env.walls[i,j] != 1-env.is_blocked_habitat(habitat_xy):
                    diff_set.append((i,j))
        
        fig, ax = plt.subplots()
        ax = plot_walls(env, ax=ax)
        diff_arr = np.array(diff_set)
        ax.scatter(diff_arr[:,1]/w, diff_arr[:,0]/h, color="r", zorder=2, marker="o", label="inconsistency", s=40)
        ax.legend()
        fig.savefig(fname="runs/tmp_plots/walls.jpg", dpi=300)
        plt.close(fig=fig)

    def test_steps(self):
        env = HabitatNavigationEnv(
            scene=self.scene,
            height=0,
            simulator_settings=self.simulator_settings,
            device=self.device,
            apsp_path=self.apsp_path,
            )

        env.reset()

        pass

    def test_init(self):
        env = HabitatNavigationEnv(
            scene=self.scene,
            height=0,
            simulator_settings=self.simulator_settings,
            device=self.device,
            apsp_path=self.apsp_path,
            )

if __name__ == "__main__":
    unittest.main()
