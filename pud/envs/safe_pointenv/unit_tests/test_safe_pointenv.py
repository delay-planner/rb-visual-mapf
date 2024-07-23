import unittest
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from pud.envs.safe_pointenv.safe_pointenv import (SafePointEnv,
                                                  plot_maze_grid_points,
                                                  plot_safe_walls)
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointQueueWrapper, SafeGoalConditionedPointWrapper,
    safe_env_load_fn)

"""
python pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_safe_env_load_fn

python pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_plot_safe_walls_w_grids

python pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_points_picked_in_obstacle



python -m debugpy \
    --listen localhost:5678 \
    --wait-for-client \
    pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_plot_safe_walls_w_grids
"""

class TestSafePointEnv(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = {
            #"walls": "CentralObstacle",
            #"walls": "FourRooms",
            #"walls": "L",
            #"walls": "Line",
            "walls": "LQuarter",
            #"walls": "LT",
            "resize_factor": 5,
            "thin": False,
        }
        self.cost_f_kwargs = {
            #"name": "cosine",
            "name": "constant",
            "radius": 6.,
        }
        self.precompilation_kwargs = {
            "cost_limit": 0,
        }

        self.p_env = SafePointEnv(
                    **self.env_kwargs, 
                    **self.precompilation_kwargs,
                    cost_f_args=self.cost_f_kwargs)

    def test_safe_env_load_fn(self):
        """"test env loader with wrappers
        TimeLimit is loaded by default if max_episode_steps>0
        """
        env_args = deepcopy(self.env_kwargs)
        env_args.update(self.precompilation_kwargs)
        gym_env_wrappers = [SafeGoalConditionedPointQueueWrapper]
        env = safe_env_load_fn(env_args,
                        self.cost_f_kwargs,
                        max_episode_steps=20,
                        gym_env_wrappers=gym_env_wrappers,
                        terminate_on_timeout=False,
                        )

        new_state, info = env.reset()
        self.assertTrue("cost" in info)

        num_steps = int(1e6)
        for _ in tqdm(range(int(1e6)), total=num_steps):
            at = env.action_space.sample()
            next_state, rew, done, info = env.step(at)
            self.assertTrue("cost" in info)

    def test_safe_apsp(self):
        self.assertFalse(np.allclose(self.p_env._safe_apsp["ub"], self.p_env._apsp))        
        #self.assertTrue(np.allclose(self.p_env._safe_apsp["lb"], self.p_env._apsp))

    def test_plot_safe_walls(self):
        output_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        for cost_ub in [0, 1, 2]:
            fig, ax = plt.subplots()
            ax = plot_safe_walls(walls=self.p_env._walls, cost_map=self.p_env._cost_map, cost_limit=cost_ub, ax=ax)
            fig.savefig(output_dir.joinpath("{}_resize={:0>2d}_cost={:.2f}.jpg".format(self.p_env.wall_name, self.p_env.resize_factor, cost_ub)), dpi=300)
            plt.close(fig)

    def test_plot_safe_walls_w_grids(self):
        output_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        for cost_ub in [0, 1, 2]:
            fig, ax = plt.subplots()
            ax = plot_maze_grid_points(walls=self.p_env._walls, ax=ax)
            ax = plot_safe_walls(walls=self.p_env._walls, cost_map=self.p_env._cost_map, cost_limit=cost_ub, ax=ax)
            fig.savefig(output_dir.joinpath("{}_resize={:0>2d}_cost={:.2f}_w_grids.jpg".format(self.p_env.wall_name, self.p_env.resize_factor, cost_ub)), dpi=300)
            plt.close(fig)

    def test_points_picked_in_obstacle(self):
        """all points are picked on top of obstacle, they should all be blocked"""
        self.env_kwargs["walls"] = "LQuarter"
        self.p_env = SafePointEnv(
                    **self.env_kwargs, 
                    **self.precompilation_kwargs,
                    cost_f_args=self.cost_f_kwargs)
        pnts = np.loadtxt("pud/envs/safe_pointenv/unit_tests/LQuarter_resize_5_blocks.txt", delimiter=",")
        pnts_orig = pnts * np.array([self.p_env._height, self.p_env._width])
        for p in pnts_orig:
            assert self.p_env._is_blocked(p)
        
    def test_reset(self):
        for _ in range(100):
            new_state, info = self.p_env.reset()
            sample_cost = self.p_env.get_state_cost(new_state)
            cx, cy = new_state
            self.assertTrue(sample_cost < self.p_env.cost_limit, msg="sample={}, sample cost= {}, cost map = {}, cost limit={}".format(new_state, sample_cost, self.p_env._cost_map[int(cx), int(cy)], self.p_env.cost_limit))
            self.assertTrue(not self.p_env._is_blocked(new_state))

    def test_step(self):
        s0, info = self.p_env.reset()
        at = self.p_env.action_space.sample()
        self.p_env.step(at)

    @unittest.skip("deprecated")
    def test_safe_apsp(self):
        """
        check the set of safe apsp, could be empty if done wrong (or with the wrong maze)
        """
        self.start_n_goal_candidates = {}
        mask_finite = self.p_env._safe_apsp["ub"] < np.inf
        mask_no_loop = self.p_env._safe_apsp["ub"] > 0
        mask_cands = mask_finite * mask_no_loop
        inds_cands = np.where(mask_cands)

        finite_dists = np.where(mask_finite)
        # check if the finite distances are all 0
        self.assertTrue(len(np.unique(self.p_env._safe_apsp["ub"][finite_dists])) > 1) 

        # check if there are more than 1 candiates
        self.start_n_goal_candidates["ub"] = np.column_stack(inds_cands) # x1, y1, x2, y2

        self.assertTrue(len(self.start_n_goal_candidates["ub"]) > 0)


if __name__ == '__main__':
    unittest.main()