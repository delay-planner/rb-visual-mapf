import unittest
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv, plot_safe_walls, plot_maze_grid_points
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TestSafePointEnv(unittest.TestCase):
    def setUp(self):
        env_kwargs = {
            "walls": "CentralObstacle",
            #"walls": "FourRooms",
            "resize_factor": 5,
            "thin": False,
        }
        cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.,
        }
        precompilation_kwargs = {
            "cost_limit": 1,
        }

        self.p_env = SafePointEnv(
                    **env_kwargs, 
                    **precompilation_kwargs,
                    cost_f_args=cost_f_kwargs)

    def test_safe_apsp(self):
        self.assertFalse(np.allclose(self.p_env._safe_apsp["ub"], self.p_env._apsp))        
        #self.assertTrue(np.allclose(self.p_env._safe_apsp["lb"], self.p_env._apsp))

    def test_plot_safe_walls(self):
        output_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        for cost_ub in [0, 1, 2]:
            fig, ax = plt.subplots()
            ax = plot_safe_walls(walls=self.p_env._walls, cost_map=self.p_env._cost_map, cost_limit=cost_ub, ax=ax)
            fig.savefig("pud/envs/safe_pointenv/unit_tests/outputs/{}_resize={:0>2d}_cost={:.2f}.jpg".format(self.p_env.wall_name, self.p_env.resize_factor, cost_ub), dpi=300)
            plt.close(fig)

    def test_plot_safe_walls_w_grids(self):
        output_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        for cost_ub in [0, 1, 2]:
            fig, ax = plt.subplots()
            ax = plot_maze_grid_points(walls=self.p_env._walls, ax=ax)
            ax = plot_safe_walls(walls=self.p_env._walls, cost_map=self.p_env._cost_map, cost_limit=cost_ub, ax=ax)
            fig.savefig("pud/envs/safe_pointenv/unit_tests/outputs/{}_resize={:0>2d}_cost={:.2f}_w_grids.jpg".format(self.p_env.wall_name, self.p_env.resize_factor, cost_ub), dpi=300)
            plt.close(fig)
        
    def test_reset(self):
        """
        python pud/envs/safe_pointenv/unit_tests/test_safe_pointenv.py TestSafePointEnv.test_safe_apsp
        """
        for _ in range(100):
            new_state = self.p_env.reset()
            sample_cost = self.p_env.get_state_cost(new_state)
            cx, cy = new_state
            self.assertTrue(sample_cost < self.p_env.cost_limit, msg="sample={}, sample cost= {}, cost map = {}, cost limit={}".format(new_state, sample_cost, self.p_env._cost_map[int(cx), int(cy)], self.p_env.cost_limit))
            self.assertTrue(not self.p_env._is_blocked(new_state))

    def test_step(self):
        s0 = self.p_env.reset()
        at = self.p_env.action_space.sample()
        self.p_env.step(at)

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
