import unittest
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv, plot_safe_walls
import numpy as np
import matplotlib.pyplot as plt

class TestSafePointEnv(unittest.TestCase):
    def setUp(self):
        env_kwargs = {
            "walls": "FourRooms",
            "resize_factor": 5,
            "thin": False,
        }
        cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.,
        }
        precompilation_kwargs = {
            "precompiled_cost_apsps": [0.0, 3.0],
        }

        self.p_env = SafePointEnv(
                    **env_kwargs, 
                    **precompilation_kwargs,
                    cost_f_args=cost_f_kwargs)

    def test_safe_apsp(self):
        self.assertFalse(np.allclose(self.p_env._safe_apsp[0.0], self.p_env._apsp))        
        self.assertTrue(np.allclose(self.p_env._safe_apsp[3.0], self.p_env._apsp))

    def test_plot_safe_walls(self):
        for cost_ub in [0]:
            fig, ax = plt.subplots()
            ax = plot_safe_walls(walls=self.p_env._walls, cost_map=self.p_env._cost_map, cost_upper_bound=cost_ub, ax=ax)
            fig.savefig("pud/envs/safe_pointenv/unit_tests/outputs/{}_resize={:0>2d}_cost={:.2f}.jpg".format(self.p_env.wall_name, self.p_env.resize_factor, cost_ub), dpi=300)
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
