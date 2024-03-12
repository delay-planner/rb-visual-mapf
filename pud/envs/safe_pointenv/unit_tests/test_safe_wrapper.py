import unittest
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper
import numpy as np

"""
python pud/envs/safe_pointenv/unit_tests/test_safe_wrapper.py TestSafeWrapper.test_sample_safe_start_n_goal_in_dists
"""

class TestSafeWrapper(unittest.TestCase):
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
        
        self.w_env = SafeGoalConditionedPointWrapper(self.p_env)

    def test_sample_start_n_goal(self):
        for i in range(100):
            out = self.w_env.sample_start_n_goal("ub")
            self.assertTrue(self.w_env.get_state_cost(out["s0"]) <= 0.0)
            self.assertTrue(self.w_env.get_state_cost(out["sg"]) <= 0.0)
            
            key = np.concatenate([out["s0"], out["sg"]], axis=0).astype(int)
            key = tuple(key.tolist())
            self.assertTrue(self.p_env._safe_apsp["ub"][key[0],key[1],key[2],key[3]] > 0)
            self.assertTrue(self.p_env._safe_apsp["ub"][key[0],key[1],key[2],key[3]] < np.inf)

    def test_sample_safe_start_n_goal_in_dists(self):
        for _ in range(100):
            dist_limits = (1, 10)
            out = self.w_env.sample_safe_start_n_goal_in_dists(
                min_dist=dist_limits[0],
                max_dist=dist_limits[1],
                )
            self.assertTrue(self.w_env.get_state_cost(out["s0"]) <= 0.0)
            self.assertTrue(self.w_env.get_state_cost(out["sg"]) <= 0.0)
            
            key = np.concatenate([out["s0"], out["sg"]], axis=0).astype(int)
            key = tuple(key.tolist())
            self.assertTrue(self.p_env._safe_apsp["ub"][key[0],key[1],key[2],key[3]] > dist_limits[0])
            self.assertTrue(self.p_env._safe_apsp["ub"][key[0],key[1],key[2],key[3]] < dist_limits[1])

    
    
    def test_reset(self):
        self.w_env.reset()

if __name__ == '__main__':
    unittest.main()
