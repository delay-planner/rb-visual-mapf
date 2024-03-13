import unittest
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper
from pud.algos.constrained_buffer import ConstrainedReplayBuffer

"""
python pud/algos/unit_tests/test_buffer.py TestConstrainedBuffer.test_add
"""

class TestConstrainedBuffer(unittest.TestCase):
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

    def test_add(self):
        out = self.w_env.reset()
        obs_dim = len(out["observation"])
        goal_dim = obs_dim
        action_dim = self.w_env.action_space.shape[0]
        buffer = ConstrainedReplayBuffer(
            obs_dim = obs_dim,
            goal_dim = goal_dim,
            action_dim = action_dim,
            max_size = 10,
        )
        for _ in range(50):
            state = self.w_env.reset()
            at = self.w_env.action_space.sample()
            next_state, rew, done, info = self.w_env.step(at)
            buffer.add(
                state= state,
                action= at,
                next_state= next_state,
                reward= rew,
                cost= info["cost"],
                done= done,
            )
        self.assertTrue(buffer.size == 10)


if __name__ == '__main__':
    unittest.main()
