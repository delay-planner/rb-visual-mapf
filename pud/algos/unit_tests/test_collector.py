import unittest
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper, safe_env_load_fn
from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.algos.constrained_collector import ConstrainedCollector
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.policies import GaussianPolicy
from pud.ddpg import GoalConditionedCritic
from termcolor import cprint

"""
python pud/algos/unit_tests/test_collector.py TestConstrainedCollector.test_eval_agent_n_record_init_states
"""

class TestConstrainedCollector(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = {
            "walls": "CentralObstacle",
            #"walls": "FourRooms",
            "resize_factor": 5,
            "thin": False,
            "cost_limit": 1,
        }
        self.cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.,
        }

        self.env = safe_env_load_fn(
            env_kwargs=self.env_kwargs,
            cost_f_kwargs=self.cost_f_kwargs,
            max_episode_steps=10,
            gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
            terminate_on_timeout=False,
        )

        obs_dim = self.env.observation_space['observation'].shape[0]
        goal_dim = obs_dim
        state_dim = obs_dim + goal_dim
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        agent_cfg = dict(
                discount= 1,
                ensemble_size= 3,
                num_bins= 20,
                actor_update_interval= 1,
                targets_update_interval= 5,
                tau= 0.05,
                use_distributional_rl= True,
                # cost configs
                cost_min= 0.0,
                cost_max= 2.0,
                cost_N= 20,
                cost_critic_lr= 0.001,
                )

        self.agent =  DRLDDPGLag(
                # DDPG args
                state_dim,  # concatenating obs and goal
                action_dim,
                max_action,
                CriticCls=GoalConditionedCritic,
                **agent_cfg,
            )
        self.policy = GaussianPolicy(self.agent)

        self.buffer = ConstrainedReplayBuffer(
                    obs_dim = obs_dim,
                    goal_dim = goal_dim,
                    action_dim = action_dim,
                    max_size = 40,
                )

        self.collector = ConstrainedCollector(
            policy=self.policy,
            buffer=self.buffer,
            env=self.env,
            initial_collect_steps=20,
        )

    def test_step(self):
        self.collector.step(1000)
        self.assertTrue(self.collector.num_eps > 0)

    def test_simple_optimize(self):
        """test simple train loop without Lagrange"""
        num_iterations = 10
        collect_steps = 2

        num_eps = self.collector.num_eps
        ep_cost = 0
        for i in range(1, num_iterations + 1):
            self.collector.step(collect_steps)
            self.agent.train()
            opt_info = self.agent.optimize(self.buffer, iterations=1, batch_size=64)
            cprint(opt_info, "yellow")

            if self.collector.num_eps > num_eps:
                ep_cost = self.collector.past_eps[-1]["ep_cost"]
                ep_len = self.collector.past_eps[-1]["ep_len"]
                cprint("[INFO] eps Jc='{:.2f}', eps length={}".format(ep_cost, ep_len), "green")
                num_eps = self.collector.num_eps

    def test_eval_agent_n_record_init_states(self):
        self.eval_env = safe_env_load_fn(
            env_kwargs=self.env_kwargs,
            cost_f_kwargs=self.cost_f_kwargs,
            max_episode_steps=10,
            gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
            terminate_on_timeout=True,
        )
        num_evals = 5
        eval_outputs = ConstrainedCollector.eval_agent_n_record_init_states(self.agent, self.eval_env, num_evals)


if __name__ == '__main__':
    unittest.main()
