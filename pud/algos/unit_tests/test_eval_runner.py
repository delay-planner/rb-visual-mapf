import yaml
import torch
import argparse
import numpy as np
from dotmap import DotMap

from pud.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.safe_wrappers import (
    #SafeGoalConditionedPointWrapper, 
    SafeGoalConditionedPointQueueWrapper,
    #SafeGoalConditionedPointBlendWrapper,
    safe_env_load_fn,
)
from pud.utils import set_env_seed, set_global_seed
#from pud.algos.crl_runner_v2 import train_eval, eval_pointenv_cost_constrained_dists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_PointEnv_Queue.yaml",
        help="Training configuration",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda")

    args = parser.parse_args()
    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    # For dot completion
    cfg = DotMap(cfg)

    # Override cfs from terminal
    cfg.device = args.device
    cfg.pprint()

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointQueueWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

    eval_env = safe_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )
    set_env_seed(eval_env, cfg.seed + 2)

    obs_dim = eval_env.observation_space["observation"].shape[0]  # type: ignore

    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    print(
        f"Observation dimension: {obs_dim},\n"
        f"Goal dimension: {goal_dim},\n"
        f"State dimension: {state_dim},\n"
        f"Action dimension: {action_dim},\n"
        f"Max Action: {max_action}"
    )

    agent = DRLDDPGLag(
        # DDPG args
        state_dim,  # concatenating obs and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(cfg.device),
        **cfg.agent,
    )
    agent.to(torch.device(args.device))

    from pud.policies import GaussianPolicy
    # gaussian policy seems just add exploration noise, the evaluation code
    # does not use it
    policy = GaussianPolicy(agent)

    print(agent)

    from pud.algos.constrained_collector import ConstrainedCollector as Collector
    from pud.envs.safe_pointenv.pb_sampler import sample_pbs_by_agent, calc_pairwise_cost

    pbs = sample_pbs_by_agent(
        env=eval_env,
        agent=agent,
        num_states=100,
        target_val=1.0,
        pval_f=calc_pairwise_cost,
        K=5,
        ensemble_agg="max"
    )

    eval_env.append_pbs(pbs)
    eval_stats = Collector.eval_agent_from_Q(policy=policy, eval_env=eval_env)

    ## logging
    attr = "costs"
    attr_vals = []
    attr_pred = []
    for id in eval_stats:
        attr_vals.append(
            eval_stats[id][attr]
        )
        attr_pred.append(
            eval_stats[id]["init_info"]["prediction"]
        )
        