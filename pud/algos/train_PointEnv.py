import argparse
import os
from pathlib import Path

import torch
import yaml
from dotmap import DotMap
from torch.utils.tensorboard import SummaryWriter

from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.ddpg import GoalConditionedActor, GoalConditionedCritic
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointWrapper, safe_env_load_fn)
from pud.utils import set_env_seed, set_global_seed
from pud.algos.crl_runner_v2 import train_eval, eval_pointenv_cost_constrained_dists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help='training configuration')
    parser.add_argument('--logdir',
        type=str,
        default="",
        help='override ckpt dir')
    parser.add_argument('--device',
        type=str,
        default="cpu",
        help='cpu or cuda')
    parser.add_argument('--pbar', action='store_true', help='show progress bar')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose printing/logging')
    args = parser.parse_args()

    cfg = {}
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    # for dot completion
    cfg = DotMap(cfg)

    # override cfs from terminal
    if len(args.logdir) > 0:
        cfg.ckpt_dir = args.logdir
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    
    cfg.pprint()

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(
                cfg.wrappers[wrapper_name].toDict()
            )

    env = safe_env_load_fn(
                    cfg.env.toDict(),
                    cfg.cost_function.toDict(),
                    max_episode_steps=cfg.time_limit.max_episode_steps,
                    gym_env_wrappers=gym_env_wrappers,
                    wrapper_kwargs=gym_env_wrapper_kwargs,
                    terminate_on_timeout=False,
                    )
    set_env_seed(env, cfg.seed + 1)


    eval_env = safe_env_load_fn(
                    cfg.env.toDict(),
                    cfg.cost_function.toDict(),
                    max_episode_steps=cfg.time_limit.max_episode_steps,
                    gym_env_wrappers=gym_env_wrappers,
                    wrapper_kwargs=gym_env_wrapper_kwargs,
                    terminate_on_timeout=True,
                    )
    set_env_seed(eval_env, cfg.seed + 2)

    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f'obs dim: {obs_dim}, goal dim: {goal_dim}, state dim: {state_dim}, action dim: {action_dim}, max action: {max_action}')

    agent =  DRLDDPGLag(
            # DDPG args
            state_dim,  # concatenating obs and goal
            action_dim,
            max_action,
            CriticCls=GoalConditionedCritic,
            device=torch.device(cfg.device),
            **cfg.agent,
        )
    agent.to(torch.device(args.device))
    
    print(agent)

    replay_buffer = ConstrainedReplayBuffer(obs_dim, goal_dim, action_dim, **cfg.replay_buffer)

    ## custom logging
    log_dir = Path(cfg.ckpt_dir)
    from datetime import datetime
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = log_dir.joinpath(date_time)
    ckpt_dir = log_dir.joinpath('ckpt')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bk_dir = log_dir.joinpath("bk")
    bk_dir.mkdir(parents=True, exist_ok=True)
    with open(bk_dir.joinpath("bk_config.yaml"), 'w') as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
    tb = SummaryWriter(log_dir=log_dir.as_posix())

    from pud.policies import GaussianPolicy
    # gaussian policy seems just add exploration noise, the evaluation code
    # does not use it
    policy = GaussianPolicy(agent)

    train_eval(policy,
            agent,
            replay_buffer,
            env,
            eval_env,
            eval_func=eval_pointenv_cost_constrained_dists,
            tensorboard_writer=tb,
            pbar=args.pbar,
            ckpt_dir=ckpt_dir,
            **cfg.runner,
            )
    torch.save(agent.state_dict(), 
        ckpt_dir.joinpath('agent.pth'),
        )