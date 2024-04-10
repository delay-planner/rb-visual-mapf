import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from dotmap import DotMap
from torch.utils.tensorboard.writer import SummaryWriter

from pud.ddpg import GoalConditionedCritic
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointWrapper,
    safe_env_load_fn,
)
from pud.utils import set_env_seed, set_global_seed
from pud.algos.crl_runner_v2 import train_eval, eval_pointenv_cost_constrained_dists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help="Training configuration",
    )
    parser.add_argument("--logdir", type=str, default="", help="Override ckpt dir")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument("--train", action="store_true", help="Train or test")
    parser.add_argument("--weights", type=str, default="", help="Weights file to load")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )

    args = parser.parse_args()
    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    # For dot completion
    cfg = DotMap(cfg)

    # Override cfs from terminal
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
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

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

    obs_dim = env.observation_space["observation"].shape[0]  # type: ignore

    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(
        f"Observation dimension: {obs_dim},\n"
        f"Goal dimension: {goal_dim},\n"
        f"State dimension: {state_dim},\n"
        f"Action dimension: {action_dim},\n"
        f"Max Action: {max_action}"
    )

    agent = DRLDDPGLag(
        # DDPG args
        state_dim,  # Concatenating observation and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        **cfg.agent,
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

    print(agent)

    replay_buffer = ConstrainedReplayBuffer(
        obs_dim, goal_dim, action_dim, **cfg.replay_buffer
    )

    # Custom logging
    log_dir = Path(cfg.ckpt_dir)
    from datetime import datetime

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = log_dir.joinpath(date_time)
    ckpt_dir = log_dir.joinpath("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bk_dir = log_dir.joinpath("bk")
    bk_dir.mkdir(parents=True, exist_ok=True)
    with open(bk_dir.joinpath("bk_config.yaml"), "w") as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
    tb = SummaryWriter(log_dir=log_dir.as_posix())

    train = args.train
    if train:
        from pud.policies import GaussianPolicy

        # Gaussian policy seems to just add exploration noise
        # The evaluation code does not use it
        policy = GaussianPolicy(agent)

        train_eval(
            policy,
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
        torch.save(
            agent.state_dict(),
            ckpt_dir.joinpath("agent.pth"),
        )
    elif not train:
        agent.load_state_dict(torch.load(args.weights))
        agent.eval()
        from pud.visualize import visualize_trajectory

        eval_env.duration = (
            100  # We'll give the agent lots of time to try to find the goal.
        )
        visualize_trajectory(agent, eval_env, difficulty=0.5, constrained=True)

        # We now will implement the search policy, which automatically finds these waypoints via graph search.
        # The first step is to fill the replay buffer with random data.
        from pud.algos.constrained_collector import ConstrainedCollector

        eval_env.set_sample_goal_args(  # type: ignore
            prob_constraint=0.0, min_dist=0, max_dist=np.inf, min_cost=0, max_cost=1
        )
        rb_vec = ConstrainedCollector.sample_initial_states(
            eval_env, replay_buffer.max_size
        )

        from pud.visualize import visualize_buffer

        visualize_buffer(rb_vec, eval_env)

        pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)
        from scipy.spatial import distance

        euclidean_dists = distance.pdist(rb_vec)

        # As a sanity check, we'll plot the pairwise distances between all
        # observations in the replay buffer. We expect to see a range of values
        # from 1 to 20. Distributional RL implicitly caps the maximum predicted
        # distance by the largest bin. We've used 20 bins, so the critic
        # predicts 20 for all states that are at least 20 steps away from one another.
        from pud.visualize import visualize_pairwise_dists

        visualize_pairwise_dists(pdist)

        # With these distances, we can construct a graph. Nodes in the graph are
        # observations in our replay buffer. We connect observations with edges
        # whose lengths are equal to the predicted distance between those observations.
        # Since it is hard to visualize the edge lengths, we included a slider that
        # allows you to only show edges whose predicted length is less than some threshold.

        # Our method learns a collection of critics, each of which makes an independent
        # prediction for the distance between two states. Because each network may make
        # bad predictions for pairs of states it hasn't seen before, we act in
        # a *risk-averse* manner by using the maximum predicted distance across our
        # ensemble. That is, we act pessimistically, only adding an edge
        # if *all* critics think that this pair of states is nearby.
        from pud.visualize import visualize_graph

        visualize_graph(rb_vec, eval_env, pdist)

        # We can also visualize the predictions from each critic.
        # Note that while each critic may make incorrect decisions
        # for distant states, their predictions in aggregate are correct.
        from pud.visualize import visualize_graph_ensemble

        visualize_graph_ensemble(rb_vec, eval_env, pdist)

        from pud.policies import SearchPolicy

        search_policy = SearchPolicy(agent, rb_vec, pdist=pdist, open_loop=True)
        eval_env.duration = (  # type: ignore
            300  # We'll give the agent lots of time to try to find the goal.
        )

        # Sparse graphical memory
        # from pud.policies import SparseSearchPolicy

        # search_policy = SparseSearchPolicy(
        #     agent, rb_vec, pdist=pdist, cache_pdist=True, max_search_steps=10
        # )
        # eval_env.duration = 300
        # from pud.runner import cleanup_and_eval_search_policy

        # (initial_g, initial_rb), (filtered_g, filtered_rb), (cleaned_g, cleaned_rb) = (
        #     cleanup_and_eval_search_policy(search_policy, eval_env, constrained=True)
        # )
        # from pud.visualize import visualize_full_graph

        # visualize_full_graph(initial_g, initial_rb, eval_env)
        # visualize_full_graph(filtered_g, filtered_rb, eval_env)
        # visualize_full_graph(cleaned_g, cleaned_rb, eval_env)

        # Plot the search path found by the search policy
        from pud.visualize import visualize_search_path

        visualize_search_path(
            search_policy,
            eval_env,
            difficulty=0.9,
            constrained=True,
        )

        # Now, we'll use that path to guide the agent towards the goal.
        # On the left, we plot rollouts from the baseline goal-conditioned policy.
        # On the right, we use that same policy to reach each of the waypoints
        # leading to the goal. As before, the slider allows you to change the
        # distance to the goal. Note that only the search policy is able to reach distant goals.
        from pud.visualize import visualize_compare_search

        visualize_compare_search(
            agent, search_policy, eval_env, difficulty=0.9, constrained=True
        )

        # Now, we'll use the multi-agent version of the search policy to guide the agents towards their goal.
        num_agents = 4

        visualize_search_path(
            search_policy,
            eval_env,
            difficulty=0.9,
            constrained=True,
            num_agents=num_agents,
        )

        # In order to ensure that each agent gets the same duration, we need to multiply the original duration by the
        # number of agents.
        eval_env.duration *= num_agents  # type: ignore
        visualize_compare_search(
            agent,
            search_policy,
            eval_env,
            difficulty=0.9,
            constrained=True,
            num_agents=num_agents,
        )
