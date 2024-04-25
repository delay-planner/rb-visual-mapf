import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from dotmap import DotMap

from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.algos.constrained_collector import ConstrainedCollector as Collector
from pud.algos.constrained_collector import eval_agent_from_Q
from pud.algos.crl_runner_v3 import (eval_pointenv_cost_constrained_dists,
                                     train_eval)
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.ddpg import GoalConditionedActor, GoalConditionedCritic
from pud.envs.safe_pointenv.pb_sampler import (sample_cost_pbs_by_agent,
                                               sample_pbs_by_agent)
from pud.envs.safe_pointenv.safe_pointenv import plot_safe_walls, plot_trajs
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeGoalConditionedPointBlendWrapper, SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointWrapper, safe_env_load_fn)
from pud.utils import set_env_seed, set_global_seed
import matplotlib.pyplot as plt


def setup_args_parser(parser:argparse.ArgumentParser):
    parser.add_argument('--cfg',
        type=str,
        default="configs/config_SafePointEnv.yaml",
        help='training configuration')
    parser.add_argument('--figsavedir',
        type=str,
        help='directory to save figures')
    parser.add_argument('--ckpt',
        type=str,
        help='the path to ckpt file')
    parser.add_argument('--device',
        type=str,
        default="cpu",
        help='cpu or cuda')
    parser.add_argument('--pbar', action='store_true', help='show progress bar')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose printing/logging')
    return parser


def setup_env(args:argparse.Namespace):
    cfg = {}
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    # for dot completion
    cfg = DotMap(cfg)

    # override cfs from terminal
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    
    cfg.pprint()

    figdir = Path(args.figsavedir)
    figdir.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointBlendWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointBlendWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointQueueWrapper":
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
    eval_env.set_prob_constraint(1.0)

    set_env_seed(eval_env, cfg.seed + 2)

    obs_dim = eval_env.observation_space['observation'].shape[0]
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
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
    
    ckpt_file = args.ckpt
    agent.load_state_dict(torch.load(ckpt_file))
    agent.eval()

    replay_buffer = ConstrainedReplayBuffer(obs_dim, goal_dim, action_dim, **cfg.replay_buffer)
    return dict(
                agent=agent,
                eval_env=eval_env,
                figsavedir=figdir,
                cfg=cfg,
                obs_dim=obs_dim,
                goal_dim=goal_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                replay_buffer=replay_buffer,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_args_parser(parser)
    args = parser.parse_args()
    
    setup_ret = setup_env(args)
    agent = setup_ret["agent"]
    eval_env = setup_ret["eval_env"]
    cfg = setup_ret["cfg"]
    obs_dim = setup_ret["obs_dim"]
    goal_dim = setup_ret["goal_dim"]
    state_dim = setup_ret["state_dim"]
    action_dim = setup_ret["action_dim"]
    max_action = setup_ret["max_action"]
    replay_buffer = setup_ret["replay_buffer"]
    figdir = setup_ret["figsavedir"]

    # from pud.visualize import visualize_trajectory
    # eval_env.duration = 100 # We'll give the agent lots of time to try to find the goal.
    # visualize_trajectory(agent, eval_env, difficulty=0.5)

    # We now will implement the search policy, which automatically finds these waypoints via graph search. 
    # The first step is to fill the replay buffer with random data.
    #
    #from pud.collector import Collector

    eval_env.set_sample_goal_args(
        prob_constraint=0.0, 
        min_dist=0, 
        max_dist=np.inf,
        min_cost=0.,
        max_cost=1.0,
        )
    
    # rb_vec is normalized between 0 and 1
    rb_vec = Collector.sample_initial_states(eval_env, replay_buffer.max_size)

    from pud.visualize import visualize_buffer
    visualize_buffer(rb_vec, eval_env, outpath=figdir.joinpath("vis_buffer.jpg").as_posix())

    pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None) # ensemble, rb_vec, rb_vec

    from pud.visualize import visualize_cost_graph
    visualize_cost_graph(rb_vec=rb_vec, 
        eval_env=eval_env, 
        pcost=pcost, 
        cost_limit=cfg["agent"]["cost_limit"],
        outpath=figdir.joinpath("vis_cost_graph.jpg").as_posix(),
        edges_to_display=10,
        )

    from pud.visualize import visualize_combined_graph
    visualize_combined_graph(
        rb_vec=rb_vec, 
        eval_env=eval_env, 
        pdist=pdist,
        pcost=pcost, 
        cutoff=7,
        cost_limit=cfg["agent"]["cost_limit"],
        outpath=figdir.joinpath("vis_combined_graph.jpg").as_posix(),
        edges_to_display=10,
    )

    #from scipy.spatial import distance
    # what the fuck, why is euclidean_dists needed? 
    # it has a stupid indexing: The metric dist(u=X[i], v=X[j]) 
    # is computed and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2.
    #euclidean_dists = distance.pdist(rb_vec)

    # As a sanity check, we'll plot the pairwise distances between all 
    # observations in the replay buffer. We expect to see a range of values 
    # from 1 to 20. Distributional RL implicitly caps the maximum predicted 
    # distance by the largest bin. We've used 20 bins, so the critic 
    # predicts 20 for all states that are at least 20 steps away from one another.
    # 
    from pud.visualize import (visualize_pairwise_costs,
                               visualize_pairwise_dists)
    visualize_pairwise_dists(pdist, 
        outpath=figdir.joinpath("vis_pdist.jpg").as_posix())
    visualize_pairwise_costs(pcost, 
        cost_limit=cfg.agent.cost_limit, 
        n_bins=cfg.agent.cost_N,
        outpath=figdir.joinpath("vis_pcost.jpg").as_posix(),
        )
    

    # With these distances, we can construct a graph. Nodes in the graph are 
    # observations in our replay buffer. We connect observations with edges 
    # whose lengths are equal to the predicted distance between those observations. 
    # Since it is hard to visualize the edge lengths, we included a slider that 
    # allows you to only show edges whose predicted length is less than some threshold.
    # ---
    # Our method learns a collection of critics, each of which makes an independent 
    # prediction for the distance between two states. Because each network may make 
    # bad predictions for pairs of states it hasn't seen before, we act in 
    # a *risk-averse* manner by using the maximum predicted distance across our 
    # ensemble. That is, we act pessimistically, only adding an edge 
    # if *all* critics think that this pair of states is nearby.
    #
    from pud.visualize import visualize_graph
    visualize_graph(rb_vec, 
        eval_env, 
        pdist, 
        outpath=figdir.joinpath("vis_graph.jpg").as_posix(),
        )

    # We can also visualize the predictions from each critic. 
    # Note that while each critic may make incorrect decisions 
    # for distant states, their predictions in aggregate are correct.
    # 
    from pud.visualize import visualize_graph_ensemble
    visualize_graph_ensemble(rb_vec, 
        eval_env, 
        pdist, 
        outpath=figdir.joinpath("vis_graph_ensemble.jpg").as_posix(),
        )

    # rollout trained policy and visualize trajectory
    eval_env.set_prob_constraint(1.0)
    pbs_c = sample_cost_pbs_by_agent(
                env=eval_env,
                agent=agent,
                num_states=50,
                target_val=0.5,
                min_dist=10,
                max_dist=20,
                ensemble_agg="mean",
                K=1,
            )
    eval_env.set_pbs(pb_list=pbs_c)
    num_pb_c = len(pbs_c)

    start_list = [p["start"].tolist() for p in pbs_c]
    goal_list = [p["goal"].tolist() for p in pbs_c]

    eval_records = eval_agent_from_Q(policy=agent, eval_env=eval_env, collect_trajs=True)
    #ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list = Collector.get_trajectory(policy=agent, eval_env=eval_env)
    #ep_de_normalized_obs = [eval_env.de_normalize_obs(x) for x in ep_observation_list]
    #ep_obs.append(ep_de_normalized_obs)
    list_trajs = []
    for id in eval_records.keys():
        list_trajs.append(eval_records[id]["traj"])
    
    fig, ax = plt.subplots()
    ax = plot_safe_walls(walls=eval_env.get_map(), 
            cost_map=eval_env.get_cost_map(),
            cost_limit=2.0,
            ax=ax,
            )

    ax = plot_trajs(list_trajs=list_trajs, 
        walls=eval_env.get_map(),
        ax=ax,
        starts=start_list,
        goals=goal_list,
        s=32,
        )

    fig.savefig(figdir.joinpath("test_trajs.jpg"), dpi=300)
    plt.close(fig)

    #from pud.policies import SearchPolicy
    #search_policy = SearchPolicy(agent, rb_vec, pdist=pdist, open_loop=True)
    #eval_env.duration = 300 # We'll give the agent lots of time to try to find the goal.

    # Sparse graphical memory
    # 
    #from pud.policies import SparseSearchPolicy
    #search_policy = SparseSearchPolicy(agent, rb_vec, pdist=pdist, cache_pdist=True, max_search_steps=10)
    #eval_env.duration = 300
    #
    #from pud.runner import cleanup_and_eval_search_policy
    #(initial_g, initial_rb), (filtered_g, filtered_rb), (cleaned_g, cleaned_rb) = cleanup_and_eval_search_policy(search_policy, eval_env)
    ##
    #from pud.visualize import visualize_full_graph
    #visualize_full_graph(cleaned_g, cleaned_rb, eval_env)

    # Plot the search path found by the search policy
    # 
    # from pud.visualize import visualize_search_path
    # visualize_search_path(search_policy, eval_env, difficulty=0.9)

    # Now, we'll use that path to guide the agent towards the goal. 
    # On the left, we plot rollouts from the baseline goal-conditioned policy. 
    # On the right, we use that same policy to reach each of the waypoints 
    # leading to the goal. As before, the slider allows you to change the 
    # distance to the goal. Note that only the search policy is able to reach distant goals.
    #
    # from pud.visualize import visualize_compare_search
    # visualize_compare_search(agent, search_policy, eval_env, difficulty=0.9)
