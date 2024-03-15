"""
Evaluate the accuracy and reliability of reward and cost critics
"""

import time
from typing import Dict, List, Optional, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pud.policies import GaussianPolicy
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.constrained_collector import ConstrainedCollector as Collector
from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.envs.safe_pointenv.safe_wrappers import SafeGoalConditionedPointWrapper


def train_eval(
    policy:GaussianPolicy,
    agent:DRLDDPGLag,
    replay_buffer:ConstrainedReplayBuffer,
    env,
    eval_env,
    num_iterations=int(1e6),
    initial_collect_steps=1000,
    collect_steps=1,
    opt_steps=1,
    batch_size_opt=64,
    eval_func=None, # make this a partial func
    opt_log_interval=100,
    eval_interval=10000,
    eval_distances=[2, 5, 10], # reference grouping based on estimated distances
    eval_cost_intervals=[0., 0.2, 0.5, 1.0], # grouping cost eval results
    tensorboard_writer:Optional[SummaryWriter]=None,
    verbose=True,
    ):
    """train constrained RL agent"""
    collector = Collector(policy, replay_buffer, env, initial_collect_steps=initial_collect_steps)
    collector.step(collector.initial_collect_steps)
    for i in tqdm(range(1, num_iterations + 1),total=num_iterations):
        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(replay_buffer, iterations=opt_steps, batch_size=batch_size_opt)

        # todo: update Lagrange multiplier

        if i % opt_log_interval == 0:
            if verbose:
                print(f'iteration = {i}, opt_info = {opt_info}')

        if i % eval_interval == 0:
            agent.eval()
            if verbose:
                print(f'evaluating iteration = {i}')

            sample_args = dict(
                sample_key = "ub",
            )
            eval_info = eval_func(
                agent=agent, 
                eval_env=eval_env,
                eval_distances=eval_distances,
                sample_args=sample_args,
                cost_intervals=eval_cost_intervals,
                )
            if verbose:
                print('-' * 10)

        if tensorboard_writer:
            tensorboard_writer.add_scalar("Opt/actor_loss", np.mean(opt_info["actor_loss"]), global_step=i)
            tensorboard_writer.add_scalar("Opt/critic_loss", np.mean(opt_info["critic_loss"]), global_step=i)

            if i % eval_interval == 0:
                for d_ref in eval_info:
                    # dotmap has interal attributes like "_ipython_display_"
                    if isinstance(d_ref, str) and d_ref.startswith("_"):
                        continue
                    tensorboard_writer.add_scalar("Eval_{:0>2d}/pred_dist".format(d_ref), np.mean(eval_info[d_ref]["pred_dist"]), global_step=i)
                    tensorboard_writer.add_scalar("Eval_{:0>2d}/pred_dist".format(d_ref), -np.mean(eval_info[d_ref]["returns"]), global_step=i)


def eval_pointenv_cost_constrained_dists(agent, 
        eval_env:SafeGoalConditionedPointWrapper, 
        num_evals=10, 
        eval_distances=[2, 5, 10, 20],
        cost_intervals=[0., 0.2, 0.5, 1.0],
        sample_args:Optional[dict]=None, 
        ):
    """sample starts and goals that are lower than a preset maximum cost limit (linear interpolation).

    The reference apsp in eval_env are all integers because they are computed from grid distances (no diagonal paths)
    """
    eval_stats = {
        # rewards are organized by reference distances
        "rewards": {},
        # cost is not grouped by reference distances but organized afterwards
        "costs": {
            "pred": [],
            "true": [],
        },
    } 
    
    for idx_d in range(len(eval_distances)-1):
        #todo: what if the requested distance is not available?
        min_dist = eval_distances[idx_d]
        max_dist = eval_distances[idx_d+1]
        eval_env.set_sample_goal_args(
            prob_constraint=1, 
            min_dist=min_dist, 
            max_dist=max_dist,
            **sample_args)
        
        eval_outputs = Collector.eval_agent_n_record_init_states(agent, eval_env, num_evals)

        # estimate distance-to-goal from initial states
        states = dict(observation=[], goal=[])
        dist_from_rewards = [] # not ground truth distance, but should be accurate when policy is trained
        ep_costs = []
        for key in eval_outputs.keys():
            states['observation'].append(eval_outputs[key]["init_states"]['observation'])
            states['goal'].append(eval_outputs[key]["init_states"]["goal"])
            dist_from_rewards.append(-eval_outputs[key]["rewards"])
            ep_costs.append(eval_outputs[key]["costs"])

        
        pred_dist = list(agent.get_dist_to_goal(states))
        eval_stats["rewards"][min_dist] = {
            'd_from_rewards': np.mean(dist_from_rewards),
            'std_d_from_rewards': np.std(dist_from_rewards),
            'd_pred': np.mean(pred_dist),
            'std_d_pred': np.std(pred_dist),
        }

        pred_costs = list(agent.get_cost_to_goal(states))
        eval_stats["costs"]["true"].extend(ep_costs)
        eval_stats["costs"]["pred"].extend(pred_costs)
    eval_stats["costs"]["true"] = np.array(eval_stats["costs"]["true"])
    eval_stats["costs"]["pred"] = np.array(eval_stats["costs"]["pred"])

    # regroup costs into low, mid, and high classes for easy visual
    re_masks = regroup_value_lists(
        val_list=eval_stats["costs"]["true"],
        div_intervals=cost_intervals,
        )

    eval_stats["grouped_costs"] = {}
    for lb_cost in re_masks:
        cur_mask = re_masks[lb_cost]
        eval_stats["grouped_costs"][lb_cost] = {}
        eval_stats["grouped_costs"][lb_cost]["true"] = eval_stats["costs"]["true"][cur_mask]
        eval_stats["grouped_costs"][lb_cost]["pred"] = eval_stats["costs"]["pred"][cur_mask]

    return eval_stats



def regroup_value_lists(
        val_list:np.ndarray, 
        div_intervals:Union[np.ndarray, List[float]]
        ) -> Dict[float, np.ndarray]:
    """
    returns binary masks for each group

    split a list of values into different classes for easy visualization
    
    grouping rule: class <= input < class_next goes to class
    for the last classes, input > class_end
    """
    rearranged_outs = {}

    for div_i in range(len(div_intervals)):
        div_start = div_intervals[div_i]
        if div_i == len(div_intervals)-1:
            cur_div_mask = val_list >= div_start
        else:
            div_end = div_intervals[div_i+1]

            mas_cond1 = div_start<=val_list
            mas_cond2 = val_list<div_end
            cur_div_mask = mas_cond1 * mas_cond2

        rearranged_outs[div_start] = cur_div_mask
    return rearranged_outs
