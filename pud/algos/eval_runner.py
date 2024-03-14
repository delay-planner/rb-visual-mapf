"""
Evaluate the accuracy and reliability of reward and cost critics
"""

import numpy as np
import time
from typing import Optional, List, Union, Dict
from pud.algos.constrained_collector import ConstrainedCollector as Collector

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


def eval_pointenv_cost_constrained_dists(agent, eval_env, num_evals=10, eval_distances=[2, 5, 10], cost_args:dict={}, sample_args:Optional[dict]=None, verbose:bool=True):
    """sample starts and goals that are lower than a preset maximum cost limit (linear interpolation). 
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
    
    for dist in eval_distances:
        eval_env.set_sample_goal_args(prob_constraint=1, min_dist=dist, max_dist=dist) # NOTE: samples goal distances in [min_dist, max_dist] closed interval
        eval_env.set_sample_cost_args(sample_args)
        
        eval_outputs = Collector.eval_agent_w_init_states(agent, eval_env, num_evals)

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
        eval_stats["rewards"][dist] = {
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
        div_intervals=[0., 0.2, 0.5, 1.0],
        )

    eval_stats["grouped_costs"] = {}
    for lb_cost in re_masks:
        cur_mask = re_masks[lb_cost]
        eval_stats["grouped_costs"][lb_cost] = {}
        eval_stats["grouped_costs"][lb_cost]["true"] = eval_stats["costs"]["true"][cur_mask]
        eval_stats["grouped_costs"][lb_cost]["pred"] = eval_stats["costs"]["pred"][cur_mask]

    return eval_stats