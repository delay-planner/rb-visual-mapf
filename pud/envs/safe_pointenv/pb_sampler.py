from typing import List, Union

import numpy as np

from pud.algos.data_struct import arg_topk, get_nd_inds_set, arg_group_vals
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.safe_wrappers import \
    SafeGoalConditionedPointWrapper


def calc_pairwise_cost(agent:DRLDDPGLag, rb_vec:np.ndarray, ensemble_agg="max"):
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)
    pcost_agg = None
    if ensemble_agg == "max":
        pcost_agg = np.max(pcost, axis=0)
    elif ensemble_agg == "mean":
        pcost_agg = np.mean(pcost, axis=0)
    return pcost_agg

def calc_pairwise_dist(agent:DRLDDPGLag, rb_vec:np.ndarray, ensemble_agg="max"):
    pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)
    pdist_agg = None
    if ensemble_agg == "max":
        pdist_agg = np.max(pdist, axis=0)
    elif ensemble_agg == "mean":
        pdist_agg = np.mean(pdist, axis=0)
    return pdist_agg

def sample_pbs_by_agent(
        env:SafeGoalConditionedPointWrapper, 
        agent:DRLDDPGLag, 
        num_states:int=100,
        target_val:List[float]=None,
        pval_f = None, # function to generate pairwise values (e.g., dists, cost, ...) 
        ensemble_agg:str="max",
        K:int=5, # num of samples nearest to the target metric
        ) -> List[dict]:
    """sample problems with target metrics according to the predictions of the agent

    Args:
        env (SafeGoalConditionedPointWrapper): env that contains reset_orig, which returns normalized start-goal pairs
        agent (DRLDDPGLag): 
        pval_f: (agent, rb_vec) -> pairwise values
        num_states (int, optional): _description_. Defaults to 100.
    """
    ## online generate start and goal states nearest to target cumulative costs
    rb_vec = [None] * num_states
    for i in range(num_states):
        s0, info = env.reset_orig()
        rb_vec[i] = s0
    rb_vec = np.array([x["observation"] for x in rb_vec])

    ## predict the pairwise costs
    pvals = pval_f(agent, rb_vec, ensemble_agg) # num_states x num_states
    diff = np.abs(pvals - target_val)
    inds = arg_topk(-diff, topK=K) # find K minimum entries
    #inds_set = get_nd_inds_set(inds)
    nearest_pbs = [None] * K
    for n in range(K):
        i,j = inds[0][n], inds[1][n]
        nearest_pbs[n] = {
            "start": env.de_normalize_obs(rb_vec[i]),
            "goal": env.de_normalize_obs(rb_vec[j]),
            "info": {"prediction": pvals[i,j]},
        }
    return nearest_pbs


def constrained_sampler_for_cost(
        env:SafeGoalConditionedPointWrapper, 
        agent:DRLDDPGLag, 
        num_states:int=100,
        target_val:List[float]=None,
        min_dist: float = 0,
        max_dist: float = 10,
        ensemble_agg:str="mean",
        max_attempts=100,
    ):
    """
    return a sampler function that generate pbs (start, goal)
    """
    for _ in range(max_attempts):
        s = env.sample_safe_empty_state(cost_limit=env.cost_limit)
        g = env.sample_safe_empty_state(cost_limit=env.cost_limit)
        s_hat = {
            "observation": env._normalize_obs(s),
            "goal": env._normalize_obs(g),
        }
        d_2_g = agent.get_dist_to_goal(s_hat)
        c_2_g = agent.get_cost_to_goal(s_hat)
        if d_2_g >= min_dist and d_2_g <= max_dist:
            pb = {
                "normalized_state": s_hat,
                "s": s,
                "g": g,
                "proj_d": d_2_g,
                "proj_c": c_2_g,
            }
            return pb
    return


def sample_cost_pbs_by_agent(
        env:SafeGoalConditionedPointWrapper, 
        agent:DRLDDPGLag, 
        num_states:int=100,
        target_val:List[float]=None,
        min_dist: float = 0,
        max_dist: float = 10,
        ensemble_agg:str="mean",
        K:int=5, # num of samples nearest to the target metric
        use_uncertainty:bool=True, # boost samples of high uncertainty 
        uncertainty_lb:float=0.0, 
        uncertainty_ub:float=1.0, 
    ):
    """
    filter based on distance constraints
    problems whose start and goals are seperated too far away is 
    meaningless as they are handled by the HRL
    if failed, return an empty list, because the test results would not be informative
    """
    rb_vec = [None] * num_states
    #rb_vec_start = [None] * num_states
    #rb_vec_goal = [None] * num_states
    for i in range(num_states):
        s0, info = env.reset_orig()
        rb_vec[i] = s0
        #rb_vec_start[i] = env.de_normalize_obs(s0["observation"])
        #rb_vec_goal[i] = env.de_normalize_obs(s0["goal"])
    rb_vec = np.array([x["observation"] for x in rb_vec])

    pcosts_ens = agent.get_pairwise_cost(rb_vec, aggregate=None) # num_states x num_states
    pcosts = None
    if ensemble_agg == "max":
        pcosts = np.max(pcosts_ens, axis=0)
    elif ensemble_agg == "mean":
        pcosts = np.mean(pcosts_ens, axis=0)
    
    pcosts_std, pcosts_std_mean = 0.0, 0.0
    if use_uncertainty:
        pcosts_std = np.std(pcosts_ens, axis=0)
        pcosts_std_mean = np.mean(pcosts_std)
        pcosts_std = np.clip(pcosts_std, 
            a_min=uncertainty_lb, a_max=uncertainty_ub)

    # filter based on distance constraints
    lb_mask = pcosts >= min_dist
    ub_mask = pcosts <= max_dist
    prod_mask = lb_mask * ub_mask
    
    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:    
        pcosts = calc_pairwise_cost(agent, rb_vec, ensemble_agg) # num_states x num_states
        pcosts_gInds = pcosts[gInds]
        diff = np.abs(pcosts_gInds - target_val)
        # encourage diverse samples
        if use_uncertainty: 
            diff = diff - pcosts_std.flatten()
        K = min(K, len(diff))
        mInds = arg_topk(-diff, topK=K) # find K minimum entries
        gmInds = (gInds[0][mInds], gInds[1][mInds])

        nearest_pbs = [None] * K
        for n in range(K):
            i,j = gmInds[0][n], gmInds[1][n]
            nearest_pbs[n] = {
                "start": env.de_normalize_obs(rb_vec[i]),
                "goal": env.de_normalize_obs(rb_vec[j]),
                "info": {"prediction": pcosts[i,j],
                        "proj_dist": pcosts[i,j],
                        "ensemble_std_mean": pcosts_std_mean,
                        },
                    }
        return nearest_pbs
