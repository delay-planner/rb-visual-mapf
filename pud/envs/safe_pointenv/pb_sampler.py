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
        min_dist: float = 0,
        max_dist: float = 10,
        target_val:List[float]=None,
        ensemble_agg:str="max",
        use_uncertainty:bool=True, # boost samples of high uncertainty 
        uncertainty_lb:float=0.0, 
        uncertainty_ub:float=1.0, 
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

    rb_vec_goal = [None] * num_states
    for i in range(num_states):
        s0, info = env.reset_orig()
        rb_vec_goal[i] = s0
    rb_vec_goal = np.array([x["observation"] for x in rb_vec_goal])

    ## predict the pairwise costs
    pdist = agent.get_pairwise_dist(obs_vec=rb_vec, 
                    goal_vec=rb_vec_goal, 
                    aggregate=None)
    pdist_agg = None
    if ensemble_agg == "max":
        pdist_agg = np.max(pdist, axis=0)
    elif ensemble_agg == "mean":
        pdist_agg = np.mean(pdist, axis=0)

    pdist_std, pdist_std_mean = 0.0, 0.0
    if use_uncertainty:
        pdist_std = np.std(pdist, axis=0)
        pdist_std_mean = np.mean(pdist_std)

    lb_mask = pdist_agg + pdist_std >= min_dist
    ub_mask = pdist_agg - pdist_std <= max_dist
    prod_mask = lb_mask * ub_mask

    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:
        pdist_gInds = pdist_agg[gInds]
        pdist_stds_gInds = pdist_std[gInds]
        scoring = 0.0 # smaller -> better
        if target_val is not None:
            scoring = scoring + np.abs(pdist_gInds - target_val)
        # encourage diverse samples
        if use_uncertainty: 
            scoring = scoring - np.clip(pdist_stds_gInds,
                    a_min=uncertainty_lb, a_max=uncertainty_ub)

        K = min(K, len(scoring))
        mInds = arg_topk(-scoring, topK=K) # find K minimum entries
        gmInds = (gInds[0][mInds], gInds[1][mInds])

        nearest_pbs = [None] * K
        for n in range(K):
            i,j = gmInds[0][n], gmInds[1][n]
            nearest_pbs[n] = {
                "start": env.de_normalize_obs(rb_vec[i]),
                "goal": env.de_normalize_obs(rb_vec_goal[j]),
                "info": {"prediction": pdist_agg[i,j],
                        "proj_dist": pdist_agg[i,j],
                        "ensemble_std_mean": pdist_std_mean,
                        },
                    }
    return nearest_pbs


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

    # filter based on distance constraints
    lb_mask = pcosts + pcosts_std >= min_dist
    ub_mask = pcosts - pcosts_std <= max_dist
    prod_mask = lb_mask * ub_mask
    
    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:    
        pcosts = calc_pairwise_cost(agent, rb_vec, ensemble_agg) # num_states x num_states
        pcosts_gInds = pcosts[gInds]
        pcosts_std_gInds = pcosts_std[gInds]
        scoring = 0.0
        if target_val is not None:
            scoring = np.abs(pcosts_gInds - target_val)
        # encourage diverse samples
        if use_uncertainty: 
            scoring = scoring - np.clip(pcosts_std_gInds,
                        a_min=uncertainty_lb, a_max=uncertainty_ub)
        K = min(K, len(scoring))
        mInds = arg_topk(-scoring, topK=K) # find K minimum entries
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
