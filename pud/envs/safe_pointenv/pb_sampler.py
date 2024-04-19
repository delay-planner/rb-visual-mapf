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

def sample_cost_pbs_by_agent(
        env:SafeGoalConditionedPointWrapper, 
        agent:DRLDDPGLag, 
        num_states:int=100,
        target_val:List[float]=None,
        min_dist: float = 0,
        max_dist: float = 10,
        ensemble_agg:str="mean",
        K:int=5, # num of samples nearest to the target metric
    ):
    """
    filter based on distance constraints
    problems whose start and goals are seperated too far away is 
    meaningless as they are handled by the HRL
    if failed, return an empty list, because the test results would not be informative
    
    """
    rb_vec = [None] * num_states
    for i in range(num_states):
        s0, info = env.reset_orig()
        rb_vec[i] = s0
    rb_vec = np.array([x["observation"] for x in rb_vec])

    pdists = calc_pairwise_dist(agent, rb_vec, ensemble_agg) # num_states x num_states
    # filter based on distance constraints
    lb_mask = pdists >= min_dist
    ub_mask = pdists <= max_dist
    prod_mask = lb_mask * ub_mask
    
    gInds = np.where(prod_mask)
    if len(gInds[0]) == 0:
        return []
    else:    
        pcosts = calc_pairwise_cost(agent, rb_vec, ensemble_agg) # num_states x num_states
        pcosts_gInds = pcosts[gInds]
        diff = np.abs(pcosts_gInds - target_val)
        mInds = arg_topk(-diff, topK=K) # find K minimum entries
        gmInds = (gInds[0][mInds], gInds[1][mInds])

        nearest_pbs = [None] * K
        for n in range(K):
            i,j = gmInds[0][n], gmInds[1][n]
            nearest_pbs[n] = {
                "start": env.de_normalize_obs(rb_vec[i]),
                "goal": env.de_normalize_obs(rb_vec[j]),
                "info": {"prediction": pcosts[i,j],
                        "proj_dist": pdists[i,j]
                        },
                    }
        return nearest_pbs
