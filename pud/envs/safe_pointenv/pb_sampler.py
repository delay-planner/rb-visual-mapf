from typing import List, Union

import numpy as np

from pud.algos.data_struct import arg_topk, get_nd_inds_set
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.safe_wrappers import \
    SafeGoalConditionedPointWrapper


def max_pairwise_cost(agent:DRLDDPGLag, rb_vec:np.ndarray):
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)
    pcost_max = np.max(pcost, axis=0)
    return pcost_max

def sample_pbs_by_agent(
        env:SafeGoalConditionedPointWrapper, 
        agent:DRLDDPGLag, 
        num_states:int=100,
        target_val:List[float]=None,
        pval_f = None, # function to generate pairwise values (e.g., dists, cost, ...) 
        K:int=5, # num of samples nearest to the target metric
        ):
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
    pvals = pval_f(agent, rb_vec) # num_states x num_states

    diff = np.abs(pvals - target_val)
    inds = arg_topk(-diff, topK=K) # find K minimum entries
    inds_set = get_nd_inds_set(inds)

    nearest_samples = {
        "starts": rb_vec[inds[0]],
        "goals": rb_vec[inds[1]],
        "predictions": pvals[inds],
    }
    return nearest_samples