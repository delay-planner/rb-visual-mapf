"""
Start from one grid point, construct a tree via breadth-first search expansion, stop expansion when cost-limit is violated.

Construct reference problems for specified ranges of cumulative costs

NOTE: the deviation from the reference distance depends on the maneuver flexibility (e.g., whether there exists some bottleneck passages, etc.)

Only operate on the discretized maze/graph level, no actual env stepping
"""

from numba import jit
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import pickle

#@jit(nopython=True)
@jit(forceobj=True, looplift=True)
def CBFS(g:dict, root:tuple, cost_limit:float, cost_map:np.ndarray):
    """
    g: dict of dicts, exported from networkx graph
    root: node of a tuple (i,j)

    #MIT 16.413 Lecture Note Simple Search
    #1. Initialize Q with partial path (S) as only entry; set Visited = ( );
    #2. If Q is empty, fail. Else, pick some partial path N from Q;
    #3. // If head(N) = G, return N; (goal reached!)
    #4. Else
        #a) Remove N from Q;
        #b) Find all children of head(N) (its neighbors in g) not in Visited
        #and create a one-step extension of N to each child;
        #c) Add to Q all the extended paths;
        #d) Add children of head(N) to Visited;
        #e) Go to step 2
    todo: make this function fully compatible to numba fast compile @njit
    """
    # setup
    visited = [] # visited nodes, not partial paths
    partial_paths = [] # partially expanded paths
    partial_path_costs = [] # cost of partial paths

    explored_paths = []
    explored_path_costs = []
    
    root_cost = cost_map[root[0], root[1]]    
    if root_cost > cost_limit:
        return partial_paths, partial_path_costs

    visited.append(root)
    partial_paths.append([root])
    partial_path_costs.append(root_cost)

    while len(partial_paths) > 0:
        subpath_n = partial_paths.pop(-1) # first in first out
        subpath_n_cost = partial_path_costs.pop(-1)
        explored_paths.append(subpath_n)
        explored_path_costs.append(subpath_n_cost)
        subpath_n_head = subpath_n[-1]
        for n_ch in g[subpath_n_head]:
            if n_ch in visited:
                continue
            # add one-step extension
            visited.append(n_ch)
            n_ch_cost = cost_map[n_ch[0], n_ch[1]]
            new_subpath_cost = subpath_n_cost + n_ch_cost
            if new_subpath_cost > cost_limit:
                continue
            new_subpath = subpath_n.copy()
            new_subpath.append(n_ch)
            partial_paths.insert(0, new_subpath)
            partial_path_costs.insert(0, new_subpath_cost)

    return explored_paths, explored_path_costs

@jit(forceobj=True, looplift=True)
def compile_all_pair_constrained_shortest_trajs(
        gd:dict, 
        cost_limit:float, 
        cost_map:np.ndarray,
        output_dir:Path,
    ):
    """
    store the raw/unnormalized indices of the maze
    cost_limit: speed up the plan generation as it does not have to explore the whole space
    evaluate edge cost

    gd: dict of dicts
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for root in tqdm(gd):
        out, out_cost = CBFS(gd, root, cost_limit=cost_limit, cost_map=cost_map)
        output_file = output_dir.joinpath("root={}_cost_limit={}".format(root,cost_limit))
        save_data = {
            "trajs": out,
            "costs": out_cost,
        }
        f = open(output_file, 'wb')
        pickle.dump(save_data, f)
        f.close()

