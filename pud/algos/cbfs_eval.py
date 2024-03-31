"""
Start from one grid point, construct a tree via breadth-first search expansion, stop expansion when cost-limit is violated.

Construct reference problems for specified ranges of cumulative costs

NOTE: the deviation from the reference distance depends on the maneuver flexibility (e.g., whether there exists some bottleneck passages, etc.)

Only operate on the discretized maze/graph level, no actual env stepping
"""

import pickle
import time
from pathlib import Path

import numpy as np
from numba import jit
from tqdm.auto import tqdm

from typing import Optional, List
from termcolor import cprint

def init_embedded_dict(D:dict, embeds:list=[]):
    """
    in-place init of embedded dict
    the init function should be either a list or dict
    embeds = [(key, init_function), ...]

    example: 
    DD = {}
    init_embedded_dict(DD, embeds=[(1, dict), (2, dict), (3, list)])
    init_embedded_dict(DD, embeds=[(1, dict), (2, dict), (3, list)])
    """
    tmp_D = D
    for next_key, init_f in embeds:
        if not (next_key in tmp_D):
            tmp_D[next_key] = init_f()
        
        assert isinstance(tmp_D[next_key], init_f)
        tmp_D = tmp_D[next_key]
    return 


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
        output_file = output_dir.joinpath("root={}_cost_limit={}.pkl".format(root,cost_limit))
        save_data = {
            "trajs": out, # list
            "costs": out_cost, # list
        }
        f = open(output_file, 'wb')
        pickle.dump(save_data, f)
        f.close()

def analyze_precompiled_cost_and_lengths(savedir):
    savedir:Path = Path(savedir)
    list_fs = list(savedir.iterdir())

    ## not needed unless the range of costs need to be redefined
    set_costs = set()
    set_lens = set()
    for i_f, f in tqdm(enumerate(list_fs), total=len(list_fs), desc="calc unique length and costs"):
        tmp_data = None
        with open(f, 'rb') as f:
            tmp_data = pickle.load(f)
        trajs = tmp_data["trajs"]
        costs = tmp_data["costs"]
        for i in range(len(trajs)):
            traj_i = trajs[i]
            if len(traj_i) > 1:
                cost_i = costs[i]
                set_costs.add(cost_i)
                len_i = len(traj_i)
                set_lens.add(len_i)
    return set_costs, set_lens


def catalog_precompiled_paths(savedir):
    """load all prebuilt policies for balanced sampling
    query policies based on traj distance and cost

    build a catalog of path

    parent file: root=(0,0) ...
    each parent file contains two lists: trajs, and costs
        indexing should be 0, ..., len(trajs)-1
    collect the unique the cost set
    indexing the parent file in the dir

    each entry = parent file index + index within the parent file
    """
    savedir:Path = Path(savedir)
    list_fs = list(savedir.iterdir())

    all_trajs = {} # based on costs, path lengths and then file index
    for i_f, f in tqdm(enumerate(list_fs), total=len(list_fs), desc="indexing trajs"):
        tmp_data = None
        with open(f, 'rb') as f:
            tmp_data = pickle.load(f)
        trajs = tmp_data["trajs"]
        costs = tmp_data["costs"]
        for i in range(len(trajs)):
            traj_i = trajs[i]
            if len(traj_i) > 1:
                cost_i = costs[i]
                len_i = len(traj_i)

                init_embedded_dict(
                    all_trajs, 
                    embeds=[(cost_i, dict),
                            (len_i, dict), 
                            (i_f, list),
                            ],
                )
                # append according to cost, length, and file index                
                all_trajs[cost_i][len_i][i_f].append(i)

    ## this pool is huge, save as layered inds
    with open("pud/envs/precompiles/central_obstacle.pkl", 'wb') as f:
        data_catalog = {
            "files": list_fs,
            "trajs": all_trajs,
            "parent_dir": savedir.as_posix(),
            "notes": """costs->path_length->file_index->traj_index, file_index starts from 0 based on files""",
        }
        pickle.dump(data_catalog, f)

    
    t0 = time.time()
    with open("pud/envs/precompiles/central_obstacle.pkl", 'rb') as f:
        pickle.load(f)
    print("[INFO] loading time of sample policy catalog: {}".format(time.time() - t0))

    return data_catalog

def sample_precompiled_grid_policies(
        policies: dict,
        min_cost: float,
        max_cost:float,
        min_len: float,
        max_len: float,
        ps_costs: Optional[List[float]]=None, 
        ):
    """
    load sample files on-demand
    """
    # scratch for balanced sampling
    trajs = policies["trajs"]
    files = policies["files"]
    
    list_costs = list(trajs.keys())
    if ps_costs is None:
        # uniform distribution on costs
        ps_costs = np.ones(len(list_costs)) / float(len(list_costs))
    bounded_costs, bounded_ps_costs = [], []
    for i, c in enumerate(list_costs):
        if c>=min_cost and c<=max_cost:
            bounded_costs.append(c)
            bounded_ps_costs.append(ps_costs[i])
    bounded_ps_costs = np.array(bounded_ps_costs) / np.sum(bounded_ps_costs)

    if len(bounded_costs) == 0:
        cprint("[ERROR]: cost range is empty", "red")
        return

    sample_cost = np.random.choice(bounded_costs, p=bounded_ps_costs)

    bounded_lens = [x for x in list(trajs[sample_cost].keys()) if (x >= min_len and x<=max_len)]
    if len(bounded_lens) == 0:
        return
    sample_len = np.random.choice(bounded_lens)

    sample_file_ind = np.random.choice(list(trajs[sample_cost][sample_len].keys()))
    sample_file_path = files[sample_file_ind]
    sample_traj_idx = np.random.choice(trajs[sample_cost][sample_len][sample_file_ind])

    traj_f_data = None
    with open(sample_file_path, 'rb') as f:
        traj_f_data = pickle.load(f)

    sample_traj = traj_f_data["trajs"][sample_traj_idx]
    sample_traj_cost = traj_f_data["costs"][sample_traj_idx]
    return sample_traj, sample_traj_cost
