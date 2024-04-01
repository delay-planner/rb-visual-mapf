import pickle
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from pud.algos.cbfs_eval import (CBFS, catalog_precompiled_paths,
                                 compile_all_pair_constrained_shortest_trajs,
                                 sample_precompiled_grid_policies)
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv

"""
python pud/algos/unit_tests/test_bfs_eval.py TestCBFSEval.test_catalog_precompiled_paths
python pud/algos/unit_tests/test_bfs_eval.py TestCBFSEval.test_sample_grid_traj
"""

class TestCBFSEval(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = {
            "walls": "CentralObstacle",
            #"walls": "FourRooms",
            "resize_factor": 5,
            "thin": False,
        }
        self.cost_f_kwargs = {
            "name": "cosine",
            "radius": 2.,
        }
        self.precompilation_kwargs = {
            "cost_limit": 1,
        }

        self.p_env = SafePointEnv(
                    **self.env_kwargs, 
                    **self.precompilation_kwargs,
                    cost_f_args=self.cost_f_kwargs)
        
        walls = self.p_env._walls
        cost_limit = self.precompilation_kwargs["cost_limit"]
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        # edges over the cost limit is guaranteed not to be feasible
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0:
                            continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if walls[i, j] == 1:
                            continue  # Don't add edges to walls
                        if walls[i + di, j + dj] == 1:
                            continue  # Don't add edges to walls
                        ## filtering by cost map
                        if self.p_env._cost_map[i,j] > cost_limit:
                            continue
                        if self.p_env._cost_map[i + di,j + dj] > cost_limit:
                            continue
                        g.add_edge((i, j), (i + di, j + dj))

        self.g_dict = nx.to_dict_of_dicts(G=g)

    def test_CBFS(self):
        cost_map = self.p_env._cost_map
        cost_limit = self.precompilation_kwargs["cost_limit"]
        explored_paths, explored_path_costs = CBFS(
            self.g_dict, 
            (0,0),
            cost_limit=1, 
            cost_map=cost_map,
            )

        # check the length
        path_lens = []
        for ep in explored_paths:
            path_lens.append(len(ep))
        
        # check the lengths are monotonoic
        self.assertTrue(
            np.all(np.diff(path_lens) >= 0)
        )
        
        # check the trajectory costs
        for i, ep in enumerate(explored_paths):
            traj_cost = 0.0
            for nj in ep:
                traj_cost += cost_map[nj[0], nj[1]]
            
            assert traj_cost == explored_path_costs[i]

        # check the max traj cost
        self.assertTrue(
            np.max(explored_path_costs) <= cost_limit
        )

        # check if the end nodes are unique
        end_nodes = []
        for ep in explored_paths:
            end_nodes.append(ep[-1])
        print("[Check] the terminal nodes are unique?: {}".format(
            len(set(end_nodes)) == len(end_nodes)
        ))

        self.assertTrue(
            len(set(end_nodes)) == len(end_nodes)
        )
    
    def test_compile_all_pair_constrained_shortest_trajs(self):
        cost_map = self.p_env._cost_map
        cost_limit = self.precompilation_kwargs["cost_limit"]
        output_dir = Path("pud/envs/precompiles").joinpath(
            "{}_resize_factor={:0>2d}_thin={}_cost_limit='{:.2f}'".format(
            self.env_kwargs["walls"],
            self.env_kwargs["resize_factor"],
            self.env_kwargs["thin"],
            cost_limit,
            ))
        
        compile_all_pair_constrained_shortest_trajs(
            self.g_dict, 
            cost_limit=cost_limit, 
            cost_map=cost_map,
            output_dir=output_dir,
            )
        
    def test_catalog_precompiled_paths(self):
        savedir = "pud/envs/precompiles/CentralObstacle_resize_factor=05_thin=False_cost_limit='1.00'"
        catalog_precompiled_paths(savedir=savedir)

    def test_unpack_n_sample(self):
        fp = "pud/envs/precompiles/central_obstacle.pkl"
        policies = None
        with open(fp, 'rb') as f:
            policies = pickle.load(f)

        # scratch for balanced sampling
        trajs = policies["trajs"]
        files = policies["files"]
        import time
        time_start = time.time()

        min_cost = 0.0
        max_cost = 1.0
        ps_costs = [0.25] * 4
        
        list_costs = list(trajs.keys())

        min_len = 1.0
        max_len = 10.0

        bounded_costs, bounded_ps_costs = [], []
        for i, c in enumerate(list_costs):
            if c>=min_cost and c<=max_cost:
                bounded_costs.append(c)
                bounded_ps_costs.append(ps_costs[i])

        assert len(bounded_costs) > 0

        sample_cost = np.random.choice(bounded_costs, p=bounded_ps_costs)

        bounded_lens = [x for x in list(trajs[sample_cost].keys()) if (x >= min_len and x<=max_len)]
        sample_len = np.random.choice(bounded_lens)

        sample_file_ind = np.random.choice(list(trajs[sample_cost][sample_len].keys()))
        sample_file_path = files[sample_file_ind]
        sample_traj_idx = np.random.choice(trajs[sample_cost][sample_len][sample_file_ind])

        traj_f_data = None
        with open(sample_file_path, 'rb') as f:
            traj_f_data = pickle.load(f)

        sample_traj = traj_f_data["trajs"][sample_traj_idx]
        sample_traj_cost = traj_f_data["costs"][sample_traj_idx]

        assert sample_traj_cost == sample_cost
        assert len(sample_traj) == sample_len

        print("[INFO] sample time: {}".format(time.time() - time_start))

    def test_sample_grid_traj(self):
        fp = "pud/envs/precompiles/central_obstacle_v2.pkl"
        policies = None
        with open(fp, 'rb') as f:
            policies = pickle.load(f)

        min_cost, max_cost, min_len, max_len = 0, 1, 1, 10
        out = sample_precompiled_grid_policies(policies=policies, min_cost=min_cost, max_cost=max_cost, min_len=min_len, max_len=max_len)
        assert out is not None
        assert (out[1] <= max_cost and out[1] >= min_cost)
        assert (len(out[0]) <= max_len and len(out[0]) >= min_len)

        min_cost, max_cost, min_len, max_len = 0.5, 1, 5, 5
        out = sample_precompiled_grid_policies(policies=policies, min_cost=min_cost, max_cost=max_cost, min_len=min_len, max_len=max_len)
        assert out is not None
        assert (out[1] <= max_cost and out[1] >= min_cost)
        assert (len(out[0]) <= max_len and len(out[0]) >= min_len)

        min_cost, max_cost, min_len, max_len = 0.5, 1, 6, 5
        out = sample_precompiled_grid_policies(policies=policies, min_cost=min_cost, max_cost=max_cost, min_len=min_len, max_len=max_len)
        assert out is None

        min_cost, max_cost, min_len, max_len = 0.5, 0.4, 1, 10
        out = sample_precompiled_grid_policies(policies=policies, min_cost=min_cost, max_cost=max_cost, min_len=min_len, max_len=max_len)
        assert out is None


if __name__ == '__main__':
    unittest.main()
