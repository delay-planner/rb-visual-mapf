import unittest

from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.algos.cbfs_eval import CBFS, compile_all_pair_constrained_shortest_trajs
import networkx as nx
import numpy as np
from pathlib import Path
"""
python pud/algos/unit_tests/test_bfs_eval.py TestCBFSEval.test_compile_all_pair_constrained_shortest_trajs
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

if __name__ == '__main__':
    unittest.main()
