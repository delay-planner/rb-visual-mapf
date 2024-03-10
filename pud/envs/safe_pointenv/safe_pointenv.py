from pud.envs.simple_navigation_env import PointEnv
import numpy as np
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from typing import Optional

def plot_safe_walls(walls:np.ndarray, cost_map:np.ndarray, cost_upper_bound:float, ax:plt.axes):
    walls = walls.T
    (height, width) = walls.shape
    # only plot walls
    for (i, j) in zip(*np.where(walls)):
        x = np.array([j, j+1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i+1, i+1]) / float(height)
        ax.fill_between(x, y0, y1, color='grey')
    
    # plot non-wall unsafe boxes
    for (i, j) in zip(*np.where(cost_map > cost_upper_bound)):
        if walls[i,j] == 1: # skip walls
            continue
        x = np.array([j, j+1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i+1, i+1]) / float(height)
        ax.fill_between(x, y0, y1, color='red')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    return ax


class SafePointEnv (PointEnv):
    """
    Add cost metric to PointEnv
    add cost to _asap
    """
    def __init__(self, 
                walls:str=None, 
                resize_factor:int=1,
                action_noise=1.0, 
                thin=False,
                # cost configs
                cost_f_args:dict={},
                precompiled_cost_apsps:List[float]= [0, 0.1],
                ):
        super(SafePointEnv, self).__init__(
            walls,
            resize_factor,
            action_noise,
            thin,)

        self.resize_factor = resize_factor
        self.thin = thin
        self.wall_name = walls
        
        obstacle_x, obstacle_y = np.where(self._walls == 1)
        self.obstacles = np.stack([obstacle_x, obstacle_y], axis=-1).astype(float)
        
        self.cost_f_cfg = cost_f_args
        cost_fn_name = cost_f_args.get('name')
        self.cost_function = None
        if cost_fn_name == 'cosine':
            from pud.envs.safe_pointenv.cost_functions import cost_from_cosine_distance
            import functools
            self.cost_function = functools.partial(cost_from_cosine_distance, r=self.cost_f_cfg['radius'])

            # if it is fast, leave it here to avoid another config entry to load from file
            self._cost_map = self.build_cost_map()
            self._safe_apsp = {}
            for c_up in precompiled_cost_apsps:
                self._safe_apsp[c_up] = self._compute_safe_apsp(self._walls, self._cost_map, cost_upper_bound=c_up)

    def get_map_width(self):
        return self._width
    
    def get_map_height(self):
        return self._height

    def build_cost_map(self):
        width = self.get_map_width()
        height = self.get_map_height()

        # todo: does the height and weight order matter here?
        mesh_x, mesh_y = np.meshgrid(np.arange(height), np.arange(width))
        p_x = mesh_x.ravel()
        p_y = mesh_y.ravel()
        pnts = np.transpose(np.vstack([p_x, p_y]))

        pnts_cost = np.zeros((len(p_x),), dtype=float)

        for ii, pt in enumerate(pnts):
            min_d, _  = self.dist_2_blocks(pt)
            pt_cost= self.cost_function(min_d)
            pnts_cost[ii] = pt_cost

        cost_map = pnts_cost.reshape(mesh_x.shape)
        return cost_map
    
    def _compute_safe_apsp(self, walls:np.ndarray, cost_map:np.ndarray, cost_upper_bound:float):
        """
        cost equivalent of _compute_asps
        take advantage of the knowledge that edges are bi-directional, so simply removing unsafe nodes/edges
        """
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
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
                        if cost_map[i,j] >= cost_upper_bound:
                            continue
                        if cost_map[i + di,j + dj] >= cost_upper_bound:
                            continue
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float32('inf'))
        for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
            for ((i2, j2), d) in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist
    
    def _sample_safe_empty_state(self, max_attempts=100):
        candidate_states = np.where(self._walls == 0)

        if hasattr(self, "_cost_constraints") and hasattr(self, "cost_map"):
            new_candidate_states = [[],[]]
            for cx, cy in zip(*candidate_states):
                # only sample states whose costs are lower than an upper bound
                if self.cost_map[cx, cy] < self._cost_constraints['max_cost']:
                    new_candidate_states[0].append(cx)
                    new_candidate_states[1].append(cy)
            candidate_states = new_candidate_states
            
        num_candidate_states = len(candidate_states[0])
        assert num_candidate_states > 0
        state_index = np.random.choice(num_candidate_states)

        # state extracted from grid coords, all intergers
        state_int = np.array([candidate_states[0][state_index],
                          candidate_states[1][state_index]],
                         dtype=np.float32)
        
        for i in range(max_attempts):
            state = state_int + np.random.uniform(size=2)
            if hasattr(self, "_cost_constraints") and hasattr(self, "cost_map"):
                state_cost = self.get_state_cost(state)
                if state_cost >= self._cost_constraints["max_cost"]:
                    continue
            if not self._is_blocked(state):
                return state
        # if all failed, use state_int instead
        return state_int
    
    def dist_2_blocks(self, xy:np.ndarray):
        """
        calculate the distance between a float state xy and a block state that are ints (from array indices)

        a block covers an square area of 
        block_x -- block_x+1
        block_y -- block_y+1

        Args:
            xy (np.ndarray): [x,y]
            block_xys (np.ndarray): [[block_x, block_y], ... ]

        Returns:
            float: calculated distance
            int: index of the nearest block
        
        Example:
            xy = np.array([0.5, 0.6])
            
            block_xys = np.array([[0,1],[2,5]])

        Reference: https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        
        bxs_min = self.obstacles[:,0]
        bys_min = self.obstacles[:,1]
        x, y = xy

        dxs = np.maximum(bxs_min-x, x-(bxs_min+1))
        dxs = np.maximum(dxs, 0.0)
        
        dys = np.maximum(bys_min-y, y-(bys_min+1))
        dys = np.maximum(dys, 0)

        d2 = dxs**2.0 + dys**2.0
        ind_min = np.argmin(d2)
        d_min = np.sqrt(d2[ind_min]) 

        return d_min, ind_min

    def get_state_cost(self, xy:np.ndarray):
        min_d, _ = self.dist_2_blocks(xy)
        return self.cost_function(min_d)