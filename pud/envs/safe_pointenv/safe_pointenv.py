from pud.envs.simple_navigation_env import PointEnv, GoalConditionedPointWrapper
from pud.envs.wrappers import TimeLimit
import numpy as np
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from typing import Optional
import time

def plot_safe_walls(walls:np.ndarray, cost_map:np.ndarray, cost_limit:float, ax:plt.axes):
    """
    step-wise cost limit visualization
    """
    walls = walls.T
    cost_map = cost_map.T
    (height, width) = walls.shape
    # only plot walls
    for (i, j) in zip(*np.where(walls)):
        x = np.array([j, j+1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i+1, i+1]) / float(height)
        ax.fill_between(x, y0, y1, color='grey')
    
    # plot non-wall unsafe boxes
    #for (i, j) in zip(*np.where(cost_map > cost_limit)):
    #    if walls[i,j] == 1: # skip walls
    #        continue
        # grid points are more accurate than grid boxes
        #x = np.array([j, j+1]) / float(width)
        #y0 = np.array([i, i]) / float(height)
        #y1 = np.array([i+1, i+1]) / float(height)
        #ax.fill_between(x, y0, y1, color='red', alpha=0.5)

    # scattered points are more accurate as they are state-wise estimations
    unsafe_points = np.where(cost_map > cost_limit)
    unsafe_points = np.column_stack(unsafe_points)
    ax.scatter(unsafe_points[:,1]/float(width), unsafe_points[:,0]/float(height), s=2, marker='o', c="red")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    return ax

def plot_trajs(list_trajs, walls:np.ndarray, ax:plt.axes):
    walls = walls.T
    (height, width) = walls.shape

    start_color = "#18aedb"
    end_color = "#dbbb18"
    
    """plot a list of trajs, each is a list of tuples (int states)"""
    for traj in list_trajs:
        # randomize colors
        c = np.random.rand(3,)
        for i in range(0, len(traj) - 1):
            pnt = traj[i]
            pnt_next = traj[i+1]
            x, y = pnt[1]/float(width), pnt[0]/float(height)
            xn, yn = pnt_next[1]/float(width), pnt_next[0]/float(height)

            ax.plot([x, xn], [y, yn], color=c, markersize=4)

            if i == 0:
                ax.plot([x], [y], markersize=8, color=start_color, zorder=5, marker="o", label="start")
            if i == len(traj) - 2:
                ax.plot([xn], [yn], markersize=8, color=end_color, zorder=5, marker="x", label="goal")

def plot_maze_grid_points(walls:np.ndarray, ax: plt.axes):
    walls = walls.T
    (height, width) = walls.shape
    empty_points = np.where(walls == 0)
    empty_points = np.column_stack(empty_points)
    ax.scatter(empty_points[:,1]/float(width), empty_points[:,0]/float(height), s=0.5, marker='o', c="green")
    return ax

class SafePointEnv (PointEnv):
    """
    - ensure start states are always safe. 
    - in each step, a cost is returned along with other info
    - rapidly estimate upper and lower feasible trajectory cost

    NOTE: to allow rapid estimation of trajectory cost, make sure there is zero-cost trajectory between each test start and goal states. Use plot_safe_walls method to visualize the zero step cost map. 
    """
    def __init__(self, 
                walls:str=None, 
                resize_factor:int=1,
                action_noise=1.0, 
                thin=False,
                # cost configs
                cost_f_args:dict={},
                cost_limit:float=0.5,
                verbose:bool = True,
                ):
        t0 = time.time()
        super(SafePointEnv, self).__init__(
            walls,
            resize_factor,
            action_noise,
            thin,)
        if verbose:
            print("[INFO] PointEnv setup: {} s".format(time.time() - t0))
        
        self.resize_factor = resize_factor
        self.thin = thin
        self.wall_name = walls
        self.cost_limit = cost_limit
        
        obstacle_x, obstacle_y = np.where(self._walls == 1)
        self.obstacles = np.stack([obstacle_x, obstacle_y], axis=-1).astype(float) # Nx2
        
        self.cost_f_cfg = cost_f_args
        cost_fn_name = cost_f_args.get('name')
        self.cost_function = None

        t0 = time.time()
        if cost_fn_name == 'cosine':
            from pud.envs.safe_pointenv.cost_functions import cost_from_cosine_distance
            import functools
            self.cost_function = functools.partial(cost_from_cosine_distance, r=self.cost_f_cfg['radius'])

            # NOTE: cost map is computed based on states, not trajectories/accumulated costs 
            self._cost_map = self.build_cost_map()
            
            # safe apsp is deprecated, replaced with cbfs grid policies
            #self._safe_apsp = {}

            # compute an conservative (upper-bound) apsp (step cost_limit = 0), this can be used to quickly estimate the distance between two states, and the cost of trajectory = 0
            #self._safe_apsp["ub"] = self._compute_safe_apsp(self._walls, self._cost_map, cost_limit=0.0)
            # compute an lower-bound apsp, so the accurate accumulated cost fallws between the two
            #self._safe_apsp["lb"] = self._compute_safe_apsp(self._walls, self._cost_map, cost_limit=self.cost_limit)
        self.safe_empty_states = self.gather_safe_empty_states(self.cost_limit)
        self.reset()
        print("[INFO] SafePointEnv setup: {} s".format(time.time() - t0))

    def get_map(self):
        return self._walls
    
    def get_cost_map(self):
        return self._cost_map

    def get_map_width(self):
        return self._width
    
    def get_map_height(self):
        return self._height
    
    def set_cost_limit(self, cost_limit:float):
        self.cost_limit = cost_limit

    def build_cost_map(self):
        (height, width) = self._walls.shape
        cost_map = np.ones([height, width], dtype=float) * np.inf
        
        for i in range(height):
            for j in range(width):
                min_d, _  = self.dist_2_blocks([i,j])
                cost_map[i,j]= self.cost_function(min_d)
        
        # todo: there seems exist a bug below, but not sure where
        # ! NOTE: stop writing fancy but buggy code!!
        # ! seems wrongly flipped/transposed
        # NOTE: the h,w order is (height, width) = self._walls.shape
        #mesh_x, mesh_y = np.meshgrid(np.arange(height), np.arange(width))
        #p_x = mesh_x.ravel()
        #p_y = mesh_y.ravel()
        #pnts = np.column_stack([p_x, p_y])
        #pnts_cost = np.ones((len(p_x),), dtype=float) * np.inf
        #for ii in range(len(pnts)):
        #    pt = pnts[ii]
        #    min_d, _  = self.dist_2_blocks(pt)
        #    pt_cost= self.cost_function(min_d)
        #    pnts_cost[ii] = pt_cost
        #cost_map = pnts_cost.reshape(mesh_x.shape)

        return cost_map
    
    def _compute_safe_apsp(self, walls:np.ndarray, cost_map:np.ndarray, cost_limit:float):
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
                        if cost_map[i,j] > cost_limit:
                            continue
                        if cost_map[i + di,j + dj] > cost_limit:
                            continue
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float32('inf'))
        for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
            for ((i2, j2), d) in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist
    
    def reset(self):
        if (not hasattr(self, "cost_limit")) or (not hasattr(self, "_cost_map")):
            print("[INFO] skipping the reset in PointEnv.__init__ because setup is not ready yet")
            return

        # todo: perhaps suffer from label inbalance?
        self.state = self._sample_safe_empty_state(cost_limit=self.cost_limit)
        new_state_cost = self.get_state_cost(xy=self.state)
        info = {"cost": new_state_cost}
        return self.state.copy(), info
    
    def reset_manual(self, start_state:np.ndarray):
        "manually set the start state"
        self.state = start_state
        new_state_cost = self.get_state_cost(xy=self.state)
        info = {"cost": new_state_cost}
        return self.state.copy(), info

    def gather_safe_empty_states(self, cost_limit:float):
        """
        due to the increased cost in reset, precompile a list of initial states here
        """
        empty_states = np.where(self._walls == 0)
        safe_empty_states = [[],[]]

        for cx, cy in zip(*empty_states):
            # only sample states whose costs are lower than an upper bound
            if self._cost_map[cx, cy] < cost_limit:
            #if self.get_state_cost([cx, cy]) < cost_limit: # more reliable when cost map is buggy
                safe_empty_states[0].append(cx)
                safe_empty_states[1].append(cy)

        safe_empty_states = np.column_stack(safe_empty_states) # N,d
        return safe_empty_states
    
    def _sample_safe_empty_state(self, cost_limit:float):
        """
        must take intersection with the empty states because state cost is computed from the center of the block?
        """
        #if not hasattr(self, "safe_empty_states"):
        #    print("[WARN] safe_empty_states are not pre-generated, compiling safe empty states")
        #    self.safe_empty_states = self.gather_safe_empty_states(cost_limit)

        num_candidate_states = len(self.safe_empty_states)

        idx = np.random.randint(0, num_candidate_states)
        new_state = self.safe_empty_states[idx].astype(np.float32)
        #state += np.random.uniform(size=2)

        # don't remove the checks below
        assert not self._is_blocked(new_state)
        assert self.get_state_cost(new_state) < self.cost_limit
        return new_state
    
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
    
    def _get_safe_distance(self, obs, goal):
        """Compute the shortest path distance.

        Note: This distance is *not* used for training."""
        (i1, j1) = self._discretize_state(obs)
        (i2, j2) = self._discretize_state(goal)
        return {
            "ub": self._safe_apsp["ub"][i1, j1, i2, j2],
            "lb": self._safe_apsp["lb"][i1, j1, i2, j2],
        }
    
    def step(self, action):
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        # NOTE: use the maximum cost along the action segment
        cost = self.get_state_cost(self.state)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                new_cost = self.get_state_cost(new_state)
                if cost < new_cost:
                    cost = new_cost
                if not self._is_blocked(new_state):
                    self.state = new_state

        done = False
        rew = -1.0 * np.linalg.norm(self.state)
        return self.state.copy(), rew, done, {"cost": cost}