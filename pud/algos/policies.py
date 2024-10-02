import time
import scipy
import torch
import logging
import numpy as np
import networkx as nx
from pud.mapf.cbs import CBSSolver
from pud.mapf.cbs_ds import CBSDSSolver


class BasePolicy:
    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.constraints = None if not hasattr(agent, "constraints") else agent.constraints

    def select_action(self, state):
        return self.agent.select_action(state)


class GaussianPolicy(BasePolicy):
    def __init__(self, agent, noise_scale=1.0):
        super().__init__(agent)
        self.noise_scale = noise_scale

    def select_action(self, state):
        action = super().select_action(state)
        action += np.random.normal(
            0, self.agent.max_action * self.noise_scale, size=action.shape
        ).astype(action.dtype)
        action = action.clip(-self.agent.max_action, self.agent.max_action)
        return action

# unfinished and un-needed
#class VectorCategoricalPolicy(BasePolicy):
#    def __init__(self, agent, noise_scale=1.0):
#        super().__init__(agent)
#        self.noise_scale = noise_scale

#    def select_action(self, state):
#        action = super().select_action(state)
#        noise_dist = scipy.stats.truncnorm(
#                        -self.agent.max_action-action, 
#                        self.agent.max_action-action,
#                        loc=np.zeros_like(action),
#                        scale=self.agent.max_action*self.noise_scale)
#        noise = noise_dist.rvs(size=action.shape).astype(action.dtype)
#        action = action + noise
#        action = action.clip(-self.agent.max_action, self.agent.max_action)
#        return action

class VectorGaussianPolicy(BasePolicy):
    def __init__(self, agent, noise_scale=1.0):
        super().__init__(agent)
        self.noise_scale = noise_scale

    def select_action(self, state):
        action = super().select_action(state)
        noise_dist = scipy.stats.truncnorm(
                        -self.agent.max_action-action, 
                        self.agent.max_action-action,
                        loc=np.zeros_like(action),
                        scale=self.agent.max_action*self.noise_scale)
        noise = noise_dist.rvs(size=action.shape).astype(action.dtype)
        
        action = action + noise
        action = action.clip(-self.agent.max_action, self.agent.max_action)
        return action


class SearchPolicy(BasePolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        """
        Args:
            rb_vec: a replay buffer vector storing the observations that will be used as nodes in the graph
            pdist: a matrix of dimension len(rb_vec) x len(rb_vec) where pdist[i,j] gives the distance going from
                   rb_vec[i] to rb_vec[j]
            max_search_steps: (int)
            open_loop: if True, only performs search once at the beginning of the episode
            weighted_path_planning: whether or not to use edge weights when planning a shortest path from start to goal
            no_waypoint_hopping: if True, will not try to proceed to goal until all waypoints have been reached
        """
        super().__init__(agent=agent, **kwargs)
        self.rb_vec = rb_vec
        self.pdist = pdist

        self.aggregate = aggregate
        self.open_loop = open_loop
        self.max_search_steps = max_search_steps
        self.weighted_path_planning = weighted_path_planning

        self.cleanup = False
        self.attempt_cutoff = 3 * max_search_steps
        self.no_waypoint_hopping = no_waypoint_hopping
        self.waypoint_consistency_cutoff = waypoint_consistency_cutoff

        self.build_rb_graph(self.rb_vec)
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(
                self.rb_vec,
                aggregate=self.aggregate,
                max_search_steps=self.max_search_steps,
                masked=True,
            )
            self.rb_distances = scipy.sparse.csgraph.floyd_warshall(
                pdist2, directed=True
            )
        self.reset_stats()

    def __str__(self):
        s = f"{self.__class__.__name__} (|V|={self.g.number_of_nodes()}, |E|={self.g.number_of_edges()})"
        return s

    def reset_stats(self):
        self.stats = dict(
            localization_fails=0,
            path_planning_fails=0,
            graph_search_time=0.0,
            path_planning_attempts=0,
        )

    def get_stats(self):
        return self.stats

    def set_cleanup(
        self, cleanup
    ):  # if True, will prune edges when fail to reach waypoint after `attempt_cutoff`
        self.cleanup = cleanup

    def build_rb_graph(self, rb_vec):
        g = nx.DiGraph()
        assert self.pdist is not None, "Pairwise distances not provided"
        pdist_combined = np.max(self.pdist, axis=0)

        for i, s_i in enumerate(rb_vec):
            for j, s_j in enumerate(rb_vec):
                length = pdist_combined[i, j]
                if length < self.max_search_steps:
                    g.add_edge(i, j, weight=length)
        self.g = g

    def get_pairwise_dist_to_rb(self, state, masked=True):
        start_to_rb_dist = self.agent.get_pairwise_dist(
            [state["observation"]],
            self.rb_vec,
            aggregate=self.aggregate,
            max_search_steps=self.max_search_steps,
            masked=masked,
        )
        rb_to_goal_dist = self.agent.get_pairwise_dist(
            self.rb_vec,
            [state["goal"]],
            aggregate=self.aggregate,
            max_search_steps=self.max_search_steps,
            masked=masked,
        )
        return start_to_rb_dist, rb_to_goal_dist

    def get_closest_waypoint(self, state):
        """
        For closed loop replanning at each step. Uses the precomputed distances
        `rb_distances` b/w states in `rb_vec`
        """
        obs_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        # (B x A), (A x B)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum(
            [
                np.expand_dims(obs_to_rb_dist, 2),
                np.expand_dims(self.rb_distances, 0),
                np.expand_dims(np.transpose(rb_to_goal_dist), 1),
            ]
        )  # elementwise sum

        # We assume a batch size of 1.
        min_search_dist = np.min(search_dist)
        waypoint_index = np.argmin(np.min(search_dist, axis=2), axis=1)[0]
        waypoint = self.rb_vec[waypoint_index]

        return waypoint, waypoint_index, min_search_dist

    def construct_planning_graph(
        self, state, planning_graph=None, start_id="start", goal_id="goal"
    ):
        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        if planning_graph is None:
            planning_graph = self.g.copy()

        for i, (dist_from_start, dist_to_goal) in enumerate(
            zip(start_to_rb_dist.flatten(), rb_to_goal_dist.flatten())
        ):
            if dist_from_start < self.max_search_steps:
                planning_graph.add_edge(start_id, i, weight=dist_from_start)
            if dist_to_goal < self.max_search_steps:
                planning_graph.add_edge(i, goal_id, weight=dist_to_goal)

        if not np.any(start_to_rb_dist < self.max_search_steps) or not np.any(
            rb_to_goal_dist < self.max_search_steps
        ):
            self.stats["localization_fails"] += 1

        return planning_graph

    def get_path(self, state):
        g2 = self.construct_planning_graph(state)
        try:
            self.stats["path_planning_attempts"] += 1
            graph_search_start = time.perf_counter()

            if self.weighted_path_planning:
                path = nx.shortest_path(
                    g2, source="start", target="goal", weight="weight"
                )
            else:
                path = nx.shortest_path(g2, source="start", target="goal")
        except Exception as e:
            self.stats["path_planning_fails"] += 1
            raise RuntimeError(
                f"Failed to find path in graph (|V|={g2.number_of_nodes()}, |E|={g2.number_of_edges()})"
            ) from e
        finally:
            graph_search_end = time.perf_counter()
            self.stats["graph_search_time"] += graph_search_end - graph_search_start

        edge_lengths = []
        for i, j in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]["weight"])

        waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        waypoint_indices = list(path)[1:-1]
        return waypoint_indices, waypoint_to_goal_dist[1:]

    def initialize_path(self, state):
        self.waypoint_indices, self.waypoint_to_goal_dist_vec = self.get_path(state)
        self.waypoint_counter = 0
        self.waypoint_attempts = 0
        self.reached_final_waypoint = False

    def get_current_waypoint(self):
        waypoint_index = self.waypoint_indices[self.waypoint_counter]
        waypoint = self.rb_vec[waypoint_index]
        return waypoint, waypoint_index

    def get_waypoints(self):
        waypoints = [self.rb_vec[i] for i in self.waypoint_indices]
        return waypoints

    def reached_waypoint(self, dist_to_waypoint, state, waypoint_index):
        # return dist_to_waypoint < self.max_search_steps
        return dist_to_waypoint < 2.0

    def select_action(self, state):
        goal = state["goal"]
        dist_to_goal = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]

        if self.open_loop or self.cleanup:
            if state.get("first_step", False):
                self.initialize_path(state)

            if self.cleanup and (self.waypoint_attempts >= self.attempt_cutoff):
                # Prune edge and replan
                if self.waypoint_counter != 0 and not self.reached_final_waypoint:
                    src_node = self.waypoint_indices[self.waypoint_counter - 1]
                    dest_node = self.waypoint_indices[self.waypoint_counter]
                    self.g.remove_edge(src_node, dest_node)
                self.initialize_path(state)

            waypoint, waypoint_index = self.get_current_waypoint()
            state["goal"] = waypoint
            dist_to_waypoint = self.agent.get_dist_to_goal(
                {k: [v] for k, v in state.items()}
            )[0]

            if self.reached_waypoint(dist_to_waypoint, state, waypoint_index):
                if not self.reached_final_waypoint:
                    self.waypoint_attempts = 0

                self.waypoint_counter += 1
                if self.waypoint_counter >= len(self.waypoint_indices):
                    self.reached_final_waypoint = True
                    self.waypoint_counter = len(self.waypoint_indices) - 1

                waypoint, waypoint_index = self.get_current_waypoint()
                state["goal"] = waypoint
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state.items()}
                )[0]

            dist_to_goal_via_waypoint = (
                dist_to_waypoint + self.waypoint_to_goal_dist_vec[self.waypoint_counter]
            )
        else:
            # Closed loop, replan waypoint at each step
            waypoint, waypoint_index, dist_to_goal_via_waypoint = (
                self.get_closest_waypoint(state)
            )

        if (
            (self.no_waypoint_hopping and not self.reached_final_waypoint)
            or (dist_to_goal_via_waypoint < dist_to_goal)
            or (dist_to_goal > self.max_search_steps)
        ):
            state["goal"] = waypoint
            if self.open_loop:
                self.waypoint_attempts += 1
        else:
            state["goal"] = goal
        return super().select_action(state)


class ConstrainedSearchPolicy(SearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        pcost=None,
        ckpts=None,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        dist_aggregate="min",
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        """
        Args:
            pcost: a matrix of dimension len(rb_vec) x len(rb_vec) where pcost[i,j] gives the cost of going from
                   rb_vec[i] to rb_vec[j]
            max_cost_limit: (int)
            cost_aggregate: (str) aggregation function to use when computing cost from ensembles
        """
        self.pcost = pcost
        self.cost_aggregate = cost_aggregate
        self.max_cost_limit = max_cost_limit

        assert ckpts is not None, "Checkpoints for constrained and unconstrained models must be provided"
        self.ckpts = ckpts

        assert hasattr(agent, "constraints") and agent.constraints is not None, "Agent must have constraints"
        self.constraints = agent.constraints

        super().__init__(
            agent=agent,
            pdist=pdist,
            rb_vec=rb_vec,
            open_loop=open_loop,
            aggregate=dist_aggregate,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )

    def build_rb_graph(self, rb_vec):
        g = nx.DiGraph()
        assert self.pdist is not None, "Pairwise distances not provided"
        assert self.pcost is not None, "Pairwise costs not provided"
        pdist_combined = np.max(self.pdist, axis=0)
        pcost_combined = np.max(self.pcost, axis=0)

        for i, s_i in enumerate(rb_vec):
            for j, s_j in enumerate(rb_vec):
                cost = pcost_combined[i, j]
                length = pdist_combined[i, j]
                if length < self.max_search_steps and cost < self.max_cost_limit:
                    g.add_edge(i, j, weight=length)
        self.g = g

    def get_pairwise_cost_to_rb(self, state, masked=True):
        self.agent.load_state_dict(torch.load(self.ckpts["unconstrained"], map_location="cuda:0"))
        start_to_rb_cost = self.agent.get_pairwise_cost(
            [state["observation"]],
            self.rb_vec,
            aggregate=self.cost_aggregate,
        )
        rb_to_goal_cost = self.agent.get_pairwise_cost(
            self.rb_vec,
            [state["goal"]],
            aggregate=self.cost_aggregate,
        )
        self.agent.load_state_dict(torch.load(self.ckpts["constrained"], map_location="cuda:0"))
        return start_to_rb_cost, rb_to_goal_cost

    def get_closest_waypoint(self, state):
        """
        For closed loop replanning at each step. Uses the precomputed distances
        `rb_distances` b/w states in `rb_vec`
        """
        obs_to_rb_cost, _ = self.get_pairwise_cost_to_rb(state)
        obs_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        # (B x A), (A x B)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum(
            [
                np.expand_dims(obs_to_rb_dist, 2),
                np.expand_dims(self.rb_distances, 0),
                np.expand_dims(np.transpose(rb_to_goal_dist), 1),
            ]
        )  # elementwise sum

        # We assume a batch size of 1.
        not_safe = True
        while not_safe:
            min_search_dist = np.min(search_dist)
            waypoint_index = np.argmin(np.min(search_dist, axis=2), axis=1)[0]
            waypoint = self.rb_vec[waypoint_index]

            if obs_to_rb_cost[0, waypoint_index] < self.max_cost_limit:
                not_safe = False

        return waypoint, waypoint_index, min_search_dist

    def construct_planning_graph(
        self, state, planning_graph=None, start_id="start", goal_id="goal"
    ):
        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        start_to_rb_cost, rb_to_goal_cost = self.get_pairwise_cost_to_rb(state)
        if planning_graph is None:
            planning_graph = self.g.copy()

        for i, (from_start, to_goal) in enumerate(
            zip(
                zip(start_to_rb_dist.flatten(), start_to_rb_cost.flatten()),
                zip(rb_to_goal_dist.flatten(), rb_to_goal_cost.flatten())
            )
        ):
            dist_from_start, cost_from_start = from_start
            dist_to_goal, cost_to_goal = to_goal
            if dist_from_start < self.max_search_steps and cost_from_start < self.max_cost_limit:
                planning_graph.add_edge(start_id, i, weight=dist_from_start)
            if dist_to_goal < self.max_search_steps and cost_to_goal < self.max_cost_limit:
                planning_graph.add_edge(i, goal_id, weight=dist_to_goal)

        if (
            not np.any(start_to_rb_dist < self.max_search_steps)
            or not np.any(rb_to_goal_dist < self.max_search_steps)
            or not np.any(start_to_rb_cost < self.max_cost_limit)
            or not np.any(rb_to_goal_cost < self.max_cost_limit)
        ):
            self.stats["localization_fails"] += 1

        return planning_graph


class VisualSearchPolicy(SearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            aggregate=aggregate,
            open_loop=open_loop,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )

        assert isinstance(rb_vec, tuple), "rb_vec must be a tuple of (grid, vec)"
        self.rb_vec_grid, self.rb_vec = rb_vec

        self.build_rb_graph(self.rb_vec_grid)
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(
                self.rb_vec,
                aggregate=self.aggregate,
                max_search_steps=self.max_search_steps,
                masked=True,
            )
            self.rb_distances = scipy.sparse.csgraph.floyd_warshall(
                pdist2, directed=True
            )

    def get_waypoints(self):
        waypoints = [(self.rb_vec_grid[i], self.rb_vec[i]) for i in self.waypoint_indices]
        return waypoints

    def select_action(self, state):
        goal = state["goal"]
        grid_goal = state["grid"]["goal"]
        dist_to_goal = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]

        if self.open_loop or self.cleanup:
            if state.get("first_step", False):
                self.initialize_path(state)

            if self.cleanup and (self.waypoint_attempts >= self.attempt_cutoff):
                # Prune edge and replan
                if self.waypoint_counter != 0 and not self.reached_final_waypoint:
                    src_node = self.waypoint_indices[self.waypoint_counter - 1]
                    dest_node = self.waypoint_indices[self.waypoint_counter]
                    self.g.remove_edge(src_node, dest_node)
                self.initialize_path(state)

            waypoint, waypoint_index = self.get_current_waypoint()
            state["goal"] = waypoint
            state["grid"]["goal"] = self.rb_vec_grid[waypoint_index]
            dist_to_waypoint = self.agent.get_dist_to_goal(
                {k: [v] for k, v in state.items()}
            )[0]

            if self.reached_waypoint(dist_to_waypoint, state, waypoint_index):
                if not self.reached_final_waypoint:
                    self.waypoint_attempts = 0

                self.waypoint_counter += 1
                if self.waypoint_counter >= len(self.waypoint_indices):
                    self.reached_final_waypoint = True
                    self.waypoint_counter = len(self.waypoint_indices) - 1

                waypoint, waypoint_index = self.get_current_waypoint()
                state["goal"] = waypoint
                state["grid"]["goal"] = self.rb_vec_grid[waypoint_index]
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state.items()}
                )[0]

            dist_to_goal_via_waypoint = (
                dist_to_waypoint + self.waypoint_to_goal_dist_vec[self.waypoint_counter]
            )
        else:
            # Closed loop, replan waypoint at each step
            waypoint, waypoint_index, dist_to_goal_via_waypoint = (
                self.get_closest_waypoint(state)
            )

        if (
            (self.no_waypoint_hopping and not self.reached_final_waypoint)
            or (dist_to_goal_via_waypoint < dist_to_goal)
            or (dist_to_goal > self.max_search_steps)
        ):
            state["goal"] = waypoint
            state["grid"]["goal"] = self.rb_vec_grid[waypoint_index]
            if self.open_loop:
                self.waypoint_attempts += 1
        else:
            state["goal"] = goal
            state["grid"]["goal"] = grid_goal
        return super(SearchPolicy, self).select_action(state)


class VisualConstrainedSearchPolicy(ConstrainedSearchPolicy, VisualSearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        pcost=None,
        ckpts=None,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        dist_aggregate="min",
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            pdist=pdist,
            pcost=pcost,
            ckpts=ckpts,
            rb_vec=rb_vec,
            open_loop=open_loop,
            dist_aggregate=dist_aggregate,
            cost_aggregate=cost_aggregate,
            max_cost_limit=max_cost_limit,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )


class MultiAgentSearchPolicy(SearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        pdist=None,
        radius=0.1,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        disjoint_split=False,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            aggregate=aggregate,
            open_loop=open_loop,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )
        self.radius = radius
        self.n_agents = n_agents
        self.disjoint_split = disjoint_split

    def get_closest_waypoints(self, state):

        assert "composite_goals" in state.keys(), "Composite goals not present in state"
        assert len(state["composite_goals"]) == self.n_agents, "Number of composite goals not equal to number of agents"

        augmented_waypoints = []
        augmented_waypoint_indices = []
        augmented_min_search_dists = []

        for agent_id in range(self.n_agents):

            state_copy = state.copy()
            state_copy["goal"] = state["composite_goals"][agent_id]
            state_copy["observation"] = state["agent_observations"][agent_id]

            waypoint, waypoint_index, min_search_dist = self.get_closest_waypoint(state_copy)

            augmented_waypoint_indices.append(waypoint_index)

            if waypoint in augmented_waypoints:
                augmented_min_search_dists.append(0)
                augmented_waypoints.append(state["agent_waypoints"][agent_id])
            else:
                augmented_waypoints.append(waypoint)
                augmented_min_search_dists.append(min_search_dist)
                state["agent_waypoints"][agent_id] = waypoint
                state["agent_waypoints_visual"][agent_id] = waypoint
        return (
            augmented_waypoints,
            augmented_waypoint_indices,
            augmented_min_search_dists,
        )

    def construct_augmented_planning_graph(self, starts, goals):
        planning_graph = self.g.copy()
        logging.debug("Initial graph size = ", self.g.number_of_nodes())

        num_nodes = self.rb_vec.shape[0] - 1

        nodes_to_agent_maps = {}
        for agent_id, (start, goal) in enumerate(zip(starts, goals)):

            start_id = num_nodes + 1
            nodes_to_agent_maps["start" + str(agent_id)] = start_id

            goal_id = num_nodes + 2
            nodes_to_agent_maps["goal" + str(agent_id)] = goal_id

            planning_graph = self.construct_planning_graph(
                {"observation": start, "goal": goal}, planning_graph, start_id, goal_id  # type: ignore
            )

            assert planning_graph.has_node(start_id), "Start node not added to graph"
            assert planning_graph.has_node(goal_id), "Goal node not added to graph"
            num_nodes += 2

        logging.debug("Final graph size = {}".format(planning_graph.number_of_nodes()))
        return planning_graph, nodes_to_agent_maps

    def initialize_paths(self, rb_vec, starts, goals):
        graph, nodes_to_agents_maps = self.construct_augmented_planning_graph(starts, goals)

        goal_ids = [nodes_to_agents_maps["goal" + str(agent_id)] for agent_id in range(self.n_agents)]
        start_ids = [nodes_to_agents_maps["start" + str(agent_id)] for agent_id in range(self.n_agents)]

        augmented_wps = rb_vec.copy()
        for _, (start, goal) in enumerate(zip(starts, goals)):
            augmented_wps = np.vstack([augmented_wps, start, goal])

        if not self.disjoint_split:
            cbs_class = CBSSolver
        else:
            cbs_class = CBSDSSolver

        cbs_solver = cbs_class(
            graph,
            augmented_wps,
            start_ids,
            goal_ids,
            weighted=self.weighted_path_planning,
            collision_radius=self.radius,
        )
        solution = cbs_solver.find_paths()
        paths = solution["paths"]  # type: ignore

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print("Cost of solution: {}".format(solution["cost"]))  # type: ignore
            print("Number of expanded nodes: {}".format(cbs_solver.num_expanded))
            print("Number of generated nodes: {}".format(cbs_solver.num_generated))
            print("Printing the paths")
            for agent_id, path in enumerate(paths):
                print("--" * 10)
                print("Path for agent ", agent_id)
                for vertex in path:
                    if vertex in start_ids or vertex in goal_ids:
                        print("Start: ", starts[agent_id]) if vertex in start_ids else print("Goal: ", goals[agent_id])
                    else:
                        print("Vertex: ", rb_vec[vertex])

        self.augmented_waypoint_indices, self.augmented_waypoint_to_goal_dist_vec = [], []
        for path in paths:
            edge_lengths = []
            for i, j in zip(path[:-1], path[1:]):
                edge_lengths.append(graph[i][j]["weight"])
            waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]
            waypoint_indices = list(path)[1:-1]
            self.augmented_waypoint_indices.append(waypoint_indices)
            self.augmented_waypoint_to_goal_dist_vec.append(waypoint_to_goal_dist[1:])

        self.augmented_waypoint_stays = np.zeros(self.n_agents, dtype=bool)
        self.augmented_waypoint_counters = np.zeros(self.n_agents, dtype=int)
        self.augmented_waypoint_attempts = np.zeros(self.n_agents, dtype=int)
        self.augmented_reached_final_waypoints = np.zeros(self.n_agents, dtype=bool)

        self.goals = goals
        self.starts = starts
        self.goal_ids = goal_ids
        self.start_ids = start_ids
        self.nodes_to_agents_maps = nodes_to_agents_maps

    def get_current_waypoints(self):
        augmented_waypoints = []
        augmented_wp_indices = []

        for agent_id in range(self.n_agents):
            waypoint_index = self.augmented_waypoint_indices[agent_id][self.augmented_waypoint_counters[agent_id]]
            if waypoint_index > self.rb_vec.shape[0]:
                waypoint = self.starts[agent_id] if waypoint_index == self.start_ids[agent_id] else self.goals[agent_id]
            else:
                waypoint = self.rb_vec[waypoint_index]

            if agent_id > 0:
                waypoint_qvals = self.agent.get_pairwise_dist([waypoint], augmented_waypoints, aggregate=None)
                if np.any(waypoint_qvals < self.waypoint_consistency_cutoff):
                    waypoint_index = self.augmented_waypoint_indices[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                    waypoint = self.rb_vec[waypoint_index]

            augmented_waypoints.append(waypoint)
            augmented_wp_indices.append(waypoint_index)
        return augmented_waypoints, augmented_wp_indices

    def get_augmented_waypoints(self):
        augmented_waypoints = []
        for agent_id in range(self.n_agents):
            augmented_waypoints.append([self.rb_vec[j] for j in self.augmented_waypoint_indices[agent_id]])
        return augmented_waypoints

    def select_action(self, state):
        assert "composite_goals" in state.keys(), "Composite goals not present in state"
        assert len(state["composite_goals"]) == self.n_agents, "Number of composite goals not equal to number of agents"

        # Composite start and goals are the grid representation of the state
        composite_goals = state["composite_goals"]
        dist_to_composite_goals = []
        for agent_id in range(self.n_agents):
            c_goal = composite_goals[agent_id]
            state_copy = state.copy()
            state_copy["goal"] = c_goal
            state_copy["observation"] = state["agent_observations"][agent_id]
            dist_to_composite_goals.append(self.agent.get_dist_to_goal({k: [v] for k, v in state_copy.items()})[0])

        if self.open_loop or self.cleanup:

            if state.get("first_step", False):

                self.initialize_paths(self.rb_vec, state["composite_starts"], state["composite_goals"])

            if self.cleanup:
                for idx, (waypoint_counter, waypoint_attempts, reached_final_waypoint) in enumerate(
                    zip(
                        self.augmented_waypoint_counters,
                        self.augmented_waypoint_attempts,
                        self.augmented_reached_final_waypoints)
                ):
                    if (
                        waypoint_attempts >= self.attempt_cutoff
                        and waypoint_counter != 0
                        and not reached_final_waypoint
                    ):
                        src_node = self.augmented_waypoint_indices[idx][waypoint_counter - 1]
                        dest_node = self.augmented_waypoint_indices[idx][waypoint_counter]
                        self.g.remove_edge(src_node, dest_node)

                self.initialize_paths(self.rb_vec, state["composite_starts"], state["composite_goals"])

            waypoints, waypoint_indices = self.get_current_waypoints()

            dist_to_goal_via_waypoints = []
            for agent_id in range(self.n_agents):
                state_copy = state.copy()
                state_copy["goal"] = waypoints[agent_id]
                state_copy["observation"] = state["agent_observations"][agent_id]
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state_copy.items()}
                )[0]

                if self.reached_waypoint(
                    dist_to_waypoint, state_copy, waypoint_indices[agent_id]
                ):
                    if not self.augmented_reached_final_waypoints[agent_id]:
                        self.augmented_waypoint_attempts[agent_id] = 0

                    self.augmented_waypoint_counters[agent_id] += 1
                    if self.augmented_waypoint_counters[agent_id] >= len(
                        self.augmented_waypoint_indices[agent_id]
                    ):
                        self.augmented_reached_final_waypoints[agent_id] = True
                        self.augmented_waypoint_counters[agent_id] = len(self.augmented_waypoint_indices[agent_id]) - 1

                    waypoints, waypoint_indices = self.get_current_waypoints()
                    state_copy["goal"] = waypoints[agent_id]
                    dist_to_waypoint = self.agent.get_dist_to_goal(
                        {k: [v] for k, v in state_copy.items()}
                    )[0]

                dist_to_goal_via_waypoint = (
                    dist_to_waypoint
                    + self.augmented_waypoint_to_goal_dist_vec[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                )
                dist_to_goal_via_waypoints.append(dist_to_goal_via_waypoint)
        else:
            # Closed loop, replan waypoint at each step
            waypoints, waypoint_indices, dist_to_goal_via_waypoints = (
                self.get_closest_waypoints(state)
            )

        # These variables are used by the "get_trajectories" function to update agent's goals with intermediate
        # waypoints
        agent_goals = []
        agent_actions = []
        for agent_id in range(self.n_agents):
            state_copy = state.copy()
            if (
                (
                    self.no_waypoint_hopping
                    and not self.augmented_reached_final_waypoints[agent_id]
                )
                or (dist_to_goal_via_waypoints[agent_id] < dist_to_composite_goals[agent_id])
                or (dist_to_composite_goals[agent_id] > self.max_search_steps)
            ):
                state_copy["goal"] = waypoints[agent_id]
                if self.open_loop:
                    self.augmented_waypoint_attempts[agent_id] += 1
            else:
                state_copy["goal"] = composite_goals[agent_id]

            agent_goals.append(state_copy["goal"])
            state_copy["observation"] = state["agent_observations"][agent_id]

            agent_action = super(SearchPolicy, self).select_action(state_copy)
            agent_actions.append(agent_action)

        return agent_actions, agent_goals


class ConstrainedMultiAgentSearchPolicy(ConstrainedSearchPolicy, MultiAgentSearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        ckpts=None,
        pdist=None,
        pcost=None,
        radius=0.1,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        dist_aggregate="min",
        disjoint_split=False,
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            pcost=pcost,
            ckpts=ckpts,
            radius=radius,
            n_agents=n_agents,
            open_loop=open_loop,
            disjoint_split=disjoint_split,
            dist_aggregate=dist_aggregate,
            cost_aggregate=cost_aggregate,
            max_cost_limit=max_cost_limit,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )


class VisualMultiAgentSearchPolicy(MultiAgentSearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        pdist=None,
        radius=0.1,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        disjoint_split=False,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            pdist=pdist,
            agent=agent,
            rb_vec=rb_vec,
            radius=radius,
            n_agents=n_agents,
            aggregate=aggregate,
            open_loop=open_loop,
            disjoint_split=disjoint_split,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )

        assert isinstance(rb_vec, tuple), "rb_vec should be a tuple"
        self.rb_vec_grid, self.rb_vec = rb_vec

        self.build_rb_graph(self.rb_vec_grid)
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(
                self.rb_vec,
                aggregate=self.aggregate,
                max_search_steps=self.max_search_steps,
                masked=True,
            )
            self.rb_distances = scipy.sparse.csgraph.floyd_warshall(
                pdist2, directed=True
            )

    def get_closest_waypoints(self, state):

        augmented_waypoints = []
        augmented_waypoint_indices = []
        augmented_visual_waypoints = []
        augmented_min_search_dists = []

        for agent_id in range(self.n_agents):

            state_copy = state.copy()

            state_copy["goal"] = state["composite_goals"][agent_id][1]
            state_copy["grid"]["goal"] = state["composite_goals"][agent_id][0]
            state_copy["observation"] = state["agent_observations_visual"][agent_id][1]
            state_copy["grid"]["observation"] = state["agent_observations"][agent_id][0]

            waypoint, waypoint_index, min_search_dist = self.get_closest_waypoint(state_copy)
            waypoint_grid = self.rb_vec_grid[waypoint_index]

            augmented_waypoint_indices.append(waypoint_index)
            if waypoint_grid in augmented_waypoints:
                augmented_min_search_dists.append(0)
                augmented_waypoints.append(state["agent_waypoints"][agent_id][0])
                augmented_visual_waypoints.append(state["agent_waypoints"][agent_id][1])
            else:
                augmented_waypoints.append(waypoint_grid)
                augmented_visual_waypoints.append(waypoint)
                augmented_min_search_dists.append(min_search_dist)
                state["agent_waypoints"][agent_id] = (waypoint_grid, waypoint)

        return (
            (
                augmented_waypoints,
                augmented_visual_waypoints,
            ),
            augmented_waypoint_indices,
            augmented_min_search_dists,
        )

    def initialize_paths(self, rb_vec, starts, goals):

        goals_grid = [goal[0] for goal in goals]
        goals = [goal[1] for goal in goals]
        starts_grid = [start[0] for start in starts]
        starts = [start[1] for start in starts]
        graph, nodes_to_agents_maps = self.construct_augmented_planning_graph(starts, goals)

        goal_ids = [nodes_to_agents_maps["goal" + str(agent_id)] for agent_id in range(self.n_agents)]
        start_ids = [nodes_to_agents_maps["start" + str(agent_id)] for agent_id in range(self.n_agents)]

        augmented_wps = rb_vec.copy()
        for _, (start, goal) in enumerate(zip(starts_grid, goals_grid)):
            augmented_wps = np.vstack([augmented_wps, start, goal])

        if not self.disjoint_split:
            cbs_class = CBSSolver
        else:
            cbs_class = CBSDSSolver

        cbs_solver = cbs_class(
            graph,
            augmented_wps,
            start_ids,
            goal_ids,
            weighted=self.weighted_path_planning,
            collision_radius=self.radius,
        )
        solution = cbs_solver.find_paths()
        paths = solution["paths"]  # type: ignore

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print("Cost of solution: {}".format(solution["cost"]))  # type: ignore
            print("Number of expanded nodes: {}".format(cbs_solver.num_expanded))
            print("Number of generated nodes: {}".format(cbs_solver.num_generated))
            print("Printing the paths")
            for a_id, path in enumerate(paths):
                print("--" * 10)
                print("Path for agent ", a_id)
                for vertex in path:
                    if vertex in start_ids or vertex in goal_ids:
                        if vertex in start_ids:
                            print("Start: ", starts_grid[a_id])
                        else:
                            print("Goal: ", goals_grid[a_id])
                    else:
                        print("Vertex: ", rb_vec[vertex])

        self.augmented_waypoint_indices, self.augmented_waypoint_to_goal_dist_vec = [], []

        for path in paths:
            edge_lengths = []
            for i, j in zip(path[:-1], path[1:]):
                edge_lengths.append(graph[i][j]["weight"])
            waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]
            waypoint_indices = list(path)[1:-1]
            self.augmented_waypoint_indices.append(waypoint_indices)
            self.augmented_waypoint_to_goal_dist_vec.append(waypoint_to_goal_dist[1:])

        self.augmented_waypoint_stays = np.zeros(len(starts), dtype=bool)
        self.augmented_waypoint_counters = np.zeros(len(starts), dtype=int)
        self.augmented_waypoint_attempts = np.zeros(len(starts), dtype=int)
        self.augmented_reached_final_waypoints = np.zeros(len(starts), dtype=bool)

        self.goals = goals
        self.starts = starts
        self.goal_ids = goal_ids
        self.start_ids = start_ids
        self.nodes_to_agents_maps = nodes_to_agents_maps

    def get_current_waypoints(self):

        augmented_waypoints = []
        augmented_wp_indices = []
        augmented_grid_waypoints = []

        for agent_id in range(self.n_agents):
            waypoint_index = self.augmented_waypoint_indices[agent_id][self.augmented_waypoint_counters[agent_id]]
            if waypoint_index > self.rb_vec.shape[0]:
                waypoint = self.starts[agent_id] if waypoint_index == self.start_ids[agent_id] else self.goals[agent_id]
            else:
                waypoint = self.rb_vec[waypoint_index]

            if agent_id > 0:
                waypoint_qvals = self.agent.get_pairwise_dist(
                    [waypoint], augmented_waypoints, aggregate=None
                )
                if np.any(waypoint_qvals < self.waypoint_consistency_cutoff):
                    waypoint_index = self.augmented_waypoint_indices[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                    waypoint = self.rb_vec[waypoint_index]

            augmented_waypoints.append(waypoint)
            augmented_wp_indices.append(waypoint_index)
            augmented_grid_waypoints.append(self.rb_vec_grid[waypoint_index])
        return (augmented_grid_waypoints, augmented_waypoints), augmented_wp_indices

    def get_augmented_waypoints(self):
        augmented_waypoints = []
        for agent_id in range(self.n_agents):
            augmented_waypoints.append(
                [(self.rb_vec_grid[j], self.rb_vec[j]) for j in self.augmented_waypoint_indices[agent_id]]
            )
        return augmented_waypoints

    def select_action(self, state):
        assert "composite_goals" in state.keys(), "Composite goals not present in state"
        assert len(state["composite_goals"]) == self.n_agents, "Number of composite goals not equal to number of agents"

        # Composite start and goals are the grid representation of the state
        composite_goals = state["composite_goals"]
        composite_goals_grid = [goal[0] for goal in composite_goals]
        composite_goals = [goal[1] for goal in composite_goals]

        composite_starts_grid = [start[0] for start in state["composite_starts"]]
        dist_to_composite_goals = []
        for agent_id in range(self.n_agents):

            state_copy = state.copy()
            state_copy["goal"] = composite_goals[agent_id]
            state_copy["grid"]["goal"] = composite_goals_grid[agent_id]
            state_copy["observation"] = state["agent_observations"][agent_id][1]
            state_copy["grid"]["observation"] = state["agent_observations"][agent_id][0]
            dist_to_composite_goals.append(self.agent.get_dist_to_goal({k: [v] for k, v in state_copy.items()})[0])

        if self.open_loop or self.cleanup:

            if state.get("first_step", False):

                logging.debug("Composite starts: ", composite_starts_grid)
                logging.debug("Composite goals: ", composite_goals_grid)
                self.initialize_paths(self.rb_vec_grid, state["composite_starts"], state["composite_goals"])

            if self.cleanup:
                for idx, (waypoint_counter, waypoint_attempts, reached_final_waypoint) in enumerate(
                    zip(
                        self.augmented_waypoint_counters,
                        self.augmented_waypoint_attempts,
                        self.augmented_reached_final_waypoints,
                    )
                ):
                    if (
                        waypoint_attempts >= self.attempt_cutoff
                        and waypoint_counter != 0
                        and not reached_final_waypoint
                    ):
                        src_node = self.augmented_waypoint_indices[idx][
                            waypoint_counter - 1
                        ]
                        dest_node = self.augmented_waypoint_indices[idx][
                            waypoint_counter
                        ]
                        self.g.remove_edge(src_node, dest_node)

                self.initialize_paths(self.rb_vec_grid, state["composite_starts"], state["composite_goals"])

            waypoints, waypoint_indices = self.get_current_waypoints()
            waypoints_grid, waypoints = waypoints

            dist_to_goal_via_waypoints = []
            for agent_id in range(self.n_agents):
                state_copy = state.copy()

                state_copy["goal"] = waypoints[agent_id]
                state_copy["grid"]["goal"] = waypoints_grid[agent_id]
                state_copy["observation"] = state["agent_observations"][agent_id][1]
                state_copy["grid"]["observation"] = state["agent_observations"][agent_id][0]
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state_copy.items()}
                )[0]

                if self.reached_waypoint(
                    dist_to_waypoint, state_copy, waypoint_indices[agent_id]
                ):
                    if not self.augmented_reached_final_waypoints[agent_id]:
                        self.augmented_waypoint_attempts[agent_id] = 0

                    self.augmented_waypoint_counters[agent_id] += 1
                    if self.augmented_waypoint_counters[agent_id] >= len(
                        self.augmented_waypoint_indices[agent_id]
                    ):
                        self.augmented_reached_final_waypoints[agent_id] = True
                        self.augmented_waypoint_counters[agent_id] = (
                            len(self.augmented_waypoint_indices[agent_id]) - 1
                        )

                    waypoints, waypoint_indices = self.get_current_waypoints()
                    waypoints_grid, waypoints = waypoints

                    state_copy["goal"] = waypoints[agent_id]
                    state_copy["grid"]["goal"] = waypoints_grid[agent_id]
                    dist_to_waypoint = self.agent.get_dist_to_goal(
                        {k: [v] for k, v in state_copy.items()}
                    )[0]

                dist_to_goal_via_waypoint = (
                    dist_to_waypoint
                    + self.augmented_waypoint_to_goal_dist_vec[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                )
                dist_to_goal_via_waypoints.append(dist_to_goal_via_waypoint)
        else:
            # Closed loop, replan waypoint at each step
            waypoints, waypoint_indices, dist_to_goal_via_waypoints = (
                self.get_closest_waypoints(state)
            )
            waypoints_grid, waypoints = waypoints

        # These variables are used by the "get_trajectories" function to update agent's goals with intermediate
        # waypoints
        agent_goals = []
        agent_actions = []
        for agent_id in range(self.n_agents):
            state_copy = state.copy()
            if (
                (
                    self.no_waypoint_hopping
                    and not self.augmented_reached_final_waypoints[agent_id]
                )
                or (dist_to_goal_via_waypoints[agent_id] < dist_to_composite_goals[agent_id])
                or (dist_to_composite_goals[agent_id] > self.max_search_steps)
            ):
                state_copy["goal"] = waypoints[agent_id]
                state_copy["grid"]["goal"] = waypoints_grid[agent_id]
                if self.open_loop:
                    self.augmented_waypoint_attempts[agent_id] += 1
            else:
                state_copy["goal"] = composite_goals[agent_id]
                state_copy["grid"]["goal"] = composite_goals_grid[agent_id]

            agent_goals.append((state_copy["grid"]["goal"], state_copy["goal"]))
            state_copy["observation"] = state["agent_observations"][agent_id][1]
            state_copy["grid"]["observation"] = state["agent_observations"][agent_id][0]

            agent_action = super(SearchPolicy, self).select_action(state_copy)
            agent_actions.append(agent_action)

        return agent_actions, agent_goals


class VisualConstrainedMultiAgentSearchPolicy(ConstrainedSearchPolicy, VisualMultiAgentSearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        pdist=None,
        pcost=None,
        ckpts=None,
        radius=0.1,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        disjoint_split=False,
        dist_aggregate="min",
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            pcost=pcost,
            ckpts=ckpts,
            radius=radius,
            n_agents=n_agents,
            open_loop=open_loop,
            disjoint_split=disjoint_split,
            dist_aggregate=dist_aggregate,
            cost_aggregate=cost_aggregate,
            max_cost_limit=max_cost_limit,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )


class SparseSearchPolicy(SearchPolicy):

    def __init__(
        self,
        *args,
        pdist=None,
        cache_pdist=True,
        beta=0.05,
        edge_cutoff=10,
        norm_cutoff=0.05,
        consistency_cutoff=5,
        waypoint_consistency_cutoff=1.5,
        k_nearest=5,
        localize_to_nearest=True,
        open_loop=True,
        no_waypoint_hopping=True,
        **kwargs,
    ):
        """
        Note: If beta!=0 and cache_pdist=False, then dynamic construction of the
        state graph will use updated embeddings to compute distances.
        Set cache_pdist=True to use the distances of the original observations.

        Args:
            cache_pdist: if True, then uses `pdist` when building graph,
                         otherwise, compute distances with new embeddings
            beta: percentage assigned to newest embedding space observation
                  in exponential moving average / variance calculations
            edge_cutoff: draw directed edges between nodes when their qvalue
                         distance is less than edge_cutoff
            norm_cutoff: define neighbors if their embedding
                         norm distance is less than norm_cutoff
            consistency_cutoff: qval consistency cutoff
            waypoint_consistency_cutoff: waypoint qval consistency cutoff
            k_nearest: for filtering the nearest k nodes using edge weight
            localize_to_nearest: if True, will incrementally add edges with
                                 incoming start and goal nodes until path
                                 exists from start to goal; otherwise, adds
                                 all edges with incoming start and goal nodes
                                 that have distance less than `max_search_steps`
        """
        self.cache_pdist = cache_pdist
        self.beta = beta
        self.edge_cutoff = edge_cutoff
        self.norm_cutoff = norm_cutoff
        self.consistency_cutoff = consistency_cutoff
        self.waypoint_consistency_cutoff = waypoint_consistency_cutoff
        self.k_nearest = k_nearest
        self.localize_to_nearest = localize_to_nearest
        super().__init__(
            *args,
            pdist=pdist,
            open_loop=open_loop,
            no_waypoint_hopping=no_waypoint_hopping,
            **kwargs,
        )

    def filter_keep_k_nearest(self):
        """
        For each node in the graph, keeps only the k outgoing edges with lowest weight.
        """
        for node in self.g.nodes():
            edges = list(self.g.edges(nbunch=node, data="weight", default=np.inf))  # type: ignore
            edges.sort(key=lambda x: x[2])
            try:
                edges_to_remove = edges[self.k_nearest :]
            except IndexError:
                edges_to_remove = []
            self.g.remove_edges_from(edges_to_remove)

    def construct_planning_graph(self, state):
        if not self.localize_to_nearest:
            return super().construct_planning_graph(state)

        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        start_to_rb_dist, rb_to_goal_dist = (
            start_to_rb_dist.flatten(),
            rb_to_goal_dist.flatten(),
        )
        planning_graph = self.g.copy()

        sorted_start_indices = np.argsort(start_to_rb_dist)
        sorted_goal_indices = np.argsort(rb_to_goal_dist)
        neighbors_added = 0
        while neighbors_added < len(start_to_rb_dist):
            i = sorted_start_indices[neighbors_added]
            j = sorted_goal_indices[neighbors_added]
            planning_graph.add_edge("start", i, weight=start_to_rb_dist[i])
            planning_graph.add_edge(j, "goal", weight=rb_to_goal_dist[j])
            try:
                nx.shortest_path(planning_graph, source="start", target="goal")
                break
            except nx.NetworkXNoPath:
                neighbors_added += 1

        if not np.any(start_to_rb_dist < self.max_search_steps) or not np.any(
            rb_to_goal_dist < self.max_search_steps
        ):
            self.stats["localization_fails"] += 1
        return planning_graph

    def construct_augmented_planning_graph(self, starts, goals):
        if not self.localize_to_nearest:
            return super().construct_augmented_planning_graph(starts, goals)

        planning_graph = self.g.copy()
        print("Initial graph size = ", self.g.number_of_nodes())
        nodes_to_agent_maps = {}
        for idx, (start, goal) in enumerate(zip(starts, goals)):
            start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(
                {"observation": start, "goal": goal}
            )
            start_to_rb_dist, rb_to_goal_dist = (
                start_to_rb_dist.flatten(),
                rb_to_goal_dist.flatten(),
            )

            sorted_start_indices = np.argsort(start_to_rb_dist)
            sorted_goal_indices = np.argsort(rb_to_goal_dist)
            neighbors_added = 0
            num_nodes = planning_graph.number_of_nodes() - 1
            while neighbors_added < len(start_to_rb_dist):
                i = sorted_start_indices[neighbors_added]
                j = sorted_goal_indices[neighbors_added]
                planning_graph.add_edge(
                    # "start" + str(idx), i, weight=start_to_rb_dist[i]
                    num_nodes + 1,
                    i,
                    weight=start_to_rb_dist[i],
                )
                # planning_graph.add_edge(j, "goal" + str(idx), weight=rb_to_goal_dist[j])
                planning_graph.add_edge(j, num_nodes + 2, weight=rb_to_goal_dist[j])
                nodes_to_agent_maps["start" + str(idx)] = num_nodes + 1
                nodes_to_agent_maps["goal" + str(idx)] = num_nodes + 2
                try:
                    nx.shortest_path(
                        planning_graph,
                        # source="start" + str(idx),
                        # target="goal" + str(idx),
                        source=num_nodes + 1,
                        target=num_nodes + 2,
                    )
                    break
                except nx.NetworkXNoPath:
                    neighbors_added += 1

            if not np.any(start_to_rb_dist < self.max_search_steps) or not np.any(
                rb_to_goal_dist < self.max_search_steps
            ):
                self.stats["localization_fails"] += 1
        print("Final graph size = ", planning_graph.number_of_nodes())
        return planning_graph, nodes_to_agent_maps

    def reached_waypoint(self, dist_to_waypoint, state, waypoint_index):
        waypoint_qvals_combined = np.max(self.pdist, axis=0)[waypoint_index, :]
        obs_qvals = self.agent.get_pairwise_dist(
            [state["observation"]], self.rb_vec, aggregate=None
        )
        obs_qvals_combined = np.max(obs_qvals, axis=0).flatten()
        qval_diffs = waypoint_qvals_combined - obs_qvals_combined
        qval_inconsistency = np.linalg.norm(qval_diffs, np.inf)
        return qval_inconsistency < self.waypoint_consistency_cutoff

    def build_rb_graph(self):
        """
        Performs dynamic graph building.
        Args:
            edge_cutoff: draw directed edges between nodes when their qvalue
                         distance is less than edge_cutoff
            beta: percentage assigned to newest embedding space observation
                  in exponential moving average / variance calculations
        """
        if self.cache_pdist:
            self.cached_pdist = self.pdist.copy()

        self.g = nx.DiGraph()
        embeddings_to_add = self.rb_vec.copy()
        for i, embedding in enumerate(embeddings_to_add):
            self.update_graph(embedding, cache_index=i)

        self.cache_pdist = False
        self.cached_pdist = None

    def update_graph(
        self, embedding, cache_index=None
    ):  # Merge with existing node or create new node
        if self.cache_pdist:
            assert cache_index is not None

        if self.g.number_of_nodes() == 0:
            self.g.add_node(cache_index)
            self.rb_vec = np.array([embedding])
            self.rb_variances = np.zeros((1))

            if self.cache_pdist:
                self.pdist = self.get_cached_pairwise_dist(
                    np.array([cache_index]), np.array([cache_index])
                )
            else:
                self.pdist = self.agent.get_pairwise_dist(self.rb_vec, aggregate=None)

            if self.cache_pdist:
                self.cache_indices = np.array([cache_index])
        else:
            # Localize to nearest neighbors in embedding space
            neighbor_indices = np.arange(len(self.rb_vec))[
                self.norm_consistency(embedding, self.rb_vec)
            ]

            # Get maximum distances (i.e., minimum qvalues)
            if self.cache_pdist:
                embedding_to_rb = self.get_cached_pairwise_dist(
                    np.array([cache_index]), self.cache_indices
                )
                rb_to_embedding = self.get_cached_pairwise_dist(
                    self.cache_indices, np.array([cache_index])
                )
            else:
                embedding_to_rb = self.agent.get_pairwise_dist(
                    [embedding], self.rb_vec, aggregate=None
                )
                rb_to_embedding = self.agent.get_pairwise_dist(
                    self.rb_vec, [embedding], aggregate=None
                )

            pdist_combined = np.max(self.pdist, axis=0)
            embedding_to_rb_combined = np.max(embedding_to_rb, axis=0).flatten()
            rb_to_embedding_combined = np.max(rb_to_embedding, axis=0).flatten()

            # Try to merge with a neighbor based on qvalue consistency
            merged = False
            for neighbor in neighbor_indices:
                # Merge if qvalues are consistent
                if self.qvalue_consistency(
                    neighbor,
                    pdist_combined,
                    embedding_to_rb_combined,
                    rb_to_embedding_combined,
                ):
                    difference_from_avg = embedding - self.rb_vec[neighbor]
                    self.rb_vec[neighbor] = (
                        self.rb_vec[neighbor] + self.beta * difference_from_avg
                    )
                    self.rb_variances[neighbor] = (1 - self.beta) * (
                        self.rb_variances[neighbor]
                        + self.beta * np.sum(difference_from_avg**2)
                    )
                    merged = True
                    break

            # Add node if cannot merge
            if not merged:
                # Add node to graph
                new_index = self.g.number_of_nodes()
                in_indices = np.arange(new_index)[
                    rb_to_embedding_combined < self.edge_cutoff
                ]
                in_weights = rb_to_embedding_combined[in_indices]
                out_indices = np.arange(new_index)[
                    embedding_to_rb_combined < self.edge_cutoff
                ]
                out_weights = embedding_to_rb_combined[out_indices]
                self.g.add_node(new_index)
                self.g.add_weighted_edges_from(
                    zip(in_indices, [new_index] * len(in_indices), in_weights)
                )
                self.g.add_weighted_edges_from(
                    zip([new_index] * len(out_indices), out_indices, out_weights)
                )

                # The only qvalue distance we don't yet have is the new node to itself.
                # Can concatenate qvalues we already have to save |V|^2 qvalue query.
                # Used to update sparse_pdist
                if self.cache_pdist:
                    embedding_to_embedding = self.get_cached_pairwise_dist(
                        np.array([cache_index]), np.array([cache_index])
                    )
                else:
                    embedding_to_embedding = self.agent.get_pairwise_dist(
                        [embedding], [embedding], aggregate=None
                    )

                # Add node to other attributes
                self.rb_vec = np.concatenate((self.rb_vec, [embedding]), axis=0)
                self.rb_variances = np.append(self.rb_variances, [0])
                self.pdist = np.concatenate((self.pdist, embedding_to_rb), axis=1)
                self.pdist = np.concatenate(
                    (
                        self.pdist,
                        np.concatenate(
                            (rb_to_embedding, embedding_to_embedding), axis=1
                        ),
                    ),
                    axis=2,
                )
                if self.cache_pdist:
                    assert cache_index is not None
                    self.cache_indices = np.append(self.cache_indices, cache_index)

    def get_cached_pairwise_dist(self, row_indices, col_indices):
        assert len(row_indices.shape) == len(col_indices.shape) == 1
        row_entries = row_indices.shape[0]
        col_entries = col_indices.shape[0]
        row_advanced_index = np.tile(row_indices, (col_entries, 1)).T
        col_advanced_index = np.tile(col_indices, (row_entries, 1))
        assert self.cached_pdist is not None
        if len(self.cached_pdist.shape) == 2:
            return self.cached_pdist[row_advanced_index, col_advanced_index]
        elif len(self.cached_pdist.shape) == 3:
            return self.cached_pdist[:, row_advanced_index, col_advanced_index]
        else:
            raise RuntimeError("Cached pdist has unrecognized shape")

    def norm_consistency(self, embedding, embeddings):
        differences = embeddings - embedding
        inconsistency = np.linalg.norm(differences, axis=1)
        return inconsistency < self.norm_cutoff

    def qvalue_consistency(
        self,
        neighbor_index,
        pdist_combined,
        rb_to_embedding_combined,
        embedding_to_rb_combined,
    ):
        # Find adjacent nodes
        in_indices = np.array(list(self.g.predecessors(neighbor_index)))
        out_indices = np.array(list(self.g.successors(neighbor_index)))

        # Be conservative about merging in this edge case
        if len(in_indices) == 0 and len(out_indices) == 0:
            return False

        # Calculate qvalues with adjacent nodes
        if len(in_indices) != 0:
            existing_in_qvals = pdist_combined[in_indices, neighbor_index]
            new_in_qvals = rb_to_embedding_combined[in_indices]
        else:
            existing_in_qvals = np.array([])
            new_in_qvals = np.array([])
        if len(out_indices) != 0:
            existing_out_qvals = pdist_combined[neighbor_index, out_indices]
            new_out_qvals = embedding_to_rb_combined[out_indices]
        else:
            existing_out_qvals = np.array([])
            new_out_qvals = np.array([])
        existing_qvals = np.append(existing_in_qvals, existing_out_qvals)
        new_qvals = np.append(new_in_qvals, new_out_qvals)

        # Measure qvalue consistency
        qval_diffs = new_qvals - existing_qvals
        qval_inconsistency = np.linalg.norm(qval_diffs, np.inf)

        # Determine if consistent using cutoff
        return qval_inconsistency < self.consistency_cutoff

    def get_goal_in_rb(self):
        goal_index = np.random.randint(low=0, high=self.rb_vec.shape[0])
        return self.rb_vec[goal_index].copy()
