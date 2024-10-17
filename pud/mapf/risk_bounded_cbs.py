import time
import heapq
import random
import logging
import numpy as np
from networkx import Graph
from typing import List, Union
from numpy.typing import NDArray

from pud.mapf.cbs import detect_collisions, disjoint_split, get_location
from pud.mapf.mapf_exceptions import MAPFError, MAPFErrorCodes
from pud.mapf.single_agent_planner import (
    RiskNode,
    a_star_with_ds,
    build_constraint_table_with_ds,
    compute_cost,
    compute_heuristics,
    compute_heuristics_v2,
    compute_sum_of_costs,
    extract_path,
    is_constrained_with_ds,
    risk_budgeted_a_star_with_ds,
)


class BudgetAllocater(object):

    def __init__(self, strategy: str, risk_bound: float):
        self.strategy = strategy
        self.risk_bound = risk_bound

    def allocate(self, paths: List[List[int]]):
        raise NotImplementedError


class UniformBudgetAllocater(BudgetAllocater):

    def __init__(self, graph: Graph, risk_bound: float):
        super().__init__("uniform", risk_bound)

    def allocate(self, paths):
        self.num_agents = len(paths)
        return [self.risk_bound / self.num_agents] * self.num_agents


class InverseUtilityBudgetAllocater(BudgetAllocater):
    def __init__(self, graph: Graph, risk_bound: float):
        self.graph = graph
        super().__init__("inverse_utility", risk_bound)

    def allocate(self, paths):
        risk_allocation = []
        self.num_agents = len(paths)
        path_utility = [compute_cost(path, self.graph) for path in paths]
        inverse_utility = [1 / utility for utility in path_utility]
        inverse_sum = sum(inverse_utility)
        for agent in range(self.num_agents):
            risk_allocation.append(self.risk_bound * inverse_utility[agent] / inverse_sum)
        assert np.isclose(sum(risk_allocation), self.risk_bound)
        return risk_allocation


class UtilityBudgetAllocater(BudgetAllocater):
    def __init__(self, graph: Graph, risk_bound: float):
        self.graph = graph
        super().__init__("utility", risk_bound)

    def allocate(self, paths):
        risk_allocation = []
        self.num_agents = len(paths)
        path_utility = [compute_cost(path, self.graph) for path in paths]
        utility_sum = sum(path_utility)
        for agent in range(self.num_agents):
            risk_allocation.append(self.risk_bound * path_utility[agent] / utility_sum)
        assert np.isclose(sum(risk_allocation), self.risk_bound)
        return risk_allocation


class RiskBoundedCBSSolver(object):
    def __init__(
        self,
        graph: Graph,
        graph_waypoints: NDArray,
        starts: List[int],
        goals: List[int],
        risk_bound: float,
        weighted: str = "",
        max_time: int = 600,
        collision_radius=0.1,
        disjoint: bool = True,
        seed: Union[int, None] = None,
        budget_allocator: str = "utility",
    ):

        random.seed(0)

        self.graph = graph
        self.graph_waypoints = graph_waypoints

        self.starts = starts
        self.goals = goals
        self.risk_bound = risk_bound

        self.num_expanded = 0
        self.num_generated = 0
        self.seed = seed
        self.max_time = max_time
        self.disjoint = disjoint
        self.weighted = weighted
        self.num_agents = len(starts)
        self.collision_radius = collision_radius

        self.open_list = []
        self.cost_heuristics = []
        self.distance_heuristics = []
        for goal in self.goals:
            self.distance_heuristics.append(
                compute_heuristics(self.graph.copy(), goal, weighted="")
            )
            self.cost_heuristics.append(
                compute_heuristics(self.graph.copy(), goal, weighted="cost")
            )

        self.budget_allocator_cls = None
        if budget_allocator == "uniform":
            self.budget_allocator_cls = UniformBudgetAllocater
        elif budget_allocator == "utility":
            self.budget_allocator_cls = UtilityBudgetAllocater
        elif budget_allocator == "inverse_utility":
            self.budget_allocator_cls = InverseUtilityBudgetAllocater
        else:
            raise RuntimeError(MAPFError(MAPFErrorCodes.INVALID_BUDGET_ALLOCATER))

        self.budget_allocator = self.budget_allocator_cls(self.graph, self.risk_bound)

        # Add self-loops
        for node in self.graph.nodes:
            self.graph.add_edge(node, node, weight=0, cost=0)

    def min_feasible_cost(self, agent, constraints):
        agent_path = a_star_with_ds(
            agent,
            self.graph,
            self.starts[agent],
            self.goals[agent],
            self.cost_heuristics[agent],
            constraints,
            weighted="cost",
            max_time=self.max_time // self.num_agents,
        )
        if type(agent_path) is MAPFErrorCodes:
            return agent_path
        else:
            risk_value = compute_cost(agent_path, self.graph, "cost")  # type: ignore
            return risk_value

    def reallocate_risk(self, node, failings_agents):
        passing_agents = [i for i in range(self.num_agents) if i not in failings_agents]

        required_budgets = {}
        for agent in failings_agents:
            minimum_cost = self.min_feasible_cost(agent, node["constraints"])
            if type(minimum_cost) is MAPFErrorCodes:
                return minimum_cost
            required_budgets[agent] = minimum_cost

        additional_budget = sum(required_budgets.values()) - sum(node["risk_allocation"][i] for i in failings_agents)
        budget_available = 0
        min_feasible_costs = {}
        for agent in passing_agents:
            min_feasible_costs[agent] = self.min_feasible_cost(agent, node["constraints"])
            if type(min_feasible_costs[agent]) is MAPFErrorCodes:
                return min_feasible_costs[agent]
            budget_available += node["risk_allocation"][agent] - min_feasible_costs[agent]

        if additional_budget > budget_available:
            return MAPFErrorCodes.BUDGET_MISMATCH

        new_allocation = node["risk_allocation"].copy()
        for agent in failings_agents:
            new_allocation[agent] = required_budgets[agent]

        for agent in passing_agents:
            max_reduction = node["risk_allocation"][agent] - min_feasible_costs[agent]
            reduction = min(max_reduction, additional_budget)
            new_allocation[agent] -= reduction
            additional_budget -= reduction
            if additional_budget <= 0:
                break

        # TODO: At this point the passing agents should have valid paths! If not check them
        return new_allocation

    def find_paths(self):
        start_time = time.time()
        logging.debug("Finding paths using Risk Bounded CBS solver")

        root = {
            "cost": 0,
            "paths": [],
            "collisions": [],
            "constraints": [],
            "risk_allocation": [],
        }

        for i in range(self.num_agents):
            logging.debug("Computing paths for agent {}".format(i))
            agent_path = a_star_with_ds(
                i,
                self.graph,
                self.starts[i],
                self.goals[i],
                self.distance_heuristics[i],
                root["constraints"],
                weighted=self.weighted,
                max_time=self.max_time // self.num_agents,
            )
            if type(agent_path) is MAPFErrorCodes:
                raise RuntimeError(MAPFError(MAPFErrorCodes.NO_INIT_PATH, agent_path)["message"])

            root["paths"].append(agent_path)

        root["cost"] = compute_sum_of_costs(
            root["paths"], self.graph, weighted=self.weighted
        )
        root["collisions"] = detect_collisions(
            root["paths"], self.graph_waypoints, self.collision_radius
        )
        root["risk_allocation"] = self.budget_allocator.allocate(root["paths"])
        logging.info("Risk allocation {}".format(root["risk_allocation"]))

        heapq.heappush(
            self.open_list,
            (root["cost"], len(root["collisions"]), self.num_generated, root),
        )
        logging.debug("Generated: {}".format(self.num_generated))
        self.num_generated += 1

        astar_success = [False for _ in range(self.num_agents)]
        while len(self.open_list) > 0 and time.time() - start_time < self.max_time:

            current_node = heapq.heappop(self.open_list)[-1]
            logging.debug("Expanding: {}".format(current_node))
            self.num_expanded += 1

            if not all(astar_success):
                for i in range(self.num_agents):
                    agent_path = self.risk_budgeted_a_star_with_ds(
                        i,
                        current_node["constraints"],
                        current_node["risk_allocation"][i],
                        weighted=self.weighted,
                        max_time=self.max_time // self.num_agents,
                    )
                    if type(agent_path) is not MAPFErrorCodes:
                        astar_success[i] = True
                        current_node["paths"][i] = agent_path
                    else:
                        logging.debug(MAPFError(MAPFErrorCodes.NO_PATH, agent_path)["message"])

            if all(astar_success):
                # If we reach here then the a_star was able to find paths for all the agents
                # within their allocated risk-budget. Now we need to focus on de-conflicting the paths
                current_node["collisions"] = detect_collisions(
                    current_node["paths"], self.graph_waypoints, self.collision_radius
                )

                if len(current_node["collisions"]) == 0:
                    logging.debug("Found paths with no collisions with node {}".format(current_node))
                    return current_node

                collision = random.choice(current_node["collisions"])
                constraints = disjoint_split(collision)

                for constraint in constraints:
                    successor = {
                        "cost": 0,
                        "paths": current_node["paths"].copy(),
                        "collisions": [],
                        "constraints": [constraint],
                        "risk_allocation": current_node["risk_allocation"],
                    }

                    for c in current_node["constraints"]:
                        if c not in successor["constraints"]:
                            successor["constraints"].append(c)

                    constraint_agent = constraint["agent_id"]
                    agent_path = self.risk_budgeted_a_star_with_ds(
                        constraint_agent,
                        successor["constraints"],
                        current_node["risk_allocation"][constraint_agent],
                        weighted=self.weighted,
                        max_time=self.max_time // self.num_agents,
                    )

                    if type(agent_path) is not MAPFErrorCodes:
                        astar_success[constraint_agent] = True
                        successor["paths"][constraint_agent] = agent_path

                        skip = False
                        if constraint["positive"]:
                            violating_agents = []
                            for agent in range(self.num_agents):
                                if agent == constraint_agent:
                                    continue

                                current_location = get_location(successor["paths"][agent], constraint["timestep"])
                                previous_location = get_location(successor["paths"][agent], constraint["timestep"] - 1)

                                if len(constraint["location"]) == 1:
                                    if (constraint["location"][0] == current_location):
                                        violating_agents.append(agent)
                                else:
                                    successor_path_location = [previous_location, current_location]
                                    if (
                                        constraint["location"] == successor_path_location[::-1]
                                        or constraint["location"][0] == successor_path_location[0]
                                        or constraint["location"][1] == successor_path_location[1]
                                    ):
                                        violating_agents.append(agent)

                            for agent in violating_agents:
                                if agent == constraint_agent:
                                    continue

                                agent_path = self.risk_budgeted_a_star_with_ds(
                                    agent,
                                    successor["constraints"],
                                    current_node["risk_allocation"][agent],
                                    weighted=self.weighted,
                                    max_time=self.max_time // self.num_agents,
                                )
                                if agent_path is None:
                                    skip = True
                                    break
                                else:
                                    astar_success[agent] = True
                                    successor["paths"][agent] = agent_path

                            if skip:
                                continue

                        successor["cost"] = compute_sum_of_costs(
                            successor["paths"], self.graph, weighted=self.weighted
                        )
                        successor["collisions"] = detect_collisions(
                            successor["paths"], self.graph_waypoints, self.collision_radius
                        )

                        heapq.heappush(
                            self.open_list,
                            (successor["cost"], len(successor["collisions"]), self.num_generated, successor),
                        )
                        logging.debug("Generated: {}".format(self.num_generated))
                        self.num_generated += 1

            else:

                # A star call to some agent failed. We need to recompute the risk allocation
                # and re-insert the node into the open list

                logging.info(
                    "A-Star failed for agents: {}".format([i for i in range(self.num_agents) if not astar_success[i]])
                )
                failings_agents = [i for i in range(self.num_agents) if not astar_success[i]]
                new_allocation = self.reallocate_risk(current_node, failings_agents)

                if type(new_allocation) is not MAPFErrorCodes:
                    current_node["risk_allocation"] = new_allocation
                    heapq.heappush(
                        self.open_list,
                        (current_node["cost"], len(current_node["collisions"]), self.num_generated, current_node),
                    )
                    logging.debug("Generated: {}".format(self.num_generated))
                    self.num_generated += 1

                    logging.info("Risk allocation {}".format(current_node["risk_allocation"]))
                else:
                    logging.debug(MAPFError(new_allocation)["message"])

        if time.time() - start_time >= self.max_time:
            raise RuntimeError(MAPFError(MAPFErrorCodes.TIMELIMIT_REACHED))
        else:
            raise RuntimeError(MAPFError(MAPFErrorCodes.NO_PATH))

    def risk_budgeted_a_star_with_ds(self, agent_id: int, constraints, risk_budget: float, weighted: str = "", max_time: int = 300):
        start_time = time.time()

        open_list = []
        closed_list = {}

        num_expanded = 0
        num_generated = 0
        max_constraints = 0

        if self.starts[agent_id] not in self.distance_heuristics[agent_id]:
            return MAPFErrorCodes.START_GOAL_DISCONNECT

        h_value = self.distance_heuristics[agent_id][self.starts[agent_id]]
        constraint_table = build_constraint_table_with_ds(constraints, agent_id)
        if constraint_table.keys():
            max_constraints = max(constraint_table.keys())

        root = RiskNode(self.starts[agent_id], 0, h_value, None, 0, 0)
        if root.location == self.goals[agent_id]:
            if root.timestep <= max_constraints:
                if not is_constrained_with_ds(
                    self.goals[agent_id],
                    self.goals[agent_id],
                    root.timestep,
                    max_constraints,
                    constraint_table,
                    agent_id,
                    goal=True,
                ):
                    max_constraints = 0

        heapq.heappush(
            open_list,
            (
                root.g_value + root.h_value,
                root.risk,
                root.h_value,
                root.location,
                num_generated,
                root,
            ),
        )
        num_generated += 1

        closed_list[(root.location, root.timestep)] = root

        while len(open_list) != 0 and time.time() - start_time < max_time:

            current_node = heapq.heappop(open_list)[-1]
            num_expanded += 1

            # If we have reached the goal and the goal is not constrained and the risk is within the budget
            if (
                current_node.location == self.goals[agent_id]
                and not is_constrained_with_ds(
                    self.goals[agent_id],
                    self.goals[agent_id],
                    current_node.timestep,
                    max_constraints,
                    constraint_table,
                    agent_id,
                    goal=True,
                )
                and current_node.risk <= risk_budget
            ):
                return extract_path(current_node)

            for neighbor in self.graph.neighbors(current_node.location):
                successor_location = neighbor

                if successor_location == current_node.location:
                    successor_risk = current_node.risk
                    successor = RiskNode(
                        successor_location,
                        current_node.g_value + 1,
                        current_node.h_value,
                        current_node,
                        current_node.timestep + 1,
                        successor_risk,
                    )
                else:
                    successor_gadd = (
                        float(self.graph[current_node.location][successor_location][weighted])
                        if len(weighted) > 0
                        else 1
                    )

                    successor_risk = current_node.risk + float(self.graph[current_node.location][successor_location]["cost"])
                    if successor_risk > risk_budget:
                        continue

                    successor = RiskNode(
                        successor_location,
                        current_node.g_value + successor_gadd,
                        self.distance_heuristics[agent_id][successor_location],
                        current_node,
                        current_node.timestep + 1,
                        successor_risk,
                    )

                if is_constrained_with_ds(
                    current_node.location,
                    successor.location,
                    successor.timestep,
                    max_constraints,
                    constraint_table,
                    agent_id,
                ):
                    continue

                if (successor.location, successor.timestep) in closed_list:
                    existing_node = closed_list[(successor.location, successor.timestep)]
                    if (
                        successor.g_value + successor.h_value
                        <= existing_node.g_value + existing_node.h_value
                        and successor.g_value <= existing_node.g_value
                        and successor.risk < existing_node.risk
                    ):
                        # logging.debug(f"Updating node {successor.location}")
                        closed_list[(successor.location, successor.timestep)] = successor
                        heapq.heappush(
                            open_list,
                            (
                                successor.g_value + successor.h_value,
                                successor.risk,
                                successor.h_value,
                                successor.location,
                                num_generated,
                                successor,
                            ),
                        )
                        num_generated += 1
                else:
                    # logging.debug(f"Adding node {successor.location}")
                    closed_list[(successor.location, successor.timestep)] = successor
                    heapq.heappush(
                        open_list,
                        (
                            successor.g_value + successor.h_value,
                            successor.risk,
                            successor.h_value,
                            successor.location,
                            num_generated,
                            successor,
                        ),
                    )
                    num_generated += 1

            del current_node

        if time.time() - start_time > max_time:
            return MAPFErrorCodes.TIMELIMIT_REACHED
        else:
            return MAPFErrorCodes.NO_PATH
