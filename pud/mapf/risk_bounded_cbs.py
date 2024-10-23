import time
import heapq
import random
import logging
import numpy as np
from networkx import Graph
from typing import Dict, List, Tuple, Union
from numpy.typing import NDArray

from pud.mapf.cbs import detect_collisions, disjoint_split, get_location
from pud.mapf.mapf_exceptions import MAPFError, MAPFErrorCodes
from pud.mapf.single_agent_planner import (
    Node,
    RiskNode,
    extract_path,
    compute_heuristics,
    is_constrained_with_ds,
    build_constraint_table_with_ds,
)


class BudgetAllocater(object):

    def __init__(self, strategy: str, risk_bound: float):
        self.strategy = strategy
        self.risk_bound = risk_bound

    def allocate(self, paths: List[List[int]]):
        raise NotImplementedError


class UniformBudgetAllocater(BudgetAllocater):

    def __init__(self, risk_bound: float):
        super().__init__("uniform", risk_bound)

    def allocate(self, paths, utility=None):
        self.num_agents = len(paths)
        return [self.risk_bound / self.num_agents] * self.num_agents


class InverseUtilityBudgetAllocater(BudgetAllocater):
    def __init__(self, risk_bound: float):
        super().__init__("inverse_utility", risk_bound)

    def allocate(self, paths, utility):
        risk_allocation = []
        self.num_agents = len(paths)
        inverse_utility = [1 / utility for utility in utility]
        inverse_sum = sum(inverse_utility)
        for agent in range(self.num_agents):
            risk_allocation.append(
                self.risk_bound * inverse_utility[agent] / inverse_sum
            )
        assert np.isclose(sum(risk_allocation), self.risk_bound)
        return risk_allocation


class UtilityBudgetAllocater(BudgetAllocater):
    def __init__(self, risk_bound: float):
        super().__init__("utility", risk_bound)

    def allocate(self, paths, utility):
        risk_allocation = []
        self.num_agents = len(paths)
        utility_sum = sum(utility)
        for agent in range(self.num_agents):
            risk_allocation.append(self.risk_bound * utility[agent] / utility_sum)
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
        disjoint: bool = True,
        collision_radius: float = 0.1,
        seed: Union[int, None] = None,
        budget_allocator: str = "utility",
    ):

        if seed is not None:
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

        self.budget_allocator = self.budget_allocator_cls(self.risk_bound)

        # Add self-loops
        for node in self.graph.nodes:
            self.graph.add_edge(node, node, weight=0, cost=0)

    def compute_cost(self, path: List[int], weighted: str = "") -> float:
        cost = 0
        for i in range(len(path) - 1):
            cost += (
                float(self.graph[path[i]][path[i + 1]][weighted])
                if len(weighted) > 0
                else 1
            )
        return cost

    def compute_sum_of_costs(self, paths: List[List[int]], weighted: str = "") -> float:
        sum_of_costs = 0
        for path in paths:
            sum_of_costs += self.compute_cost(path, weighted)
        return sum_of_costs

    def min_feasible_cost(
        self, agent: int, constraints: List[Dict]
    ) -> Union[float, MAPFErrorCodes]:
        agent_path = self.a_star_with_ds(
            agent,
            constraints,
            cost_h=True,
            weighted="cost",
            max_time=self.max_time // self.num_agents,
        )
        if type(agent_path) is MAPFErrorCodes:
            error_code = agent_path
            return error_code
        else:
            assert agent_path is not MAPFErrorCodes.NO_PATH
            assert agent_path is not MAPFErrorCodes.TIMELIMIT_REACHED
            assert agent_path is not MAPFErrorCodes.START_GOAL_DISCONNECT

            risk_value = self.compute_cost(agent_path, "cost")
            return risk_value

    def reallocate_risk(
        self,
        failings_agents: List[int],
        constraints: List[Dict],
        original_risk_allocation: List[float],
    ) -> Union[Tuple[List[float], int], MAPFErrorCodes]:

        passing_agents = [i for i in range(self.num_agents) if i not in failings_agents]

        required_budgets = {}
        for agent in failings_agents:
            minimum_cost = self.min_feasible_cost(agent, constraints)

            # If the agent is not able to find a risk-based path within the constraints
            if type(minimum_cost) is MAPFErrorCodes:
                error_code = minimum_cost
                return error_code

            required_budgets[agent] = minimum_cost

        additional_budget = sum(required_budgets.values()) - sum(
            original_risk_allocation[agent] for agent in failings_agents
        )

        budget_available = 0
        passing_agent_minimum_feasible_budgets = {}
        for agent in passing_agents:
            passing_agent_minimum_feasible_budgets[agent] = self.min_feasible_cost(
                agent, constraints
            )

            # If the agent is not able to find a risk-based path within the constraints
            if type(passing_agent_minimum_feasible_budgets[agent]) is MAPFErrorCodes:
                error_code = passing_agent_minimum_feasible_budgets[agent]
                return error_code

            budget_available += (
                original_risk_allocation[agent]
                - passing_agent_minimum_feasible_budgets[agent]
            )

        # If the cumulative minimum feasible risk allocations of passing agents are less than the required budget
        # for the failing agents, then we no reallocation of the risk will make the failing agents pass without
        # violating the risk-budgeted paths of the passing agents
        if additional_budget > budget_available:
            return MAPFErrorCodes.BUDGET_MISMATCH

        num_agents_allocation_modified = 0
        new_allocation = original_risk_allocation.copy()
        for agent in failings_agents:
            num_agents_allocation_modified += 1
            new_allocation[agent] = required_budgets[agent]

        for agent in passing_agents:
            num_agents_allocation_modified += 1
            max_reduction = (
                original_risk_allocation[agent]
                - passing_agent_minimum_feasible_budgets[agent]
            )
            reduction = min(max_reduction, additional_budget)
            new_allocation[agent] -= reduction
            additional_budget -= reduction
            if additional_budget <= 0:
                break

        return new_allocation, num_agents_allocation_modified

    def extract_violating_agents(self, successor: Dict, constraint: Dict) -> List[int]:

        violating_agents = []
        for agent in range(self.num_agents):

            if agent == constraint["agent_id"]:
                continue

            current_location = get_location(
                successor["paths"][agent], constraint["timestep"]
            )
            previous_location = get_location(
                successor["paths"][agent],
                constraint["timestep"] - 1,
            )

            if len(constraint["location"]) == 1:
                if constraint["location"][0] == current_location:
                    violating_agents.append(agent)
            else:
                successor_path_location = [
                    previous_location,
                    current_location,
                ]
                if (
                    constraint["location"]
                    == successor_path_location[::-1]
                    or constraint["location"][0]
                    == successor_path_location[0]
                    or constraint["location"][1]
                    == successor_path_location[1]
                ):
                    violating_agents.append(agent)

        logging.debug("Violating agents are {}".format(violating_agents))
        return violating_agents

    def find_paths(self):
        start_time = time.time()
        logging.debug("Finding paths using Risk Bounded CBS solver")

        # Define the root of the risk-bounded CBS tree
        root = {
            "cost": 0,
            "paths": [],
            "collisions": [],
            "constraints": [],
            "risk_allocation": [],
        }

        # Compute the paths for each agent ignoring the risk constraints
        for agent in range(self.num_agents):
            logging.debug("Computing paths for agent {}".format(agent))
            agent_path = self.a_star_with_ds(
                agent,
                root["constraints"],
                weighted=self.weighted,
                max_time=self.max_time // self.num_agents,
            )
            if type(agent_path) is MAPFErrorCodes:
                error_code = agent_path
                raise RuntimeError(
                    MAPFError(MAPFErrorCodes.NO_INIT_PATH, error_code)["message"]
                )

            root["paths"].append(agent_path)

        # Cost and collisions are not computed since the current paths are ignoring the risk constraints
        root["cost"] = 0
        root["collisions"] = 0

        # Compute the risk allocation for each agent based on their current utility
        root["risk_allocation"] = self.budget_allocator.allocate(
            root["paths"], [self.compute_cost(path) for path in root["paths"]]
        )
        logging.info("Risk allocation {}".format(root["risk_allocation"]))

        # The open list tracks nodes for the CBS tree based on how many agents' risk allocation was changed
        # The root's risk allocation is the initial risk allocation so no agents' risk allocation was changed
        # The last element in the tuple keep track of the agents that failed to find paths within their risk allocation
        heapq.heappush(
            self.open_list,
            (0, self.num_generated, root, [False] * self.num_agents),
        )
        logging.debug("Generated: {}".format(self.num_generated))
        self.num_generated += 1

        # Run the loop till timeout or the open list is empty
        while len(self.open_list) > 0 and time.time() - start_time < self.max_time:

            # The second last element in the open-list tuple is the node itself
            current_node, agent_failure_status = heapq.heappop(self.open_list)[-2:]
            logging.debug("Expanding: {}".format(current_node))
            self.num_expanded += 1

            # Recompute the paths for the agents with updated risk-allocation
            if not all(agent_failure_status):
                logging.debug("Computing paths for agents that are within risk-budget")
                for agent in range(self.num_agents):
                    agent_path = self.risk_budgeted_a_star_with_ds(
                        agent,
                        current_node["constraints"],
                        current_node["risk_allocation"][agent],
                        weighted=self.weighted,
                        max_time=self.max_time // self.num_agents,
                    )
                    if type(agent_path) is not MAPFErrorCodes:
                        agent_failure_status[agent] = True
                        current_node["paths"][agent] = agent_path
                        current_node["cost"] += self.compute_cost(agent_path, self.weighted)  # type: ignore
                    else:
                        logging.debug(
                            MAPFError(MAPFErrorCodes.NO_PATH, agent_path)["message"]
                        )

            # If all agents were able to find paths within their risk allocation then we need to proceed
            # with conflict resolution step of CBS
            if all(agent_failure_status):
                # If we reach here then the a_star was able to find paths for all the agents
                # within their allocated risk-budget. Now we need to focus on de-conflicting the paths
                current_node["collisions"] = detect_collisions(
                    current_node["paths"], self.graph_waypoints, self.collision_radius
                )

                # If there are no conflicts then all the agents' paths are risk-bounded and conflict-free!
                if len(current_node["collisions"]) == 0:
                    logging.debug(
                        "Found paths with no collisions with node {}".format(
                            current_node
                        )
                    )
                    return current_node

                # Choose a random conflict and split the CBS tree based on the conflict
                collision = random.choice(current_node["collisions"])
                constraints = disjoint_split(collision)

                for constraint in constraints:

                    logging.debug("Tackling constraint {}".format(constraint))
                    # One branch of the CBS tree
                    successor = {
                        "cost": 0,
                        "paths": current_node["paths"].copy(),
                        "collisions": [],
                        "constraints": [constraint],
                        "risk_allocation": current_node["risk_allocation"],
                    }

                    # Copy the constraints from the parent node
                    for c in current_node["constraints"]:
                        if c not in successor["constraints"]:
                            successor["constraints"].append(c)

                    # Find a path for the conflicting agent with its assigned risk-budget
                    constraint_agent = constraint["agent_id"]
                    agent_path = self.risk_budgeted_a_star_with_ds(
                        constraint_agent,
                        successor["constraints"],
                        current_node["risk_allocation"][constraint_agent],
                        weighted=self.weighted,
                        max_time=self.max_time // self.num_agents,
                    )

                    # If the path is found then we need to recompute the paths for the other agents
                    # whose path might be affected by this new path
                    if type(agent_path) is not MAPFErrorCodes:
                        # Update the path of the conflicting agent in this branch's CBS node
                        successor["paths"][constraint_agent] = agent_path
                        agent_failure_status[constraint_agent] = True

                        if constraint["positive"]:
                            # Extract the agents whose paths are affected by the new path of the conflicting agent
                            violating_agents = self.extract_violating_agents(
                                successor, constraint
                            )
                            for agent in violating_agents:
                                if agent == constraint_agent:
                                    continue

                                # Recompute their paths with their risk allocation
                                agent_path = self.risk_budgeted_a_star_with_ds(
                                    agent,
                                    successor["constraints"],
                                    current_node["risk_allocation"][agent],
                                    weighted=self.weighted,
                                    max_time=self.max_time // self.num_agents,
                                )

                                # If their path is found then update their path in this branch's CBS node
                                if type(agent_path) is not MAPFErrorCodes:
                                    successor["paths"][agent] = agent_path
                                    agent_failure_status[agent] = True
                                else:
                                    # If we cannot find a path for this violating agent then keep track of them so
                                    # that we can recompute the risk allocation and re-insert the node into the open
                                    # list
                                    agent_failure_status[agent] = False

                            # If some of the violating agents were not able to find paths within their risk allocation
                            # then we need to recompute the risk allocation and re-insert the node into the open list
                            # with the current state of the CBS tree stored in the node so that later it can be expanded
                            # to compute the paths for the violating agents with the updated risk-allocation
                            if not all(agent_failure_status):
                                failing_violating_agents = [
                                    agent for agent in violating_agents if not agent_failure_status[agent]
                                ]
                                new_allocation = self.reallocate_risk(
                                    failing_violating_agents,
                                    successor["constraints"],
                                    current_node["risk_allocation"],
                                )

                                # If the reallocation of the risk is successful then we need to re-insert the node
                                if type(new_allocation) is not MAPFErrorCodes:
                                    new_allocation, num_agents_allocation_changed = new_allocation  # type: ignore
                                    successor["risk_allocation"] = new_allocation

                                    heapq.heappush(
                                        self.open_list,
                                        (
                                            num_agents_allocation_changed,
                                            self.num_generated,
                                            successor,
                                            agent_failure_status
                                        ),
                                    )
                                    logging.debug("Violating agents paths not found so reallocated risk")
                                    logging.debug("Violating agents: {}".format(failing_violating_agents))
                                    logging.debug(
                                        "Generated: {}".format(self.num_generated)
                                    )
                                    logging.debug("{}".format(successor))
                                    self.num_generated += 1
                                else:
                                    # If the reallocation of the risk is not successful then we cannot find a solution
                                    # from this branch of the CBS tree that satisfies the risk allocation and the
                                    # constraint imposed on it
                                    error_code = new_allocation
                                    logging.debug(MAPFError(error_code)["message"])
                                    continue
                            else:
                                # If all the violating agents were able to find paths within their risk allocation
                                # and the updated constraint then we need to update the cost and collisions of the
                                # successor node and insert it into the open list so that later it can be expanded
                                # to resolve other potential conflicts
                                successor["cost"] = self.compute_sum_of_costs(
                                    successor["paths"], weighted=self.weighted
                                )
                                successor["collisions"] = detect_collisions(
                                    successor["paths"],
                                    self.graph_waypoints,
                                    self.collision_radius,
                                )

                                heapq.heappush(
                                    self.open_list,
                                    (
                                        0,
                                        self.num_generated,
                                        successor,
                                        agent_failure_status,
                                    ),
                                )
                                logging.debug("Generated: {}".format(self.num_generated))
                                logging.debug("{}".format(successor))
                                self.num_generated += 1
                    else:
                        # If the path is not found for the conflicting agent within its risk allocation
                        # then we need to recompute the risk allocation for the failing agent and check if
                        # the failure of the agent is due to the risk allocation. If the failure is due to the
                        # risk allocation then we need to recompute the risk allocation and re-insert the node
                        # into the open list with the current state of the CBS tree stored in the node
                        agent_failure_status[constraint_agent] = False
                        new_allocation = self.reallocate_risk(
                            [constraint_agent],
                            successor["constraints"],
                            current_node["risk_allocation"],
                        )

                        # If the reallocation of the risk is successful then we need to re-insert the node
                        if type(new_allocation) is not MAPFErrorCodes:
                            new_allocation, num_agents_allocation_changed = new_allocation  # type: ignore
                            successor["risk_allocation"] = new_allocation

                            # Cost and collisions are not computed since the current paths include an invalid path for
                            # the conflicting agent

                            heapq.heappush(
                                self.open_list,
                                (
                                    num_agents_allocation_changed,
                                    self.num_generated,
                                    successor,
                                    agent_failure_status
                                ),
                            )
                            logging.debug("Constraint agent's path not found so reallocated risk")
                            logging.debug("Generated: {}".format(self.num_generated))
                            logging.debug("{}".format(successor))
                            self.num_generated += 1
                        else:
                            # If the reallocation of the risk is not successful then we cannot find a solution from
                            # this branch of the CBS tree
                            error_code = new_allocation
                            logging.debug(MAPFError(error_code)["message"])
            else:

                # A star call to some agents failed. We need to recompute the risk allocation
                # and re-insert the node into the open list with the current state of the CBS tree
                # stored in the node so that later it can be expanded to compute the paths for the
                # failing agents with the updated risk-allocation

                failings_agents = [
                    agent for agent in range(self.num_agents) if not agent_failure_status[agent]
                ]
                logging.info("A-Star failed for agents: {}".format(failings_agents))

                # Recompute the risk allocation for the failing agents
                new_allocation = self.reallocate_risk(
                    failings_agents,
                    current_node["constraints"],
                    current_node["risk_allocation"],
                )

                # If the reallocation of the risk is successful then we need to re-insert the node
                if type(new_allocation) is not MAPFErrorCodes:
                    new_allocation, num_agents_allocation_changed = new_allocation  # type: ignore
                    current_node["risk_allocation"] = new_allocation

                    heapq.heappush(
                        self.open_list,
                        (
                            num_agents_allocation_changed,
                            self.num_generated,
                            current_node,
                            agent_failure_status
                        ),
                    )
                    logging.debug("Generated: {}".format(self.num_generated))
                    logging.debug("{}".format(current_node))
                    self.num_generated += 1

                    logging.info(
                        "Risk allocation {}".format(current_node["risk_allocation"])
                    )
                else:
                    # If the reallocation of the risk is not successful then we cannot find a solution from
                    # this branch of the CBS tree that satisfies the risk allocation and the constraint imposed
                    # on it
                    error_code = new_allocation
                    logging.debug(MAPFError(error_code)["message"])

        # If we terminate the search due to timeout the explicitly return the timeout error code
        # otherwise return the no path error code
        if time.time() - start_time >= self.max_time:
            raise RuntimeError(MAPFError(MAPFErrorCodes.TIMELIMIT_REACHED))
        else:
            raise RuntimeError(MAPFError(MAPFErrorCodes.NO_PATH))

    def a_star_with_ds(
        self,
        agent_id: int,
        constraints: List[Dict],
        cost_h: bool = False,
        weighted: str = "",
        max_time: int = 300,
    ):
        start_time = time.time()

        open_list = []
        closed_list = {}

        num_expanded = 0
        num_generated = 0
        max_constraints = 0

        if not cost_h:
            if self.starts[agent_id] not in self.distance_heuristics[agent_id]:
                return MAPFErrorCodes.START_GOAL_DISCONNECT
            h_value = self.distance_heuristics[agent_id][self.starts[agent_id]]
        else:
            if self.starts[agent_id] not in self.cost_heuristics[agent_id]:
                return MAPFErrorCodes.START_GOAL_DISCONNECT
            h_value = self.cost_heuristics[agent_id][self.starts[agent_id]]

        constraint_table = build_constraint_table_with_ds(constraints, agent_id)
        if constraint_table.keys():
            max_constraints = max(constraint_table.keys())

        root = Node(self.starts[agent_id], 0, h_value, None, 0)
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
                root.h_value,
                root.location,
                num_generated,
                root,
            ),
        )
        num_generated += 1

        closed_list[(root.location, root.timestep)] = root

        while len(open_list) != 0 and time.time() - start_time < max_time:

            current_node = heapq.heappop(open_list)[4]
            num_expanded += 1

            if current_node.location == self.goals[
                agent_id
            ] and not is_constrained_with_ds(
                self.goals[agent_id],
                self.goals[agent_id],
                current_node.timestep,
                max_constraints,
                constraint_table,
                agent_id,
                goal=True,
            ):
                return extract_path(current_node)

            for neighbor in self.graph.neighbors(current_node.location):
                successor_location = neighbor

                if successor_location == current_node.location:
                    successor = Node(
                        successor_location,
                        current_node.g_value + 1,
                        current_node.h_value,
                        current_node,
                        current_node.timestep + 1,
                    )
                else:
                    successor_gadd = (
                        float(
                            self.graph[current_node.location][successor_location][
                                weighted
                            ]
                        )
                        if len(weighted) > 0
                        else 1
                    )
                    if not cost_h:
                        h_val = self.distance_heuristics[agent_id][successor_location]
                    else:
                        h_val = self.cost_heuristics[agent_id][successor_location]
                    successor = Node(
                        successor_location,
                        current_node.g_value + successor_gadd,
                        h_val,
                        current_node,
                        current_node.timestep + 1,
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
                    existing_node = closed_list[
                        (successor.location, successor.timestep)
                    ]
                    if (
                        successor.g_value + successor.h_value
                        < existing_node.g_value + existing_node.h_value
                        and successor.g_value < existing_node.g_value
                    ):
                        # logging.debug(f"Updating node {successor.location}")
                        closed_list[(successor.location, successor.timestep)] = (
                            successor
                        )
                        heapq.heappush(
                            open_list,
                            (
                                successor.g_value + successor.h_value,
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
                            successor.h_value,
                            successor.location,
                            num_generated,
                            successor,
                        ),
                    )
                    num_generated += 1

        if time.time() - start_time > max_time:
            return MAPFErrorCodes.TIMELIMIT_REACHED
        else:
            return MAPFErrorCodes.NO_PATH

    def risk_budgeted_a_star_with_ds(
        self,
        agent_id: int,
        constraints,
        risk_budget: float,
        weighted: str = "",
        max_time: int = 300,
    ):
        start_time = time.time()

        logging.debug("\nComputing path for agent {}".format(agent_id))
        logging.debug("Risk budget allocated is {}".format(risk_budget))
        logging.debug("Constraints applied are {}\n".format(constraints))

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

        logging.debug("\nConstraint table is {}".format(constraint_table))

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
                        float(
                            self.graph[current_node.location][successor_location][
                                weighted
                            ]
                        )
                        if len(weighted) > 0
                        else 1
                    )

                    successor_risk = current_node.risk + float(
                        self.graph[current_node.location][successor_location]["cost"]
                    )
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
                    existing_node = closed_list[
                        (successor.location, successor.timestep)
                    ]
                    if (
                        successor.g_value + successor.h_value
                        <= existing_node.g_value + existing_node.h_value
                        and successor.g_value <= existing_node.g_value
                        and successor.risk < existing_node.risk
                    ):
                        # logging.debug(f"Updating node {successor.location}")
                        closed_list[(successor.location, successor.timestep)] = (
                            successor
                        )
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
