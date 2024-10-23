from __future__ import annotations
import time
import heapq
import logging
from networkx import Graph
from typing import Dict, List, Union
from pud.mapf.mapf_exceptions import MAPFErrorCodes


def compute_cost(path: List[int], graph: Graph, weighted: str = ""):
    """
    Compute the cost of the path
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += float(graph[path[i]][path[i + 1]][weighted]) if len(weighted) > 0 else 1
    return cost


def compute_sum_of_costs(
    paths: List[List[int]], graph: Graph, weighted: str = ""
) -> float:
    """
    Compute the sum of costs of the paths
    """
    sum_of_costs = 0
    for path in paths:
        sum_of_costs += compute_cost(path, graph, weighted)
    return sum_of_costs


def compute_heuristics(graph: Graph, goal: int, weighted: str = ""):
    """
    Compute the heuristic for each node in the graph
    """
    graph = graph.to_undirected()
    open_list = [(0, goal)]
    heuristics = {goal: 0}
    closed_list = set()

    while open_list:
        cost, location = heapq.heappop(open_list)

        if location in closed_list:
            continue
        closed_list.add(location)

        for neighbor in graph.neighbors(location):
            edge_cost = float(graph[location][neighbor][weighted]) if weighted else 1
            successor_cost = cost + edge_cost

            if neighbor not in heuristics or heuristics[neighbor] > successor_cost:
                heuristics[neighbor] = successor_cost  # type: ignore
                heapq.heappush(open_list, (successor_cost, neighbor))  # type: ignore

    return heuristics


def build_constraint_table(
    constraints: List[Dict], agent_id: int
) -> Dict[int, List[Dict]]:

    constraint_table = {}
    for constraint in constraints:
        if constraint["agent_id"] == agent_id:
            timestep = constraint["timestep"]
            if timestep not in constraint_table:
                constraint_table[timestep] = [constraint]
            else:
                constraint_table[timestep].append(constraint)
    return constraint_table


def build_constraint_table_with_ds(
    constraints: List[Dict], agent_id: int
) -> Dict[int, List[Dict]]:

    constraint_table = {}
    if not constraints:
        return constraint_table

    for constraint in constraints:
        timestep = constraint["timestep"]
        timestep_constraint = []
        if timestep in constraint_table:
            timestep_constraint = constraint_table[timestep]

        if constraint["positive"] and constraint["agent_id"] == agent_id:
            timestep_constraint.append(constraint)
            constraint_table[timestep] = timestep_constraint
        elif not constraint["positive"] and constraint["agent_id"] == agent_id:
            timestep_constraint.append(constraint)
            constraint_table[timestep] = timestep_constraint
        elif constraint["positive"]:
            negative_constraint = constraint.copy()
            negative_constraint["agent_id"] = agent_id
            if len(negative_constraint["location"]) == 2:
                negative_constraint["location"] = negative_constraint["location"][::-1]
            negative_constraint["positive"] = False
            timestep_constraint.append(negative_constraint)
            constraint_table[timestep] = timestep_constraint

    return constraint_table


def is_constrained(
    current_location: int | str,
    next_location: int | str,
    timestep: int,
    constraint_table: Dict[int, List[Dict]],
    goal: bool = False,
):

    if not goal:
        if timestep in constraint_table:
            for constraint in constraint_table[timestep]:
                if [next_location] == constraint["location"] or [
                    current_location,
                    next_location,
                ] == constraint["location"]:
                    return True
        else:
            flattened_constraints = []
            constraints = [
                constraint
                for timestep_idx, constraint in constraint_table.items()
                if timestep_idx < timestep
            ]
            for constraint in constraints:
                for c in constraint:
                    flattened_constraints.append(c)
            for constraint in flattened_constraints:
                if [next_location] == constraint["location"] and constraint["final"]:
                    return True
    else:
        flattened_constraints = []
        constraints = [
            constraint
            for timestep_idx, constraint in constraint_table.items()
            if timestep_idx > timestep
        ]
        for constraint in constraints:
            for c in constraint:
                flattened_constraints.append(c)
        for constraint in flattened_constraints:
            if [next_location] == constraint["location"]:
                return True

    return False


def is_constrained_with_ds(
    current_location: int | str,
    next_location: int | str,
    timestep: int,
    max_timestep: int,
    constraint_table: Dict[int, List[Dict]],
    constraint_agent: int,
    goal: bool = False,
):
    if not goal:
        if timestep not in constraint_table:
            return False

        for constraint in constraint_table[timestep]:
            if constraint_agent == constraint["agent_id"]:
                if len(constraint["location"]) == 1:
                    if (
                        constraint["positive"]
                        and next_location != constraint["location"][0]
                    ):
                        return True
                    elif (
                        not constraint["positive"]
                        and next_location == constraint["location"][0]
                    ):
                        return True
                else:
                    if constraint["positive"] and constraint["location"] != [
                        current_location,
                        next_location,
                    ]:
                        return True
                    if not constraint["positive"] and constraint["location"] == [
                        current_location,
                        next_location,
                    ]:
                        return True
    else:
        for t in range(timestep + 1, max_timestep + 1):
            if t not in constraint_table:
                continue
            for constraint in constraint_table[t]:

                if constraint_agent == constraint["agent_id"]:
                    if len(constraint["location"]) == 1:
                        if (
                            constraint["positive"]
                            and current_location != constraint["location"][0]
                        ):
                            return True
                        elif (
                            not constraint["positive"]
                            and current_location == constraint["location"][0]
                        ):
                            return True
    return False


def extract_path(goal_node: Node) -> List[int]:
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.location)
        current_node = current_node.parent
    path.reverse()
    return path


class Node:
    def __init__(self, location, g_value, h_value, parent, timestep):
        self.parent = parent
        self.g_value = g_value
        self.h_value = h_value
        self.location = location
        self.timestep = timestep

    def __lt__(self, other):
        if self.g_value + self.h_value == other.g_value + other.h_value:
            if self.h_value == other.h_value:
                return self.timestep < other.timestep
            return self.h_value < other.h_value
        return self.g_value + self.h_value < other.g_value + other.h_value


class RiskNode(Node):
    def __init__(self, location, g_value, h_value, parent, timestep, risk):
        self.risk = risk
        super().__init__(location, g_value, h_value, parent, timestep)

    def __lt__(self, other):
        if self.g_value + self.h_value == other.g_value + other.h_value:
            if self.risk == other.risk:
                if self.h_value == other.h_value:
                    return self.timestep < other.timestep
                return self.h_value < other.h_value
            return self.risk < other.risk
        return self.g_value + self.h_value < other.g_value + other.h_value


def a_star(
    agent_id: int,
    graph: Graph,
    start: int,
    goal: int,
    heuristics: Dict[int, float],
    constraints,
    weighted: str = "",
    max_time: int = 300,
) -> Union[List[int], MAPFErrorCodes]:

    open_list = []
    closed_list = {}

    if start not in heuristics:
        return MAPFErrorCodes.START_GOAL_DISCONNECT

    h_value = heuristics[start]
    constraint_table = build_constraint_table(constraints, agent_id)

    root = Node(start, 0, h_value, None, 0)
    heapq.heappush(
        open_list,
        (root.g_value + root.h_value, root.h_value, root.location, root),
    )

    closed_list[(root.location, root.timestep)] = root

    # Add self-loops
    for node in graph.nodes:
        graph.add_edge(node, node, weight=0, cost=0)

    start_time = time.time()
    while len(open_list) != 0 and time.time() - start_time < max_time:

        logging.debug(f"Size of open List: {len(open_list)}")
        current_node = heapq.heappop(open_list)[3]
        logging.debug(f"Current Node: {current_node.location}")
        logging.debug(f"Current Timestep: {current_node.timestep}")
        if current_node.location == goal and not is_constrained(
            goal, goal, current_node.timestep, constraint_table, goal=True
        ):
            return extract_path(current_node)

        for neighbor in graph.neighbors(current_node.location):
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
                    float(graph[current_node.location][successor_location][weighted])
                    if len(weighted) > 0
                    else 1
                )
                successor = Node(
                    successor_location,
                    current_node.g_value + successor_gadd,
                    heuristics[successor_location],
                    current_node,
                    current_node.timestep + 1,
                )

            if is_constrained(
                current_node.location,
                successor.location,
                successor.timestep,
                constraint_table,
            ):
                continue

            if (successor.location, successor.timestep) in closed_list:
                existing_node = closed_list[(successor.location, successor.timestep)]
                if (
                    successor.g_value + successor.h_value
                    < existing_node.g_value + existing_node.h_value
                ):
                    # logging.debug(f"Updating node {successor.location}")
                    closed_list[(successor.location, successor.timestep)] = successor
                    heapq.heappush(
                        open_list,
                        (
                            successor.g_value + successor.h_value,
                            successor.h_value,
                            successor.location,
                            successor,
                        ),
                    )
            else:
                # logging.debug(f"Adding node {successor.location}")
                closed_list[(successor.location, successor.timestep)] = successor
                heapq.heappush(
                    open_list,
                    (
                        successor.g_value + successor.h_value,
                        successor.h_value,
                        successor.location,
                        successor,
                    ),
                )

    if time.time() - start_time > max_time:
        return MAPFErrorCodes.TIMELIMIT_REACHED
    else:
        return MAPFErrorCodes.NO_PATH


def a_star_with_ds(
    agent_id: int,
    graph: Graph,
    start: int,
    goal: int,
    heuristics: Dict[int, float],
    constraints,
    weighted: str = "",
    max_time: int = 300,
):
    start_time = time.time()

    open_list = []
    closed_list = {}

    num_expanded = 0
    num_generated = 0
    max_constraints = 0

    if start not in heuristics:
        return MAPFErrorCodes.START_GOAL_DISCONNECT

    h_value = heuristics[start]
    constraint_table = build_constraint_table_with_ds(constraints, agent_id)
    if constraint_table.keys():
        max_constraints = max(constraint_table.keys())

    root = Node(start, 0, h_value, None, 0)
    if root.location == goal:
        if root.timestep <= max_constraints:
            if not is_constrained_with_ds(
                goal,
                goal,
                root.timestep,
                max_constraints,
                constraint_table,
                agent_id,
                goal=True,
            ):
                max_constraints = 0

    heapq.heappush(
        open_list,
        (root.g_value + root.h_value, root.h_value, root.location, num_generated, root),
    )
    num_generated += 1

    closed_list[(root.location, root.timestep)] = root

    # Add self-loops
    for node in graph.nodes:
        graph.add_edge(node, node, weight=0, cost=0)

    while len(open_list) != 0 and time.time() - start_time < max_time:

        current_node = heapq.heappop(open_list)[4]
        num_expanded += 1

        if current_node.location == goal and not is_constrained_with_ds(
            goal,
            goal,
            current_node.timestep,
            max_constraints,
            constraint_table,
            agent_id,
            goal=True,
        ):
            return extract_path(current_node)

        for neighbor in graph.neighbors(current_node.location):
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
                    float(graph[current_node.location][successor_location][weighted])
                    if len(weighted) > 0
                    else 1
                )
                successor = Node(
                    successor_location,
                    current_node.g_value + successor_gadd,
                    heuristics[successor_location],
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
                existing_node = closed_list[(successor.location, successor.timestep)]
                if (
                    successor.g_value + successor.h_value
                    < existing_node.g_value + existing_node.h_value
                    and successor.g_value < existing_node.g_value
                ):
                    # logging.debug(f"Updating node {successor.location}")
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
    agent_id: int,
    graph: Graph,
    start: int,
    goal: int,
    heuristics: Dict[int, float],
    constraints,
    risk_budget: float,
    weighted: str = "",
    max_time: int = 300,
):
    start_time = time.time()

    open_list = []
    closed_list = {}

    num_expanded = 0
    num_generated = 0
    max_constraints = 0

    if start not in heuristics:
        return MAPFErrorCodes.START_GOAL_DISCONNECT

    h_value = heuristics[start]
    constraint_table = build_constraint_table_with_ds(constraints, agent_id)
    if constraint_table.keys():
        max_constraints = max(constraint_table.keys())

    root = RiskNode(start, 0, h_value, None, 0, 0)
    if root.location == goal:
        if root.timestep <= max_constraints:
            if not is_constrained_with_ds(
                goal,
                goal,
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

    # Add self-loops
    for node in graph.nodes:
        graph.add_edge(node, node, weight=0, cost=0)

    while len(open_list) != 0 and time.time() - start_time < max_time:

        current_node = heapq.heappop(open_list)[-1]
        num_expanded += 1

        # If we have reached the goal and the goal is not constrained and the risk is within the budget
        if (
            current_node.location == goal
            and not is_constrained_with_ds(
                goal,
                goal,
                current_node.timestep,
                max_constraints,
                constraint_table,
                agent_id,
                goal=True,
            )
            and current_node.risk <= risk_budget
        ):
            return extract_path(current_node)

        for neighbor in graph.neighbors(current_node.location):
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
                    float(graph[current_node.location][successor_location][weighted])
                    if len(weighted) > 0
                    else 1
                )

                successor_risk = current_node.risk + float(graph[current_node.location][successor_location]["cost"])
                if successor_risk > risk_budget:
                    continue

                successor = RiskNode(
                    successor_location,
                    current_node.g_value + successor_gadd,
                    heuristics[successor_location],
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

