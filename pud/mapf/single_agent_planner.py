from __future__ import annotations
import copy
import time
import heapq
import numpy as np
from networkx import Graph
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union
from pud.mapf.mapf_exceptions import MAPFErrorCodes
from pud.mapf.utils import PrioritySet, dominate_or_equal


def compute_cost(path: List[int], graph: Graph, edge_attribute: str = ""):
    """
    Compute the cost of the path
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += (
            float(graph[path[i]][path[i + 1]][edge_attribute])
            if len(edge_attribute) > 0
            else 1
        )
    return cost


def compute_sum_of_costs(
    paths: List[List[int]], graph: Graph, edge_attribute: str = ""
) -> float:
    """
    Compute the sum of costs of the paths
    """
    sum_of_costs = 0
    for path in paths:
        sum_of_costs += compute_cost(path, graph, edge_attribute)
    return sum_of_costs


def compute_heuristics(graph: Graph, goal: int, edge_attribute: str = "step"):
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
            edge_cost = graph[location][neighbor][edge_attribute]
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
    def __init__(self, id, location, g_value, h_value, parent, timestep):
        self.id = id
        self.parent = parent
        self.g_value = g_value
        self.h_value = h_value
        self.location = location
        self.timestep = timestep

    def __lt__(self, other):
        if self.g_value + self.h_value == other.g_value + other.h_value:
            if self.h_value == other.h_value:
                if self.timestep == other.timestep:
                    if self.location == other.location:
                        return self.id < other.id
                    return self.location < other.location
                return self.timestep < other.timestep
            return self.h_value < other.h_value
        return self.g_value + self.h_value < other.g_value + other.h_value


class RiskNode(Node):
    def __init__(self, id, location, g_value, h_value, parent, timestep, risk):
        self.risk = risk
        super().__init__(id, location, g_value, h_value, parent, timestep)

    def __lt__(self, other):
        if self.g_value + self.h_value == other.g_value + other.h_value:
            if self.risk == other.risk:
                if self.h_value == other.h_value:
                    if self.timestep == other.timestep:
                        if self.location == other.location:
                            return self.id < other.id
                        return self.location < other.location
                    return self.timestep < other.timestep
                return self.h_value < other.h_value
            return self.risk < other.risk
        return self.g_value + self.h_value < other.g_value + other.h_value


class AStar(object):
    def __init__(
        self, graph: Graph, agent_id: int, start: int, goal: int, config: Dict
    ):
        self.goal = goal
        self.start = start
        self.graph = graph
        self.agent_id = agent_id

        self.num_expanded = 0
        self.num_generated = 0

        self.splitter = config["split_strategy"]
        self.max_distance = config["max_distance"]
        self.use_experience = config["use_experience"]
        self.edge_attributes = config["edge_attributes"]

        self.open_list = []
        self.closed_list = {}

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = compute_heuristics(
                self.graph, self.goal, edge_attribute
            )

    def reset(self) -> None:
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list = []
        self.closed_list = {}

    def push_node(self, node: Node) -> None:
        heapq.heappush(
            self.open_list,
            (
                node.g_value + node.h_value,
                node.h_value,
                node,
            ),
        )
        self.num_generated += 1

    def successor_generator(
        self, current_node: Node, neighbor_location: int, edge_attribute: str = "step"
    ) -> Node:
        if neighbor_location == current_node.location:
            # We are waiting here
            successor = Node(
                parent=current_node,
                id=self.num_generated,
                location=neighbor_location,
                h_value=current_node.h_value,
                g_value=current_node.g_value + 1,
                timestep=current_node.timestep + 1,
            )
        else:
            successor_gadd = self.graph[current_node.location][neighbor_location][
                edge_attribute
            ]

            if edge_attribute == "cost":
                # This ensures that the agent gives more priority to the cost of the edge
                successor_gadd *= 3 * self.max_distance
                # This ensures that the agent makes progress even when the edge attribute is zero
                successor_gadd += 1

            successor = Node(
                parent=current_node,
                id=self.num_generated,
                location=neighbor_location,
                g_value=current_node.g_value + successor_gadd,
                timestep=current_node.timestep + 1,
                h_value=self.heuristic[edge_attribute][neighbor_location],
            )

        return successor

    def add_child(
        self,
        current_node: Node,
        neighbor_location: int,
        constraint_table: Dict[int, List[Dict]],
        max_constraint: int,
        edge_attribute: str = "step",
    ) -> Union[Node, None]:
        successor = self.successor_generator(
            current_node, neighbor_location, edge_attribute
        )

        if is_constrained_with_ds(
            current_node.location,
            successor.location,
            successor.timestep,
            max_constraint,
            constraint_table,
            self.agent_id,
        ):
            return

        if (successor.location, successor.timestep) in self.closed_list:
            existing_node = self.closed_list[(successor.location, successor.timestep)]
            if (
                successor.g_value + successor.h_value
                < existing_node.g_value + existing_node.h_value
                and successor.g_value < existing_node.g_value
            ):
                self.closed_list[(successor.location, successor.timestep)] = successor
                self.push_node(successor)
        else:
            self.closed_list[(successor.location, successor.timestep)] = successor
            self.push_node(successor)

        return successor

    def find_path(
        self,
        constraints: List[Dict],
        experience: Union[List[int], None] = None,
        max_time: int = 300,
        edge_attribute: str = "step",
    ) -> Union[List[int], MAPFErrorCodes]:
        start_time = time.time()
        self.reset()

        if self.start not in self.heuristic[edge_attribute]:
            return MAPFErrorCodes.START_GOAL_DISCONNECT

        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )

        max_constraint = 0
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())

        root = Node(
            g_value=0,
            timestep=0,
            parent=None,
            location=self.start,
            id=self.num_generated,
            h_value=self.heuristic[edge_attribute][self.start],
        )

        if root.location == self.goal:
            if root.timestep <= max_constraint:
                if not is_constrained_with_ds(
                    self.goal,
                    self.goal,
                    root.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                    goal=True,
                ):
                    max_constraint = 0

        self.push_node(root)

        if (
            self.use_experience
            and experience is not None
            and root.location in experience
        ):
            self.push_partial_experience(
                state=root,
                experience=experience,
                max_constraint=max_constraint,
                constraint_table=constraint_table,
                edge_attribute=edge_attribute,
            )
        self.closed_list[(root.location, root.timestep)] = root

        while len(self.open_list) != 0 and time.time() - start_time < max_time:

            current_node = heapq.heappop(self.open_list)[-1]
            self.num_expanded += 1

            if current_node.location == self.goal and not is_constrained_with_ds(
                self.goal,
                self.goal,
                current_node.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
                goal=True,
            ):
                return extract_path(current_node)

            if (
                self.use_experience
                and experience is not None
                and current_node.location in experience
            ):
                self.push_partial_experience(
                    state=current_node,
                    experience=experience,
                    max_constraint=max_constraint,
                    constraint_table=constraint_table,
                    edge_attribute=edge_attribute,
                )

            for neighbor in self.graph.neighbors(current_node.location):
                _ = self.add_child(
                    current_node,
                    neighbor,
                    constraint_table,
                    max_constraint,
                    edge_attribute,
                )

        if time.time() - start_time > max_time:
            return MAPFErrorCodes.TIMELIMIT_REACHED
        else:
            return MAPFErrorCodes.NO_PATH

    def push_partial_experience(
        self,
        state: Node,
        experience: List[int],
        max_constraint: int,
        constraint_table: Dict[int, List[Dict]],
        edge_attribute: str = "step",
    ) -> None:
        for idx, location in enumerate(experience):
            if location == state.location:
                break
        if idx == len(experience) - 1:
            return

        prev_successor = state
        experience_suffix = experience[idx + 1:]

        for successor_location in experience_suffix:
            successor = self.add_child(
                prev_successor,
                successor_location,
                constraint_table,
                max_constraint,
                edge_attribute,
            )
            if successor is None:
                break
            prev_successor = successor


class RiskBudgetedAStar(AStar):
    def __init__(
        self, graph: Graph, agent_id: int, start: int, goal: int, config: Dict
    ):
        super().__init__(graph, agent_id, start, goal, config)

    def push_constrained_node(self, node: RiskNode) -> None:
        heapq.heappush(
            self.open_list,
            (
                node.g_value + node.h_value,
                node.risk,
                node.h_value,
                node,
            ),
        )
        self.num_generated += 1

    def constrained_successor_generator(
        self,
        current_node: RiskNode,
        neighbor_location: int,
        edge_attribute: str = "step",
    ) -> RiskNode:
        if neighbor_location == current_node.location:
            successor_risk = current_node.risk
            successor = RiskNode(
                parent=current_node,
                risk=successor_risk,
                id=self.num_generated,
                location=neighbor_location,
                h_value=current_node.h_value,
                g_value=current_node.g_value + 1,
                timestep=current_node.timestep + 1,
            )
        else:
            successor_gadd = self.graph[current_node.location][neighbor_location][
                edge_attribute
            ]
            successor_risk = (
                current_node.risk
                + self.graph[current_node.location][neighbor_location]["cost"]
            )
            successor = RiskNode(
                parent=current_node,
                risk=successor_risk,
                id=self.num_generated,
                location=neighbor_location,
                timestep=current_node.timestep + 1,
                g_value=current_node.g_value + successor_gadd,
                h_value=self.heuristic[edge_attribute][neighbor_location],
            )

        return successor

    def add_constrained_child(
        self,
        current_node: RiskNode,
        neighbor_location: int,
        risk_budget: float,
        constraint_table: Dict[int, List[Dict]],
        max_constraint: int,
        edge_attribute: str = "step",
    ) -> Union[RiskNode, None]:
        successor = self.constrained_successor_generator(
            current_node, neighbor_location, edge_attribute
        )

        if (
            is_constrained_with_ds(
                current_node.location,
                successor.location,
                successor.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
            )
            or successor.risk > risk_budget
        ):
            return

        if (successor.location, successor.timestep) in self.closed_list:
            existing_node = self.closed_list[(successor.location, successor.timestep)]
            if (
                successor.g_value + successor.h_value
                <= existing_node.g_value + existing_node.h_value
                and successor.g_value <= existing_node.g_value
                and successor.risk < existing_node.risk
            ):
                self.closed_list[(successor.location, successor.timestep)] = successor
                self.push_constrained_node(successor)
        else:
            self.closed_list[(successor.location, successor.timestep)] = successor
            self.push_constrained_node(successor)

        return successor

    def find_constrained_path(
        self,
        constraints: List[Dict],
        risk_budget: float,
        experience: Union[List[int], None] = None,
        max_time: int = 300,
        edge_attribute: str = "step",
    ) -> Union[List[int], MAPFErrorCodes]:
        start_time = time.time()
        self.reset()

        if self.start not in self.heuristic[edge_attribute]:
            return MAPFErrorCodes.START_GOAL_DISCONNECT

        max_constraint = 0
        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())

        root = RiskNode(
            risk=0,
            g_value=0,
            timestep=0,
            parent=None,
            location=self.start,
            id=self.num_generated,
            h_value=self.heuristic[edge_attribute][self.start],
        )

        if root.location == self.goal:
            if root.timestep <= max_constraint:
                if not is_constrained_with_ds(
                    self.goal,
                    self.goal,
                    root.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                    goal=True,
                ):
                    max_constraint = 0

        self.push_node(root)

        if (
            self.use_experience
            and experience is not None
            and root.location in experience
        ):
            self.push_constrained_partial_experience(
                state=root,
                experience=experience,
                risk_budget=risk_budget,
                constraint_table=constraint_table,
                max_constraint=max_constraint,
                edge_attribute=edge_attribute,
            )
        self.closed_list[(root.location, root.timestep)] = root

        while len(self.open_list) != 0 and time.time() - start_time < max_time:

            current_node = heapq.heappop(self.open_list)[-1]
            self.num_expanded += 1

            # If we have reached the goal and the goal is not constrained and the risk is within the budget
            if (
                current_node.location == self.goal
                and not is_constrained_with_ds(
                    self.goal,
                    self.goal,
                    current_node.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                    goal=True,
                )
                and current_node.risk <= risk_budget
            ):
                return extract_path(current_node)

            if (
                self.use_experience
                and experience is not None
                and current_node.location in experience
            ):
                self.push_constrained_partial_experience(
                    state=current_node,
                    experience=experience,
                    risk_budget=risk_budget,
                    constraint_table=constraint_table,
                    max_constraint=max_constraint,
                    edge_attribute=edge_attribute,
                )

            for neighbor in self.graph.neighbors(current_node.location):
                self.add_constrained_child(
                    current_node,
                    neighbor,
                    risk_budget,
                    constraint_table,
                    max_constraint,
                    edge_attribute,
                )

        if time.time() - start_time > max_time:
            return MAPFErrorCodes.TIMELIMIT_REACHED
        else:
            return MAPFErrorCodes.NO_PATH

    def push_constrained_partial_experience(
        self,
        state: RiskNode,
        risk_budget: float,
        experience: List[int],
        max_constraint: int,
        constraint_table: Dict[int, List[Dict]],
        edge_attribute: str = "step",
    ) -> None:
        for idx, location in enumerate(experience):
            if location == state.location:
                break

        if idx == len(experience) - 1:
            return

        prev_successor = state
        experience_suffix = experience[idx + 1:]

        for successor_location in experience_suffix:
            successor = self.add_constrained_child(
                current_node=prev_successor,
                neighbor_location=successor_location,
                risk_budget=risk_budget,
                constraint_table=constraint_table,
                max_constraint=max_constraint,
                edge_attribute=edge_attribute,
            )
            if successor is None:
                break
            prev_successor = successor


class LagrangianAStar(AStar):
    def __init__(
        self,
        graph: Graph,
        agent_id: int,
        start: int,
        goal: int,
        lagrangian: float,
        config: Dict,
    ):
        super().__init__(graph, agent_id, start, goal, config)
        self.lagrangian = lagrangian

        assert (
            len(self.edge_attributes) == 2
        ), "Lagrangian A* requires two edge attributes"
        assert (
            "step" in self.edge_attributes
        ), "Lagrangian A* requires step edge attribute"
        assert (
            "cost" in self.edge_attributes
        ), "Lagrangian A* requires cost edge attribute"

        self.heuristic["step"] = self.compute_heuristics()

    def compute_heuristics(self):
        graph = self.graph.to_undirected()
        open_list = [(0, self.goal)]
        heuristics = {self.goal: 0}
        closed_list = set()

        while open_list:
            cost, location = heapq.heappop(open_list)

            if location in closed_list:
                continue
            closed_list.add(location)

            for neighbor in graph.neighbors(location):
                edge_cost = (
                    graph[location][neighbor]["step"]
                    + self.lagrangian * graph[location][neighbor]["cost"]
                )
                successor_cost = cost + edge_cost

                if neighbor not in heuristics or heuristics[neighbor] > successor_cost:
                    heuristics[neighbor] = successor_cost  # type: ignore
                    heapq.heappush(open_list, (successor_cost, neighbor))  # type: ignore

        return heuristics

    def successor_generator(
        self, current_node: Node, neighbor_location: int, edge_attribute: str = "step"
    ) -> Node:
        successor_gadd = (
            self.graph[current_node.location][neighbor_location][edge_attribute]
            + self.lagrangian
            * self.graph[current_node.location][neighbor_location]["cost"]
        )
        successor_hvalue = (
            self.heuristic[edge_attribute][neighbor_location]
            if neighbor_location != current_node.location
            else current_node.h_value
        )
        successor = Node(
            parent=current_node,
            id=self.num_generated,
            location=neighbor_location,
            h_value=successor_hvalue,
            g_value=current_node.g_value + successor_gadd,
            timestep=current_node.timestep + 1,
        )
        return successor


class MultiObjectiveNode(object):
    def __init__(
        self,
        id: int,
        location: int,
        g_vector: NDArray,
        h_vector: NDArray,
        parent: MultiObjectiveNode | None,
    ):
        self.id = id
        self.g_vector = g_vector
        self.h_vector = h_vector
        self.location = location
        self.parent = parent

    def __lt__(self, other: MultiObjectiveNode) -> bool:
        if np.all(self.g_vector + self.h_vector == other.g_vector + other.h_vector):
            if np.all(self.h_vector == other.h_vector):
                if self.location == other.location:
                    return self.id < other.id
                return self.location < other.location
            return bool(np.all(self.h_vector < other.h_vector))
        return bool(
            np.all(self.g_vector + self.h_vector < other.g_vector + other.h_vector)
        )


class BiObjectiveAStar(object):
    def __init__(
        self, graph: Graph, agent_id: int, start: int, goal: int, config: Dict
    ):
        self.goal = goal
        self.start = start
        self.agent_id = agent_id

        self.graph = graph

        self.num_expanded = 0
        self.num_generated = 0

        self.splitter = config["split_strategy"]
        self.cost_dim = len(config["edge_attributes"])
        self.edge_attributes = config["edge_attributes"]

        self.open_list = PrioritySet()
        self.best_secondary_gs = {}  # location -> g_2_min

        self.solution_nodes = []

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = compute_heuristics(
                self.graph, self.goal, edge_attribute
            )

    def get_frontier_key(self, state: MultiObjectiveNode) -> float:
        return state.location

    def get_heuristic(self, location: int) -> NDArray:
        heuristic = np.zeros(self.cost_dim)
        for idx, edge_attribute in enumerate(self.edge_attributes):
            heuristic[idx] = self.heuristic[edge_attribute][location]
        return heuristic

    def reset(self) -> None:
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list = PrioritySet()
        self.best_secondary_gs = {}

        self.start_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.start,
            g_vector=np.zeros(self.cost_dim),
            h_vector=self.get_heuristic(self.start),
            parent=None,
        )
        self.num_generated += 1
        self.goal_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.goal,
            g_vector=np.ones(self.cost_dim) * np.inf,
            h_vector=self.get_heuristic(self.goal),
            parent=None,
        )

        goal_key = self.get_frontier_key(self.goal_state)
        start_key = self.get_frontier_key(self.start_state)
        self.best_secondary_gs[goal_key] = np.inf
        self.best_secondary_gs[start_key] = np.inf

    def find_path(self, max_time: int = 300):
        start_time = time.time()
        self.reset()

        f_value = self.start_state.g_vector + self.start_state.h_vector
        priority_key = tuple(f_value), self.start_state
        self.open_list.add(priority_key, self.start_state.id)

        while self.open_list.size() != 0 and time.time() - start_time < max_time:
            priority_key, current_node_id = self.open_list.pop()

            current_node = priority_key[-1]
            current_node_key = self.get_frontier_key(current_node)
            self.num_expanded += 1

            if (
                current_node_key in self.best_secondary_gs
                and current_node.g_vector[-1]
                >= self.best_secondary_gs[current_node_key]
            ) or (
                current_node.g_vector[-1] + current_node.h_vector[-1]
                >= self.best_secondary_gs[self.get_frontier_key(self.goal_state)]
            ):
                continue

            self.best_secondary_gs[current_node_key] = current_node.g_vector[-1]

            if current_node.location == self.goal_state.location:
                self.solution_nodes.append(current_node)
                continue

            for neighbor in self.graph.neighbors(current_node.location):
                successor_gadd_vector = []
                for edge_attribute in self.edge_attributes:
                    successor_gadd_vector.append(
                        self.graph[current_node.location][neighbor][edge_attribute]
                    )
                if current_node.location == neighbor:
                    successor_gadd_vector[self.edge_attributes.index("step")] += 1
                successor = MultiObjectiveNode(
                    id=self.num_generated,
                    location=neighbor,
                    g_vector=current_node.g_vector + np.array(successor_gadd_vector),
                    h_vector=self.get_heuristic(neighbor),
                    parent=current_node,
                )

                successor_key = self.get_frontier_key(successor)
                if (
                    successor_key in self.best_secondary_gs
                    and successor.g_vector[-1] >= self.best_secondary_gs[successor_key]
                ) or (
                    successor.g_vector[-1] + successor.h_vector[-1]
                    >= self.best_secondary_gs[self.get_frontier_key(self.goal_state)]
                ):
                    continue

                successor_priority_key = (
                    tuple(successor.g_vector + successor.h_vector),
                    successor,
                )
                self.open_list.add(successor_priority_key, successor.id)
                self.num_generated += 1

        paths = {}
        cost_vectors = {}
        for node in self.solution_nodes:
            path = []
            current_node = node
            cost_vectors[node.id] = node.g_vector
            while current_node is not None:
                path.append(current_node.location)
                current_node = current_node.parent
            path.reverse()
            paths[node.id] = path

        return paths, cost_vectors


class BiObjectiveSpaceTimeAStar(BiObjectiveAStar):
    def __init__(
        self, graph: Graph, agent_id: int, start: int, goal: int, config: Dict
    ):
        super().__init__(graph, agent_id, start, goal, config)

    def reset(self):
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list = PrioritySet()
        self.closed_list = {}
        self.best_secondary_gs = {}

        self.solution_nodes = set()

        self.start_state = MultiObjectiveSpaceTimeNode(
            id=self.num_generated,
            location=self.start,
            g_vector=np.zeros(self.cost_dim),
            h_vector=self.get_heuristic(self.start),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

        self.goal_state = MultiObjectiveSpaceTimeNode(
            id=self.num_generated,
            location=self.goal,
            g_vector=np.ones(self.cost_dim) * np.inf,
            h_vector=self.get_heuristic(self.goal),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

        goal_key = self.get_frontier_key(self.goal_state)
        start_key = self.get_frontier_key(self.start_state)
        self.best_secondary_gs[goal_key] = np.inf
        self.best_secondary_gs[start_key] = np.inf

        for node in self.graph:
            self.best_secondary_gs[node] = np.inf

    def find_path(
        self, constraints: List[Dict], max_time: int = 300
    ) -> Tuple[Dict, Dict]:
        start_time = time.time()
        self.reset()

        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )

        max_constraint = 0
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())

        f_value = self.start_state.g_vector + self.start_state.h_vector
        priority_key = tuple(f_value), self.start_state
        self.open_list.add(priority_key, self.start_state.id)
        self.closed_list[self.start_state.id] = self.start_state

        while self.open_list.size() != 0 and time.time() - start_time < max_time:
            priority_key, current_node_id = self.open_list.pop()

            current_node = priority_key[-1]
            current_node_key = self.get_frontier_key(current_node)
            self.num_expanded += 1

            if (
                current_node_key in self.best_secondary_gs
                and current_node.g_vector[-1]
                >= self.best_secondary_gs[current_node_key]
            ) or (
                current_node.g_vector[-1] + current_node.h_vector[-1]
                >= self.best_secondary_gs[self.get_frontier_key(self.goal_state)]
            ):
                continue

            self.best_secondary_gs[current_node_key] = current_node.g_vector[-1]

            if current_node.location == self.goal and not is_constrained_with_ds(
                self.goal,
                self.goal,
                current_node.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
                goal=True,
            ):
                self.solution_nodes.add(current_node.id)
                temporary_set = copy.deepcopy(self.solution_nodes)
                for goal_state_id in temporary_set:
                    if goal_state_id == current_node.id:
                        continue
                    goal_state = self.closed_list[goal_state_id]
                    if goal_state.g_vector[-1] >= current_node.g_vector[-1]:
                        self.solution_nodes.remove(goal_state_id)
                continue

            for neighbor in self.graph.neighbors(current_node.location):
                successor_gadd_vector = []
                for edge_attribute in self.edge_attributes:
                    successor_gadd_vector.append(
                        self.graph[current_node.location][neighbor][edge_attribute]
                    )
                if current_node.location == neighbor:
                    successor_gadd_vector[self.edge_attributes.index("step")] += 1

                successor = MultiObjectiveSpaceTimeNode(
                    id=self.num_generated,
                    location=neighbor,
                    g_vector=current_node.g_vector + np.array(successor_gadd_vector),
                    h_vector=self.get_heuristic(neighbor),
                    parent=current_node,
                    timestep=current_node.timestep + 1,
                )

                if is_constrained_with_ds(
                    current_node.location,
                    successor.location,
                    successor.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                ):
                    continue

                successor_key = self.get_frontier_key(successor)
                if (
                    successor_key in self.best_secondary_gs
                    and successor.g_vector[-1] >= self.best_secondary_gs[successor_key]
                ) or (
                    successor.g_vector[-1] + successor.h_vector[-1]
                    >= self.best_secondary_gs[self.get_frontier_key(self.goal_state)]
                ):
                    continue

                successor_priority_key = (
                    tuple(successor.g_vector + successor.h_vector),
                    successor,
                )
                self.open_list.add(successor_priority_key, successor.id)
                self.closed_list[successor.id] = successor
                self.num_generated += 1

        paths = {}
        cost_vectors = {}
        for node_id in self.solution_nodes:
            path = []
            current_node = self.closed_list[node_id]
            cost_vectors[node_id] = current_node.g_vector
            while current_node is not None:
                path.append(current_node.location)
                current_node = current_node.parent
            path.reverse()
            paths[node_id] = path

        return paths, cost_vectors


class MultiObjectiveAStar(object):
    def __init__(
        self, graph: Graph, agent_id: int, start: int, goal: int, config: Dict
    ):
        self.goal = goal
        self.start = start
        self.agent_id = agent_id

        self.graph = graph

        self.num_expanded = 0
        self.num_generated = 0

        self.splitter = config["split_strategy"]
        self.cost_dim = len(config["edge_attributes"])
        self.edge_attributes = config["edge_attributes"]

        self.open_list = PrioritySet()
        self.closed_list = {}
        self.frontier_map = {}

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = compute_heuristics(
                self.graph, self.goal, edge_attribute
            )

    def get_frontier_key(self, state: MultiObjectiveNode) -> float:
        return state.location

    def get_heuristic(self, location: int) -> NDArray:
        heuristic = np.zeros(self.cost_dim)
        for idx, edge_attribute in enumerate(self.edge_attributes):
            heuristic[idx] = self.heuristic[edge_attribute][location]
        return heuristic

    def push_node(self, node: MultiObjectiveNode) -> None:
        f_value = np.sum(node.g_vector + node.h_vector)
        self.open_list.add((f_value, np.sum(node.h_vector), node), node.id)
        self.add_to_frontier(node)
        self.num_generated += 1

    def filter_frontier_state(self, state: MultiObjectiveNode) -> bool:
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            return False
        for existing_state_id in self.frontier_map[state_key]:
            if existing_state_id == state.id:
                continue
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(existing_state.g_vector, state.g_vector):
                return True
        return False

    def filter_goal_state(self, state: MultiObjectiveNode) -> bool:
        goal_state_key = self.get_frontier_key(self.goal_state)
        if goal_state_key not in self.frontier_map:
            return False
        for existing_state_id in self.frontier_map[goal_state_key]:
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(
                existing_state.g_vector + existing_state.h_vector,
                state.g_vector + state.h_vector,
            ):
                return True
            if dominate_or_equal(existing_state.g_vector, state.g_vector):
                return True
        return False

    def filter_state(self, state: MultiObjectiveNode) -> bool:
        if self.filter_frontier_state(state):
            return True
        if self.filter_goal_state(state):
            return True
        return False

    def pruning(self, state: MultiObjectiveNode) -> bool:
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            return False
        for existing_state_id in self.frontier_map[state_key]:
            if existing_state_id == state.id:
                continue
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(existing_state.g_vector, state.g_vector):
                return True
        return False

    def reset(self) -> None:
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list = PrioritySet()
        self.closed_list = {}
        self.frontier_map = {}

        self.start_state = MultiObjectiveNode(
            id=self.num_generated,
            location=self.start,
            g_vector=np.zeros(self.cost_dim),
            h_vector=self.get_heuristic(self.start),
            parent=None,
        )
        self.num_generated += 1
        self.goal_state = MultiObjectiveNode(
            id=1,
            location=self.goal,
            g_vector=np.ones(self.cost_dim) * np.inf,
            h_vector=self.get_heuristic(self.goal),
            parent=None,
        )
        self.num_generated += 1

    def add_to_frontier(self, state: MultiObjectiveNode) -> None:
        self.closed_list[state.id] = state
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            self.frontier_map[state_key] = set()
            self.frontier_map[state_key].add(state.id)
        else:
            self.refine_frontier(state)
            self.frontier_map[state_key].add(state.id)

    def refine_frontier(self, state: MultiObjectiveNode) -> None:
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            return
        temporary_frontier = copy.deepcopy(self.frontier_map[state_key])
        for existing_state_id in temporary_frontier:
            if existing_state_id == state.id:
                continue
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(state.g_vector, existing_state.g_vector):
                self.frontier_map[state_key].remove(existing_state_id)
                self.open_list.remove(existing_state_id)

    def reconstruct_paths(self) -> Dict:
        paths = {}
        goal_state_key = self.get_frontier_key(self.goal_state)
        if goal_state_key not in self.frontier_map:
            return paths
        for goal_state_id in self.frontier_map[goal_state_key]:
            goal_state = self.closed_list[goal_state_id]
            path = []
            current_state = goal_state
            while current_state is not None:
                path.append(current_state.location)
                current_state = current_state.parent
            path.reverse()
            paths[goal_state_id] = path
        return paths

    def find_path(self, max_time: int = 300):
        start_time = time.time()
        self.reset()
        self.push_node(self.start_state)

        while self.open_list.size() != 0 and time.time() - start_time < max_time:
            _, current_node_id = self.open_list.pop()
            current_node = self.closed_list[current_node_id]
            self.num_expanded += 1

            if self.filter_state(current_node):
                continue

            for neighbor in self.graph.neighbors(current_node.location):
                successor_gadd_vector = []
                for edge_attribute in self.edge_attributes:
                    successor_gadd_vector.append(
                        self.graph[current_node.location][neighbor][edge_attribute]
                    )
                successor = MultiObjectiveNode(
                    id=self.num_generated,
                    location=neighbor,
                    g_vector=current_node.g_vector + np.array(successor_gadd_vector),
                    h_vector=self.get_heuristic(neighbor),
                    parent=current_node,
                )

                if self.filter_state(successor):
                    continue

                if not self.pruning(successor):
                    self.push_node(successor)

        paths = self.reconstruct_paths()
        cost_vectors = {}
        for id, _ in paths.items():
            cost_vectors[id] = self.closed_list[id].g_vector
        return paths, cost_vectors


class MultiObjectiveSpaceTimeNode(MultiObjectiveNode):
    def __init__(
        self,
        id: int,
        location: int,
        g_vector: NDArray,
        h_vector: NDArray,
        parent: MultiObjectiveSpaceTimeNode | None,
        timestep: int,
    ):
        super().__init__(id, location, g_vector, h_vector, parent)
        self.timestep = timestep

    def __lt__(self, other: MultiObjectiveSpaceTimeNode) -> bool:
        if np.all(self.g_vector + self.h_vector == other.g_vector + other.h_vector):
            if np.all(self.h_vector == other.h_vector):
                if self.timestep == other.timestep:
                    if self.location == other.location:
                        return self.id < other.id
                    return self.location < other.location
                return self.timestep < other.timestep
            return bool(np.all(self.h_vector < other.h_vector))
        return bool(
            np.all(self.g_vector + self.h_vector < other.g_vector + other.h_vector)
        )


class MultiObjectiveSpaceTimeAStar(MultiObjectiveAStar):
    def __init__(
        self, graph: Graph, agent_id: int, start: int, goal: int, config: Dict
    ):
        super().__init__(graph, agent_id, start, goal, config)
        self.goal_states = set()

        self.heuristic = {}
        for edge_attribute in self.edge_attributes:
            self.heuristic[edge_attribute] = compute_heuristics(
                self.graph, self.goal, edge_attribute
            )

    def reset(self):
        super().reset()
        self.num_generated = 0
        self.goal_states = set()

        self.start_state = MultiObjectiveSpaceTimeNode(
            id=self.num_generated,
            location=self.start,
            g_vector=np.zeros(self.cost_dim),
            h_vector=self.get_heuristic(self.start),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1
        self.goal_state = MultiObjectiveSpaceTimeNode(
            id=1,
            location=self.goal,
            g_vector=np.ones(self.cost_dim) * np.inf,
            h_vector=self.get_heuristic(self.goal),
            parent=None,
            timestep=0,
        )
        self.num_generated += 1

    def filter_frontier_state(self, state: MultiObjectiveSpaceTimeNode) -> bool:
        state_key = self.get_frontier_key(state)
        if state_key not in self.frontier_map:
            return False
        for existing_state_id in self.frontier_map[state_key]:
            if existing_state_id == state.id:
                continue
            existing_state = self.closed_list[existing_state_id]
            if dominate_or_equal(
                existing_state.g_vector + existing_state.h_vector,
                state.g_vector + state.h_vector,
            ):
                return True
            if dominate_or_equal(existing_state.g_vector, state.g_vector):
                return True
        return False

    def filter_goal_state(self, state: MultiObjectiveSpaceTimeNode) -> bool:
        for goal_state_id in self.goal_states:
            goal_state = self.closed_list[goal_state_id]
            if dominate_or_equal(
                goal_state.g_vector + goal_state.h_vector,
                state.g_vector + state.h_vector,
            ):
                return True
            if dominate_or_equal(goal_state.g_vector, state.g_vector):
                return True
        return False

    def get_heuristic(self, location: int) -> NDArray:
        h_val = np.zeros(self.cost_dim)
        for idx, edge_attribute in enumerate(self.edge_attributes):
            h_val[idx] = self.heuristic[edge_attribute][location]
        return h_val

    def get_frontier_key(self, state: MultiObjectiveSpaceTimeNode) -> Tuple[int, int]:
        return (state.location, state.timestep)

    def refine_reached_goals(self, state: MultiObjectiveSpaceTimeNode) -> None:
        temporary_set = copy.deepcopy(self.goal_states)
        for goal_state_id in temporary_set:
            if goal_state_id == state.id:
                continue
            goal_state = self.closed_list[goal_state_id]
            if dominate_or_equal(state.g_vector, goal_state.g_vector):
                self.goal_states.remove(goal_state_id)

    def reconstruct_paths(self) -> Dict:
        paths = {}
        for goal_state_id in self.goal_states:
            goal_state = self.closed_list[goal_state_id]
            path = []
            current_state = goal_state
            while current_state is not None:
                path.append(current_state.location)
                current_state = current_state.parent
            path.reverse()
            paths[goal_state_id] = path
        return paths

    def find_path(
        self, constraints: List[Dict], max_time: int = 300
    ) -> Tuple[Dict, Dict]:
        start_time = time.time()
        self.reset()

        if self.splitter == "standard":
            constraint_table = build_constraint_table(constraints, self.agent_id)
        elif self.splitter == "disjoint":
            constraint_table = build_constraint_table_with_ds(
                constraints, self.agent_id
            )

        max_constraint = 0
        if constraint_table.keys():
            max_constraint = max(constraint_table.keys())
        self.push_node(self.start_state)

        while self.open_list.size() != 0 and time.time() - start_time < max_time:
            _, current_node_id = self.open_list.pop()
            current_node = self.closed_list[current_node_id]
            self.num_expanded += 1

            if self.filter_state(current_node):
                continue

            if current_node.location == self.goal and not is_constrained_with_ds(
                self.goal,
                self.goal,
                current_node.timestep,
                max_constraint,
                constraint_table,
                self.agent_id,
                goal=True,
            ):
                self.goal_states.add(current_node.id)
                self.refine_reached_goals(current_node)

            for neighbor in self.graph.neighbors(current_node.location):
                successor_gadd_vector = []
                for edge_attribute in self.edge_attributes:
                    successor_gadd_vector.append(
                        self.graph[current_node.location][neighbor][edge_attribute]
                    )
                if current_node.location == neighbor:
                    successor_gadd_vector[self.edge_attributes.index("step")] += 1
                successor = MultiObjectiveSpaceTimeNode(
                    id=self.num_generated,
                    location=neighbor,
                    g_vector=current_node.g_vector + np.array(successor_gadd_vector),
                    h_vector=self.get_heuristic(neighbor),
                    parent=current_node,
                    timestep=current_node.timestep + 1,
                )

                if is_constrained_with_ds(
                    current_node.location,
                    successor.location,
                    successor.timestep,
                    max_constraint,
                    constraint_table,
                    self.agent_id,
                ):
                    continue

                if self.filter_state(successor):
                    continue

                if not self.pruning(successor):
                    self.push_node(successor)

        paths = self.reconstruct_paths()
        cost_vectors = {}
        for id, _ in paths.items():
            cost_vectors[id] = self.closed_list[id].g_vector
        return paths, cost_vectors
