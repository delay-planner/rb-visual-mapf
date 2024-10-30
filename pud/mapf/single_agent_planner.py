from __future__ import annotations
import time
import heapq
from networkx import Graph
from typing import Dict, List, Union
from pud.mapf.mapf_exceptions import MAPFErrorCodes


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
            if edge_attribute == "cost":
                # This ensures that the agent gives more priority to the cost of the edge
                successor_gadd *= 3 * self.max_distance
                # This ensures that the agent makes progress even when the edge attribute is zero
                successor_gadd += 1

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
