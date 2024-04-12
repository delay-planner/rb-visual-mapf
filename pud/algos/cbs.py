import heapq
import random
import numpy as np
import networkx as nx
from networkx import Graph
from numpy.typing import NDArray
from typing import List, Union, Tuple, Dict
from pud.algos.single_agent_planner import (
    a_star,
    compute_heuristics,
    compute_sum_of_costs,
)


def detect_collision(
    pathA: List[int], pathB: List[int], graph_waypoints: NDArray
) -> Union[List[Tuple[List, int, str]], None]:

    collisions = []
    path1 = pathA.copy()
    path2 = pathB.copy()
    if len(path1) >= len(path2):
        short_path = path2
        long_path = path1
    else:
        short_path = path1
        long_path = path2

    for _ in range(len(long_path) - len(short_path)):
        short_path.append(short_path[-1])

    for timestep in range(len(path1)):
        # if path1[timestep] == path2[timestep]:
        #     collisions.append(([path1[timestep]], timestep, "vertex"))
        #     # return [path1[timestep]], timestep, "vertex"
        # if timestep < len(path1) - 1:
        #     if (
        #         path1[timestep] == path2[timestep + 1]
        #         and path1[timestep + 1] == path2[timestep]
        #     ):
        #         collisions.append(
        #             ([path1[timestep], path1[timestep + 1]], timestep + 1, "edge")
        #         )
        #         # return (
        #         #     [path1[timestep], path1[timestep + 1]],
        #         #     timestep + 1,
        #         #     "edge",
        #         # )
        if (
            np.linalg.norm(
                graph_waypoints[path1[timestep]] - graph_waypoints[path2[timestep]]
            )
            <= 0.5
        ):
            print("Colliding the low-level space")
            collisions.append(([path1[timestep]], timestep, "vertex"))
            # return [path1[timestep]], timestep, "vertex"
        if timestep < len(path1) - 1:
            if (
                np.linalg.norm(
                    graph_waypoints[path1[timestep]]
                    - graph_waypoints[path2[timestep + 1]]
                )
                <= 0.5
                and np.linalg.norm(
                    graph_waypoints[path1[timestep + 1]]
                    - graph_waypoints[path2[timestep]]
                )
                <= 0.5
            ):
                collisions.append(
                    (
                        [
                            path1[timestep],
                            path1[timestep + 1],
                            path2[timestep],
                            path2[timestep + 1],
                        ],
                        timestep + 1,
                        "edge",
                    )
                )
                # return (
                #     [path1[timestep], path1[timestep + 1], path2[timestep], path2[timestep + 1]],
                #     timestep + 1,
                #     "edge",
                # )

    return None


def detect_collisions(paths: List[List[int]], graph_waypoints: NDArray) -> List[Dict]:
    agg_collisions = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            collisions = detect_collision(paths[i], paths[j], graph_waypoints)
            # collision = detect_collision(paths[i], paths[j])
            if collisions is not None:
                for collision in collisions:
                    agg_collisions.append(
                        {
                            "agent_A": i,
                            "agent_B": j,
                            "location": collision[0],
                            "timestep": collision[1],
                            "type": collision[2],
                        }
                    )
    return agg_collisions


def standard_split(collision: Dict) -> List[Dict]:
    constraints = []

    if collision["type"] == "vertex":
        constraints.append(
            {
                "agent_id": collision["agent_A"],
                "location": collision["location"],
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )
        constraints.append(
            {
                "agent_id": collision["agent_B"],
                "location": collision["location"],
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )
    elif collision["type"] == "edge":
        constraints.append(
            {
                "agent_id": collision["agent_A"],
                "location": collision["location"],
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )
        constraints.append(
            {
                "agent_id": collision["agent_B"],
                "location": list(reversed(collision["location"])),
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )

    return constraints


def disjoint_split(collision: Dict) -> List[Dict]:
    agents = [collision["agent_A"], collision["agent_B"]]
    agent_choice = random.randint(0, 1)
    agent = agents[agent_choice]
    location = (
        collision["location"]
        if agent_choice == 0
        else list(reversed(collision["location"]))
    )
    return [
        {
            "agent_id": agent,
            "location": location,
            "timestep": collision["timestep"],
            "positive": True,
            "final": False,
        },
        {
            "agent_id": agent,
            "location": location,
            "timestep": collision["timestep"],
            "positive": False,
            "final": False,
        },
    ]


def to_inflate(
    constraint: Dict,
    graph: Graph,
    paths: List[List[int]],
    radius: int = 3,
):

    for idx, other_agent_path in enumerate(paths):
        if idx == constraint["agent_id"]:
            continue

        nearest_nodes = nx.ego_graph(
            graph, constraint["location"][0], radius=radius
        ).nodes
        for node in nearest_nodes:
            test_path = nx.shortest_path(
                graph, source=constraint["location"][0], target=node
            )
            for i in range(len(test_path) - 1):
                if test_path[i + 1] in other_agent_path[constraint["timestep"] :]:
                    return True
    return False


class CBSSolver(object):

    def __init__(
        self,
        graph: Graph,
        graph_waypoints: NDArray,
        starts: List[int],
        goals: List[int],
        disjoint: bool = False,
        seed: Union[int, None] = None,
        max_steps: Union[int, None] = None,
    ):

        if seed is not None:
            random.seed(seed)
        self.graph = graph
        self.goals = goals
        self.starts = starts
        self.disjoint = disjoint
        self.num_agents = len(starts)
        self.graph_waypoints = graph_waypoints

        self.open_list = []
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(self.graph, goal))

        self.num_expanded = 0
        self.num_generated = 0

    def find_paths(self) -> List[List[int]]:
        print("Finding paths using CBS Solver")
        root = {
            "cost": 0,
            "paths": [],
            "collisions": [],
            "constraints": [],
        }

        for i in range(self.num_agents):
            agent_path = a_star(
                i,
                self.graph,
                self.starts[i],
                self.goals[i],
                self.heuristics[i],
                root["constraints"],
            )
            if agent_path is None:
                raise BaseException("No path found for agent {}".format(i))

            root["paths"].append(agent_path)

        root["cost"] = compute_sum_of_costs(root["paths"], self.graph)
        root["collisions"] = detect_collisions(root["paths"], self.graph_waypoints)

        print(root["collisions"])
        for collision in root["collisions"]:
            print(standard_split(collision))

        heapq.heappush(
            self.open_list,
            (root["cost"], len(root["collisions"]), self.num_generated, root),
        )
        print("Generated: ", self.num_generated)
        self.num_generated += 1

        while len(self.open_list) > 0:
            id, current_node = heapq.heappop(self.open_list)[2:]
            print("Expanded: ", id)
            self.num_expanded += 1

            if len(current_node["collisions"]) == 0:
                return current_node["paths"]

            collision = random.choice(current_node["collisions"])
            constraints = (
                disjoint_split(collision)
                if self.disjoint
                else standard_split(collision)
            )

            for constraint in constraints:
                successor = {
                    "cost": 0,
                    "paths": current_node["paths"].copy(),
                    "collisions": [],
                    "constraints": [*current_node["constraints"], constraint],
                }

                # radius = 3
                # inflate = to_inflate(constraint, self.graph, successor["paths"], radius)
                # if inflate:
                #     print("Inflated!!")
                #     print("---" * 50)

                # update_h = self.heuristics[constraint["agent_id"]].copy()
                # if inflate:
                #     for key, value in update_h.items():
                #         if value < radius:
                #             update_h[key] = value + 10

                agent_path = a_star(
                    constraint["agent_id"],
                    self.graph,
                    self.starts[constraint["agent_id"]],
                    self.goals[constraint["agent_id"]],
                    (
                        self.heuristics[constraint["agent_id"]]
                        # if not inflate
                        # else update_h
                    ),
                    successor["constraints"],
                )

                skip = False
                if agent_path is None:
                    raise BaseException(
                        "No path found for agent {}".format(constraint["agent_id"])
                    )
                else:
                    print(agent_path)
                    successor["paths"][constraint["agent_id"]] = agent_path
                    if constraint["positive"]:
                        violating_agents = []
                        if len(constraint["location"]) == 1:
                            for agent in range(self.num_agents):
                                if (
                                    constraint["location"][0]
                                    == successor["paths"][agent][constraint["timestep"]]
                                ):
                                    violating_agents.append(agent)
                        else:
                            for agent in range(self.num_agents):
                                if (
                                    constraint["location"]
                                    == [
                                        successor["paths"][agent][
                                            constraint["timestep"] - 1
                                        ],
                                        successor["paths"][agent][
                                            constraint["timestep"]
                                        ],
                                    ]
                                    or constraint["location"][0]
                                    == successor["paths"][agent][
                                        constraint["timestep"] - 1
                                    ]
                                    or constraint["location"][1]
                                    == successor["paths"][agent][constraint["timestep"]]
                                ):
                                    violating_agents.append(agent)

                        for agent in violating_agents:
                            constraint_copy = constraint.copy()
                            constraint_copy["agent_id"] = agent
                            constraint_copy["positive"] = False
                            successor["constraints"].append(constraint_copy)
                            agent_path = a_star(
                                agent,
                                self.graph,
                                self.starts[agent],
                                self.goals[agent],
                                self.heuristics[agent],
                                successor["constraints"],
                            )

                            if agent_path is None:
                                skip = True
                                break
                            else:
                                print(agent_path)
                                successor["paths"][agent] = agent_path

                    if not skip:
                        successor["collisions"] = detect_collisions(
                            successor["paths"], self.graph_waypoints
                        )
                        successor["cost"] = compute_sum_of_costs(
                            successor["paths"], self.graph
                        )
                        heapq.heappush(
                            self.open_list,
                            (
                                successor["cost"],
                                len(successor["collisions"]),
                                self.num_generated,
                                successor,
                            ),
                        )
                        print("Generated: ", self.num_generated)
                        self.num_generated += 1

        raise BaseException("No solution found")
