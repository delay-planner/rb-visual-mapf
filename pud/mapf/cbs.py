import heapq
import random
import logging
import time
import numpy as np
from networkx import Graph
from numpy.typing import NDArray
from typing import List, Union, Dict
from pud.mapf.single_agent_planner import (
    a_star,
    compute_heuristics,
    compute_sum_of_costs,
)


def location_collision(path1: List[int], path2: List[int], timestep: int):

    position1 = get_location(path1, timestep)
    position2 = get_location(path2, timestep)

    if position1 == position2:
        return [position1], timestep, "vertex"
    if timestep < len(path1) - 1:
        next_position1 = get_location(path1, timestep + 1)
        next_position2 = get_location(path2, timestep + 1)
        if position1 == next_position2 and position2 == next_position1:
            return [position1, next_position1], timestep + 1, "edge"

    return None


def radius_collision(path1: List[int], path2: List[int], timestep: int, graph_waypoints: NDArray, radius: float = 0.1):
    if (
        np.linalg.norm(
            graph_waypoints[path1[timestep]] - graph_waypoints[path2[timestep]]
        )
        <= radius
    ):
        return [path1[timestep]], timestep, "vertex"

    if timestep < len(path1) - 1:
        if (
            np.linalg.norm(
                graph_waypoints[path1[timestep]]
                - graph_waypoints[path2[timestep + 1]]
            )
            <= radius
            and np.linalg.norm(
                graph_waypoints[path1[timestep + 1]]
                - graph_waypoints[path2[timestep]]
            )
            <= radius
        ):
            return (
                [path1[timestep], path1[timestep + 1]],
                timestep + 1,
                "edge",
            )

    return None


def detect_collision(
    pathA: List[int], pathB: List[int], graph_waypoints: NDArray, collision_radius=0.1
):

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
        collided = None
        if collision_radius > 0:
            collided = radius_collision(path1, path2, timestep, graph_waypoints, collision_radius)
        else:
            collided = location_collision(path1, path2, timestep)

        if collided is not None:
            return collided

    return None


def detect_collisions(
    paths: List[List[int]], graph_waypoints: NDArray, collision_radius=0.1
) -> List[Dict]:
    agg_collisions = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            collisions = detect_collision(
                paths[i], paths[j], graph_waypoints, collision_radius
            )
            if collisions is not None:
                agg_collisions.append(
                    {
                        "agent_A": i,
                        "agent_B": j,
                        "location": collisions[0],
                        "timestep": collisions[1],
                        "type": collisions[2],
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


def get_location(path, timestep):
    if timestep < 0:
        return path[0]
    elif timestep < len(path):
        return path[timestep]
    else:
        return path[-1]


class CBSSolver(object):

    def __init__(
        self,
        graph: Graph,
        graph_waypoints: NDArray,
        starts: List[int],
        goals: List[int],
        disjoint: bool = False,
        seed: Union[int, None] = None,
        weighted: bool = False,
        collision_radius=0.1,
        max_time: int = 300,
    ):

        if seed is not None:
            random.seed(seed)
        self.graph = graph
        self.goals = goals
        self.max_time = max_time
        self.starts = starts
        self.weighted = weighted
        self.disjoint = disjoint
        self.num_agents = len(starts)
        self.graph_waypoints = graph_waypoints
        self.collision_radius = collision_radius

        self.open_list = []
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(
                compute_heuristics(self.graph, goal, weighted=self.weighted)
            )

        self.num_expanded = 0
        self.num_generated = 0

    def find_paths(self) -> List[List[int]]:
        self.start_time = time.time()
        logging.debug("Finding paths using CBS Solver")
        root = {
            "cost": 0,
            "paths": [],
            "collisions": [],
            "constraints": [],
        }

        for i in range(self.num_agents):
            logging.debug("Computing paths for agent {}".format(i))
            agent_path = a_star(
                i,
                self.graph,
                self.starts[i],
                self.goals[i],
                self.heuristics[i],
                root["constraints"],
                weighted=self.weighted,
            )
            if agent_path is None:
                raise RuntimeError("No path found for agent {}".format(i))

            root["paths"].append(agent_path)

        root["cost"] = compute_sum_of_costs(root["paths"], self.graph, weighted=self.weighted)
        root["collisions"] = detect_collisions(
            root["paths"], self.graph_waypoints, self.collision_radius
        )

        heapq.heappush(
            self.open_list,
            (root["cost"], len(root["collisions"]), self.num_generated, root),
        )
        logging.debug("Generated: {}".format(self.num_generated))
        self.num_generated += 1

        while len(self.open_list) > 0 and time.time() - self.start_time < self.max_time:
            id, current_node = heapq.heappop(self.open_list)[2:]
            logging.debug("Expanded: {}".format(id))
            self.num_expanded += 1

            if len(current_node["collisions"]) == 0:
                return current_node

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

                agent_path = a_star(
                    constraint["agent_id"],
                    self.graph,
                    self.starts[constraint["agent_id"]],
                    self.goals[constraint["agent_id"]],
                    self.heuristics[constraint["agent_id"]],
                    successor["constraints"],
                    weighted=self.weighted,
                )

                skip = False
                if agent_path is None:
                    raise RuntimeError(
                        "No path found for agent {}".format(constraint["agent_id"])
                    )
                else:
                    successor["paths"][constraint["agent_id"]] = agent_path
                    if constraint["positive"]:
                        violating_agents = []
                        if len(constraint["location"]) == 1:
                            for agent in range(self.num_agents):
                                if (
                                    constraint["location"][0]
                                    == get_location(successor["paths"]["agent"], constraint["timestep"])
                                ):
                                    violating_agents.append(agent)
                        else:
                            for agent in range(self.num_agents):
                                successor_path_location = [
                                    get_location(successor["paths"][agent], constraint["timestep"] - 1),
                                    get_location(successor["paths"][agent], constraint["timestep"])
                                ]
                                if (
                                    constraint["location"] == successor_path_location
                                    or constraint["location"][0] == successor_path_location[0]
                                    or constraint["location"][1] == successor_path_location[1]
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
                                weighted=self.weighted,
                            )

                            if agent_path is None:
                                skip = True
                                break
                            else:
                                successor["paths"][agent] = agent_path

                    if not skip:
                        successor["collisions"] = detect_collisions(
                            successor["paths"], self.graph_waypoints, self.collision_radius
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
                        logging.debug("Generated: {}".format(self.num_generated))
                        self.num_generated += 1

        raise RuntimeError("Timelimit exceeded")
