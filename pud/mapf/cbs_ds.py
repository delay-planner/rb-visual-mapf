import heapq
import random
import logging
import time
from networkx import Graph
from typing import List, Union
from numpy.typing import NDArray

from pud.mapf.cbs import CBSSolver, detect_collisions, get_location
from pud.mapf.single_agent_planner import (
    a_star_with_ds,
    compute_sum_of_costs,
)


def disjoint_split(collision):
    constraints = []
    choice = random.randint(0, 1)
    agents = [collision["agent_A"], collision["agent_B"]]
    agent = agents[choice]

    if len(collision["location"]) == 1:  # Vertex conflict
        constraints.append({
            "agent_id": agent,
            "location": collision["location"],
            "timestep": collision["timestep"],
            "positive": True,
            "final": False,
        })
        constraints.append({
            "agent_id": agent,
            "location": collision["location"],
            "timestep": collision["timestep"],
            "positive": False,
            "final": False,
        })
    else:  # Edge conflict
        location = collision["location"] if choice == 0 else list(reversed(collision["location"]))
        constraints.append({
            "agent_id": agent,
            "location": location,
            "timestep": collision["timestep"],
            "positive": True,
            "final": False,
        })
        constraints.append({
            "agent_id": agent,
            "location": location,
            "timestep": collision["timestep"],
            "positive": False,
            "final": False,
        })

    return constraints


class CBSDSSolver(CBSSolver):

    def __init__(
        self,
        graph: Graph,
        graph_waypoints: NDArray,
        starts: List[int],
        goals: List[int],
        disjoint: bool = True,
        seed: Union[int, None] = None,
        weighted: bool = False,
        collision_radius=0.1,
        max_time: int = 300,
    ):
        assert disjoint is True, "CBS-DS only supports disjoint splitting"
        super().__init__(
            seed=seed,
            graph=graph,
            goals=goals,
            starts=starts,
            disjoint=disjoint,
            weighted=weighted,
            max_time=max_time,
            graph_waypoints=graph_waypoints,
            collision_radius=collision_radius,
        )

    def find_paths(self):
        self.start_time = time.time()
        logging.debug("Finding paths using CBS-DS solver")
        root = {
            "cost": 0,
            "paths": [],
            "collisions": [],
            "constraints": [],
        }

        for i in range(self.num_agents):
            logging.debug("Computing paths for agent {}".format(i))
            agent_path = a_star_with_ds(
                i,
                self.graph,
                self.starts[i],
                self.goals[i],
                self.heuristics[i],
                root["constraints"],
                weighted=self.weighted,
            )
            if agent_path is None:
                return RuntimeError("No path found for agent {}".format(i))

            root["paths"].append(agent_path)

        root["cost"] = compute_sum_of_costs(root["paths"], self.graph, weighted=self.weighted)
        root["collisions"] = detect_collisions(root["paths"], self.graph_waypoints, self.collision_radius)

        heapq.heappush(
            self.open_list,
            (root["cost"], len(root["collisions"]), self.num_generated, root)
        )
        logging.debug("Generated: {}".format(self.num_generated))
        self.num_generated += 1

        while len(self.open_list) > 0 and time.time() - self.start_time < self.max_time:
            id, current_node = heapq.heappop(self.open_list)[2:]
            logging.debug("Expanding: {}".format(id))
            self.num_expanded += 1

            if len(current_node["collisions"]) == 0:
                return current_node

            collision = random.choice(current_node["collisions"])
            constraints = disjoint_split(collision)

            for constraint in constraints:
                successor = {
                    "cost": 0,
                    "paths": current_node["paths"].copy(),
                    "collisions": [],
                    "constraints": [constraint],
                }

                for c in current_node["constraints"]:
                    if c not in successor["constraints"]:
                        successor["constraints"].append(c)

                constraint_agent = constraint["agent_id"]
                agent_path = a_star_with_ds(
                    constraint_agent,
                    self.graph,
                    self.starts[constraint_agent],
                    self.goals[constraint_agent],
                    self.heuristics[constraint_agent],
                    successor["constraints"],
                    weighted=self.weighted,
                )

                if agent_path is not None:
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

                            agent_path = a_star_with_ds(
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
                            else:
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
                        (successor["cost"], len(successor["collisions"]), self.num_generated, successor)
                    )
                    logging.debug("Generated: {}".format(self.num_generated))
                    self.num_generated += 1

        return RuntimeError("Timelimit exceeded")
