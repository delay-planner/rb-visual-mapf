import heapq
import random
import numpy as np
from typing import List, Dict
from numpy.typing import NDArray


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


def radius_collision(
    path1: List[int],
    path2: List[int],
    timestep: int,
    graph_waypoints: NDArray,
    radius: float = 0.1,
):
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
                graph_waypoints[path1[timestep]] - graph_waypoints[path2[timestep + 1]]
            )
            <= radius
            and np.linalg.norm(
                graph_waypoints[path1[timestep + 1]] - graph_waypoints[path2[timestep]]
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
            collided = radius_collision(
                path1, path2, timestep, graph_waypoints, collision_radius
            )
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


# Functions and Classes used by Multi-Objective CBS


class PrioritySet(object):
    def __init__(self):
        self.set = set()
        self.heap = []

    def add(self, priority, item):
        if item not in self.set:
            self.set.add(item)
            heapq.heappush(self.heap, (priority, item))

    def pop(self):
        priority, item = heapq.heappop(self.heap)
        while item not in self.set:
            priority, item = heapq.heappop(self.heap)
        self.set.remove(item)
        return priority, item

    def size(self):
        return len(self.set)

    def has(self, item):
        return item in self.set

    def remove(self, item):
        if item not in self.set:
            return False
        self.set.remove(item)
        return True


def less_dominant(vecA: NDArray, vecB: NDArray):
    # return np.all(vecA <= vecB)
    exist_strictly_less = False
    for idx in range(len(vecA)):
        if vecA[idx] > vecB[idx] + 1e-6:
            return False
        else:
            if vecA[idx] < vecB[idx] - 1e-6:
                exist_strictly_less = True
    return exist_strictly_less


def equal(vecA: NDArray, vecB: NDArray):
    # return np.all(vecA == vecB)
    for idx in range(len(vecA)):
        if abs(vecA[idx] - vecB[idx]) > 1e-6:
            return False
    return True


def dominate_or_equal(vecA: NDArray, vecB: NDArray):
    if less_dominant(vecA, vecB) or equal(vecA, vecB):
        return True
    return False
