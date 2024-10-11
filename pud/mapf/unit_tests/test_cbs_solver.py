import unittest
import numpy as np
import networkx as nx
from pud.mapf.cbs_ds import CBSDSSolver
from pud.mapf.cbs import CBSSolver, detect_collisions
from pud.mapf.single_agent_planner import compute_cost
from pud.mapf.risk_bounded_cbs import RiskBoundedCBSSolver

"""
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_cbs_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_cbsds_paths
python pud/mapf/unit_tests/test_cbs_solver.py TestCBSSolver.test_risk_bounded_cbs_paths
"""


class TestCBSSolver(unittest.TestCase):
    def setUp(self):
        self.filename = "pud/mapf/unit_tests/test_cbs_input.txt"

    def load_problem(self, filename):
        f = open(
            self.filename,
            "r",
        )
        line = f.readline()
        rows, columns = [int(x) for x in line.split(" ")]
        rows = int(rows)
        columns = int(columns)

        self.G = nx.empty_graph(0, create_using=nx.DiGraph)
        self.graph_waypoints = []

        boolean_map = []
        for _ in range(rows):
            line = f.readline()
            boolean_map.append([])
            for cell in line:
                if cell == "@":
                    boolean_map[-1].append(True)
                elif cell == ".":
                    boolean_map[-1].append(False)

        boolean_map = np.array(boolean_map)

        line = f.readline()
        self.num_agents = int(line)

        starts = []
        goals = []
        for a in range(self.num_agents):
            line = f.readline()
            start_x, start_y, goal_x, goal_y = [int(x) for x in line.split(" ")]
            starts.append((start_x, start_y))
            goals.append((goal_x, goal_y))

            self.graph_waypoints.append([start_x, start_y])
            self.graph_waypoints.append([goal_x, goal_y])

        f.close()

        for node in range(boolean_map.shape[0] * boolean_map.shape[1]):
            node_x, node_y = node // boolean_map.shape[1], node % boolean_map.shape[1]
            self.graph_waypoints.append([node_x, node_y])
            potential_neighbors = [
                (node_x - 1, node_y),
                (node_x + 1, node_y),
                (node_x, node_y - 1),
                (node_x, node_y + 1),
            ]

            for neighbor in potential_neighbors:
                if (
                    neighbor[0] >= 0
                    and neighbor[0] < boolean_map.shape[0]
                    and neighbor[1] >= 0
                    and neighbor[1] < boolean_map.shape[1]
                    and not boolean_map[neighbor[0], neighbor[1]]
                ):
                    # Check if the neighbor's neighbor is blocked
                    neighbor_potential_neighbors = [
                        (neighbor[0] - 1, neighbor[1]),
                        (neighbor[0] + 1, neighbor[1]),
                        (neighbor[0], neighbor[1] - 1),
                        (neighbor[0], neighbor[1] + 1),
                    ]
                    blocked = False
                    for neighbor_neighbor in neighbor_potential_neighbors:
                        if (
                            neighbor_neighbor[0] >= 0
                            and neighbor_neighbor[0] < boolean_map.shape[0]
                            and neighbor_neighbor[1] >= 0
                            and neighbor_neighbor[1] < boolean_map.shape[1]
                            and boolean_map[neighbor_neighbor[0], neighbor_neighbor[1]]
                        ):
                            blocked = True
                            break

                    if not blocked:
                        self.G.add_edge(
                            node, neighbor[0] * boolean_map.shape[1] + neighbor[1], weight=1, cost=1
                        )
                    else:
                        self.G.add_edge(
                            node, neighbor[0] * boolean_map.shape[1] + neighbor[1], weight=1, cost=2
                        )

        self.start_ids, self.goal_ids = [], []
        for start_node in starts:
            start_node = start_node[0] * boolean_map.shape[1] + start_node[1]
            self.start_ids.append(start_node)
        for goal_node in goals:
            goal_node = goal_node[0] * boolean_map.shape[1] + goal_node[1]
            self.goal_ids.append(goal_node)

        self.graph_waypoints = np.array(self.graph_waypoints)

    def test_cbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        solver = CBSSolver(self.G, self.graph_waypoints, self.start_ids, self.goal_ids, seed=0, collision_radius=0.0)
        solution = solver.find_paths()
        paths = solution["paths"]  # type: ignore
        print(paths)

        self.assertTrue(len(paths) == 5)
        self.assertTrue(solution["cost"] == 41)  # type: ignore
        self.assertTrue(detect_collisions(paths, self.graph_waypoints, 0.0) == [])

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

    def test_cbsds_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        solver = CBSDSSolver(self.G, self.graph_waypoints, self.start_ids, self.goal_ids, seed=0, collision_radius=0.0)
        solution = solver.find_paths()
        paths = solution["paths"]  # type: ignore
        print(paths)

        for idx, path in enumerate(paths):
            print("Cost of path for agent {}: {}".format(idx, compute_cost(path, self.G, "cost")))

        self.assertTrue(len(paths) == 5)  # type: ignore
        self.assertTrue(solution["cost"] == 41)  # type: ignore
        self.assertTrue(detect_collisions(paths, self.graph_waypoints, 0.0) == [])  # type: ignore

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))

    def test_risk_bounded_cbs_paths(self):
        self.load_problem(self.filename)
        self.graph_waypoints = np.array(self.graph_waypoints)
        solver = RiskBoundedCBSSolver(
            self.G,
            self.graph_waypoints,
            self.start_ids,
            self.goal_ids,
            risk_bound=53,
            seed=0,
            collision_radius=0.0
        )
        solution = solver.find_paths()
        paths = solution["paths"]  # type: ignore
        print(paths)

        for idx, path in enumerate(paths):
            print("Cost of path for agent {}: {}".format(idx, compute_cost(path, self.G, "cost")))

        self.assertTrue(len(paths) == 5)  # type: ignore
        self.assertTrue(solution["cost"] == 41)  # type: ignore
        self.assertTrue(detect_collisions(paths, self.graph_waypoints, 0.0) == [])  # type: ignore

        print("Number of expanded nodes: {}".format(solver.num_expanded))
        print("Number of generated nodes: {}".format(solver.num_generated))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    unittest.main()
