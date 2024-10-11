import logging
import numpy as np
import networkx as nx
from pud.mapf.single_agent_planner import compute_cost
from pud.mapf.risk_bounded_cbs import RiskBoundedCBSSolver

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filename = "pud/mapf/unit_tests/test_cbs_input.txt"

    f = open(
        filename,
        "r",
    )
    line = f.readline()
    rows, columns = [int(x) for x in line.split(" ")]
    rows = int(rows)
    columns = int(columns)

    G = nx.empty_graph(0, create_using=nx.DiGraph)
    graph_waypoints = []

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
    num_agents = int(line)

    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        start_x, start_y, goal_x, goal_y = [int(x) for x in line.split(" ")]
        starts.append((start_x, start_y))
        goals.append((goal_x, goal_y))

        graph_waypoints.append([start_x, start_y])
        graph_waypoints.append([goal_x, goal_y])

    f.close()

    for node in range(boolean_map.shape[0] * boolean_map.shape[1]):
        node_x, node_y = node // boolean_map.shape[1], node % boolean_map.shape[1]
        graph_waypoints.append([node_x, node_y])
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
                    G.add_edge(
                        node, neighbor[0] * boolean_map.shape[1] + neighbor[1], weight=1, cost=1
                    )
                else:
                    G.add_edge(
                        node, neighbor[0] * boolean_map.shape[1] + neighbor[1], weight=1, cost=2
                    )

    start_ids, goal_ids = [], []
    for start_node in starts:
        start_node = start_node[0] * boolean_map.shape[1] + start_node[1]
        start_ids.append(start_node)
    for goal_node in goals:
        goal_node = goal_node[0] * boolean_map.shape[1] + goal_node[1]
        goal_ids.append(goal_node)

    graph_waypoints = np.array(graph_waypoints)

    solver = RiskBoundedCBSSolver(
        G,
        graph_waypoints,
        start_ids,
        goal_ids,
        risk_bound=51,
        seed=0,
        collision_radius=0.0
    )
    solution = solver.find_paths()
    paths = solution["paths"]  # type: ignore
    print(paths)

    for idx, path in enumerate(paths):
        print("Cost of path for agent {}: {}".format(idx, compute_cost(path, G, "cost")))

    # assertTrue(len(paths) == 5)  # type: ignore
    # assertTrue(solution["cost"] == 41)  # type: ignore
    # assertTrue(detect_collisions(paths, self.graph_waypoints, 0.0) == [])  # type: ignore

    print("Number of expanded nodes: {}".format(solver.num_expanded))
    print("Number of generated nodes: {}".format(solver.num_generated))
