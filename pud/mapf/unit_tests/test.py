import time
import logging
import numpy as np
import networkx as nx
from pud.mapf.single_agent_planner import compute_cost
from pud.mapf.risk_bounded_cbs import RiskBoundedCBSSolver

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
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

    risky_nodes = []
    for node in range(boolean_map.shape[0] * boolean_map.shape[1]):
        node_x, node_y = node // boolean_map.shape[1], node % boolean_map.shape[1]
        if boolean_map[node_x, node_y]:
            potential_neighbors = [
                (node_x - 1, node_y),
                (node_x + 1, node_y),
                (node_x, node_y - 1),
                (node_x, node_y + 1),
            ]
            for unsafe_neighbor in potential_neighbors:
                if (
                    unsafe_neighbor[0] >= 0
                    and unsafe_neighbor[0] < boolean_map.shape[0]
                    and unsafe_neighbor[1] >= 0
                    and unsafe_neighbor[1] < boolean_map.shape[1]
                    and not boolean_map[unsafe_neighbor[0], unsafe_neighbor[1]]
                ):
                    risky_nodes.append(unsafe_neighbor[0] * boolean_map.shape[1] + unsafe_neighbor[1])

    for node in range(boolean_map.shape[0] * boolean_map.shape[1]):
        node_x, node_y = node // boolean_map.shape[1], node % boolean_map.shape[1]
        if boolean_map[node_x, node_y]:
            continue
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
                if node in risky_nodes or neighbor[0] * boolean_map.shape[1] + neighbor[1] in risky_nodes:
                    G.add_edge(
                        node,
                        neighbor[0] * boolean_map.shape[1] + neighbor[1],
                        weight=1,
                        cost=1
                    )
                else:
                    G.add_edge(
                        node, neighbor[0] * boolean_map.shape[1] + neighbor[1], weight=1, cost=0
                    )

    start_ids, goal_ids = [], []
    for start_node in starts:
        start_node = start_node[0] * boolean_map.shape[1] + start_node[1]
        start_ids.append(start_node)
    for goal_node in goals:
        goal_node = goal_node[0] * boolean_map.shape[1] + goal_node[1]
        goal_ids.append(goal_node)

    graph_waypoints = np.array(graph_waypoints)
    risky_nodes = list(set(risky_nodes))

    start = time.time()
    solver = RiskBoundedCBSSolver(
        G,
        np.array(risky_nodes),
        graph_waypoints,
        start_ids,
        goal_ids,
        risk_bound=4,
        seed=0,
        collision_radius=0.0,
        max_time=10000,
        use_experience=True,
        use_cardinality=True
    )
    solution = solver.find_paths()
    print("Time taken: {}".format(time.time() - start))
    paths = solution["paths"]  # type: ignore
    print(paths)

    for idx, path in enumerate(paths):
        print("Cost of path for agent {}: {}".format(idx, solver.compute_cost(path, "cost")))

    # assertTrue(len(paths) == 5)  # type: ignore
    # assertTrue(solution["cost"] == 41)  # type: ignore
    # assertTrue(detect_collisions(paths, self.graph_waypoints, 0.0) == [])  # type: ignore

    print("Number of expanded nodes: {}".format(solver.num_expanded))
    print("Number of generated nodes: {}".format(solver.num_generated))
