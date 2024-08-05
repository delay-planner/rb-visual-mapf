import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pud.collector import Collector
from pud.utils import set_env_seed, set_global_seed
from pud.envs.habitat_navigation_env import plot_wall
from pud.algos.constrained_collector import ConstrainedCollector
from pud.envs.safe_pointenv.safe_wrappers import set_safe_env_difficulty
from pud.envs.simple_navigation_env import plot_walls, set_env_difficulty
from pud.envs.safe_pointenv.safe_pointenv import plot_safe_walls, plot_trajs

AGENT_COLORS = [
    ("darkblue", "blue"),
    ("darkred", "red"),
    ("darkgreen", "green"),
    ("darkorange", "orange"),
    ("darkmagenta", "magenta"),
    ("darkcyan", "cyan"),
    ("black", "gray"),
]


def plot_agent_paths(a_id, start, goal, obs, title, ax, wps=None):

    ax.plot(obs[:, 0], obs[:, 1], "o-", c=AGENT_COLORS[a_id][1], alpha=0.3)
    ax.scatter([start[0]], [start[1]], marker="+", c=AGENT_COLORS[a_id][0], s=200, label="Start " + str(a_id))
    ax.scatter([obs[-1, 0]], [obs[-1, 1]], marker="x", c=AGENT_COLORS[a_id][0], s=200, label="End " + str(a_id))
    ax.scatter([goal[0]], [goal[1]], marker="*", c=AGENT_COLORS[a_id][0], s=200, label="Goal " + str(a_id))
    if wps is not None:
        ax.plot(wps[:, 0], wps[:, 1], "s-", c=AGENT_COLORS[a_id][1], alpha=0.3, label="Waypoint " + str(a_id))
    ax.set_title(title, fontsize=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -1.15), ncol=4, fontsize=16)
    return ax


def visualize_trajectory(agent, eval_env, difficulty=0.5, outpath=""):

    constrained = hasattr(agent, "constraints") and agent.constraints is not None

    if constrained:
        cost_constraints = agent.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(8, 4))
    for col_index in range(2):

        ax = plt.subplot(1, 2, col_index + 1)
        ax = plot_walls(eval_env.walls, ax)
        collector_cls = ConstrainedCollector if constrained else Collector
        goal, observations_list, _, _ = collector_cls.get_trajectory(agent, eval_env)

        obs_vec = np.array(observations_list)
        print(f"Trajectory {col_index}")
        print(f"Start: {obs_vec[0]}")
        print(f"Goal: {goal}")
        print(f"Steps: {obs_vec.shape[0]}")

        ax = plot_agent_paths(0, obs_vec[0], goal, obs_vec, "Trajectory", ax)

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], "b-o", alpha=0.3)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_buffer(rb_vec, eval_env, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    ax = plot_walls(eval_env.walls, ax=ax)
    ax.scatter(rb_vec[:, 0], rb_vec[:, 1])
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_pairwise_dists(pdist, outpath=""):
    plt.figure(figsize=(6, 3))
    plt.hist(pdist.flatten(), bins=range(20))
    plt.xlabel('Predicted Distance')
    plt.ylabel('Number of (s, g) pairs')
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_pairwise_costs(pdist, cost_limit, n_bins=20, outpath=""):
    plt.figure(figsize=(6, 3))
    plt.hist(pdist.flatten(), bins=np.linspace(0, cost_limit, n_bins))
    plt.xlabel('Predicted Costs')
    plt.ylabel('Number of (s, g) pairs')
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_eval_records(eval_records, eval_env, ax, starts=[], goals=[], use_pbar=False, color=None):

    list_trajs = []
    for id in eval_records.keys():
        list_trajs.append(eval_records[id]["traj"])

    ax = plot_safe_walls(
        walls=eval_env.get_map(),
        cost_map=eval_env.get_cost_map(),
        cost_limit=eval_env.cost_limit,
        ax=ax
    )

    ax = plot_trajs(
        ax=ax,
        s=32,
        goals=goals,
        starts=starts,
        traj_color=color,
        use_pbar=use_pbar,
        list_trajs=list_trajs,
        walls=eval_env.get_map(),
    )

    return ax


def visualize_problems(eval_env,  ax, starts=[], goals=[]):
    return visualize_eval_records(eval_records={}, eval_env=eval_env, ax=ax, starts=starts, goals=goals)


def visualize_graph(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    plot_walls(eval_env.walls, ax)
    ax.scatter(*rb_vec.T)

    pdist_combined = np.max(pdist, axis=0)
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff:
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_cost_graph(rb_vec, eval_env, pcost, cost_limit, outpath="", edges_to_display=8):
    # Plot the edges that are deemed unsafe
    pcost_combined = np.max(pcost, axis=0)  # rb_vec, rb_vec
    safe_mask = pcost_combined < cost_limit
    ind_v, _ = np.where(safe_mask)
    print("Ratio of predicted unsafe edges: {:.2f}%".format(100. * len(ind_v) / np.prod(safe_mask.shape)))
    assert len(ind_v) == len(ind_v)

    fig, ax = plt.subplots()
    plot_safe_walls(eval_env.get_map(), eval_env.get_cost_map(), cost_limit=cost_limit, ax=ax)
    ax.scatter(rb_vec[:, 0], rb_vec[:, 1])

    pbar = tqdm(total=len(rb_vec))
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pcost_combined[i])):
            if count < edges_to_display and pcost_combined[i, j] < cost_limit:
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='g', alpha=0.5)
        pbar.update()

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_combined_graph(rb_vec, eval_env, pdist, pcost, cost_limit, cutoff=7, outpath="", edges_to_display=8):
    """
    Plot edges that are both within the cutoff distance and cost limit shorter edges are prioritized
    rb_vec, pdist, pcost: (ensemble_size, N, N)
    """

    fig, ax = plt.subplots()
    ax.scatter(*rb_vec.T)
    plot_safe_walls(eval_env.get_map(), eval_env.get_cost_map(), cost_limit=cost_limit, ax=ax)

    pbar = tqdm(total=len(rb_vec))
    pdist_combined = np.max(pdist, axis=0)
    pcost_combined = np.max(pcost, axis=0)  # rb_vec, rb_vec
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff and pcost_combined[i, j] < cost_limit:
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='g', alpha=0.4)
        pbar.update()

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_graph_ensemble(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):
    ensemble_size = pdist.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=ensemble_size, figsize=(5 * ensemble_size, 4))

    for col_index in range(ensemble_size):
        ax[col_index] = plot_walls(eval_env.walls, ax=ax[col_index])
        ax[col_index].set_title("critic %d" % (col_index + 1))
        ax[col_index].scatter(*rb_vec.T)
        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if count < edges_to_display and pdist[col_index, i, j] < cutoff:
                    s_j = rb_vec[j]
                    ax[col_index].plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_full_graph(g, rb_vec, eval_env, outpath=""):
    _, ax = plt.subplots(figsize=(6, 6))
    ax = plot_walls(eval_env.walls, ax)
    ax.scatter(rb_vec[g.nodes, 0], rb_vec[g.nodes, 1])

    edges_to_plot = g.edges
    edges_to_plot = np.array(list(edges_to_plot))

    for i, j in edges_to_plot:
        s_i = rb_vec[i]
        s_j = rb_vec[j]
        ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.title(f"|V|={g.number_of_nodes()}, |E|={len(edges_to_plot)}")
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_path(paths, eval_env, filename, plot_handles, save_fig=False, waypoints=None):

    num_agents = len(paths)
    fig, ax = plot_handles

    ends = []
    lines = []
    starts = []
    waypoint_lines = []

    for agent in range(num_agents):
        (line,) = ax.plot([], [], "o-", lw=2, c=AGENT_COLORS[agent][1], alpha=0.7)
        (start,) = ax.plot([], [], "x", lw=2, c=AGENT_COLORS[agent][0], alpha=0.7)
        (end,) = ax.plot([], [], "*", lw=2, c=AGENT_COLORS[agent][0], alpha=0.7)
        if waypoints is not None:
            (waypoint_line,) = ax.plot([], [], "s-", lw=2, c=AGENT_COLORS[agent][1], alpha=0.7)
            waypoint_lines.append(waypoint_line)
        ends.append(end)
        lines.append(line)
        starts.append(start)

    def init():
        for i, path in enumerate(paths):
            starts[i].set_data(path[0][0], path[0][1])
            ends[i].set_data(path[-1][0], path[-1][1])
        return starts

    def update(frame):
        for i, path in enumerate(paths):
            if frame >= len(path):
                continue
            x_data = [point[0] for point in path[: frame + 1]]
            y_data = [point[1] for point in path[: frame + 1]]
            lines[i].set_data(x_data, y_data)
            if waypoints is not None:
                wps_x_data = [point[0] for point in waypoints[i][: frame + 1]]
                wps_y_data = [point[1] for point in waypoints[i][: frame + 1]]
                waypoint_lines[i].set_data(wps_x_data, wps_y_data)

        return lines if waypoints is None else lines + waypoint_lines

    frames = max(len(path) for path in paths)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False)
    if save_fig:
        ani.save(filename, writer="pillow", fps=1)


def visualize_search_path_single_agent(search_policy, eval_env, difficulty=0.5, outpath=""):

    constrained = search_policy.constraints is not None

    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:
        state = eval_env.reset()
        state, _ = state if constrained else (state, None)

        goal = state["goal"]
        start = state["observation"]
        search_policy.select_action(state)
        waypoints = search_policy.get_waypoints()
    else:
        collector_cls = ConstrainedCollector if constrained else Collector
        goal, observations, waypoints, _ = collector_cls.get_trajectory(search_policy, eval_env)
        start = observations[0]

    ax = plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls, ax)

    waypoint_vec = np.array(waypoints)

    print(f"Start: {start}")
    print(f"Waypoints: {waypoint_vec}")
    print(f"Goal: {goal}")
    print(f"Steps: {waypoint_vec.shape}")
    print("-" * 10)

    ax = plot_agent_paths(0, start, goal, waypoint_vec, "Search", ax)
    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_search_path_multi_agent(search_policy, eval_env, num_agents, difficulty=0.5, outpath=""):

    constrained = search_policy.constraints is not None
    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:

        state = eval_env.reset()
        state, _ = state if constrained else (state, None)

        agent_goal = [state["goal"]]
        agent_start = [state["observation"]]

        # Mutable objects
        state["agent_waypoints"] = agent_goal.copy()
        state["agent_observations"] = agent_start.copy()

        goals = agent_goal.copy()
        starts = agent_start.copy()

        for _ in range(num_agents - 1):

            agent_state = eval_env.reset()

            agent_goal = [agent_state["goal"]]
            agent_start = [agent_state["observation"]]

            goals.extend(agent_goal.copy())
            starts.extend(agent_start.copy())
            state["agent_waypoints"].append(agent_goal.copy())
            state["agent_observations"].extend(agent_start.copy())

        # Immutable objects - Should not be modified ever!
        state["composite_goals"] = goals.copy()
        state["composite_starts"] = starts.copy()
        print("Sampled the required starts and goals")

        search_policy.select_actions(state)
        waypoints = search_policy.get_augmented_waypoints()
    else:
        collector_cls = ConstrainedCollector if constrained else Collector
        goals, observations, waypoints, _ = collector_cls.get_trajectory(search_policy, eval_env)
        starts = observations[0]

    ax = plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls, ax)

    agent_waypoints = []
    for agent_id in range(num_agents):

        agent_goal = goals[agent_id]
        agent_start = starts[agent_id]

        waypoint_vec = np.array(waypoints[agent_id])
        agent_waypoints.append(waypoint_vec)

        print(f"Agent: {agent_id}")
        print(f"Start: {agent_start}")
        print(f"Waypoints: {waypoint_vec}")
        print(f"Goal: {agent_goal}")
        print(f"Steps: {waypoint_vec.shape}")
        print("-" * 10)

        ax = plot_agent_paths(agent_id, agent_start, agent_goal, waypoint_vec, "Search", ax)

    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
        save_path = outpath[:-4] + ".gif"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plot_walls(eval_env.walls, ax)
        visualize_path(agent_waypoints, eval_env, (fig, ax), save_path, save_fig=True)
    else:
        plt.show()


def visualize_search_path(search_policy, eval_env, difficulty=0.5, outpath="", num_agents=None):
    if num_agents is None:
        visualize_search_path_single_agent(search_policy, eval_env, difficulty, outpath)
    else:
        visualize_search_path_multi_agent(search_policy, eval_env, num_agents, difficulty, outpath)


def visualize_compare_search_single_agent(agent, search_policy, eval_env, difficulty=0.5, seed=0, outpath=""):

    constrained = search_policy.constraints is not None
    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(12, 5))

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = plot_walls(eval_env.walls, ax)

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        policy = search_policy if use_search else agent

        collector_cls = ConstrainedCollector if constrained else Collector
        goal, observations, waypoints, _ = collector_cls.get_trajectory(policy, eval_env)

        start = observations[0]
        obs_vec = np.array(observations)
        waypoint_vec = np.array(waypoints) if use_search else None

        print(f"Policy: {title}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Steps: {obs_vec.shape[0] - 1}")
        print("-" * 10)

        ax = plot_agent_paths(0, start, goal, obs_vec, title, ax, waypoint_vec)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_compare_search_multi_agent(agent, search_policy, eval_env, n_agents, difficulty=0.5, seed=0, outpath=""):

    constrained = search_policy.constraints is not None
    if constrained:
        cost_constraints = search_policy.constraints
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(12, 5))

    search_waypoints = []
    search_observations = []
    no_search_observations = []

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = plot_walls(eval_env.walls, ax)

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        policy = search_policy if use_search else agent
        threshold = search_policy.radius
        collector_cls = ConstrainedCollector if constrained else Collector

        if col_index == 0:
            starts, goals, observations, waypoints, _ = collector_cls.get_trajectories(
                policy, eval_env, n_agents, threshold=threshold
            )
        else:
            _, _, observations, waypoints, _ = collector_cls.get_trajectories(
                policy, eval_env, n_agents, starts, goals, threshold=threshold
            )

        print(f"Policy: {title}")
        for agent_id in range(n_agents):

            agent_goal = goals[agent_id]
            agent_start = observations[agent_id][0]

            obs_vec = np.array(observations[agent_id])
            waypoint_vec = np.array(waypoints[agent_id]) if use_search else None

            print(f"Agent: {agent_id}")
            print(f"Start: {agent_start}")
            if use_search:
                search_waypoints.append(waypoint_vec)
                search_observations.append(obs_vec)
                print(f"Waypoints: {waypoint_vec}")
            else:
                no_search_observations.append(obs_vec)
            print(f"Goal: {agent_goal}")
            print(f"Steps: {obs_vec.shape[0] - 1}")
            print("-" * 10)

            ax = plot_agent_paths(agent_id, agent_start, agent_goal, obs_vec, title, ax, waypoint_vec)

    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)

        save_path = outpath[:-4] + "_search.gif"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(search_observations, eval_env, (fig, ax), save_path, save_fig=True, waypoints=search_waypoints)

        save_path = outpath[:-4] + "_no_search.gif"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(no_search_observations, eval_env, (fig, ax), save_path, save_fig=True)
    else:
        plt.show()


def visualize_compare_search(agent, search_policy, eval_env, difficulty=0.5, seed=0, outpath="", num_agents=None):

    if num_agents is None:
        visualize_compare_search_single_agent(agent, search_policy, eval_env, difficulty, seed, outpath)
    else:
        visualize_compare_search_multi_agent(agent, search_policy, eval_env, num_agents, difficulty, seed, outpath)
