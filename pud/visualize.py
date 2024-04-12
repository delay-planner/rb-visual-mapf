import numpy as np
import matplotlib.pyplot as plt
from pud.collector import Collector
from matplotlib.animation import FuncAnimation
from pud.utils import set_global_seed, set_env_seed
from pud.algos.constrained_collector import ConstrainedCollector
from pud.envs.simple_navigation_env import plot_walls, set_env_difficulty
from pud.envs.safe_pointenv.safe_wrappers import (
    SafeTimeLimit,
    SafeGoalConditionedPointWrapper,
    set_safe_env_difficulty,
)


def visualize_trajectory(
    agent,
    eval_env,
    difficulty=0.5,
    outpath="",
    cost_constraints: dict = {},
    constrained=False,
):
    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(8, 4))
    for col_index in range(2):
        plt.subplot(1, 2, col_index + 1)
        plot_walls(eval_env.walls)
        if constrained:
            goal, observations_list, _, _ = ConstrainedCollector.get_trajectory(
                agent, eval_env
            )
        else:
            goal, observations_list, _, _ = Collector.get_trajectory(agent, eval_env)
        obs_vec = np.array(observations_list)

        print(f"traj {col_index}, num steps: {len(obs_vec)}")

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], "b-o", alpha=0.3)
        plt.scatter(
            [obs_vec[0, 0]],
            [obs_vec[0, 1]],
            marker="+",
            color="red",
            s=200,
            label="start",
        )
        plt.scatter(
            [obs_vec[-1, 0]],
            [obs_vec[-1, 1]],
            marker="+",
            color="green",
            s=200,
            label="end",
        )
        plt.scatter(
            [goal[0]], [goal[1]], marker="*", color="green", s=200, label="goal"
        )
        if col_index == 0:
            plt.legend(loc="lower left", bbox_to_anchor=(0.3, 1), ncol=3, fontsize=16)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def visualize_buffer(rb_vec, eval_env, outpath: str = ""):
    plt.figure(figsize=(6, 6))
    plt.scatter(*rb_vec.T)
    plot_walls(eval_env.walls)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def visualize_pairwise_dists(pdist, outpath=""):
    plt.figure(figsize=(6, 3))
    plt.hist(pdist.flatten(), bins=range(20))
    plt.xlabel('predicted distance')
    plt.ylabel('number of (s, g) pairs')
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()

def visualize_pairwise_costs(pdist, cost_limit:float, n_bins:int=20, outpath=""):
    plt.figure(figsize=(6, 3))    
    plt.hist(pdist.flatten(), bins=np.linspace(0, cost_limit, n_bins))
    plt.xlabel('predicted costs')
    plt.ylabel('number of (s, g) pairs')
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def visualize_graph(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):
    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)
    pdist_combined = np.max(pdist, axis=0)
    plt.scatter(*rb_vec.T)
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff:
                s_j = rb_vec[j]
                plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()

def visualize_cost_graph(
            rb_vec, 
            eval_env, 
            pcost, 
            cost_limit:float, 
            outpath:str="",
            edges_to_display:int=8,
        ):
    # plot the edges that are deemed unsafe
    pcost_combined = np.max(pcost, axis=0) # rb_vec, rb_vec
    safe_mask = pcost_combined < cost_limit
    ind_v, ind_w = np.where(safe_mask)
    print("ratio of predicted unsafe edges: {:.2f}%".format(100.*len(ind_v)/np.prod(safe_mask.shape)))
    assert len(ind_v) == len(ind_v)
    fig, ax = plt.subplots()
    plot_safe_walls(eval_env.get_map(), 
            eval_env.get_cost_map(), 
            cost_limit=cost_limit, 
            ax=ax)
    ax.scatter(*rb_vec.T)
    pbar = tqdm(total=len(rb_vec))
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pcost_combined[i])):
            if count < edges_to_display and pcost_combined[i, j] < cost_limit:
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='g', alpha=0.5)
        pbar.update()
    if len(outpath) > 0:
        fig.savefig(outpath, dpi=300)
    else:
        plt.show()
    plt.close(fig)

def visualize_combined_graph(
            rb_vec: np.ndarray, # ensemble, N, N 
            eval_env, 
            pdist: np.ndarray, # ensemble, N, N 
            pcost: np.ndarray, # ensemble, N, N 
            cost_limit:float, 
            cutoff=7,
            outpath:str="",
            edges_to_display:int=8,
    ):
    """plot edges that are both within the cutoff distance and cost limit
    shorter edges are prioritized
    """
    fig, ax = plt.subplots()
    ax.scatter(*rb_vec.T)
    plot_safe_walls(eval_env.get_map(), 
            eval_env.get_cost_map(), 
            cost_limit=cost_limit, 
            ax=ax)
    
    pbar = tqdm(total=len(rb_vec))
    pdist_combined = np.max(pdist, axis=0)
    pcost_combined = np.max(pcost, axis=0) # rb_vec, rb_vec
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff and pcost_combined[i, j] < cost_limit:
                s_j = rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='g', alpha=0.4)
        pbar.update()

    if len(outpath) > 0:
        fig.savefig(outpath, dpi=300)
    else:
        plt.show()
    plt.close(fig)

def visualize_graph_ensemble(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):
    ensemble_size = pdist.shape[0]
    plt.figure(figsize=(5 * ensemble_size, 4))
    for col_index in range(ensemble_size):
        plt.subplot(1, ensemble_size, col_index + 1)
        plot_walls(eval_env.walls)
        plt.title("critic %d" % (col_index + 1))

        plt.scatter(*rb_vec.T)
        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if count < edges_to_display and pdist[col_index, i, j] < cutoff:
                    s_j = rb_vec[j]
                    plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def visualize_full_graph(g, rb_vec, eval_env, outpath=""):
    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)
    plt.scatter(rb_vec[g.nodes, 0], rb_vec[g.nodes, 1])

    edges_to_plot = g.edges
    edges_to_plot = np.array(list(edges_to_plot))

    for i, j in edges_to_plot:
        s_i = rb_vec[i]
        s_j = rb_vec[j]
        plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.title(f"|V|={g.number_of_nodes()}, |E|={len(edges_to_plot)}")
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def visualize_search_path(
    search_policy,
    eval_env,
    difficulty=0.5,
    outpath="",
    cost_constraints: dict = {},
    constrained=False,
    num_agents=None,
):
    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:

        if num_agents is None:
            if constrained:
                state, info = eval_env.reset()
            else:
                state = eval_env.reset()
            start = state["observation"]
            goal = state["goal"]
            search_policy.select_action(state)
            waypoints = search_policy.get_waypoints()
        else:
            if constrained:
                state, info = eval_env.reset()
            else:
                state = eval_env.reset()
            state["agent_observations"] = [state["observation"]]
            state["agent_waypoints"] = [state["goal"]]
            starts = [state["observation"]]
            goals = [state["goal"]]
            for _ in range(num_agents - 1):
                new_obs = eval_env.env.env._sample_safe_empty_state(
                    cost_limit=eval_env.env.env.cost_limit
                )
                new_goal = None
                count = 0
                while new_goal is None:
                    new_obs = eval_env.env.env._sample_safe_empty_state(
                        cost_limit=eval_env.env.env.cost_limit
                    )
                    (new_obs, new_goal) = eval_env.env._sample_goal(new_obs)
                    count += 1
                    if count > 1000:
                        print("WARNING: Unable to find goal within constraints.")
                new_obs = eval_env.env._normalize_obs(new_obs)
                new_goal = eval_env.env._normalize_obs(new_goal)
                starts.append(new_obs)
                goals.append(new_goal)
                state["agent_observations"].append(new_obs)
                state["agent_waypoints"].append(new_goal)
            state["composite_starts"] = starts
            state["composite_goals"] = goals
            print("Sampled the required starts and goals")
            search_policy.select_multiple_actions(state)
            waypoints = search_policy.get_augmented_waypoints()

    else:
        if constrained:
            goal, observations, waypoints, _ = ConstrainedCollector.get_trajectory(
                search_policy, eval_env
            )
        else:
            goal, observations, waypoints, _ = Collector.get_trajectory(
                search_policy, eval_env
            )
        start = observations[0]

    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)

    if num_agents is None:
        waypoint_vec = np.array(waypoints)

        print(f"waypoints: {waypoint_vec}")
        print(f"waypoints shape: {waypoint_vec.shape}")
        print(f"start: {start}")
        print(f"goal: {goal}")

        plt.scatter(
            [start[0]], [start[1]], marker="+", color="red", s=200, label="start"
        )
        plt.scatter(
            [goal[0]], [goal[1]], marker="*", color="green", s=200, label="goal"
        )
        plt.plot(
            waypoint_vec[:, 0], waypoint_vec[:, 1], "y-s", alpha=0.3, label="waypoint"
        )
        plt.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=16)
        if len(outpath) > 0:
            plt.savefig(outpath, dpi=300)
        else:
            plt.show()
    else:
        print(f"waypoints: {waypoints}")
        for agent in range(num_agents):
            waypoint_vec = np.array(waypoints[agent])
            print(f"waypoints for agent {agent} shape: {waypoint_vec.shape}")
        print(f"starts: {starts}")
        print(f"goals: {goals}")

        # agent_colors = ["blue", "purple", "orange", "cyan", "magenta", "black"]
        agent_colors = [
            ("darkblue", "blue"),
            ("darkred", "red"),
            ("darkgreen", "green"),
            ("darkorange", "orange"),
            ("darkmagenta", "magenta"),
            ("darkcyan", "cyan"),
            ("black", "gray"),
        ]

        wp_paths = []
        for i in range(num_agents):
            waypoint_vec = np.array(waypoints[i])
            wp_paths.append(waypoint_vec)
            plt.scatter(
                [starts[i][0]],
                [starts[i][1]],
                marker="+",
                color=agent_colors[i][0],
                s=200,
                label="start" + str(i),
            )
            plt.scatter(
                [goals[i][0]],
                [goals[i][1]],
                marker="*",
                color=agent_colors[i][0],
                s=200,
                label="goal" + str(i),
            )
            plt.plot(
                waypoint_vec[:, 0],
                waypoint_vec[:, 1],
                # "y-s",
                color=agent_colors[i][1],
                marker="s",
                linestyle="-",
                alpha=0.3,
                label="waypoint" + str(i),
            )
            plt.legend(
                loc="lower left", bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=16
            )
        plt.show()

        visualize_path(wp_paths, eval_env, "multi_agent_search.gif", save_fig=True)


def visualize_path(paths, eval_env, filename: str, save_fig: bool = False, wps=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    walls = eval_env.walls.T
    (height, width) = walls.shape
    for i, j in zip(*np.where(walls)):
        x = np.array([j, j + 1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i + 1, i + 1]) / float(height)
        ax.fill_between(x, y0, y1, color="grey")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.xticks([])
    # ax.yticks([])

    if wps is not None:
        marker = "s"
    else:
        marker = "o"

    # agent_colors = ["g", "b", "r", "deeppink", "y", "m", "c", "w"]
    agent_colors = [
        ("darkblue", "blue"),
        ("darkred", "red"),
        ("darkgreen", "green"),
        ("darkorange", "orange"),
        ("darkmagenta", "magenta"),
        ("darkcyan", "cyan"),
        ("black", "gray"),
    ]
    num_of_agents = len(paths)

    lines = []
    if wps is not None:
        wps_lines = []
    starts = []
    ends = []
    for agent in range(num_of_agents):
        (line,) = ax.plot(
            [],
            [],
            lw=2,
            color=agent_colors[agent][1],
            ls="-",
            marker="o",
            alpha=0.7,
        )
        (start,) = ax.plot(
            [], [], lw=2, color=agent_colors[agent][0], marker="x", alpha=0.7
        )
        (end,) = ax.plot(
            [], [], lw=2, color=agent_colors[agent][0], marker="*", alpha=0.7
        )
        if wps is not None:
            (wps_line,) = ax.plot(
                [],
                [],
                lw=2,
                color=agent_colors[agent][1],
                ls="-",
                marker="s",
                alpha=0.7,
            )
            wps_lines.append(wps_line)
        lines.append(line)
        starts.append(start)
        ends.append(end)

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
            if wps is not None:
                wps_x_data = [point[0] for point in wps[i][: frame + 1]]
                wps_y_data = [point[1] for point in wps[i][: frame + 1]]
                wps_lines[i].set_data(wps_x_data, wps_y_data)
        if wps is not None:
            return lines + wps_lines
        else:
            return lines

    frames = max(len(path) for path in paths)
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, repeat=False
    )

    if save_fig:
        ani.save(filename, writer="pillow", fps=1)
    plt.show()


def visualize_compare_search(
    agent,
    search_policy,
    eval_env,
    difficulty=0.5,
    seed=0,
    outpath="",
    cost_constraints: dict = {},
    constrained=False,
    num_agents=None,
):

    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(12, 5))

    if num_agents is not None:
        search_obs = []
        search_wps = []
        no_search_obs = []

    for col_index in range(2):
        title = "no search" if col_index == 1 else "search"
        plt.subplot(1, 2, col_index + 1)
        plot_walls(eval_env.walls)
        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        if use_search:
            policy = search_policy
        else:
            policy = agent
        if constrained:
            if num_agents is not None:

                if use_search:
                    starts, goal, observations, waypoints, _ = (
                        ConstrainedCollector.get_trajectories(
                            policy, eval_env, num_agents
                        )
                    )
                else:
                    _, goal, observations, waypoints, _ = (
                        ConstrainedCollector.get_trajectories(
                            policy, eval_env, num_agents, starts, goal
                        )
                    )
            else:
                goal, observations, waypoints, _ = ConstrainedCollector.get_trajectory(
                    policy, eval_env
                )
        else:
            goal, observations, waypoints, _ = Collector.get_trajectory(
                policy, eval_env
            )

        if num_agents is None:
            start = observations[0]

            obs_vec = np.array(observations)
            waypoint_vec = np.array(waypoints)

            print(f"policy: {title}")
            print(f"start: {start}")
            print(f"goal: {goal}")
            print(f"steps: {obs_vec.shape[0] - 1}")
            print("-" * 10)

            plt.plot(obs_vec[:, 0], obs_vec[:, 1], "b-o", alpha=0.3)
            plt.scatter(
                [start[0]], [start[1]], marker="+", color="red", s=200, label="start"
            )
            plt.scatter(
                [obs_vec[-1, 0]],
                [obs_vec[-1, 1]],
                marker="+",
                color="green",
                s=200,
                label="end",
            )
            plt.scatter(
                [goal[0]], [goal[1]], marker="*", color="green", s=200, label="goal"
            )
            plt.title(title, fontsize=24)

            if use_search:
                plt.plot(
                    waypoint_vec[:, 0],
                    waypoint_vec[:, 1],
                    "y-s",
                    alpha=0.3,
                    label="waypoint",
                )
                plt.legend(
                    loc="lower left", bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16
                )
            if len(outpath) > 0:
                plt.savefig(outpath, dpi=300)
        else:
            agent_colors = [
                ("darkblue", "blue"),
                ("darkred", "red"),
                ("darkgreen", "green"),
                ("darkorange", "orange"),
                ("darkmagenta", "magenta"),
                ("darkcyan", "cyan"),
                ("black", "gray"),
            ]
            for agent_id in range(num_agents):
                start = observations[agent_id][0]
                agent_goal = goal[agent_id]
                obs_vec = np.array(observations[agent_id])
                waypoint_vec = np.array(waypoints[agent_id])

                if not use_search:
                    no_search_obs.append(obs_vec)
                else:
                    search_obs.append(obs_vec)
                    search_wps.append(waypoint_vec)

                print(f"policy: {title}")
                print(f"agent: {agent_id}")
                print(f"start: {start}")
                print(f"goal: {agent_goal}")
                print(f"waypoints: {waypoint_vec}")
                print(f"steps: {obs_vec.shape[0] - 1}")
                print("-" * 10)

                plt.plot(
                    obs_vec[:, 0],
                    obs_vec[:, 1],
                    color=agent_colors[agent_id][1],
                    marker="o",
                    linestyle="-",
                    alpha=0.3,
                )
                plt.scatter(
                    [start[0]],
                    [start[1]],
                    marker="+",
                    color=agent_colors[agent_id][0],
                    s=200,
                    label="start" + str(agent_id),
                )
                plt.scatter(
                    [obs_vec[-1, 0]],
                    [obs_vec[-1, 1]],
                    marker="+",
                    color=agent_colors[agent_id][0],
                    s=200,
                    label="end" + str(agent_id),
                )
                plt.scatter(
                    [agent_goal[0]],
                    [agent_goal[1]],
                    marker="*",
                    color=agent_colors[agent_id][0],
                    s=200,
                    label="goal" + str(agent_id),
                )
                plt.title(title, fontsize=24)

                if use_search:
                    plt.plot(
                        waypoint_vec[:, 0],
                        waypoint_vec[:, 1],
                        color=agent_colors[agent_id][1],
                        marker="s",
                        linestyle="-",
                        alpha=0.3,
                        label="waypoint" + str(agent_id),
                    )
                    # plt.legend(
                    #     loc="lower left",
                    #     bbox_to_anchor=(-0.8, -0.15),
                    #     ncol=4,
                    #     fontsize=16,
                    # )
    plt.show()

    if num_agents is not None:
        # visualize_path(
        #     no_search_obs,
        #     eval_env,
        #     "no_search.gif",
        #     save_fig=True,
        # )
        visualize_path(
            search_obs,
            eval_env,
            "search.gif",
            save_fig=True,
            wps=search_wps,
        )
