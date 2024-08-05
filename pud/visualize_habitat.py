import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

from pud.collector import Collector
from pud.utils import set_env_seed, set_global_seed
from pud.envs.habitat_navigation_env import plot_wall
from pud.visualize import plot_agent_paths, visualize_path
from pud.envs.simple_navigation_env import set_env_difficulty


def visualize_buffer(rb_vec, eval_env, outpath: str = ""):

    _, ax = plt.subplots()
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape

    scaled_rb_vec = rb_vec / np.array([height, width])
    ax.scatter(*scaled_rb_vec.T)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_graph(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):

    _, ax = plt.subplots()
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape

    scaled_rb_vec = rb_vec / np.array([height, width])
    ax.scatter(*scaled_rb_vec.T)

    pdist_combined = np.max(pdist, axis=0)
    for i, s_i in enumerate(scaled_rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff:
                s_j = scaled_rb_vec[j]
                ax.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_graph_ensemble(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8, outpath=""):

    ensemble_size = pdist.shape[0]
    plt.figure(figsize=(5 * ensemble_size, 4))
    height, width = eval_env.walls.shape
    rb_vec = rb_vec / np.array([height, width])

    for col_index in range(ensemble_size):

        ax = plt.subplot(1, ensemble_size, col_index + 1)
        ax = plot_wall(eval_env.walls, ax)
        plt.title("Critic %d" % (col_index + 1))
        plt.scatter(*rb_vec.T)

        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if count < edges_to_display and pdist[col_index, i, j] < cutoff:
                    s_j = rb_vec[j]
                    plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c="k", alpha=0.5)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_habitat_agent(frames, outpath, bs=1, border_color=(1, 0, 0, 0)):

    N, H, W, C = frames[0].shape
    final_height = H + 2 * bs
    final_width = W * N + bs * (N + 1)

    final_frames = []
    for frame in frames:

        ff = np.zeros((final_height, final_width, C), dtype=frame.dtype)
        for i in range(C):
            ff[:, :, i] = border_color[i]
        for i in range(N):
            start_x = bs + i * (W + bs)
            ff[bs : H + bs, start_x : start_x + W, :] = frame[i]  # noqa
            final_frames.append(ff)

    clip = ImageSequenceClip(final_frames[:-1], fps=10)
    clip.write_videofile(outpath, fps=10)

    from PIL import Image

    goal_image = Image.fromarray(final_frames[-1])
    goal_image.save(outpath[:-4] + "_goal.png")


def visualize_search_path_single_agent(search_policy, eval_env, outpath="", difficulty=0.5):

    set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:

        state = eval_env.reset()
        goal = (state["grid"]["goal"], state["goal"])
        start = (state["grid"]["observation"], state["observation"])
        search_policy.select_action(state)
        waypoints = search_policy.get_waypoints()
    else:
        goal, observations, waypoints, _ = Collector.get_trajectory(search_policy, eval_env, habitat=True)
        start = observations[0]

    _, ax = plt.subplots()
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape
    normalizing_factor = np.array([height, width])

    goal_grid, goal_visual = goal
    goal_grid = goal_grid / normalizing_factor

    start_grid, start_visual = start
    start_grid = start_grid / normalizing_factor

    waypoints_grid = np.array([wp[0] for wp in waypoints]) / normalizing_factor
    waypoints_visual = [wp[1] for wp in waypoints]

    print(f"Start: {start_grid}")
    print(f"Waypoints: {waypoints_grid}")
    print(f"Goal: {goal_grid}")
    print(f"Steps: {waypoints_grid.shape}")
    print("-" * 10)

    ax = plot_agent_paths(0, start_grid, goal_grid, waypoints_grid, "Search", ax)
    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
        visualize_habitat_agent(
            [start_visual, *waypoints_visual, goal_visual], outpath[:-4] + ".mp4"
        )
    else:
        plt.show()


def visualize_search_path_multi_agent(search_policy, eval_env, num_agents, outpath="", difficulty=0.5):

    set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:

        state = eval_env.reset()
        # Use the sampled start and goal for the first agent
        agent_goal = [(state["grid"]["goal"], state["goal"])]
        agent_start = [(state["grid"]["observation"], state["observation"])]

        # Mutable objects
        state["agent_waypoints"] = agent_goal.copy()
        state["agent_observations"] = agent_start.copy()

        goals = agent_goal.copy()
        starts = agent_start.copy()

        # Sample the starts and goals for the other agents
        for _ in range(num_agents - 1):

            agent_state = eval_env.reset()
            agent_goal = [(agent_state["grid"]["goal"], agent_state["goal"])]
            agent_start = [
                (agent_state["grid"]["observation"], agent_state["observation"])
            ]

            # Add the new observations and goals to the state
            goals.extend(agent_goal.copy())
            starts.extend(agent_start.copy())
            state["agent_waypoints"].extend(agent_goal.copy())
            state["agent_observations"].extend(agent_start.copy())

        # Immutable objects - Should not change ever!
        state["composite_goals"] = goals.copy()
        state["composite_starts"] = starts.copy()
        print("Sampled the required starts and goals")

        search_policy.select_action(state)
        waypoints = search_policy.get_augmented_waypoints()

    else:
        threshold = search_policy.radius
        (
            starts,
            goals,
            _,
            waypoints,
            _,
        ) = Collector.get_trajectories(search_policy, eval_env, num_agents, habitat=True, threshold=threshold)

    _, ax = plt.subplots()
    ax = plot_wall(eval_env.walls, ax)
    height, width = eval_env.walls.shape
    normalizing_factor = np.array([height, width])

    agent_waypoints = []

    for agent_id in range(num_agents):

        agent_goal = goals[agent_id]
        agent_goal_grid, agent_goal_visual = agent_goal
        agent_goal_grid = agent_goal_grid / normalizing_factor

        agent_start = starts[agent_id]
        agent_start_grid, agent_start_visual = agent_start
        agent_start_grid = agent_start_grid / normalizing_factor

        waypoints_grid = np.array([wp[0] for wp in waypoints[agent_id]]) / normalizing_factor
        agent_waypoints.append(waypoints_grid)

        waypoints_visual = [wp[1] for wp in waypoints[agent_id]]

        print(f"Agent: {agent_id}")
        print(f"Start: {agent_start_grid}")
        print(f"Waypoints: {waypoints_grid}")
        print(f"Goal: {agent_goal_grid}")
        print(f"Steps: {waypoints_grid.shape[0] - 1}")
        print("-" * 10)

        ax = plot_agent_paths(
            agent_id,
            agent_start_grid,
            agent_goal_grid,
            waypoints_grid,
            "Search",
            ax
        )

        if len(outpath) > 0:
            save_path = outpath[:-4] + f"_agent_{agent_id}.mp4"
            visualize_habitat_agent([*waypoints_visual, agent_goal_visual], save_path)

    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)
        save_path = outpath[:-4] + ".gif"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(agent_waypoints, eval_env, save_path, (fig, ax), save_fig=True)
    else:
        plt.show()


def visualize_search_path(search_policy, eval_env, outpath="", difficulty=0.5, num_agents=None):

    if num_agents is None:
        visualize_search_path_single_agent(search_policy, eval_env, outpath, difficulty)
    else:
        visualize_search_path_multi_agent(search_policy, eval_env, num_agents, outpath, difficulty)


def visualize_compare_search_single_agent(agent, search_policy, eval_env, seed=0, outpath="", difficulty=0.5):

    set_env_difficulty(eval_env, difficulty)
    plt.figure(figsize=(12, 5))

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = plot_wall(eval_env.walls, ax)
        height, width = eval_env.walls.shape
        normalizing_factor = np.array([height, width])

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        policy = search_policy if use_search else agent

        goal, observations, waypoints, _ = Collector.get_trajectory(policy, eval_env, habitat=True)

        goal_grid, goal_visual = goal
        goal_grid = goal_grid / normalizing_factor

        start = observations[0]
        start_grid, start_visual = start
        start_grid = start_grid / normalizing_factor

        observations_grid = np.array([obs[0] for obs in observations]) / normalizing_factor
        observations_visual = [obs[1] for obs in observations]

        waypoints_grid = np.array([wp[0] for wp in waypoints]) / normalizing_factor
        waypoints_grid = waypoints_grid if use_search else None

        print(f"Policy: {title}")
        print(f"Start: {start_grid}")
        print(f"Goal: {goal_grid}")
        print(f"Steps: {observations_grid.shape[0] - 1}")
        print("-" * 10)

        ax = plot_agent_paths(0, start_grid, goal_grid, observations_grid, title, ax, waypoints_grid)

        if len(outpath) > 0:
            save_path = outpath[:-4] + "_search.mp4" if use_search else outpath[:-4] + "_no_search.mp4"
            visualize_habitat_agent([*observations_visual, goal_visual], save_path)

    plt.savefig(outpath, dpi=300) if len(outpath) > 0 else plt.show()


def visualize_compare_search_multi_agent(agent, search_policy, eval_env, n_agents, seed=0, outpath="", difficulty=0.5):

    set_env_difficulty(eval_env, difficulty)
    plt.figure(figsize=(12, 5))

    search_waypoints = []
    search_observations = []
    no_search_observations = []

    for col_index in range(2):

        title = "No Search" if col_index == 1 else "Search"

        ax = plt.subplot(1, 2, col_index + 1)
        ax = plot_wall(eval_env.walls, ax)
        height, width = eval_env.walls.shape
        normalizing_factor = np.array([height, width])

        use_search = col_index == 0

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        threshold = search_policy.radius
        policy = search_policy if use_search else agent

        if col_index == 0:
            (
                starts,
                goals,
                observations,
                waypoints,
                _,
            ) = Collector.get_trajectories(policy, eval_env, n_agents, habitat=True, threshold=threshold)
        else:
            (
                _,
                _,
                observations,
                waypoints,
                _,
            ) = Collector.get_trajectories(policy, eval_env, n_agents, starts, goals, habitat=True, threshold=threshold)

        print(f"Policy: {title}")
        for agent_id in range(n_agents):

            agent_goal = goals[agent_id]
            agent_goal_grid, agent_goal_visual = agent_goal
            agent_goal_grid = agent_goal_grid / normalizing_factor

            agent_start = observations[agent_id][0]
            agent_start_grid, agent_start_visual = agent_start
            agent_start_grid = agent_start_grid / normalizing_factor

            observations_grid = np.array([obs[0] for obs in observations[agent_id]]) / normalizing_factor
            observations_visual = [obs[1] for obs in observations[agent_id]]

            waypoints_grid = np.array([wp[0] for wp in waypoints[agent_id]]) / normalizing_factor
            waypoints_grid = waypoints_grid if use_search else None

            print(f"Agent: {agent_id}")
            print(f"Start: {agent_start_grid}")
            if use_search:
                search_waypoints.append(waypoints_grid)
                search_observations.append(observations_grid)
                print(f"Waypoints: {waypoints_grid}")
            else:
                no_search_observations.append(observations_grid)
            print(f"Goal: {agent_goal_grid}")
            print(f"Steps: {observations_grid.shape[0] - 1}")
            print("-" * 10)

            ax = plot_agent_paths(
                agent_id,
                agent_start_grid,
                agent_goal_grid,
                observations_grid,
                title,
                ax,
                waypoints_grid
            )

            if len(outpath) > 0:
                if use_search:
                    save_path = outpath[:-4] + f"_search_agent_{agent_id}.mp4"
                else:
                    save_path = outpath[:-4] + f"_no_search_agent_{agent_id}.mp4"
                visualize_habitat_agent([*observations_visual, agent_goal_visual], save_path)

    if len(outpath) > 0:
        plt.savefig(outpath, dpi=300)

        save_path = outpath[:-4] + "_search.gif"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(search_observations, eval_env, save_path, (fig, ax), save_fig=True, waypoints=search_waypoints)

        save_path = outpath[:-4] + "_no_search.gif"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plot_wall(eval_env.walls, ax)
        visualize_path(no_search_observations, eval_env, save_path, (fig, ax), save_fig=True)
    else:
        plt.show()


def visualize_compare_search(agent, search_policy, eval_env, seed=0, outpath="", difficulty=0.5, num_agents=None):

    if num_agents is None:
        visualize_compare_search_single_agent(agent, search_policy, eval_env, seed, outpath, difficulty)
    else:
        visualize_compare_search_multi_agent(agent, search_policy, eval_env, num_agents, seed, outpath, difficulty)
