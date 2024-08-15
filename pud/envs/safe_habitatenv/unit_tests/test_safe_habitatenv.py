import unittest
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt

from pud.envs.safe_habitatenv.safe_habitatenv import (
    SafeHabitatNavigationEnv,
    display_map,
)
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    safe_habitat_env_load_fn,
)

"""
python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.test_reset

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.test_plot_safe_environment


"""


class TestSafeHabitatEnv(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = dict(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "cosine", "radius": 2.0},
            cost_limit=1.0,
        )

        self.habitat_env = SafeHabitatNavigationEnv(**self.env_kwargs)

    def test_safe_habitat_env_load_fn(self):
        """
        Test the environment loader with wrappers
        TimeLimit is loaded by default if max_episode_steps > 0
        """

        env_args = deepcopy(self.env_kwargs)
        env_args.update(self.precompilation_kwargs)
        gym_env_wrappers = [SafeGoalConditionedHabitatPointWrapper]
        env = safe_habitat_env_load_fn(
            env_args,
            self.cost_fn_kwargs,
            max_episode_steps=20,
            gym_env_wrappers=gym_env_wrappers,  # type: ignore
            terminate_on_timeout=False,
        )

        state, info = env.reset()  # type: ignore
        self.assertTrue("cost" in info)
        self.assertTrue("goal" in state)
        self.assertTrue("observation" in state)
        self.assertTrue(state["observation"].shape == (4, 256, 256, 4))  # type: ignore

        for _ in range(40):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.assertTrue("cost" in info)
            self.assertTrue("goal" in next_state)
            self.assertTrue("observation" in next_state)
            self.assertTrue(next_state["observation"].shape == (4, 256, 256, 4))

    def test_plot_safe_environment(self):
        output_dir = Path("pud/envs/safe_habitatenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        title = "Safe Habitat Test Visual"
        for cost_ub in [0, 1, 2]:
            fig, ax = plt.subplots()
            _ = display_map(
                self.habitat_env.walls,
                title,
                self.habitat_env.cost_map,
                cost_ub,  # type: ignore
                ax=ax,
            )
            fig.savefig(
                "pud/envs/safe_habitatenv/unit_tests/outputs/{}_cost={:.2f}.jpg".format(
                    title, cost_ub
                ),
                dpi=300,
            )
            plt.close(fig)

    @unittest.skip("Skipping the test as it is not necessary")
    def test_observations(self):
        output_dir = Path("pud/envs/safe_habitatenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_arrays = []
        titles = ["Forward", "Right", "Backward", "Left"]

        env_args = deepcopy(self.env_kwargs)
        env_args.update
        gym_env_wrappers = [SafeGoalConditionedHabitatPointWrapper]
        env = safe_habitat_env_load_fn(
            env_args,
            self.cost_fn_kwargs,
            max_episode_steps=20,
            gym_env_wrappers=gym_env_wrappers,  # type: ignore
            terminate_on_timeout=False,
        )

        done = False
        state, info = env.reset()  # type: ignore
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            observations = next_state["observation"]

            image_arr = []
            for idx, obs in enumerate(observations):
                rgb_img = Image.fromarray(obs, mode="RGBA")
                image_arr.append(rgb_img)
            image_arrays.append(image_arr)

        fig = plt.figure(figsize=(12, 8))

        image_counter = 1
        image_arr = []
        for image_direction in range(len(image_arrays[0])):
            ax = plt.subplot(1, 4, image_direction + 1)
            ax.axis("off")
            ax.set_title(titles[image_direction])
            ax_im = ax.imshow(image_arrays[0][image_direction])
            image_arr.append(ax_im)

        def update(*args):
            nonlocal image_counter
            if image_counter >= len(image_arrays) - 1:  # type: ignore
                image_counter = 0  # type: ignore
            else:
                image_counter += 1
            print("Image counter = ", image_counter)
            print("Length of image arrays = ", len(image_arrays))
            for j, image in enumerate(image_arrays[image_counter]):
                image_arr[j].set_array(image)
            return image_arr

        ani = FuncAnimation(
            fig, update, fargs=(image_counter,), frames=len(image_arrays), blit=True
        )
        ani.save(
            "pud/envs/safe_habitatenv/unit_tests/outputs/observations.gif",
            writer="pillow",
            fps=1,
        )

    @unittest.skip("Skipping the test as it is not necessary")
    def test_topdown_view(self):
        output_dir = Path("pud/envs/safe_habitatenv/unit_tests/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectory = []

        state, info = self.habitat_env.reset()  # type: ignore
        agent_position = self.habitat_env.get_xy_in_habitat()
        agent_grid_position = self.habitat_env.get_grid_xy_from_habitat_xy(agent_position)
        trajectory.append(np.array([agent_grid_position[0], agent_grid_position[1]]))
        for _ in range(30):
            action = self.habitat_env.action_space.sample()
            next_state, reward, done, info = self.habitat_env.step(action)
            agent_position = self.habitat_env.get_xy_in_habitat()
            agent_grid_position = self.habitat_env.get_grid_xy_from_habitat_xy(agent_position)
            trajectory.append(
                np.array([agent_grid_position[0], agent_grid_position[1]])
            )

            self.assertTrue(not self.habitat_env._is_blocked(agent_position))

        title = "Skokloster Castle"
        fig, ax = plt.subplots()
        ax = display_map(
            self.habitat_env.walls,
            title,
            self.habitat_env.cost_map,
            1.0,
            ax=ax,
            key_points=np.array(trajectory),
        )
        fig.savefig(
            "pud/envs/safe_habitatenv/unit_tests/outputs/topdown_view_{}_trajectory.jpg".format(
                title
            ),
            dpi=300,
        )

    def test_reset(self):
        import IPython
        IPython.embed(colors="Linux")
        for _ in range(100):
            state, info = self.habitat_env.reset()  # type: ignore
            agent_position = self.habitat_env.get_xy_in_habitat()
            cx, cy = self.habitat_env.get_grid_xy_from_habitat_xy(agent_position)
            sample_cost = self.habitat_env.get_state_cost(np.array([cx, cy]))
            self.assertTrue(
                sample_cost < self.habitat_env.cost_limit,
                msg="Sample = {}, Sample Cost = {}, Cost Map = {}, Cost Limit = {}".format(
                    (cx, cy),
                    sample_cost,
                    self.habitat_env.cost_map[int(cx), int(cy)],
                    self.habitat_env.cost_limit,
                ),
            )

            self.assertTrue(not self.habitat_env._is_blocked(agent_position))

    def test_step(self):
        s0, info = self.habitat_env.reset()  # type: ignore
        action = self.habitat_env.action_space.sample()
        self.habitat_env.step(action)

    def test_sim_to_grid_and_back_conversions(self):

        for _ in range(100):
            state, info = self.habitat_env.reset()  # type: ignore
            agent_position = self.habitat_env.get_xy_in_habitat()
            agent_grid_position_x, agent_grid_position_y = (
                self.habitat_env.get_grid_xy_from_habitat_xy(agent_position)
            )
            converted_agent_position_x, converted_agent_position_y = (
                self.habitat_env.get_xy_in_habitat_from_xy_in_grid(
                    (agent_grid_position_x, agent_grid_position_y)
                )
            )
            converted_agent_position = np.array(
                [converted_agent_position_x, converted_agent_position_y]
            )
            reconverted_agent_grid_position_x, reconverted_agent_grid_position_y = (
                self.habitat_env.get_grid_xy_from_habitat_xy(converted_agent_position)
            )
            self.assertTrue(np.allclose(agent_position, converted_agent_position))
            self.assertTrue(
                agent_grid_position_x == reconverted_agent_grid_position_x
                and agent_grid_position_y == reconverted_agent_grid_position_y
            )


if __name__ == "__main__":
    unittest.main()
