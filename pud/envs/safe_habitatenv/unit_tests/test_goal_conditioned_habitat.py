import unittest
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper, habitat_env_load_fn, HabitatNavigationEnv
from pud.algos.visual_buffer import VisualReplayBuffer

scene = "scene_datasets/habitat-test-scenes/skokloster-castle.glb"
device = "cpu"
simulator_settings = dict(
    scene= "scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    width= 64,
    height= 64,
    default_agent= 0,
    sensor_height= 1.5,
)
apsp_path = "pud/envs/safe_habitatenv/apsps/skokloster/apsp.pickle"

env = habitat_env_load_fn(
            scene=scene,
            height=0,
            action_noise=1.0,
            terminate_on_timeout=False,
            simulator_settings=simulator_settings,
            max_episode_steps=20,
            apsp_path=apsp_path,
            gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),
            device=device,
        )


latent_dimensions = 512
obs_dim = env.observation_space["observation"].shape  # type: ignore
goal_dim = env.observation_space["goal"].shape  # type: ignore
state_dim = (
    latent_dimensions * obs_dim[0] * 2
)  # For each image along cardinal directions and the same for the goal

action_dim = env.action_space.shape[0]  # type: ignore
max_action = float(env.action_space.high[0])  # type: ignore

buffer = VisualReplayBuffer(
    obs_dim=obs_dim,
    goal_dim=goal_dim,
    action_dim=action_dim,
    max_size=1000,
    )

class TestGoalConditionedHabitatEnv(unittest.TestCase):
    def setUp(self):
        import IPython
        IPython.embed(colors="Linux")

        env

    def test_something(self):
        pass

if __name__ == "__main__":
    unittest.main()
