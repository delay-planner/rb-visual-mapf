import unittest
from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper, habitat_env_load_fn, HabitatNavigationEnv

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


class TestGoalConditionedHabitatEnv(unittest.TestCase):
    def setUp(self):
        import IPython
        IPython.embed(colors="Linux")

        env

    def test_something(self):
        pass

if __name__ == "__main__":
    unittest.main()
