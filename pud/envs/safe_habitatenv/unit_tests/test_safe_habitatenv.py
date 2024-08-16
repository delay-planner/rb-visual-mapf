import unittest

from pud.envs.safe_habitatenv.safe_habitatenv import (
    SafeHabitatNavigationEnv, 
)

"""

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.reset

python pud/envs/safe_habitatenv/unit_tests/test_safe_habitatenv.py TestSafeHabitatEnv.stepD
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

    def reset(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "cosine", "radius": 2.0},
            cost_limit=10.0,
        )
        env.reset()

    def step(self):
        env = SafeHabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
            device="cuda:1",
            cost_f_args={"name": "cosine", "radius": 2.0},
            cost_limit=10.0,
        )
        s0, info = env.reset()  # type: ignore
        action = env.action_space.sample()
        env.step(action)


if __name__ == "__main__":
    unittest.main()
