import unittest
import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

"""
mostly copied from 
habitat-sim/examples/tutorials/nb_python/ReplicaCAD_quickstart.py
"""

"""
python pud/envs/safe_habitatenv/unit_tests/test_replica_cad_barebone.py TestReplicaCADBarebone.test_replica_cad_in_habitat_env

python pud/envs/safe_habitatenv/unit_tests/test_replica_cad_barebone.py TestReplicaCADBarebone.vis_handed_crafted_waypoints

opencv-python is problem

perhaps flip the image upside down?
"""


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Specify the location of the scene dataset
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]
    if "override_scene_light_defaults" in settings:
        sim_cfg.override_scene_light_defaults = settings[
            "override_scene_light_defaults"
        ]
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
    color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
    color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_1st_person_spec.resolution = [
        settings["height"],
        settings["width"],
    ]
    color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_1st_person_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_1st_person_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_default_settings():
    settings = {
        "width": 1280,  # Spatial resolution of the observations
        "height": 720,
        "scene_dataset": "replica cad dataset path",  # dataset path
        "scene": "sc1_staging_00",  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": 0.0,  # sensor pitch (x rotation in rads)
        "seed": 1,
        "enable_physics": False,  # enable dynamics simulation
    }
    return settings

class TestReplicaCADBarebone(unittest.TestCase):
    def test_construct_sim(self):
        scene_dataset = "external_data/replica_cad/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"
        settings = make_default_settings()
        settings["scene_dataset"] = scene_dataset

        cfg = make_cfg(settings)
        sim = habitat_sim.Simulator(cfg)

    def test_replica_cad_in_habitat_env(self):
        from pud.envs.habitat_navigation_env import HabitatNavigationEnv
        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
        )
        print("walls shape: {}".format(env._walls.shape))

        walls = env._walls.copy()
        fig, ax = plt.subplots()
        # 1 is navigatbale, 0 is obstacle
        # convert to the convention of pointenv
        walls = 1 - walls
        walls = walls.T
        (height, width) = walls.shape
        # only plot walls
        for (i, j) in zip(*np.where(walls)):
            x = np.array([j, j+1]) / float(width)
            y0 = np.array([i, i]) / float(height)
            y1 = np.array([i+1, i+1]) / float(height)
            ax.fill_between(x, y0, y1, color='grey')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

        fig.savefig("runs/tmp_plots/replicad_cad_3d.jpg", dpi=300)

    def vis_handed_crafted_waypoints(self):
        from pud.envs.habitat_navigation_env import HabitatNavigationEnv
        env = HabitatNavigationEnv(
            env_type="ReplicaCAD",
            sensor_type="rgb",
        )

        height, width = env._walls.shape
        waypoints = np.loadtxt("runs/tmp_plots/waypoints.txt", delimiter=",")
        #waypoints = np.fliplr(waypoints)
        # waypoints in 2d grid
        waypoints = waypoints * np.array([height,width], dtype=float)
        obs_at_waypoints = [env.get_sensor_obs_at_grid_xy(wp) for wp in waypoints]
        
        assert env.sensor_type == "rgb"
        pbar = tqdm(total=len(obs_at_waypoints))
        for i_obs, obs_cat in enumerate(obs_at_waypoints):
            fig, ax = plt.subplots(nrows=2, ncols=2)
            for i in range(4):
                ax[i%2,i//2].imshow((obs_cat[i]).astype(dtype="uint8"))

            target_dir = Path("runs/tmp_plots/trace_bounds/")
            fig_path = target_dir.joinpath("trace_bounds_{:0>2d}.jpg".format(i_obs))
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            pbar.update()

if __name__ == "__main__":
    unittest.main()
