import unittest
import habitat_sim
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

if __name__ == "__main__":
    unittest.main()
