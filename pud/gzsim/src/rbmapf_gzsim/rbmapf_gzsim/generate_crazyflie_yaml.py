import sys
import yaml
import numpy as np
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--num_drones', required=True)
    parser.add_argument('--starts_file', required=True)
    args = parser.parse_args()

    num_drones = int(args.num_drones)
    starts = np.loadtxt(args.starts_file, delimiter=',')
    if starts.ndim == 1:
        starts = starts.reshape(1, -1)
    data = {
        'fileversion': 3,
        'robots': {},
        'robot_types': {
            'cf21': {
                'motion_capture': {
                    'tracking': 'librigidbodytracker',
                    'marker': 'default_single_marker',
                    'dynamics': 'fast_cf'   # 'default'
                },
                'big_quad': False,
                'battery': {
                    'voltage_warning': 3.8,
                    'voltage_critical': 3.7
                }
            }
        },
        'all': {
            'firmware_logging': {
                'enabled': True,
                'default_topics': {
                    'pose': {'frequency': 10},
                    'status': {'frequency': 1}
                }
            },
            'firmware_params': {
                'commander': {'enHighLevel': 1},
                'stabilizer': {'estimator': 2, 'controller': 2},
                'locSrv': {'extPosStdDev': 1e-3, 'extQuatStdDev': 0.5e-1},
                # 'colAv': {'enable': 1}
            },
            'reference_frame': 'world',
            'broadcasts': {
                'num_repeats': 15,
                'delay_between_repeats_ms': 1
            }
        }
    }

    initial_positions = [
        # [0.0, 0.0, 0.0],
        # [3.185, 1.792, 0.781], # Old small marker
        [3.15, 1.86, 0.8], # New medium marker
        # [3.13, 1.95, 0.8],  # New big marker
        # [0.0, -1.0, 0.0],
        # [3.220, 1.080, 0.782],
        [3.22, 1.00, 0.8],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ]

    for idx in range(1, num_drones + 1):
        drone_id = f'cf{idx}'
        x, y = float(starts[idx - 1, 0]), float(starts[idx - 1, 1])
        uri_suffix = f'E7E7E7E7{str(idx + 2).zfill(2)}'  # TODO: Change the idx + 1 back to idx later!!
        # Need this to avoid issues with drone flipping. Not sure why. 
        # One reason is that the original starts are not actually where 
        # the drone takesoff as its hard to place them exactly there in
        # the highbay
        initial_position = initial_positions[(idx - 1) % len(initial_positions)]

        data['robots'][drone_id] = {
            'enabled': True,
            'uri': f'radio://0/80/2M/{uri_suffix}',
            'initial_position': initial_position,
            'type': 'cf21'
        }

    yaml_data = yaml.dump(data, sort_keys=False)
    with open(args.output_path, "w") as f:
        f.write(yaml_data)

if __name__ == "__main__":
    main()