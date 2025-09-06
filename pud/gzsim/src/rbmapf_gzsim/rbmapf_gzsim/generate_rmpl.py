# Generate RMPL for a multi-drone multi-mission scenario
from argparse import ArgumentParser

number_to_name = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

parser = ArgumentParser(description="Generate RMPL for multi-drone missions")
parser.add_argument("--num-drones", type=str, default=1, help="Number of drones")
parser.add_argument("--num-missions", type=str, default=1, help="Number of missions")
parser.add_argument("--max-mission-time", type=str, default=600, help="Maximum time for each mission in seconds")
parser.add_argument("--min-mission-time", type=str, default=1, help="Minimum time for each mission in seconds")
parser.add_argument("--max-sync-time", type=str, default=600, help="Maximum time for synchronization in seconds")
parser.add_argument("--min-sync-time", type=str, default=1, help="Minimum time for synchronization in seconds")
parser.add_argument("--max-land-time", type=str, default=5, help="Maximum time for landing in seconds")
parser.add_argument("--min-land-time", type=str, default=1, help="Minimum time for landing in seconds")
parser.add_argument("--output-rmpl-path", type=str, default="./src/rbmapf_gzsim/models/rmpls/mission.lisp",
                    help="Path to save the generated RMPL file")
args, _ = parser.parse_known_args()


args.num_drones = int(args.num_drones)
args.num_missions = int(args.num_missions)
args.max_sync_time = int(args.max_sync_time)
args.min_sync_time = int(args.min_sync_time)
args.max_land_time = int(args.max_land_time)
args.min_land_time = int(args.min_land_time)
args.max_mission_time = int(args.max_mission_time)
args.min_mission_time = int(args.min_mission_time)


def generate_rmpl(
    missions=args.num_missions,
    drones=args.num_drones,
    max_mission_time=args.max_mission_time,
    min_mission_time=args.min_mission_time,
    max_sync_time=args.max_sync_time,
    min_sync_time=args.min_sync_time,
    max_land_time=args.max_land_time,
    min_land_time=args.min_land_time,
    mission_rmpl_path=args.output_rmpl_path
):
    """Generate RMPL content for a multi-drone multi-mission scenario."""

    rmpl_content = """
(defpackage #:scenario1)
(in-package #:scenario1)
    """

    # Add a mission for each drone and mission combination
    for mission in range(1, missions + 1):
        for drone in range(1, drones + 1):
            rmpl_content += f"""
(define-control-program start-mission-{number_to_name[mission]}-drone-{number_to_name[drone]} ()
    (declare (primitive)
    (duration (simple :lower-bound {min_mission_time} :upper-bound {max_mission_time})
    :contingent t
)))
            """

        # Add a sync after every mission except the last one
        if mission < missions:
            rmpl_content += f"""
(define-control-program sync-{number_to_name[mission]} ()
    (declare (primitive)
    (duration (simple :lower-bound {min_sync_time} :upper-bound {max_sync_time})
    :contingent t
)))
            """

    # Add a land control program for each drone
    for drone in range(1, drones + 1):
        rmpl_content += f"""
(define-control-program land-drone-{number_to_name[drone]} ()
    (declare (primitive)
    (duration (simple :lower-bound {min_land_time} :upper-bound {max_land_time})
)))
        """

    if drones == 1:
        for n_mission in range(1, 2*missions + 2):
            rmpl_content += f"""
(define-control-program znoop-{number_to_name[n_mission]} ()
    (declare (primitive)
    (duration (simple :lower-bound 0 :upper-bound 0)
)))
            """

    # For each mission, create a parallel sequence that starts the mission for each drone,
    mission_sequences = []
    for mission in range(1, missions + 1):
        parallel_sequence = ""

        # Create embedded parallel sequences (for N drones, need N-1 parallel sequences)
        for _ in range(drones-1):
            parallel_sequence += """
(parallel (:slack t)
            """

        # Add just the first drone
        parallel_sequence += f"""
(start-mission-{number_to_name[mission]}-drone-{number_to_name[1]})
        """

        # For all remaining drones, add them to the parallel sequence and close the parallel sequence
        for drone in range(2, drones + 1):
            parallel_sequence += f"""
(start-mission-{number_to_name[mission]}-drone-{number_to_name[drone]})
            """
            parallel_sequence += ")"
        mission_sequences.append(parallel_sequence)

        # Add a sync after each mission except the last one
        if mission < missions:
            if drones == 1:
                mission_sequences.append(f"(znoop-{number_to_name[2*mission]})")

            mission_sequences.append(f"""
(sync-{number_to_name[mission]})
            """)

            if drones == 1:
                mission_sequences.append(f"(znoop-{number_to_name[2*mission + 1]})")

    # Add the final sequence that lands all drones
    land_sequence = ""
    for _ in range(drones - 1):
        land_sequence += """(parallel (:slack t)"""
    land_sequence += f"""
(land-drone-{number_to_name[1]})
    """

    for drone in range(2, drones + 1):
        land_sequence += f"""
(land-drone-{number_to_name[drone]})
        """
        land_sequence += ")"
    mission_sequences.append(land_sequence)

    # Combine all sequences into the main control program
    rmpl_content += f"""
(define-control-program main ()
    (with-temporal-constraint (simple-temporal :upper-bound {(max_mission_time + max_sync_time) * missions})
    (sequence (:slack nil)
    """
    rmpl_content += "\n".join(mission_sequences)
    rmpl_content += """
))
)
    """

    # Write the RMPL content to a file
    with open(mission_rmpl_path, "w") as file:
        file.write(rmpl_content)


if __name__ == "__main__":
    generate_rmpl()
