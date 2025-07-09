#!/bin/bash

# Configuration for sampling
NUM_SAMPLES=50
VISUAL_FLAG="--visual"

# List of environments to process (skip staging_08)
envs=(
    "sc0_staging_20"
    "sc3_staging_05"
    "sc3_staging_11"
    "sc3_staging_15"
)

# Different agent counts to run for
agent_counts=(1 5 10)

# Difficulties to run
difficulties=("easy" "medium" "hard")

for env in "${envs[@]}"; do
    echo "Processing environment: $env"
    
    # Set paths based on env
    case "$env" in
        "sc0_staging_20")
            unconstrained_ckpt=models/SC0_Staging_20/ckpt/ckpt_0482500
            config=models/SC0_Staging_20/lag/2024-09-11-19-43-42/bk/config.yaml
            constrained_ckpt=models/SC0_Staging_20/lag/2024-09-11-19-43-42/ckpt/ckpt_0250000
            ;;
        "sc3_staging_05")
            unconstrained_ckpt=models/SC3_Staging_05/ckpt/ckpt_0490000
            config=models/SC3_Staging_05/lag/2024-09-11-19-44-18/bk/config.yaml
            constrained_ckpt=models/SC3_Staging_05/lag/2024-09-11-19-44-18/ckpt/ckpt_0207500
            ;;
        "sc3_staging_11")
            unconstrained_ckpt=models/SC3_Staging_11/ckpt/ckpt_0722500
            config=models/SC3_Staging_11/lag/2024-09-11-15-53-23/bk/config.yaml
            constrained_ckpt=models/SC3_Staging_11/lag/2024-09-11-15-53-23/ckpt/ckpt_0460000
            ;;
        "sc3_staging_15")
            unconstrained_ckpt=models/SC3_Staging_15/ckpt/ckpt_0565000
            config=models/SC3_Staging_15/lag/2024-09-11-19-44-43/bk/config.yaml
            constrained_ckpt=models/SC3_Staging_15/lag/2024-09-11-19-44-43/ckpt/ckpt_0247500
            ;;
        *)
            echo "Error: Unknown environment $env"
            continue
            ;;
    esac

    for diff in "${difficulties[@]}"; do
        echo "  Difficulty: $diff"
        problem_set_file=pud/plots/data/${env}/${diff}.npz

        for num_agents in "${agent_counts[@]}"; do
            echo "    Running for num_agents: $num_agents"
            python pud/plots/preprocess_problems.py \
                --config_file $config \
                --constrained_ckpt_file $constrained_ckpt \
                --unconstrained_ckpt_file $unconstrained_ckpt \
                --problem_set_file $problem_set_file \
                --num_agents $num_agents \
                --traj_difficulty $diff \
                $VISUAL_FLAG \
                --num_samples $NUM_SAMPLES
            echo "    Completed num_agents: $num_agents"
        done
        echo "  Finished difficulty: $diff"
        echo "  ---------------------"
    done

    echo "Finished all difficulties and agent counts for $env"
    echo "=========================="

done
