#!/bin/bash

env=$1
num_samples=50
config_file=$2
agents=(1 5 10 20)
constrained_ckpt_file=$4
unconstrained_ckpt_file=$3
problem_types=("hard" "medium" "easy")
method_types=(
    "collect_bounds_data"
    "unconstrained" 
    "unconstrained_reward_search" 
    "constrained" 
    "constrained_reward_search" 
    "constrained_risk_search" 
    "full_constrained_reward_search" 
    "full_constrained_risk_search" 
    "lagrangian_search" 
    "biobjective_search"
    "risk_budgeted_search"
    "risk_bounded_uniform_search"
    "risk_bounded_utility_search"
    "risk_bounded_inverse_utility_search"
)

env_options=("centerdot" "sc0_staging_20" "sc2_staging_08" "sc3_staging_05" "sc3_staging_11" "sc3_staging_15") 
if [[ ! " ${env_options[@]} " =~ " ${env} " ]]; then
    echo $baseline
    echo "Error: Invalid env provided. Please choose from: ${env_options[*]}"
    exit 1
fi

if [[ $env == *"staging"* ]]; then
    visual="--visual"
    agents=(1 5 10)
else
    visual=""
fi

collect_trajectories() {
    while true; do
        python -u pud/plots/collect_safe_trajectory_records.py --config_file ${config_file} --unconstrained_ckpt_file ${unconstrained_ckpt_file} --constrained_ckpt_file ${constrained_ckpt_file} --load_problem_set --problem_set_file ${problem_set_file} --num_samples ${num_samples} --method_type ${method_type} --num_agents ${num_agent} ${visual} --traj_difficulty ${problem_type}
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Script completed successfully."
            break
        fi
        echo "Script crashed with exit code $EXIT_CODE. Restarting..." >&2
    sleep 1
    done
}

# Sample problems
if [[ $5 = true ]]; then
    for problem_type in "${problem_types[@]}"; do
        printf "%*s\n" 100 | tr ' ' '*'
        echo "Sampling problems for ${env} on ${problem_type} problems"
        python -u pud/plots/collect_safe_trajectory_records.py --config_file ${config_file} --unconstrained_ckpt_file ${unconstrained_ckpt_file} --constrained_ckpt_file ${constrained_ckpt_file} --collect_trajs --traj_difficulty ${problem_type} --num_samples ${num_samples} --num_agents 25 ${visual}
    done
fi

# Collect the trajectories
for problem_type in "${problem_types[@]}"; do
    problem_set_file=pud/plots/data/${env}/${problem_type}.npz
    for num_agent in "${agents[@]}"; do
        printf "%*s\n" 100 | tr ' ' '-'
        echo "Collecting trajectories for ${env} with ${num_agent} agents on ${problem_type} problems"
        printf "%*s\n" 100 | tr ' ' '-'
        for method_type in "${method_types[@]}"; do
            printf "%*s\n" 50 | tr ' ' '*'
            echo "Method type: ${method_type}"
            printf "%*s\n" 50 | tr ' ' '*'
            if [[ $method_type == *"ds" ]] && [ $num_agent -eq 1 ]; then
                continue
            fi
            collect_trajectories
        done
    done
done
