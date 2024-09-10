# !/bin/sh

env=hatbitat
comment="debug_safe"
SLURM_JOB_ID=local_debug
experiment_dir="runs_debug/results"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

cd "${project_root}"

#debugger_port=5679

ckpt="runs/hatbitat/job_26896380_visual_cost_correct_flag/2024-08-28-03-58-20/ckpt/ckpt_0335000"
config="runs/hatbitat/job_26896380_visual_cost_correct_flag/2024-08-28-03-58-20/bk/config.yaml"
illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sci_02_staging_08_linear_r1.txt"


ckpt="runs/results/habitat/job_local_sc0_staging_20/2024-09-09-05-58-37/ckpt/ckpt_0480000"
config="runs/results/habitat/job_local_sc0_staging_20/2024-09-09-05-58-37/bk/config.yaml"
illustration_pbs="pud/envs/safe_habitatenv/illustration_set/sc0_staging_20_linear_r1.txt"


lambda_lr=1
collect_steps=20
eval_interval=2500  # 5000 | 10
num_iterations=600000
cost_limit=10.0

sampler_cost_bounds="5-40"
sampler_dist_bounds="0-5"
sampler_K=10
sampler_std_ub=1

device="cuda:0" # must use GPU cluster

# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/algos/train_safe_habitat_lag.py \
        --cfg $config \
        --ckpt $ckpt \
        --collect_steps $collect_steps \
        --eval_interval 20 \
        --cost_limit $cost_limit \
        --lambda_lr $lambda_lr \
        --num_iterations $num_iterations \
        --device ${device} \
        --illustration_pb_file ${illustration_pbs} \
        --sampler_cost_bounds $sampler_cost_bounds \
        --sampler_dist_bounds $sampler_dist_bounds \
        --sampler_K $sampler_K \
        --sampler_std_ub $sampler_std_ub \
        --visual \
        --pbar
else
    echo "[INFO] running in normal mode"
    python pud/algos/train_safe_habitat_lag.py \
            --cfg $config \
            --ckpt $ckpt \
            --collect_steps $collect_steps \
            --eval_interval $eval_interval \
            --cost_limit $cost_limit \
            --lambda_lr $lambda_lr \
            --num_iterations $num_iterations \
            --device ${device} \
            --illustration_pb_file ${illustration_pbs} \
            --sampler_cost_bounds $sampler_cost_bounds \
            --sampler_dist_bounds $sampler_dist_bounds \
            --sampler_K $sampler_K \
            --sampler_std_ub $sampler_std_ub \
            --visual
fi