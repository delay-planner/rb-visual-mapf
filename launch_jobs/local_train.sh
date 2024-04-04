# !/bin/sh

comment="cfg_init"
SLURM_JOB_ID=local
experiment_dir="runs/results"
log_dir=${experiment_dir}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

config="configs/config_SafePointEnv.yaml"
#config="configs/config_SafePointEnv_debug.yaml"

cd "${project_root}"
python pud/algos/train_PointEnv.py \
    --cfg $config \
    --logdir ${log_dir} \
    --device cuda:0 \
    --pbar