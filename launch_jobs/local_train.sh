# !/bin/sh

#env="FourRooms"
env="CentralObstacle"

comment="rev_pb_sampler"
SLURM_JOB_ID=local
experiment_dir="runs/results"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

#config="configs/config_SafePointEnv.yaml"
config="configs/config_PointEnv_Queue.yaml"
#config="configs/config_PointEnv_Queue_debug.yaml"
#config="configs/config_SafePointEnv_debug.yaml"

#device="cpu"
device="cuda:0"

cd "${project_root}"
python pud/algos/train_PointEnv.py \
    --cfg $config \
    --env $env \
    --logdir ${log_dir} \
    --device ${device} \
    --visual \
    --pbar