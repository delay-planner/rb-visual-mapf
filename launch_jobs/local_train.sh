# !/bin/sh

#env="FourRooms"
env="CentralObstacle"

comment="rev_pb_sampler_max_1_debug"
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

#debugger_port=5678

cost_max=1.0
cost_N=20

# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/algos/train_PointEnv.py \
        --cfg $config \
        --env $env \
        --cost_max $cost_max \
        --cost_N $cost_N \
        --logdir ${log_dir} \
        --device ${device} \
        --visual \
        --pbar
else
    echo "[INFO] running in normal mode"
    python pud/algos/train_PointEnv.py \
        --cfg $config \
        --env $env \
        --cost_max $cost_max \
        --cost_N $cost_N \
        --logdir ${log_dir} \
        --device ${device} \
        --visual \
        --pbar
fi


