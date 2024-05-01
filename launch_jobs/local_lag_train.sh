# !/bin/sh

#env="FourRooms"
env="CentralObstacle"

#device="cpu"
device="cuda:0"

ckpt="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/ckpt/ckpt_0300000"
config="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/bk/bk_config.yaml"

ckpt="runs/results/CentralObstacle/job_local_rev_pb_sampler_max_2_debug/2024-04-29-16-42-03/ckpt/ckpt_0300000"
config="runs/results/CentralObstacle/job_local_rev_pb_sampler_max_2_debug/2024-04-29-16-42-03/bk/bk_config.yaml"

cd "${project_root}"

#debugger_port=5679
illustration_pb_file="pud/envs/safe_pointenv/illustration_set/CentralObstacle.txt"

additional_comment=""

lambda_lr=0.001
cost_limit=0

if [[ -n ${debugger_port} ]]; then
    echo "[INFO] running in debug mode"
    echo "[Action Needed] need to launch the debugger client at port ${debugger_port}"
    python -m debugpy \
        --listen localhost:${debugger_port} \
        --wait-for-client \
        pud/algos/train_lag_policy.py \
            --cfg $config \
            --ckpt $ckpt \
            --device ${device} \
            --cost_limit $cost_limit \
            --illustration_pb_file $illustration_pb_file \
            --lambda_lr $lambda_lr \
            --additional_comment $additional_comment \
            --visual \
            --pbar
else
    echo "[INFO] running in normal mode"
    python pud/algos/train_lag_policy.py \
        --cfg $config \
        --ckpt $ckpt \
        --device ${device} \
        --cost_limit $cost_limit \
        --illustration_pb_file $illustration_pb_file \
        --lambda_lr $lambda_lr \
        --additional_comment $additional_comment \
        --visual \
        --pbar
fi