# !/bin/sh

#env="FourRooms"
env="CentralObstacle"

#device="cpu"
device="cuda:0"

ckpt="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/ckpt/ckpt_0300000"
config="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/bk/bk_config.yaml"

cd "${project_root}"

#debugger_port=5679

lambda_lr=1

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
            --lambda_lr $lambda_lr \
            --visual \
            --pbar
else
    echo "[INFO] running in normal mode"
    python pud/algos/train_lag_policy.py \
        --cfg $config \
        --ckpt $ckpt \
        --device ${device} \
        --lambda_lr $lambda_lr \
        --visual \
        --pbar
fi