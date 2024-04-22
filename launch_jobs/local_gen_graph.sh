# !/bin/sh

ckpt_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/ckpt/ckpt_0300000"
cfg_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/bk/bk_config.yaml"
figdir="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/visuals"

python pud/algos/gen_graph.py \
    --cfg $cfg_path \
    --ckpt $ckpt_path \
    --figsavedir $figdir