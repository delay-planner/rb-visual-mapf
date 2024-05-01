# !/bin/sh

ckpt_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/ckpt/ckpt_0260000"
cfg_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/bk/bk_config.yaml"
figdir="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/visuals"


ckpt_path="runs/results/CentralObstacle/job_local_rev_pb_sampler_max_2_debug/2024-04-29-16-42-03/ckpt/ckpt_0300000"
cfg_path="runs/results/CentralObstacle/job_local_rev_pb_sampler_max_2_debug/2024-04-29-16-42-03/bk/bk_config.yaml"
figdir="runs/results/CentralObstacle/job_local_rev_pb_sampler_max_2_debug/2024-04-29-16-42-03/"

problem_file="pud/envs/safe_pointenv/illustration_set/CentralObstacle.txt"

figname="pb_CentralObstacle.jpg"

python pud/algos/unit_tests/eval_illus_ps.py \
    --cfg $cfg_path \
    --ckpt $ckpt_path \
    --figsavedir $figdir \
    --problem_file $problem_file \
    --figname $figname