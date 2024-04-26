# !/bin/sh

ckpt_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/ckpt/ckpt_0260000"
cfg_path="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/bk/bk_config.yaml"
figdir="runs/results/job_local_self_train_eval_with_lag/2024-04-20-18-31-00/visuals"

K=100
N=200
target=0.8
min_dist=0
max_dist=10
figname="pb_${target}_dist_${min_dist}-${max_dist}_K=${K}_N=${N}.jpg"

python pud/algos/unit_tests/vis_sampler.py \
    --cfg $cfg_path \
    --ckpt $ckpt_path \
    --figsavedir $figdir \
    --K $K \
    --N $N \
    --target $target \
    --min_dist $min_dist \
    --max_dist $max_dist \
    --figname $figname