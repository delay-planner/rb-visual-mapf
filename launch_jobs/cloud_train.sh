#!/bin/bash 

# Slurm sbatch options 
#SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs/logs/job.log-%j
#SBATCH -c 8
#++++SBATCH --exclusive

# Loading the required module 
source /etc/profile 
module load anaconda/2022b

# load my light pyenv
MYPYENVROOT=/home/gridsan/mfeng1/my_python_user_bases
MYPYTHONENV=ccrl
MYPYTHONUSERBASE="${MYPYENVROOT}/$MYPYTHONENV"
export PATH=/home/gridsan/mfeng1/my_python_user_bases/$MYPYTHONENV/bin:$PATH
export PYTHONUSERBASE=$MYPYTHONUSERBASE

project_root=/home/gridsan/mfeng1/git_repos/cc-sorb
export PYTHONPATH=$project_root:$PYTHONPATH

comment="short_range"

experiment_dir="runs/results"
log_dir=${experiment_dir}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

export TQDM_DISABLE=1

cd "${project_root}"
python pud/algos/train_PointEnv.py \
    --cfg configs/config_SafePointEnv.yaml \
    --logdir ${log_dir}