#!/bin/sh
# Slurm sbatch options 
#SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs_debug/logs/job.log-%j
#SBATCH -c 8

##+++++++++
#SBATCH --exclusive
#SBATCH -o /home/gridsan/mfeng1/git_repos/cc-sorb/runs/logs/job.log-%j
# Loading the required module 
source /etc/profile 
module load anaconda/2023a-pytorch
module load cuda/11.8
#mypyenv_load ccrl

# load my light pyenv
MYPYENVROOT=/home/gridsan/mfeng1/my_python_user_bases
MYPYTHONENV=sorb
MYPYTHONUSERBASE="${MYPYENVROOT}/$MYPYTHONENV"
export PATH=/home/gridsan/mfeng1/my_python_user_bases/$MYPYTHONENV/bin:$PATH
export PYTHONUSERBASE=$MYPYTHONUSERBASE

project_root=/home/gridsan/mfeng1/git_repos/cc-sorb
export PYTHONPATH=$project_root:$PYTHONPATH
## -----------------------------------------------------------------------------
env=hatbitat
comment=""
#SLURM_JOB_ID=local_vec
experiment_dir="runs_debug"
log_dir=${experiment_dir}/${env}/job_${SLURM_JOB_ID}_${comment}

echo "project root directory: ${project_root}"
echo "experiment directory: ${log_dir}"

#config=configs/config_SafeHabitatEnv.yaml
#config=configs/config_SafeHabitatEnv_Queue_debug.yaml
#config=configs/config_HabitatEnv.yaml
config=configs/config_HabitatReplicaCAD.yaml
device="cpu"
#device="cuda:0"

cd "${project_root}"

cost_name="linear"
cost_radius=10.0
num_envs=8
embedding_size=256

python pud/envs/safe_habitatenv/unit_tests/train_uvddpg_vec_habitat.py \
    --cfg $config \
    --cost_name $cost_name \
    --cost_radius $cost_radius \
    --logdir ${log_dir} \
    --device ${device} \
    --visual \
    --num_envs ${num_envs} \
    --embedding_size $embedding_size
    #--pbar