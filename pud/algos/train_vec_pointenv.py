from pud.dependencies import *
from pud.utils import set_global_seed, set_env_seed, AttrDict
import yaml
from dotmap import DotMap
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from termcolor import cprint
import shutil
from typing import List

def setup_logger(root_dir:str, subdir_names:List[str], tag_time:bool=False, verbose=True):
    log_dir = Path(root_dir)
    if tag_time:
        from datetime import datetime
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = log_dir.joinpath(date_time)

    log_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        cprint("[Logger] root directory: {}".format(log_dir.as_posix()), "green")
    logger = {"log_dir": log_dir}
    for name in subdir_names:
        subdir = log_dir.joinpath(name)
        subdir.mkdir(parents=True, exist_ok=True)
        logger[name] = subdir

    return logger

num_envs = 4

cfg = {}
with open(sys.argv[-1], 'r') as f:
    cfg = yaml.safe_load(f)
# for dot completion
cfg = DotMap(cfg)
cfg.num_envs = num_envs
cfg.pprint()
set_global_seed(cfg.seed)

# Custom Logging
logger = setup_logger(
    root_dir=cfg.ckpt_dir, 
    subdir_names=["ckpt", "tfevent", "bk",], # "imgs"
    tag_time=True,
    )
with open(logger["bk"].joinpath("config.yaml"), "w") as f:
    yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
logger["tb"] = SummaryWriter(log_dir=logger["tfevent"].as_posix())

from pud.envs.simple_navigation_env import env_load_fn

envs = [
    env_load_fn(
    cfg.env.env_name, cfg.env.max_episode_steps,
    resize_factor=cfg.env.resize_factor,
    terminate_on_timeout=False,
    thin=cfg.env.thin
    ) for _ in range(num_envs)
]
for i in range(num_envs):
    set_env_seed(envs[i], cfg.seed + i)
    #set_env_seed(envs[i], cfg.seed)
env = envs[0] # to help initialize other modules

eval_env = env_load_fn(cfg.env.env_name, cfg.env.max_episode_steps,
                       resize_factor=cfg.env.resize_factor,
                       terminate_on_timeout=True,
                       thin=cfg.env.thin)
set_env_seed(eval_env, cfg.seed + num_envs + 1)
#set_env_seed(eval_env, cfg.seed + 1)

obs_dim = env.observation_space['observation'].shape[0]
goal_dim = obs_dim
state_dim = obs_dim + goal_dim
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print(f'obs dim: {obs_dim}, goal dim: {goal_dim}, state dim: {state_dim}, action dim: {action_dim}, max action: {max_action}')

from pud.ddpg import UVFDDPG
agent = UVFDDPG(
    state_dim, # concatenating obs and goal
    action_dim,
    max_action,
    **cfg.agent,
)
print(agent)

#agent.to(device="cuda:1")

from pud.buffer import ReplayBuffer
cfg.replay_buffer.max_size = cfg.replay_buffer.max_size * num_envs
replay_buffer = ReplayBuffer(obs_dim, goal_dim, action_dim, **cfg.replay_buffer)


logger["eval_distances"] = [2, 5, 10]

if True:
    from pud.policies import GaussianPolicy
    policy = GaussianPolicy(agent)

    #from pud.runner import train_eval, eval_pointenv_dists
    from pud.runner_vec import train_eval, eval_pointenv_dists
    train_eval(policy,
               agent,
               replay_buffer,
               env=envs,
               eval_env=eval_env,
               eval_func=eval_pointenv_dists,
               logger=logger,
               **cfg.runner,
              )
    torch.save(agent.state_dict(), logger["ckpt"].joinpath("agent.pth"))
