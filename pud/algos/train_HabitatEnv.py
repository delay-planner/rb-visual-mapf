import yaml
import torch
import argparse
from pathlib import Path
from dotmap import DotMap
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

from pud.policies import GaussianPolicy
from pud.algos.crl_runner_v2 import train_eval
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.ddpg import GoalConditionedCritic, AutoEncoder
from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointWrapper,
    SafeGoalConditionedHabitatPointQueueWrapper,
    safe_habitat_env_load_fn,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config_SafeHabitatEnv.yaml",
        help="Training configuration",
    )
    parser.add_argument("--cost_name",
        type=str,
        default="",
        help="override cost type")
    parser.add_argument("--cost_radius",
        type=float,
        default=-1,
        help="override cost type")
    parser.add_argument("--cost_max",
        type=float,
        default=-1,
        help="override if > 0")
    parser.add_argument("--cost_N",
        type=int,
        default=0,
        help="override if > 0")
    parser.add_argument("--ckpt",
        type=str,
        default="",
        help="if non-empty, load the checkpoint into agent model")
    parser.add_argument("--i_start",
        type=int,
        default=1,
        help="override the start iteration index, for resume training")
    parser.add_argument("--collect_steps",
        type=int,
        default=-1,
        help="override the number of steps per agent update")
    parser.add_argument("--num_iterations",
        type=int,
        default=-1,
        help="override num of iterations")
    parser.add_argument("--cost_limit",
        type=float,
        default=-1,
        help="override cost limit")
    parser.add_argument("--lambda_lr",
        type=float,
        default=-1,
        help="override lagrange lr")
    parser.add_argument("--illustration_pb_file",
        type=str,
        help="problems that serve as illustration and evaluation set")
    parser.add_argument("--logdir", type=str, default="", help="Override ckpt dir")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--pbar", action="store_true", help="Show progress bar")
    parser.add_argument("--visual", action="store_true", help="generate and save visual trajs")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose printing/logging"
    )
    args = parser.parse_args()

    cfg = {}
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)


    # Override cfs from terminal
    if len(args.logdir) > 0:
        cfg.ckpt_dir = args.logdir
    if len(args.cost_name) > 0:
        cfg.cost_function.name = args.cost_name
    if args.cost_radius > 0:
        cfg.cost_function.radius = args.cost_radius
    if args.cost_max > 0:
        cfg.agent.cost_max = args.cost_max
    if args.cost_N > 0:
        cfg.agent.cost_N = args.cost_N
    if args.lambda_lr > 0:
        cfg.agent.lambda_lr = args.lambda_lr
    if args.cost_limit >= 0:
        cfg.agent.cost_limit = args.cost_limit
    if args.num_iterations > 0:
        cfg.runner.num_iterations = args.num_iterations
    if args.collect_steps > 0:
        cfg.runner.collect_steps = args.collect_steps
    cfg.runner.verbose = args.verbose
    cfg.device = args.device
    cfg.pprint()

    # Custom Logging
    log_dir = Path(cfg.ckpt_dir)
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = log_dir.joinpath(date_time)
    ckpt_dir = log_dir.joinpath("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tfevent_dir = log_dir.joinpath("tfevent")
    tfevent_dir.mkdir(parents=True, exist_ok=True)
    bk_dir = log_dir.joinpath("bk")
    bk_dir.mkdir(parents=True, exist_ok=True)
    with open(bk_dir.joinpath("config.yaml"), "w") as f:
        yaml.safe_dump(data=cfg.toDict(), stream=f, allow_unicode=True, indent=4)
    tb = SummaryWriter(log_dir=tfevent_dir.as_posix())


    set_global_seed(cfg.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in cfg.wrappers:
        if wrapper_name == "SafeGoalConditionedHabitatPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedHabitatPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedHabitatPointQueueWrapper)
            gym_env_wrapper_kwargs.append(cfg.wrappers[wrapper_name].toDict())

    cfg.env.use_gpu = args.device!="cpu"
    env = safe_habitat_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=False,
    )

    set_env_seed(env, cfg.seed + 1)

    eval_env = safe_habitat_env_load_fn(
        cfg.env.toDict(),
        cfg.cost_function.toDict(),
        max_episode_steps=cfg.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,  # type: ignore
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=False,
    )
    set_env_seed(eval_env, cfg.seed + 2)

    # TODO: Need to be able to consume this observation data
    latent_dimensions = 512
    obs_dim = env.observation_space["observation"].shape  # type: ignore
    goal_dim = env.observation_space["goal"].shape  # type: ignore
    state_dim = (
        latent_dimensions * obs_dim[0] * 2
    )  # For each image along cardinal directions and the same for the goal

    action_dim = env.action_space.shape[0]  # type: ignore
    max_action = float(env.action_space.high[0])  # type: ignore

    print(
        f"Observation dim: {obs_dim},\n"
        f"Goal dim: {goal_dim},\n"
        f"State dim: {state_dim},\n"
        f"Action dim: {action_dim},\n"
        f"Max action: {max_action}"
    )

    image_params = {
        "input_channels": obs_dim[-1],
    }

    agent = DRLDDPGLag(
        state_dim,
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(cfg.device),
        **cfg.agent,
        AutoEncoderCls=AutoEncoder,
        obs_dim=obs_dim,
        latent_dimension=latent_dimensions,
    )
    agent.to(torch.device(args.device))

    print("Agent ", agent)

    replay_buffer = ConstrainedReplayBuffer(
        obs_dim, goal_dim, action_dim, **cfg.replay_buffer
    )

    policy = GaussianPolicy(agent)

    train_eval(
        policy,
        agent,
        replay_buffer,
        env,
        eval_env,
        eval_func=None,
        tensorboard_writer=tb,
        pbar=args.pbar,
        ckpt_dir=ckpt_dir,
        **cfg.runner,
    )
    torch.save(agent.state_dict(), ckpt_dir.joinpath("agent.pth"))
