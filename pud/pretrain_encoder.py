from pud.visual_encode import VisualEncoder, VisualDecoder

from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper, habitat_env_load_fn
import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim",
        type=int,
        default=256,
        help="")
    parser.add_argument("--scene",
        type=str,
        default="scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        help="")
    parser.add_argument("--apsp_path",
        type=str,
        default="pud/envs/safe_habitatenv/apsps/skokloster/apsp.pickle",
        help="")
    parser.add_argument("--train_itrs",
        type=int,
        default=1000,
        help="num of training itrs")
    parser.add_argument("--lr",
        type=float,
        default=1e-3,
        help="learning rate")
    parser.add_argument("--save_after_steps",
        type=int,
        default=1000,
        help="save after every x steps")
    parser.add_argument("--logdir",
        type=str,
        help="dir to save checkpoints and training curves")
    parser.add_argument("--train_batch_size",
        type=int,
        default=4,
        help="batch size")
    parser.add_argument("--device",
        type=str,
        default="cpu",
        help="cpu or cuda:x")
    
    args = parser.parse_args()


    log_dir_p = Path(args.logdir)
    log_dir_p.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = log_dir_p.joinpath("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_enc_dir = ckpt_dir.joinpath("enc")
    ckpt_enc_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dec_dir = ckpt_dir.joinpath("dec")
    ckpt_dec_dir.mkdir(parents=True, exist_ok=True)

    recon_dir = log_dir_p.joinpath("reconstruction")
    recon_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = log_dir_p.joinpath("tb")
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir.as_posix())

    scene = args.scene
    device = "cpu"
    simulator_settings = dict(
        scene= "scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        width= 64,
        height= 64,
        default_agent= 0,
        sensor_height= 1.5,
    )
    env = habitat_env_load_fn(
                scene=scene,
                height=0,
                action_noise=1.0,
                terminate_on_timeout=False,
                simulator_settings=simulator_settings,
                max_episode_steps=20,
                apsp_path=args.apsp_path,
                gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),
                device=args.device,
            )

    latent_dimensions = args.emb_dim
    obs_dim = env.observation_space["observation"].shape  # type: ignore
    goal_dim = env.observation_space["goal"].shape  # type: ignore

    action_dim = env.action_space.shape[0]  # type: ignore
    max_action = float(env.action_space.high[0])  # type: ignore


    enc = VisualEncoder(embedding_size=args.emb_dim)
    dec = VisualDecoder(emb_size=args.emb_dim)


    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(params=list(enc.parameters())+list(dec.parameters()),
                        lr=args.lr)


    train_batch_size = args.train_batch_size
    num_itrs = args.train_itrs

    for i in tqdm(range(num_itrs),total=num_itrs, disable=True):
        obs = np.zeros([train_batch_size*4, 64, 64 ,4])
        for j in range(train_batch_size):
            state = env.reset()
            obs[j*4:j*4+4] = state["observation"]
        obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2).float()
        emb = enc(obs_t)
        rec = dec(emb)

        loss = loss_fn(rec, obs_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar(
            tag="reconstruction loss",
            scalar_value=loss.item(),
            global_step=i,
        )

        if i % args.save_after_steps == 0:

            enc_path = ckpt_enc_dir.joinpath("enc_{:0>6d}.ckpt".format(i))
            torch.save(enc, enc_path.as_posix())

            dec_path = ckpt_dec_dir.joinpath("dec_{:0>6d}.ckpt".format(i))
            torch.save(dec, dec_path.as_posix())

            recon_path = recon_dir.joinpath("recon_{:0>6d}.jpg".format(i))
            #todo: write images from array







