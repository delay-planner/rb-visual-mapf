from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper, habitat_env_load_fn
from pud.algos.visual_buffer import VisualReplayBuffer
from pud.visual_encode import VisualEncoder, VisualDecoder
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

scene = "scene_datasets/habitat-test-scenes/skokloster-castle.glb"
device = "cpu"
simulator_settings = dict(
    scene= "scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    width= 64,
    height= 64,
    default_agent= 0,
    sensor_height= 1.5,
)
apsp_path = "pud/envs/safe_habitatenv/apsps/skokloster/apsp.pickle"

env = habitat_env_load_fn(
            scene=scene,
            height=0,
            action_noise=1.0,
            terminate_on_timeout=False,
            simulator_settings=simulator_settings,
            max_episode_steps=20,
            apsp_path=apsp_path,
            gym_env_wrappers=(GoalConditionedHabitatPointWrapper,),
            device=device,
        )


latent_dimensions = 512
obs_dim = env.observation_space["observation"].shape  # type: ignore
goal_dim = env.observation_space["goal"].shape  # type: ignore
state_dim = (
    latent_dimensions * obs_dim[0] * 2
)  # For each image along cardinal directions and the same for the goal

action_dim = env.action_space.shape[0]  # type: ignore
max_action = float(env.action_space.high[0])  # type: ignore

buffer = VisualReplayBuffer(
    obs_dim=obs_dim,
    goal_dim=goal_dim,
    action_dim=action_dim,
    max_size=1000,
    )

venc = VisualEncoder()
vdec = VisualDecoder()

opt = torch.optim.Adam(params=list(venc.parameters()) + list(vdec.parameters()), lr=1e-3)

gpu = "cuda:1"

device=torch.device(gpu)

print("using GPU: {}".format(torch.cuda.get_device_name("gpu")))

venc.to(device)
vdec.to(device)

train_batch_size = 1
train_itrs = int(1e6)

loss_fn = nn.MSELoss(reduction="mean")

tb = SummaryWriter(log_dir="temp/")

for itr in range(train_itrs):
    # Note: this is NOT goal conditioned, only pretrain the state encoder and decoder
    state_batch = torch.zeros(train_batch_size*4, 4, 64, 64)

    for i in range(train_batch_size):
        state = env.reset()
        obs = state["observation"]
        obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2).float() # 4,4,64,64
        state_batch[i*4:i*4+4] = obs_t

    # batch encode
    state_batch = state_batch.to(device)
    emb_batch = venc(state_batch)
    reconstruction_batch = vdec(emb_batch)

    loss = loss_fn(reconstruction_batch, state_batch)
    #loss_flatten = torch.flatten(loss, start_dim=1)
    #loss_per_img = torch.mean(loss_flatten, dim=-1)
    #loss_avg = torch.mean(loss_per_img)
    loss.backward()
    opt.step()
    tb.add_scalar(tag="Reconstruction MSE Mean", scalar_value=loss.item(), global_step=itr)

    #print("itr {}: {:.2f}".format(itr, loss))



