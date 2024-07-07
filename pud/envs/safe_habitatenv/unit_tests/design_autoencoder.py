from pud.envs.habitat_navigation_env import GoalConditionedHabitatPointWrapper, habitat_env_load_fn, HabitatNavigationEnv
from pud.algos.visual_buffer import VisualReplayBuffer
from pud.visual_models import VisualEncoder, VisualDecoder
import torch
from torch import nn
from pud.algos.data_struct import inp_to_device
import numpy as np

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

state = env.reset()
obs = state["observation"]
#obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2).float()


class VisualEncoder(nn.Module):
    def __init__(self,
            in_channels:int=4,
            embedding_size:int=256,
            act_fn=nn.SELU,
            ):
        """
        conv are performed on individual images (4 directions)
        -> image embeddings
        state embedding <- MLP(4 x image embeddings)
        """
        super(VisualEncoder, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16, 
                kernel_size=8, 
                stride=4
            ),  # 64x64 -> 15x15
            act_fn(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4),  # 15x15 -> 3x3
            act_fn(),
            nn.Flatten(),
        )
        # 4 direction obs in batch dim x embedding size
        self.l1 = nn.Linear(4*32*3*3, embedding_size)

    def forward(self, x):
        out = self.conv_net(x)
        num_cat_channels, _ = out.shape
        assert num_cat_channels%4 == 0
        batch_size = int(num_cat_channels/4)
        out = out.reshape(batch_size, -1)
        out = self.l1(out)
        return out
    
class VisualDecoder (nn.Module):
    def __init__(self, 
            input_channel:int=4,
            embedding_size:int=256,
            act_fn=nn.SELU,
            ):
        """
        4 * image embeddings <- MLP (state embedding)
        each image emedding is deconved into full images (4x64x64)
        """
        super(VisualDecoder, self).__init__()
        self.l1 = nn.Linear(embedding_size, 4*32*3*3)
        self.deconv1 = nn.ConvTranspose2d(
                            in_channels=32,
                            out_channels=16,
                            kernel_size=4,
                            stride=4,)
        
        self.output_size_1 = torch.Size([-1, 16, 15, 15])
            
        self.a1 = act_fn()
        self.deconv2 = nn.ConvTranspose2d(
                            in_channels=16,
                            out_channels=input_channel,
                            kernel_size=8,
                            stride=4,
                            )
        # batch_size, num_channel, image sizes
        self.output_size_2 = torch.Size([-1, 4, 64, 64])

    def forward(self, emb):
        batch_size, emb_size = emb.shape
        out = self.l1(emb)
        out = out.reshape([batch_size*4, 32, 3, 3])
        out = self.deconv1(out, output_size=self.output_size_1)
        out = self.deconv2(out, output_size=self.output_size_2)
        return out
        
import IPython
IPython.embed(colors="Linux")

# --- test batch encoding ----

oo1 = torch.rand(12, 4, 64, 64).float()
oo2 =torch.rand(4, 4, 64, 64).float()

e1 = VisualEncoder()
d1 = VisualDecoder()

emb = e1(oo1)
roo1 = d1(emb)


e1 = e1.float()
tmp = e1(oo1)

e1.get_latent_state(state, torch.device("cpu"))

emb = e1(state)
emb.shape

rtol = 1e-4
with torch.no_grad():
    eq1 = torch.allclose(
        e1(oo1)[0:1], e1(oo1[:4]), rtol=rtol
    )

    eq2 = torch.allclose(
        e1(oo1)[1:2], e1(oo1[4:8]), rtol=rtol
    )

    eq3 = torch.allclose(
        e1(oo1)[2:3], e1(oo1[8:]), rtol=rtol
    )

    eq4 = torch.allclose(
        e1(oo1)[1:2], e1(oo2), rtol=rtol
    )

    print(eq1, eq2, eq3, eq4)


e1(oo2)

# design decoder
# decoder seems not need non-linear activation
# reference: https://stackoverflow.com/questions/54313572/why-is-there-no-activation-function-in-upsampling-layer-of-u-net

import IPython
IPython.embed(colors="LightBG")

emb = e1(oo1) # batch_size (e.g.,12/4=3), 256

# make sure the embedding is image-specific, even for goal-conditioned encoders
input_channels = 4
conv1 = nn.Sequential(
    nn.Conv2d(
        in_channels=input_channels,
        out_channels=16, 
        kernel_size=8, 
        stride=4
    ),  # 64x64 -> 15x15
)

conv2 =  nn.Conv2d(16, 32, kernel_size=4, stride=4)  # 15x15 -> 3x3

obs_t.shape

obs_t = oo1
out1_1 = conv1(obs_t)
out1_1.shape  # (3x4), 16, 15, 15
out1_1_2 = conv2(out1_1)

#l1_size = list(out1_1_2.shape)[:] # 4, 32, 3, 3
#l1 = nn.Linear(np.prod(l1_size), 256)

l1 = nn.Linear(32*3*3, )

out1_2 = torch.flatten(out1_1_2, start_dim=1)

out1_2r = out1_2.reshape(2, 4*288)

out1_2.shape

torch.allclose(
    torch.flatten(out1_2[:4]),
    out1_2r[0]
)

torch.allclose(
    torch.flatten(out1_2[4:]),
    out1_2r[1]
)



## decode
l2 = nn.Linear(256, 4*32*3*3)
out2 = l2(emb)

#torch.Size([4, 32, 3, 3])
out2_1 = out2.reshape([-1, 32, 3, 3])
out2_1.shape # 3*4, 32, 3, 3

# verify reshape makes sense
torch.allclose(out2[0].flatten(), out2_1[0:4].flatten())

ctn1 = nn.ConvTranspose2d(
    in_channels=32,
    out_channels=16,
    kernel_size=4,
    stride=4,
)
out2_2 = ctn1(out2_1, output_size=torch.Size([-1, 16, 15, 15]))
out2_2.shape

ctn2 = nn.ConvTranspose2d(
    in_channels=16,
    out_channels=4,
    kernel_size=8,
    stride=4,
)

out2_3 = ctn2(out2_2, output_size=[-1, 4, 64, 64])
out2_3.shape


## auto encoder
#class TestGoalConditionedHabitatEnv(unittest.TestCase):
#    def setUp(self):
#        import IPython
#        IPython.embed(colors="Linux")

#        env

#    def test_something(self):
#        pass

#if __name__ == "__main__":
#    unittest.main()
