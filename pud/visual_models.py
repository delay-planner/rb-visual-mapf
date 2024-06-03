import torch
from torch import nn
from typing import Union
from torch.nn import functional as F
import numpy as np
from pud.utils import variance_initializer_
from pud.algos.data_struct import inp_to_device
from pud.ddpg import UVFDDPG


class VisualEncoder(nn.Module):
    """
    conv are performed on individual images (4 directions)
    -> image embeddings
    state embedding <- MLP(4 x image embeddings)
    """
    def __init__(self,
            in_channels:int=4,
            embedding_size:int=256,
            act_fn=nn.SELU,
            ):
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

    def forward(self, x:torch.Tensor):
        x = x.permute(0, 3, 1, 2) # batch_dim, channel_dim, *image_size
        out = self.conv_net(x)
        num_cat_channels, _ = out.shape
        assert num_cat_channels%4 == 0
        batch_size = int(num_cat_channels/4)
        out = out.reshape(batch_size, -1)
        out = self.l1(out)
        return out
    
    def get_latent_state(self, 
            image_state:Union[torch.Tensor, dict, np.ndarray],
            device:torch.device):
        """convert image state to latent state"""
        # move input to torch device
        inp = inp_to_device(image_state, device=device)

        if torch.is_tensor(inp):
            return self.forward(inp)
        elif isinstance(inp, dict):
            latent_obs = self.forward(inp["observation"].float())
            latent_goal = self.forward(inp["goal"].float())
            latent_state = {}
            for key in inp:
                if key == "observation":
                    latent_state[key] = latent_obs
                elif key == "goal":
                    latent_state[key] = latent_goal
                else:
                    latent_state[key] = inp[key]
            return latent_state

class VisualDecoder (nn.Module):
    def __init__(self, emb_size:int=256):
        super(VisualDecoder, self).__init__()

        self.l_emb = nn.Linear(emb_size, 4*32*3*3)
        self.deconv1 = nn.ConvTranspose2d(
                            in_channels=32,
                            out_channels=16,
                            kernel_size=4,
                            stride=4,
                        )
        self.deconv2 = nn.ConvTranspose2d(
                    in_channels=16,
                    out_channels=4,
                    kernel_size=8,
                    stride=4,
                    )

    def forward(self, emb:torch.Tensor):
        batch_size, emb_dim = emb.shape
        out = self.l_emb(emb)
        #torch.Size([4, 32, 3, 3])
        out = out.reshape([batch_size*4, 32, 3, 3])
        out = self.deconv1(out, output_size=torch.Size([4, 16, 15, 15]))
        out = self.deconv2(out, output_size=[batch_size*4, 4, 64, 64])
        # batch_dim, channel_dim, image size 1, image size 2
        # to 
        # batch_dim, image size 1, image size 2, channel dim
        out = out.permute(0, 2, 3, 1) # batch_dim, channel_dim, *image_size
        return out
    


class VisualUVFDDPG (UVFDDPG):
    def __init__(
            self,
            # encoder args
            in_channels:int,
            embedding_size:int,
            act_fn,
            device:str,
            # policy args
            uvfddpg_kwargs:dict,
            ):
        super(VisualUVFDDPG, self).__init__(**uvfddpg_kwargs)
        self.encoder = VisualEncoder(
            in_channels=in_channels,
            embedding_size=embedding_size,
            act_fn=act_fn,
        )
        self.device = torch.device(device)

    def select_action(self, state):
        latent_state = self.encoder.get_latent_state(state, device=self.device)
        with torch.no_grad():
            return self.actor(latent_state).cpu().detach().numpy().flatten()

    def get_q_values(self, state, aggregate='mean'):
        latent_state = self.encoder.get_latent_state(state, device=self.device)
        return super().get_q_values(latent_state, aggregate)