import torch
from torch import nn
from typing import Union
from torch.nn import functional as F
from pud.utils import variance_initializer_


class VisualEncoder(nn.Module):
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

    def forward(self, x):
        out = self.conv_net(x)
        num_cat_channels, _ = out.shape
        assert num_cat_channels%4 == 0
        batch_size = int(num_cat_channels/4)
        out = out.reshape(batch_size, -1)
        out = self.l1(out)
        return out
    
    #def reset_parameters(self):
    #    pass
