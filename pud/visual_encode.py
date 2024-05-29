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