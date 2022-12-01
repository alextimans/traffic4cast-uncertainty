# base code from https://github.com/iarai/NeurIPS2021-traffic4cast
# which is a copy from https://github.com/mie-lab/traffic4cast/blob/aea6f90e8884c01689c84255c99e96d2b58dc470/models/unet.py
# which is based on https://github.com/jvanvugt/pytorch-unet.

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 depth: int = 5,
                 layer_out_pow2: int = 6,
                 padding: bool = False,
                 batch_norm: bool = False,
                 up_mode: str = "upconv"):

        """
        Using the default arguments will yield the exact version used
        in the original U-Net paper (Ronneberger et al. 2015).
    
        Parameters
        ----------
        in_channels: int 
            Nr of input channels.
        out_channels: int 
            Nr of output channels.
        depth: int
            Depth of the network.
        layer_out_pow2: int
            Multiple of 2 used to generate output channel counts for each layer.
            Nr of output channels in 1st layer is 2**layer_out_pow2.
            Nr of output channels in later layers is 2**(layer_out_pow2 + layer depth)
        padding: bool 
            If True, apply padding such that the input shape
            is the same as the output. This may introduce artifacts
        batch_norm: bool 
            Use BatchNorm after layers with an activation.
        up_mode: str 
            One of 'upconv' or 'upsample'.
            'upconv' will use transposed convolutions with learnable params.
            'upsample' will use bilinear upsampling.
        """

        super().__init__()

        self.depth = depth
        self.padding = padding

        # Down path (Convolutions)
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            layer_out = 2 ** (layer_out_pow2 + i)
            self.down_path.append(
                UNetConvBlock(in_size=prev_channels,
                              out_size=layer_out,
                              padding=padding,
                              batch_norm=batch_norm)
                )
            prev_channels = layer_out

        # Up path (Transpose convolutions)
        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            layer_out = 2 ** (layer_out_pow2 + i)
            self.up_path.append(
                UNetUpBlock(in_size=prev_channels,
                            out_size=layer_out,
                            up_mode=up_mode,
                            padding=padding,
                            batch_norm=batch_norm)
                )
            prev_channels = layer_out

        self.final = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        down_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        x_skip = []

        for i, down_block in enumerate(self.down_path):
            x = down_block(x)
            if i != (self.depth - 1):
                x_skip.append(x) # Skip connections
                x = down_pool(x)

        x_skip.reverse()
        for i, up_block in enumerate(self.up_path):
            x = up_block(x, x_skip[i])

        out = self.final(x)

        return out


class UNetConvBlock(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 padding: bool,
                 batch_norm: bool):

        super(UNetConvBlock, self).__init__()
        block = []
        activation = nn.ReLU()

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(activation)
        if batch_norm: # Design choice: BN after activation
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(activation)
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x) -> torch.Tensor:
        out = self.block(x)

        return out


class UNetUpBlock(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 up_mode: str,
                 padding: bool,
                 batch_norm: bool):

        super(UNetUpBlock, self).__init__()

        assert up_mode in ("upconv", "upsample"), "Select up_mode from ['upconv', 'upsample']"
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def _center_crop(self, x, target_size):
        _, _, x_height, x_width = x.size()
        _, _, target_height, target_width = target_size

        diff_y = (x_height - target_height) // 2
        diff_x = (x_width - target_width) // 2

        crop = x[:, :,
                 diff_y:(diff_y + target_height),
                 diff_x:(diff_x + target_width)]

        return crop

    def forward(self, x, x_skip) -> torch.Tensor:
        x = self.up(x)
        x_skip_crop = self._center_crop(x_skip, target_size=x.shape)
        out = torch.cat([x, x_skip_crop], 1) # Should it be [x_skip_crop, x] ?
        out = self.conv_block(out)

        return out


class UNetTransformer:
    
    """
    Transformer for T4CDataset <-> UNet interaction.
    
    unet_pre_transform:
        - stack time and channels into one dimension
        - zeropad2d data to ensure same input and output sizes for UNet
        since rounding errors for uneven dimensions occur;
        pad input with zeroes s.t. (H, W) -> (H + pad, W + pad)
    
    unet_post_transform:
        - unstack time and channel dimensions
        - re-crop to original dimensions
    """

    @staticmethod
    def unet_pre_transform(
            data: Union[np.ndarray, torch.Tensor],
            zeropad2d: Optional[Tuple[int, int, int, int]] = None,
            stack_channels_on_time: bool = True,
            batch_dim: bool = False,
            from_numpy: bool = False,
            **kwargs) -> torch.Tensor:

        if from_numpy:
            data = torch.from_numpy(data).to(dtype=torch.float)

        if not batch_dim: # e.g. (12, H, W, Ch) -> (1, 12, H, W, Ch)
            data = torch.unsqueeze(data, dim=0)

        if stack_channels_on_time:
            data = UNetTransformer.transform_stack_channels_on_time(data)

        if zeropad2d is not None:
            zeropad2d = torch.nn.ZeroPad2d(zeropad2d)
            data = zeropad2d(data)

        if not batch_dim: # e.g. (1, 12, H, W, Ch) -> (12, H, W, Ch)
            data = torch.squeeze(data, dim=0)

        return data

    @staticmethod
    def unet_post_transform(
            data: torch.Tensor,
            crop: Optional[Tuple[int, int, int, int]] = None,
            unstack_channels_on_time: bool = True,
            batch_dim: bool = False,
            **kwargs) -> torch.Tensor:

        if not batch_dim:
            data = torch.unsqueeze(data, dim=0)

        if crop is not None:
            _, _, height, width = data.shape
            left, right, top, bottom = crop
            right = width - right
            bottom = height - bottom
            data = data[:, :, top:bottom, left:right]

        if unstack_channels_on_time:
            data = UNetTransformer.transform_unstack_channels_on_time(data)

        if not batch_dim:
            data = torch.squeeze(data, dim=0)

        return data

    @staticmethod
    def transform_stack_channels_on_time(data: torch.Tensor) -> torch.Tensor:
        
        """
        Combine time and channel dimensions:
        e.g. (k, 12, H, W, Ch) -> (k, 12 * Ch, H, W)
        """

        batch, time_steps, height, width, channels = data.shape

        # (k, 12, H, W, Ch) -> (k, 12, Ch, H, W)
        data = torch.movedim(data, 4, 2) #torch.moveaxis

        # (k, 12, Ch, H, W) -> (k, 12 * Ch, H, W)
        data = torch.reshape(data, (batch, time_steps * channels, height, width))

        return data

    @staticmethod
    def transform_unstack_channels_on_time(data: torch.Tensor,
                                           channels: int = 8) -> torch.Tensor:

        """
        Uncombine time and channel dimensions:
        e.g. (k, 12 * Ch, H, W) -> (k, 12, H, W, Ch)
        """

        batch, stacked, height, width = data.shape

        time_steps = int(stacked / channels)

        # (k, 12 * Ch, H, W) -> (k, 12, Ch, H, W)
        data = torch.reshape(data, (batch, time_steps, channels, height, width))

        # (k, 12, Ch, H, W) -> (k, 12, H, W, Ch)
        data = torch.movedim(data, 2, 4) #torch.moveaxis

        return data
